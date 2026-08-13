"""Deterministic road-context scenario substrate for ROTR.

Extends the platform's scene conventions (REUSE, read-only:
bevfusion.scenes DT / EGO_SPEED / seeded default_rng discipline) with what
bevfusion deliberately does not model: lane geometry with permitted
maneuvers, intersection topology (controlled/uncontrolled, signals/signs),
crosswalks, and multi-actor right-of-way states + intent.

A bank is a pure function of (n_scenarios, seed, model_version): the same
inputs produce byte-identical scenarios. `model_version` selects a stack
profile that determines — deterministically per scenario — whether the
stack COMMITS each planted violation opportunity; this is how
baseline-vs-candidate regression banks are materialized (the bank is "the
log of that software driving the same scenario set", i.e. resimulation).

Planted truth includes NON-violations that superficially resemble
violations (legal assertive merge; ego yielding while holding right-of-way)
so the rule engine's false-accusation rate is measurable.

All numeric thresholds/kinematics here are ILLUSTRATIVE (synthetic
substrate); the generator's contract, not its constants, is the deliverable.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# REUSE (landed, read-only): platform time base and nominal ego speed.
from sensorflow.bevfusion.scenes import DT, EGO_SPEED

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr.models import (
    Actor, ActorObservation, ActorState, Crosswalk, EgoState, Environment,
    Lane, PlannedPoint, PlantedTruth, Provenance, ROTRScenario, RoadContext,
)

N_FRAMES = 70
X0 = -10.0
V_NOM = float(EGO_SPEED)          # 10 m/s
LANE_W = 3.5

# Intersection geometry shared by intersection templates.
STOP_LINE_X = 38.0
CW_X_MIN, CW_X_MAX = 39.5, 44.5   # crosswalk span along x
BOX_X_MIN, BOX_X_MAX = 39.5, 52.0

DATASET_VERSION = "rotr-bank-1"
GENERATOR_VERSION = "rotr-scenes-1.0.0"

# Stack profiles: per-cause commit probability for planted violation
# opportunities. Deterministic per scenario (seeded rng draw).
MODEL_PROFILES: Dict[str, Dict[str, float]] = {
    "stack-v1": {"perception": 0.90, "prediction": 0.90, "planning": 0.50,
                 "localization": 1.00, "map": 1.00, "control": 0.80,
                 "data_label": 1.00},
    "stack-v2-improved": {"perception": 0.25, "prediction": 0.90,
                          "planning": 0.50, "localization": 1.00,
                          "map": 1.00, "control": 0.80, "data_label": 1.00},
    "stack-v3-planning-regression": {"perception": 0.25, "prediction": 0.90,
                                     "planning": 1.00, "localization": 1.00,
                                     "map": 1.00, "control": 0.80,
                                     "data_label": 1.00},
}
DEFAULT_MODEL = "stack-v1"


# ------------------------------------------------------------ contexts


def _lane(lane_id: str, y: float, maneuvers: List[str],
          restricted_to: Optional[str] = None) -> Lane:
    return Lane(lane_id=lane_id, center_y=y, permitted_maneuvers=maneuvers,
                restricted_to=restricted_to)


def _ctx_uncontrolled() -> RoadContext:
    return RoadContext(
        context_id="uncontrolled-xing", intersection_type="uncontrolled",
        control="none", stop_line_x=STOP_LINE_X,
        intersection_x_min=BOX_X_MIN, intersection_x_max=BOX_X_MAX,
        lanes=[_lane("L0", 0.0, ["STRAIGHT", "LANE_CHANGE"]),
               _lane("L-1", -LANE_W, ["STRAIGHT", "LANE_CHANGE"])],
        crosswalks=[Crosswalk(crosswalk_id="cw1", x_min=CW_X_MIN, x_max=CW_X_MAX)])


def _ctx_bus(restricted: bool) -> RoadContext:
    return RoadContext(
        context_id="bus-corridor", intersection_type="none", control="none",
        lanes=[_lane("L0", 0.0, ["STRAIGHT", "LANE_CHANGE"]),
               _lane("LBUS", -LANE_W, ["STRAIGHT"],
                     restricted_to="bus" if restricted else None)])


def _ctx_turnlane() -> RoadContext:
    return RoadContext(
        context_id="turn-lane-xing", intersection_type="uncontrolled",
        control="none", stop_line_x=STOP_LINE_X,
        intersection_x_min=BOX_X_MIN, intersection_x_max=BOX_X_MAX,
        lanes=[_lane("L0", 0.0, ["STRAIGHT", "LANE_CHANGE"]),
               _lane("LT", LANE_W, ["LEFT_TURN"])])


def _ctx_signal(mapped_signal: bool) -> RoadContext:
    return RoadContext(
        context_id="signal-xing", intersection_type="controlled",
        control="signal" if mapped_signal else "none",
        signal_state_for_ego="red" if mapped_signal else None,
        stop_line_x=STOP_LINE_X,
        intersection_x_min=BOX_X_MIN, intersection_x_max=BOX_X_MAX,
        lanes=[_lane("L0", 0.0, ["STRAIGHT"])],
        crosswalks=[Crosswalk(crosswalk_id="cw1", x_min=CW_X_MIN, x_max=CW_X_MAX)])


def _ctx_stopsign() -> RoadContext:
    return RoadContext(
        context_id="stopsign-xing", intersection_type="controlled",
        control="stop_sign", stop_line_x=STOP_LINE_X,
        intersection_x_min=BOX_X_MIN, intersection_x_max=BOX_X_MAX,
        lanes=[_lane("L0", 0.0, ["STRAIGHT"])],
        crosswalks=[Crosswalk(crosswalk_id="cw1", x_min=CW_X_MIN, x_max=CW_X_MAX)])


def _ctx_merge() -> RoadContext:
    return RoadContext(
        context_id="merge-corridor", intersection_type="none", control="none",
        lanes=[_lane("L0", 0.0, ["STRAIGHT", "LANE_CHANGE"]),
               _lane("L-1", -LANE_W, ["STRAIGHT", "LANE_CHANGE"])])


# ------------------------------------------------------------ trajectories


def _traj_const(v: float = V_NOM, y: float = 0.0) -> List[Dict]:
    return [{"t": fi * DT, "x": X0 + v * fi * DT, "y": y, "speed": v,
             "accel": 0.0} for fi in range(N_FRAMES)]


def _traj_stop(x_stop: float, decel: float = 3.0, v0: float = V_NOM,
               y: float = 0.0, resume_t: Optional[float] = None,
               resume_accel: float = 2.0) -> List[Dict]:
    """Cruise, brake to a stop at x_stop, optionally resume."""
    d_brake = v0 * v0 / (2 * decel)
    x_brake = x_stop - d_brake
    pts, x, v, a = [], X0, v0, 0.0
    for fi in range(N_FRAMES):
        t = fi * DT
        pts.append({"t": t, "x": x, "y": y, "speed": v, "accel": a})
        if v > 0 and x >= x_brake:
            a = -decel
        elif v <= 0 and resume_t is not None and t >= resume_t:
            a = resume_accel
        elif v <= 0:
            a = 0.0
        v = max(0.0, min(v0, v + a * DT))
        if v == 0.0:
            a = 0.0
        x += v * DT
    return pts


def _traj_rolling(x_line: float, v_min: float = 3.0, v0: float = V_NOM,
                  y: float = 0.0) -> List[Dict]:
    """Slow to v_min at the stop line, never stop, resume to v0."""
    decel = 2.5
    d = (v0 * v0 - v_min * v_min) / (2 * decel)
    x_brake = x_line - d
    pts, x, v, a = [], X0, v0, 0.0
    for fi in range(N_FRAMES):
        t = fi * DT
        pts.append({"t": t, "x": x, "y": y, "speed": v, "accel": a})
        if x >= x_brake and v > v_min and x < x_line:
            a = -decel
        elif x >= x_line and v < v0:
            a = 2.0
        else:
            a = 0.0
        v = max(v_min if x >= x_brake else 0.0, min(v0, v + a * DT))
        x += v * DT
    return pts


def _traj_lane_change(t_start: float, t_end: float, y_from: float,
                      y_to: float, v: float = V_NOM) -> List[Dict]:
    pts = []
    for fi in range(N_FRAMES):
        t = fi * DT
        if t <= t_start:
            y = y_from
        elif t >= t_end:
            y = y_to
        else:
            frac = (t - t_start) / (t_end - t_start)
            y = y_from + (y_to - y_from) * (1 - math.cos(math.pi * frac)) / 2
        pts.append({"t": t, "x": X0 + v * t, "y": y, "speed": v, "accel": 0.0})
    return pts


def _ego_states(pts: List[Dict], context: RoadContext,
                loc_offset_y: float = 0.0) -> List[EgoState]:
    """Attach true lane association + localization (believed) view."""
    out = []
    for p in pts:
        lane = _nearest_lane(context, p["y"])
        out.append(EgoState(
            t=p["t"], x=p["x"], y=p["y"], yaw=0.0, speed=p["speed"],
            accel=p["accel"], lane_id=lane.lane_id if lane else None,
            believed_x=p["x"], believed_y=p["y"] + loc_offset_y,
            believed_lane_id=(_nearest_lane(context, p["y"] + loc_offset_y).lane_id
                              if _nearest_lane(context, p["y"] + loc_offset_y)
                              else None)))
    return out


def _nearest_lane(context: RoadContext, y: float) -> Optional[Lane]:
    best, best_d = None, 1e9
    for lane in context.lanes:
        d = abs(lane.center_y - y)
        if d < best_d:
            best, best_d = lane, d
    return best if best_d <= LANE_W * 0.6 else None


def _planned(pts: List[Dict]) -> List[PlannedPoint]:
    return [PlannedPoint(t=p["t"], x=p["x"], y=p["y"], v=p["speed"])
            for p in pts]


# ------------------------------------------------------------ actors


def _observe(rng: np.random.Generator, st: Dict, cls: str,
             mode: str) -> Optional[ActorObservation]:
    """Plant the stack's observation of a GT actor state.

    mode: good | missed | late(handled by caller) | absent(None=telemetry gap)
    """
    if mode == "absent":
        return None
    if mode == "missed":
        return ActorObservation(detected=False)
    return ActorObservation(
        detected=True,
        x=st["x"] + float(rng.normal(0, 0.12)),
        y=st["y"] + float(rng.normal(0, 0.08)),
        class_name=cls)


def _actor(rng: np.random.Generator, actor_id: str, cls: str,
           dims: List[float], row: str, intent: str,
           predicted_intent: Optional[str],
           state_fn, obs_mode_fn) -> Actor:
    states = []
    for fi in range(N_FRAMES):
        t = fi * DT
        x, y, vx, vy, yaw = state_fn(t)
        st = {"x": x, "y": y}
        states.append(ActorState(
            t=t, x=x, y=y, vx=vx, vy=vy, yaw=yaw,
            observation=_observe(rng, st, cls, obs_mode_fn(t))))
    return Actor(actor_id=actor_id, class_name=cls, dims=dims,
                 right_of_way=row, intent=intent,
                 predicted_intent=predicted_intent, states=states)


def _crossing_ped_fn(x: float = 44.9, y0: float = 6.0, vy: float = -1.4,
                     y_end: float = -7.0):
    # x=44.9 places the crossing at the far edge of the crosswalk so a
    # non-yielding ego produces a deterministic NEAR-miss (sub-meter
    # clearance), not a synthetic collision.
    def fn(t: float) -> Tuple[float, float, float, float, float]:
        y = max(y_end, y0 + vy * t)
        moving = y > y_end
        return x, y, 0.0, vy if moving else 0.0, -math.pi / 2
    return fn


def _waiting_ped_fn(x: float = 42.5, y: float = 5.5):
    def fn(t: float):
        return x, y, 0.0, 0.0, -math.pi / 2
    return fn


def _crossing_vehicle_fn(x: float = 45.5, y0: float = 20.0, vy: float = -2.8):
    def fn(t: float):
        return x, y0 + vy * t, 0.0, vy, -math.pi / 2
    return fn


def _trailing_vehicle_fn(x0: float, vx: float = 13.0,
                         brake_t: Optional[float] = None,
                         brake_a: float = 2.5, v_min: float = 8.5):
    def fn(t: float):
        if brake_t is None or t <= brake_t:
            return x0 + vx * t, -LANE_W, vx, 0.0, 0.0
        tb = t - brake_t
        v = max(v_min, vx - brake_a * tb)
        t_ramp = (vx - v_min) / brake_a
        if tb <= t_ramp:
            x = x0 + vx * brake_t + vx * tb - 0.5 * brake_a * tb * tb
        else:
            x = (x0 + vx * brake_t + vx * t_ramp - 0.5 * brake_a * t_ramp ** 2
                 + v_min * (tb - t_ramp))
        return x, -LANE_W, v, 0.0, 0.0
    return fn


ALWAYS_GOOD = lambda t: "good"          # noqa: E731
NEVER_SEEN = lambda t: "missed"         # noqa: E731
TELEMETRY_GAP = lambda t: "absent"      # noqa: E731


# ------------------------------------------------------------ templates
#
# Each template returns the scenario pieces given (rng, committed).
# `cause` is the planted causal layer for the committed variant.

VEH_DIMS = [4.5, 1.9, 1.6]
PED_DIMS = [0.6, 0.6, 1.75]


def _tpl_fail_yield_ped(rng, committed: bool, cause: str) -> Dict:
    ctx = _ctx_uncontrolled()
    obs_mode = NEVER_SEEN if (committed and cause == "perception") else \
        (TELEMETRY_GAP if (committed and cause == "data_label") else ALWAYS_GOOD)
    predicted = None if cause == "data_label" else \
        ("WAITING" if (committed and cause == "prediction") else "CROSSING")
    ped = _actor(rng, "ped-1", "pedestrian", PED_DIMS, "HAS_ROW", "CROSSING",
                 predicted, _crossing_ped_fn(), obs_mode)
    if committed:
        exec_pts = _traj_const()
        plan_pts = exec_pts
    else:
        exec_pts = _traj_stop(STOP_LINE_X - 0.5, resume_t=6.5)
        plan_pts = exec_pts
    ego = _ego_states(exec_pts, ctx)
    if committed and cause == "data_label":
        # Planted record-integrity fault: disordered timestamps mid-approach.
        for k in range(30, 34):
            ego[k].t = ego[k].t - 0.35
    return {"map_context": ctx, "actual_context": ctx, "ego": ego,
            "planned": _planned(plan_pts), "actors": [ped],
            "expected_rule": "R-YIELD-PED-01",
            "description": "pedestrian crossing at uncontrolled intersection; "
                           + ("ego proceeds without yielding" if committed
                              else "ego yields correctly")}


def _tpl_restricted_path(rng, committed: bool, cause: str) -> Dict:
    actual = _ctx_bus(restricted=True)
    mapped = _ctx_bus(restricted=(cause != "map"))
    if committed:
        pts = _traj_lane_change(2.0, 3.5, 0.0, -LANE_W)
    else:
        pts = _traj_const()
    return {"map_context": mapped, "actual_context": actual,
            "ego": _ego_states(pts, actual), "planned": _planned(pts),
            "actors": [],
            "expected_rule": "R-PATH-RESTRICT-02",
            "description": "bus-restricted lane; "
                           + ("ego enters restricted path" if committed
                              else "ego stays in general lane")
                           + (" (restriction missing from map)"
                              if cause == "map" else "")}


def _tpl_wrong_lane(rng, committed: bool, cause: str) -> Dict:
    ctx = _ctx_turnlane()
    # Ego actually drives the left-turn-only lane straight through; its
    # localization believes it is in the through lane (offset -LANE_W).
    pts = _traj_const(y=LANE_W)
    ego = _ego_states(pts, ctx, loc_offset_y=-LANE_W)
    return {"map_context": ctx, "actual_context": ctx, "ego": ego,
            "planned": _planned(pts), "actors": [],
            "expected_rule": "R-LANE-MANEUVER-03",
            "description": "ego proceeds straight from a left-turn-only lane "
                           "(wrong lane association from localization error)"}


def _tpl_signal_conflict(rng, committed: bool, cause: str) -> Dict:
    actual = _ctx_signal(mapped_signal=True)
    mapped = _ctx_signal(mapped_signal=(cause != "map"))
    obs_mode = NEVER_SEEN if (committed and cause == "perception") else ALWAYS_GOOD
    veh = _actor(rng, "xveh-1", "vehicle", VEH_DIMS, "HAS_ROW", "CROSSING",
                 "CROSSING", _crossing_vehicle_fn(), obs_mode)
    if committed:
        pts = _traj_const()
    else:
        pts = _traj_stop(STOP_LINE_X - 0.5)
    return {"map_context": mapped, "actual_context": actual,
            "ego": _ego_states(pts, actual), "planned": _planned(pts),
            "actors": [veh],
            "expected_rule": "R-INT-CONFLICT-04",
            "description": "red signal with conflicting cross traffic; "
                           + ("ego enters on red" if committed
                              else "ego stops at the line")
                           + (" (signal unmapped)" if cause == "map" else "")}


def _tpl_unsafe_merge(rng, committed: bool, cause: str) -> Dict:
    ctx = _ctx_merge()
    if committed:
        veh = _actor(rng, "mveh-1", "vehicle", VEH_DIMS, "HAS_ROW",
                     "PROCEEDING", "PROCEEDING",
                     _trailing_vehicle_fn(-31.5, brake_t=4.6), ALWAYS_GOOD)
        pts = _traj_lane_change(3.0, 4.5, 0.0, -LANE_W)
    else:
        veh = _actor(rng, "mveh-1", "vehicle", VEH_DIMS, "HAS_ROW",
                     "PROCEEDING", "PROCEEDING",
                     _trailing_vehicle_fn(-58.0), ALWAYS_GOOD)
        pts = _traj_const()
    return {"map_context": ctx, "actual_context": ctx,
            "ego": _ego_states(pts, ctx), "planned": _planned(pts),
            "actors": [veh],
            "expected_rule": "R-MERGE-GAP-05",
            "description": "merge ahead of right-of-way traffic with "
                           + ("insufficient gap" if committed else "safe gap")}


def _tpl_stop_overshoot(rng, committed: bool, cause: str) -> Dict:
    ctx = _ctx_stopsign()
    ped = _actor(rng, "ped-w1", "pedestrian", PED_DIMS, "NONE", "WAITING",
                 "WAITING", _waiting_ped_fn(), ALWAYS_GOOD)
    plan_pts = _traj_stop(STOP_LINE_X - 0.5, resume_t=5.6)
    if committed:
        exec_pts = _traj_rolling(STOP_LINE_X)      # actuation under-brakes
    else:
        exec_pts = plan_pts
    return {"map_context": ctx, "actual_context": ctx,
            "ego": _ego_states(exec_pts, ctx), "planned": _planned(plan_pts),
            "actors": [ped],
            "expected_rule": "R-STOP-06",
            "description": "stop sign; " + ("planned stop, executed roll-through "
                                            "(control tracking error)"
                                            if committed else "full stop made")}


def _tpl_legal_merge(rng, committed: bool, cause: str) -> Dict:
    d = _tpl_unsafe_merge(rng, committed=False, cause="")
    # Assertive but legal: the merge happens, with a sufficient gap.
    ctx = d["actual_context"]
    pts = _traj_lane_change(3.0, 4.5, 0.0, -LANE_W)
    d.update({"ego": _ego_states(pts, ctx), "planned": _planned(pts),
              "expected_rule": None,
              "description": "legal assertive merge (sufficient gap; planted "
                             "NON-violation that superficially resembles one)"})
    return d


def _tpl_yield_with_row(rng, committed: bool, cause: str) -> Dict:
    ctx = _ctx_uncontrolled()
    ped = _actor(rng, "ped-c1", "pedestrian", PED_DIMS, "NONE", "WAITING",
                 "WAITING", _waiting_ped_fn(), ALWAYS_GOOD)
    pts = _traj_stop(STOP_LINE_X - 0.5, resume_t=5.8)
    return {"map_context": ctx, "actual_context": ctx,
            "ego": _ego_states(pts, ctx), "planned": _planned(pts),
            "actors": [ped],
            "expected_rule": None,
            "description": "ego yields although it holds right-of-way (curb "
                           "pedestrian is waiting; planted NON-violation)"}


def _tpl_green_proceed(rng, committed: bool, cause: str) -> Dict:
    ctx = _ctx_signal(mapped_signal=True)
    ctx.signal_state_for_ego = "green"
    pts = _traj_const()
    return {"map_context": ctx, "actual_context": ctx.model_copy(deep=True),
            "ego": _ego_states(pts, ctx), "planned": _planned(pts),
            "actors": [],
            "expected_rule": None,
            "description": "normal green-light transit (planted NON-violation)"}


# (template fn, planted kind, cause layer or None for non-violations)
TEMPLATES = [
    (_tpl_fail_yield_ped, "fail_yield_pedestrian", "perception"),
    (_tpl_fail_yield_ped, "fail_yield_pedestrian", "planning"),
    (_tpl_fail_yield_ped, "fail_yield_pedestrian", "prediction"),
    (_tpl_restricted_path, "restricted_path_entry", "planning"),
    (_tpl_restricted_path, "restricted_path_entry", "map"),
    (_tpl_wrong_lane, "wrong_lane_association", "localization"),
    (_tpl_signal_conflict, "intersection_conflict", "map"),
    (_tpl_signal_conflict, "intersection_conflict", "perception"),
    (_tpl_unsafe_merge, "unsafe_merge", "planning"),
    (_tpl_stop_overshoot, "stop_overshoot", "control"),
    (_tpl_fail_yield_ped, "fail_yield_pedestrian", "data_label"),
    (_tpl_legal_merge, "legal_assertive_merge", None),
    (_tpl_yield_with_row, "yield_with_right_of_way", None),
    (_tpl_green_proceed, "green_proceed", None),
]

VISIBILITIES = ["low", "clear", "clear"]
LIGHTINGS = ["day", "dusk", "night"]
WEATHERS = ["clear", "clear", "rain"]


def generate_bank(n_scenarios: int = 28, seed: int = 7,
                  model_version: str = DEFAULT_MODEL) -> List[ROTRScenario]:
    """Deterministic scenario bank; same inputs -> identical output."""
    if model_version not in MODEL_PROFILES:
        raise ValueError(f"unknown model_version {model_version!r}; known: "
                         f"{sorted(MODEL_PROFILES)}")
    profile = MODEL_PROFILES[model_version]
    bank_id = bank_id_for(n_scenarios, seed, model_version)
    out: List[ROTRScenario] = []
    for i in range(n_scenarios):
        # Same per-sequence seeding discipline as bevfusion.scenes.
        rng = np.random.default_rng(seed * 100003 + i * 7919)
        tpl_fn, kind, cause = TEMPLATES[i % len(TEMPLATES)]
        is_opportunity = cause is not None
        committed = bool(is_opportunity
                         and rng.random() < profile.get(cause, 1.0))
        parts = tpl_fn(rng, committed, cause or "")
        env = Environment(visibility=VISIBILITIES[i % 3],
                          lighting=LIGHTINGS[i % 3],
                          weather=WEATHERS[i % 3])
        sid = f"{bank_id}-sc{i:03d}"
        out.append(ROTRScenario(
            scenario_id=sid, bank_id=bank_id, seed=seed,
            description=parts["description"], environment=env,
            map_context=parts["map_context"],
            actual_context=parts["actual_context"],
            ego=parts["ego"], planned=parts["planned"],
            actors=parts["actors"],
            planted=PlantedTruth(
                kind=kind, is_violation_opportunity=is_opportunity,
                committed=committed,
                expected_rule_id=parts["expected_rule"] if committed else None,
                cause_layer=cause if committed else None,
                notes=parts["description"]),
            provenance=Provenance(
                scenario_id=sid, dataset_version=DATASET_VERSION,
                model_version=model_version,
                software_version=f"{SOFTWARE_VERSION}/{GENERATOR_VERSION}",
                source="SYNTHETIC", confidence=1.0)))
    return out


def bank_id_for(n_scenarios: int, seed: int, model_version: str) -> str:
    blob = f"{n_scenarios}|{seed}|{model_version}|{GENERATOR_VERSION}"
    return "bank-" + hashlib.sha256(blob.encode()).hexdigest()[:10]
