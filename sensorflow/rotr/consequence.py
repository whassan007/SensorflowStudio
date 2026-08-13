"""Behavioral-consequence classification via counterfactual replay.

For each violation we construct the trajectory the vehicle WOULD have
driven with the attributed layer's inputs corrected, and diff it against
the observed trajectory. The corrected trajectory is produced per primary
layer:

* control            -> the recorded plan itself (the plan was compliant;
                        correct the actuation): engine "planned-trajectory".
* perception /       -> re-plan with the corrected world view (ground-truth
  prediction            actors): IDM-style replay. REUSE: the planner is
                        nextgen.closedloop.plan_acceleration when importable
                        (nextgen is an in-progress workstream — guarded
                        import); otherwise a self-contained IDM fallback
                        reusing sensorflow.safety.ssam_ext math (landed).
* yield/signal/stop  -> compliant-envelope stop-at-line trajectory
  rules attributed      (a rule-compliant stack brakes to the line; the
  to planning/map/      synthetic stop is labeled engine
  localization/data     "compliant-envelope").
* lane-keeping rules -> stay-in-lane replay (target y = initial y).

Classes (thresholds ILLUSTRATIVE): NO_MATERIAL_CONSEQUENCE <
DEGRADED_COMFORT < PLANNER_INTERVENTION < SAFETY_CRITICAL, with measured
trajectory divergence, TTC, PET, min clearance, stopping distance, braking
intensity and lateral deviation on BOTH branches. Surrogate caveat: these
measures prioritize triage; they never gate a release alone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

# REUSE (landed, read-only): SSAM surrogate-safety math.
from sensorflow.safety.ssam_ext import (
    projected_ttc, rect_gap, rects_overlap, zone_grid_pet,
)

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr.models import (
    CounterfactualScenario, FailureAttribution, PlannerEvaluation, Provenance,
    ROTRScenario, ROTRViolation, SafetyAssessment,
)
from sensorflow.rotr.scenes import DT, V_NOM

CONSEQUENCE_VERSION = "rotr-consequence-1.0.0"

EGO_DIMS = (4.5, 1.9)
ACTUATION_TAU_S = 0.25

# REUSE nextgen's planner when importable (in-progress workstream: guarded).
try:
    from sensorflow.nextgen.closedloop import plan_acceleration as _ng_plan
    _PLANNER, PLANNER_ENGINE = _ng_plan, "nextgen.plan_acceleration"
except Exception:                                    # pragma: no cover
    _PLANNER, PLANNER_ENGINE = None, "rotr-idm-fallback"

STOP_RULES = {"R-YIELD-PED-01", "R-INT-CONFLICT-04", "R-STOP-06"}
LANE_RULES = {"R-PATH-RESTRICT-02", "R-LANE-MANEUVER-03", "R-MERGE-GAP-05"}


# ------------------------------------------------------------ fallback planner


def _fallback_plan(ego: Dict, detections: List[Dict]) -> Dict:
    """Self-contained IDM-style longitudinal command + projected-TTC braking
    (mirrors the reused planner's contract; ssam_ext math for conflicts)."""
    v = ego["v"]
    a_cmd = 2.5 * (1.0 - (v / max(V_NOM, 0.1)) ** 4)
    intervention = False
    ego_state = {"x": ego["x"], "y": ego["y"], "speed": v,
                 "heading": ego["heading"]}
    for det in detections:
        rel_x = det["x"] - ego["x"]
        if abs(det["y"] - ego["y"]) <= 1.6 and rel_x > 0:
            gap = max(rel_x - 0.5 * (EGO_DIMS[0] + det["dims"][0]), 0.5)
            dv = v - det["vx"]
            s_star = 2.0 + max(0.0, v * 1.5 + v * dv / (2 * math.sqrt(2.5 * 3.5)))
            a_idm = 2.5 * (1.0 - (v / max(V_NOM, 0.1)) ** 4 - (s_star / gap) ** 2)
            a_cmd = min(a_cmd, a_idm)
        spd = math.hypot(det["vx"], det["vy"])
        obj = {"x": det["x"], "y": det["y"], "speed": spd,
               "heading": math.atan2(det["vy"], det["vx"]) if spd else 0.0}
        ttc = projected_ttc(ego_state, EGO_DIMS, obj,
                            (det["dims"][0], det["dims"][1]),
                            look_ahead=4.0, dt=0.1)
        if ttc is not None:
            if ttc <= 1.5:
                a_cmd, intervention = -8.0, True
            elif ttc <= 4.0:
                a_cmd = min(a_cmd, -3.5 * (4.0 - ttc) / 2.5)
                intervention = True
    return {"a_cmd": float(np.clip(a_cmd, -8.0, 2.5)),
            "intervention": intervention, "reason": "fallback"}


def _planner(ego: Dict, detections: List[Dict]) -> Dict:
    if _PLANNER is not None:
        return _PLANNER(ego, detections)
    return _fallback_plan(ego, detections)


# ------------------------------------------------------------ replay engines


def _gt_detections(scenario: ROTRScenario, fi: int, ego_x: float) -> List[Dict]:
    out = []
    for a in scenario.actors:
        s = a.states[min(fi, len(a.states) - 1)]
        if s.x - ego_x < -5.0 or math.hypot(s.x - ego_x, s.y) > 75.0:
            continue
        out.append({"instance_id": a.actor_id, "class_name": a.class_name,
                    "x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy,
                    "dims": a.dims, "yaw": s.yaw})
    return out


def _idm_replay(scenario: ROTRScenario) -> Dict:
    """Re-plan with the CORRECTED world view (ground-truth actors)."""
    e0 = scenario.ego[0]
    n = len(scenario.ego)
    ego = {"x": e0.x, "y": e0.y, "v": e0.speed, "a": 0.0, "heading": 0.0}
    traj, intervention = [], False
    for step in range(n):
        t = step * DT
        plan = _planner(ego, _gt_detections(scenario, step, ego["x"]))
        intervention = intervention or bool(plan.get("intervention"))
        alpha = DT / (DT + ACTUATION_TAU_S)
        ego["a"] += alpha * (plan["a_cmd"] - ego["a"])
        ego["v"] = max(0.0, ego["v"] + ego["a"] * DT)
        ego["x"] += ego["v"] * DT
        traj.append({"t": round(t, 2), "x": round(ego["x"], 3),
                     "y": round(ego["y"], 3), "v": round(ego["v"], 3),
                     "a": round(ego["a"], 3)})
    return {"traj": traj, "intervention": intervention,
            "engine": PLANNER_ENGINE}


def _stop_at_line_replay(scenario: ROTRScenario) -> Dict:
    """Compliant-envelope reference: brake to the stop line and hold."""
    ctx = scenario.actual_context
    line = ctx.stop_line_x if ctx.stop_line_x is not None else 38.0
    e0 = scenario.ego[0]
    v0 = max(e0.speed, 0.1)
    decel = 3.0
    x_brake = (line - 0.5) - v0 * v0 / (2 * decel)
    x, v, traj = e0.x, e0.speed, []
    for step in range(len(scenario.ego)):
        t = step * DT
        a = -decel if (x >= x_brake and v > 0) else 0.0
        traj.append({"t": round(t, 2), "x": round(x, 3), "y": round(e0.y, 3),
                     "v": round(v, 3), "a": round(a, 3)})
        v = max(0.0, v + a * DT)
        x += v * DT
    return {"traj": traj, "intervention": True, "engine": "compliant-envelope"}


def _planned_replay(scenario: ROTRScenario) -> Dict:
    traj = [{"t": round(p.t, 2), "x": round(p.x, 3), "y": round(p.y, 3),
             "v": round(p.v, 3), "a": 0.0} for p in scenario.planned]
    return {"traj": traj, "intervention": False, "engine": "planned-trajectory"}


def corrected_replay(scenario: ROTRScenario, violation: ROTRViolation,
                     attribution: FailureAttribution) -> Dict:
    primary = attribution.primary_layer
    if primary == "control" and scenario.planned:
        return {**_planned_replay(scenario), "corrected_layers": ["control"]}
    if primary in ("perception", "prediction"):
        return {**_idm_replay(scenario), "corrected_layers": [primary]}
    if violation.rule_id in STOP_RULES:
        return {**_stop_at_line_replay(scenario),
                "corrected_layers": [primary or "unknown"]}
    if violation.rule_id in LANE_RULES:
        return {**_idm_replay(scenario),
                "corrected_layers": [primary or "unknown"]}
    return {**_idm_replay(scenario), "corrected_layers": [primary or "unknown"]}


# ------------------------------------------------------------ measurement


def _assess(traj: List[Dict], scenario: ROTRScenario) -> SafetyAssessment:
    min_ttc = min_gap = None
    collision = False
    max_brake = max_lat = 0.0
    y0 = traj[0]["y"] if traj else 0.0
    brake_x: Optional[float] = None
    stop_dist: Optional[float] = None
    for k, p in enumerate(traj):
        a = p.get("a", 0.0)
        v = p.get("v", p.get("speed", 0.0))
        max_brake = max(max_brake, -a)
        max_lat = max(max_lat, abs(p["y"] - y0))
        if a < -2.0 and brake_x is None:
            brake_x = p["x"]
        if brake_x is not None and stop_dist is None and v < 0.1:
            stop_dist = p["x"] - brake_x
        ego_state = {"x": p["x"], "y": p["y"], "speed": v, "heading": 0.0}
        for actor in scenario.actors:
            s = actor.states[min(k, len(actor.states) - 1)]
            if abs(s.x - p["x"]) > 60 or abs(s.y - p["y"]) > 25:
                continue
            gap = rect_gap((p["x"], p["y"]), 0.0, *EGO_DIMS,
                           (s.x, s.y), s.yaw, actor.dims[0], actor.dims[1])
            if min_gap is None or gap < min_gap:
                min_gap = gap
            obj = {"x": s.x, "y": s.y, "speed": math.hypot(s.vx, s.vy),
                   "heading": s.yaw}
            ttc = projected_ttc(ego_state, EGO_DIMS, obj,
                                (actor.dims[0], actor.dims[1]),
                                look_ahead=4.0, dt=0.1)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc = ttc
            if rects_overlap((p["x"], p["y"]), 0.0, *EGO_DIMS,
                             (s.x, s.y), s.yaw, actor.dims[0], actor.dims[1]):
                collision = True
    pet = _min_pet(traj, scenario)
    return SafetyAssessment(
        min_ttc_s=_r(min_ttc), pet_s=_r(pet), min_clearance_m=_r(min_gap),
        stopping_distance_m=_r(stop_dist), max_braking_mps2=round(max_brake, 3),
        max_lateral_deviation_m=round(max_lat, 3), collision=collision)


def _min_pet(traj: List[Dict], scenario: ROTRScenario) -> Optional[float]:
    ego_traj = {"vehicle_id": "ego", "vehicle_type": "car",
                "length": EGO_DIMS[0], "width": EGO_DIMS[1],
                "states": [{"t": p["t"], "x": p["x"], "y": p["y"],
                            "speed": p.get("v", p.get("speed", 0.0)),
                            "heading": 0.0} for p in traj]}
    best = None
    for actor in scenario.actors:
        other = {"vehicle_id": actor.actor_id, "vehicle_type": actor.class_name,
                 "length": actor.dims[0], "width": actor.dims[1],
                 "states": [{"t": s.t, "x": s.x, "y": s.y,
                             "speed": math.hypot(s.vx, s.vy),
                             "heading": s.yaw} for s in actor.states]}
        pet = zone_grid_pet(ego_traj, other, cell=2.0)
        if pet is not None and (best is None or pet < best):
            best = pet
    return best


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 3)


# ------------------------------------------------------------ classification
# ILLUSTRATIVE thresholds (synthetic substrate).

SC_VRU_CLEARANCE_M = 1.5
SC_CLEARANCE_M = 0.5
DEGRADED_DIVERGENCE_M = 2.0
DEGRADED_BRAKING_MPS2 = 1.5


def classify(scenario: ROTRScenario, violation: ROTRViolation,
             attribution: FailureAttribution) -> CounterfactualScenario:
    replay = corrected_replay(scenario, violation, attribution)
    observed = [{"t": e.t, "x": e.x, "y": e.y, "v": e.speed, "a": e.accel}
                for e in scenario.ego]
    corrected = replay["traj"]

    n = min(len(observed), len(corrected))
    max_pos = max_spd = 0.0
    for k in range(n):
        o, c = observed[k], corrected[k]
        max_pos = max(max_pos, math.hypot(o["x"] - c["x"], o["y"] - c["y"]))
        max_spd = max(max_spd, abs(o["v"] - c["v"]))

    obs = _assess(observed, scenario)
    corr = _assess(corrected, scenario)

    vru = any(a.class_name in ("pedestrian", "cyclist")
              for a in scenario.actors if a.actor_id in set(violation.actor_ids))
    # Observed danger classifies on its own; whether the corrected branch
    # avoided it is reported as evidence, never used to downgrade severity
    # (the replay planner is simplified and must not launder real exposure).
    if (obs.collision
            or (vru and obs.min_clearance_m is not None
                and obs.min_clearance_m < SC_VRU_CLEARANCE_M)
            or (obs.min_clearance_m is not None
                and obs.min_clearance_m < SC_CLEARANCE_M)):
        cls = "SAFETY_CRITICAL"
    elif replay["intervention"] and max_pos > DEGRADED_DIVERGENCE_M:
        cls = "PLANNER_INTERVENTION"
    elif max_pos > DEGRADED_DIVERGENCE_M or \
            corr.max_braking_mps2 > DEGRADED_BRAKING_MPS2:
        cls = "DEGRADED_COMFORT"
    else:
        cls = "NO_MATERIAL_CONSEQUENCE"

    return CounterfactualScenario(
        counterfactual_id=f"{violation.violation_id}-cf",
        violation_id=violation.violation_id,
        scenario_id=scenario.scenario_id,
        corrected_layers=replay["corrected_layers"],
        consequence_class=cls,
        planner_evaluation=PlannerEvaluation(
            engine=replay["engine"],
            observed_trajectory=observed,
            corrected_trajectory=corrected,
            max_position_divergence_m=round(max_pos, 3),
            max_speed_divergence_mps=round(max_spd, 3),
            corrected_max_braking_mps2=corr.max_braking_mps2,
            corrected_intervention=bool(replay["intervention"])),
        observed_safety=obs, corrected_safety=corr,
        provenance=Provenance(
            scenario_id=scenario.scenario_id,
            dataset_version=scenario.provenance.dataset_version,
            model_version=scenario.provenance.model_version,
            software_version=f"{SOFTWARE_VERSION}/{CONSEQUENCE_VERSION}",
            source="COUNTERFACTUAL"))
