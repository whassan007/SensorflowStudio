"""Deterministic, versioned ROTR rule engine.

The engine evaluates a TRAJECTORY against an EXPECTED-BEHAVIOR ENVELOPE
derived from a road context and a set of actor views. It is deliberately
source-agnostic and is used twice:

* detection — executed ego trajectory + ACTUAL (as-built) context + ground-
  truth actor states: "did the vehicle violate right-of-the-road?"
* plan compliance (attribution.py) — planned trajectory + MAPPED context +
  the STACK'S OWN actor view (detections + predicted intents), shifted into
  the believed localization frame: "was the plan compliant given what the
  stack believed?" — the planning layer is implicated only when this fails.

This dual use is what keeps causal layers separated: the same rule text,
different evidence sources. Rules never guess causes; they emit structured
ROTRViolation records with the evidence fields that drove the decision.

All thresholds in CONFIG are ILLUSTRATIVE (synthetic substrate).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional

# REUSE (landed, read-only): rotated-rectangle separation from SSAM.
from sensorflow.safety.ssam_ext import rect_gap

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr.models import (
    Provenance, ROTRScenario, ROTRViolation, RoadContext,
)

RULESET_VERSION = "rotr-rules-1.0.0"

CONFIG: Dict[str, float] = {
    # ILLUSTRATIVE thresholds — calibrated only against the synthetic bank.
    "yield_max_transit_speed_mps": 2.0,   # faster than this through an
                                          # occupied crosswalk = not yielding
    "ped_roadway_half_width_m": 4.5,      # |y| below which a ped is "in roadway"
    "crosswalk_margin_m": 0.5,
    "restricted_min_frames": 5,
    "lane_assoc_tolerance_m": 2.1,        # 0.6 * lane width
    "signal_entry_min_speed_mps": 0.5,
    "merge_min_time_gap_s": 1.5,
    "stop_max_speed_mps": 0.7,            # above this at the line = no stop
    "stop_window_m": 15.0,
}

EGO_DIMS = (4.5, 1.9)


def ruleset_fingerprint() -> str:
    blob = json.dumps({"version": RULESET_VERSION, "config": CONFIG},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ------------------------------------------------------------ view builders


def gt_actor_views(scenario: ROTRScenario) -> List[Dict]:
    """The world as it is: ground-truth actor states + true intent."""
    return [{
        "actor_id": a.actor_id, "class_name": a.class_name, "dims": a.dims,
        "intent": a.intent, "right_of_way": a.right_of_way,
        "states": [{"t": s.t, "x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy,
                    "yaw": s.yaw} for s in a.states],
    } for a in scenario.actors]


def stack_actor_views(scenario: ROTRScenario) -> List[Dict]:
    """The world as the stack saw it: detected positions + predicted intent.

    Frames where the actor was not detected are simply absent from the view
    (the stack cannot plan around what it did not see). Actors with no
    telemetry at all (observation=None) yield an empty view.
    """
    out = []
    for a in scenario.actors:
        states = []
        for s in a.states:
            o = s.observation
            if o is None or not o.detected:
                continue
            states.append({"t": s.t, "x": o.x, "y": o.y,
                           "vx": s.vx, "vy": s.vy, "yaw": s.yaw})
        out.append({
            "actor_id": a.actor_id, "class_name": a.class_name,
            "dims": a.dims,
            "intent": a.predicted_intent,      # the STACK's belief
            "right_of_way": a.right_of_way,
            "states": states,
        })
    return out


def ego_trajectory(scenario: ROTRScenario) -> List[Dict]:
    return [{"t": e.t, "x": e.x, "y": e.y, "speed": e.speed}
            for e in scenario.ego]


def planned_trajectory_believed_frame(scenario: ROTRScenario) -> List[Dict]:
    """Plan shifted into the believed localization frame: the plan is judged
    where the stack believed the vehicle to be, not where it truly was."""
    if not scenario.planned or not scenario.ego:
        return []
    e0 = scenario.ego[0]
    dy = (e0.believed_y - e0.y) if e0.believed_y is not None else 0.0
    dx = (e0.believed_x - e0.x) if e0.believed_x is not None else 0.0
    return [{"t": p.t, "x": p.x + dx, "y": p.y + dy, "speed": p.v}
            for p in scenario.planned]


# ------------------------------------------------------------ helpers


def _nearest_lane(ctx: RoadContext, y: float, tol: float):
    best, best_d = None, 1e9
    for lane in ctx.lanes:
        d = abs(lane.center_y - y)
        if d < best_d:
            best, best_d = lane, d
    return (best, best_d) if best is not None and best_d <= tol else (None, None)


def _state_at(actor: Dict, t: float) -> Optional[Dict]:
    best, best_dt = None, 0.25
    for s in actor["states"]:
        dt = abs(s["t"] - t)
        if dt < best_dt:
            best, best_dt = s, dt
    return best


def _clearance(traj_pt: Dict, actor: Dict, st: Dict) -> float:
    return rect_gap((traj_pt["x"], traj_pt["y"]), 0.0, *EGO_DIMS,
                    (st["x"], st["y"]), st.get("yaw", 0.0),
                    actor["dims"][0], actor["dims"][1])


# ------------------------------------------------------------ rules
# Each rule: (traj, ctx, actors) -> Optional[evidence dict]. A returned dict
# means the trajectory exits the expected-behavior envelope for that rule.


def _rule_yield_ped(traj: List[Dict], ctx: RoadContext,
                    actors: List[Dict]) -> Optional[Dict]:
    """R-YIELD-PED-01: yield to a crossing pedestrian at a crosswalk."""
    if not ctx.crosswalks:
        return None
    cw = ctx.crosswalks[0]
    margin = CONFIG["crosswalk_margin_m"]
    for actor in actors:
        if actor["class_name"] not in ("pedestrian", "cyclist"):
            continue
        if actor["intent"] != "CROSSING" or actor["right_of_way"] != "HAS_ROW":
            continue
        # Ped must actually enter the roadway for the obligation to bind.
        if not any(abs(s["y"]) < 3.5 for s in actor["states"]):
            continue
        worst = None
        for p in traj:
            if not (cw.x_min - margin <= p["x"] <= cw.x_max + margin):
                continue
            st = _state_at(actor, p["t"])
            if st is None:
                continue
            if not (cw.x_min - margin <= st["x"] <= cw.x_max + margin):
                continue
            if abs(st["y"]) > CONFIG["ped_roadway_half_width_m"]:
                continue
            if p["speed"] > CONFIG["yield_max_transit_speed_mps"]:
                gap = _clearance(p, actor, st)
                if worst is None or gap < worst["min_clearance_m"]:
                    worst = {"t": round(p["t"], 2),
                             "ego_speed_mps": round(p["speed"], 2),
                             "ped_y_m": round(st["y"], 2),
                             "min_clearance_m": round(gap, 3),
                             "actor_id": actor["actor_id"]}
        if worst:
            worst["threshold_speed_mps"] = CONFIG["yield_max_transit_speed_mps"]
            return worst
    return None


def _rule_restricted_path(traj: List[Dict], ctx: RoadContext,
                          actors: List[Dict]) -> Optional[Dict]:
    """R-PATH-RESTRICT-02: do not occupy a lane restricted to another class."""
    frames_in, lane_id = 0, None
    for p in traj:
        lane, _ = _nearest_lane(ctx, p["y"], CONFIG["lane_assoc_tolerance_m"])
        if lane is not None and lane.restricted_to not in (None, "any", "car"):
            frames_in += 1
            lane_id = lane.lane_id
    if frames_in >= CONFIG["restricted_min_frames"]:
        return {"lane_id": lane_id, "frames_in_restricted": frames_in,
                "duration_s": round(frames_in * (traj[1]["t"] - traj[0]["t"]), 2)
                if len(traj) > 1 else 0.0}
    return None


def _rule_lane_maneuver(traj: List[Dict], ctx: RoadContext,
                        actors: List[Dict]) -> Optional[Dict]:
    """R-LANE-MANEUVER-03: only maneuvers permitted from the occupied lane."""
    if ctx.intersection_x_min is None or ctx.stop_line_x is None:
        return None
    entry = [p for p in traj if p["x"] < ctx.stop_line_x]
    through = [p for p in traj if p["x"] > (ctx.intersection_x_max or 0)]
    if not entry or not through:
        return None
    lane, _ = _nearest_lane(ctx, entry[-1]["y"], CONFIG["lane_assoc_tolerance_m"])
    if lane is None:
        return None
    went_straight = abs(through[0]["y"] - entry[-1]["y"]) < 1.5
    if went_straight and "STRAIGHT" not in lane.permitted_maneuvers:
        return {"lane_id": lane.lane_id, "maneuver": "STRAIGHT",
                "permitted": lane.permitted_maneuvers,
                "entry_y_m": round(entry[-1]["y"], 2)}
    return None


def _rule_signal_conflict(traj: List[Dict], ctx: RoadContext,
                          actors: List[Dict]) -> Optional[Dict]:
    """R-INT-CONFLICT-04: do not enter a signalized intersection on red."""
    if ctx.control != "signal" or ctx.signal_state_for_ego != "red":
        return None
    if ctx.intersection_x_min is None:
        return None
    for p in traj:
        if p["x"] >= ctx.intersection_x_min and \
                p["speed"] > CONFIG["signal_entry_min_speed_mps"]:
            ev = {"t": round(p["t"], 2), "entry_speed_mps": round(p["speed"], 2),
                  "signal_state": "red"}
            # Conflicting right-of-way actor evidence (severity, not verdict).
            worst_gap, worst_id = None, None
            for actor in actors:
                if actor["right_of_way"] != "HAS_ROW":
                    continue
                for q in traj:
                    if q["x"] < ctx.intersection_x_min:
                        continue
                    st = _state_at(actor, q["t"])
                    if st is None:
                        continue
                    gap = _clearance(q, actor, st)
                    if worst_gap is None or gap < worst_gap:
                        worst_gap, worst_id = gap, actor["actor_id"]
            if worst_gap is not None:
                ev["min_clearance_m"] = round(worst_gap, 3)
                ev["actor_id"] = worst_id
            return ev
    return None


def _rule_merge_gap(traj: List[Dict], ctx: RoadContext,
                    actors: List[Dict]) -> Optional[Dict]:
    """R-MERGE-GAP-05: merging in front of right-of-way traffic needs a gap."""
    tol = CONFIG["lane_assoc_tolerance_m"]
    lane_seq = []
    for p in traj:
        lane, _ = _nearest_lane(ctx, p["y"], tol)
        lane_seq.append(lane.lane_id if lane else None)
    change_idx = None
    for k in range(1, len(lane_seq)):
        if lane_seq[k] and lane_seq[k - 1] and lane_seq[k] != lane_seq[k - 1]:
            change_idx = k
    if change_idx is None:
        return None
    p = traj[change_idx]
    target = lane_seq[change_idx]
    for actor in actors:
        if actor["right_of_way"] != "HAS_ROW":
            continue
        st = _state_at(actor, p["t"])
        if st is None:
            continue
        lane, _ = _nearest_lane(ctx, st["y"], tol)
        if lane is None or lane.lane_id != target:
            continue
        if st["x"] >= p["x"]:
            continue                     # only trailing traffic holds the gap
        speed = math.hypot(st["vx"], st["vy"])
        bumper_gap = (p["x"] - st["x"]) - 0.5 * (EGO_DIMS[0] + actor["dims"][0])
        time_gap = bumper_gap / max(speed, 0.1)
        if time_gap < CONFIG["merge_min_time_gap_s"]:
            return {"t": round(p["t"], 2), "target_lane": target,
                    "actor_id": actor["actor_id"],
                    "bumper_gap_m": round(bumper_gap, 2),
                    "time_gap_s": round(time_gap, 2),
                    "threshold_time_gap_s": CONFIG["merge_min_time_gap_s"]}
    return None


def _rule_stop(traj: List[Dict], ctx: RoadContext,
               actors: List[Dict]) -> Optional[Dict]:
    """R-STOP-06: a stop control requires an actual stop before the line."""
    if ctx.control != "stop_sign" or ctx.stop_line_x is None:
        return None
    line = ctx.stop_line_x
    window = [p for p in traj
              if line - CONFIG["stop_window_m"] <= p["x"] <= line + 1.0]
    crossed = any(p["x"] > line for p in traj)
    if not window or not crossed:
        return None
    min_speed = min(p["speed"] for p in window)
    if min_speed > CONFIG["stop_max_speed_mps"]:
        return {"min_speed_in_window_mps": round(min_speed, 2),
                "stop_line_x": line,
                "threshold_mps": CONFIG["stop_max_speed_mps"]}
    return None


RULES = [
    ("R-YIELD-PED-01", "failure to yield to crossing pedestrian", _rule_yield_ped),
    ("R-PATH-RESTRICT-02", "restricted-path entry", _rule_restricted_path),
    ("R-LANE-MANEUVER-03", "maneuver not permitted from occupied lane",
     _rule_lane_maneuver),
    ("R-INT-CONFLICT-04", "entered signalized intersection on red",
     _rule_signal_conflict),
    ("R-MERGE-GAP-05", "merge with insufficient gap to right-of-way traffic",
     _rule_merge_gap),
    ("R-STOP-06", "failed to stop at stop control", _rule_stop),
]


# ------------------------------------------------------------ entry points


def evaluate_trajectory(traj: List[Dict], ctx: RoadContext,
                        actors: List[Dict]) -> List[Dict]:
    """Run every rule; return [{rule_id, description, evidence}]."""
    hits = []
    for rule_id, desc, fn in RULES:
        ev = fn(traj, ctx, actors)
        if ev is not None:
            hits.append({"rule_id": rule_id, "description": desc,
                         "evidence": ev})
    return hits


def detect(scenario: ROTRScenario) -> List[ROTRViolation]:
    """Detection: executed trajectory vs the ACTUAL world (GT evidence)."""
    traj = ego_trajectory(scenario)
    hits = evaluate_trajectory(traj, scenario.actual_context,
                               gt_actor_views(scenario))
    out = []
    for h in hits:
        ev = h["evidence"]
        vid = f"{scenario.scenario_id}-{h['rule_id']}"
        out.append(ROTRViolation(
            violation_id=vid, scenario_id=scenario.scenario_id,
            rule_id=h["rule_id"], rule_version=RULESET_VERSION,
            description=f"{h['description']} — {scenario.description}",
            actor_ids=[ev["actor_id"]] if "actor_id" in ev else [],
            t_start=float(ev.get("t", 0.0)), t_end=float(ev.get("t", 0.0)),
            evidence={**ev, "ruleset_fingerprint": ruleset_fingerprint(),
                      "thresholds_note": "ILLUSTRATIVE (synthetic substrate)"},
            confidence=0.9,
            provenance=Provenance(
                scenario_id=scenario.scenario_id,
                dataset_version=scenario.provenance.dataset_version,
                model_version=scenario.provenance.model_version,
                software_version=f"{SOFTWARE_VERSION}/{RULESET_VERSION}",
                source=scenario.provenance.source, confidence=0.9)))
    return out


def plan_violations(scenario: ROTRScenario) -> List[Dict]:
    """Plan compliance given the stack's OWN world view (mapped context,
    detections, predicted intents, believed localization frame). Used by
    attribution: a compliant plan RULES OUT the planning layer."""
    traj = planned_trajectory_believed_frame(scenario)
    if not traj:
        return []
    return evaluate_trajectory(traj, scenario.map_context,
                               stack_actor_views(scenario))
