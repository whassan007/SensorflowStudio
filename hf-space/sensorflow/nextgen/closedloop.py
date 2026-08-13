"""Closed-loop behavioral evaluation.

Extends the platform's open-loop chain (scenario -> perception -> metrics)
into scenario -> perception -> prediction -> planner -> controller ->
vehicle-response, and measures BEHAVIOR, not just detection quality.

Honest scope: the planner/controller here is a deterministic simplified
stack — IDM-style longitudinal control + PD lane-keep lateral + constant-
velocity prediction — sufficient to measure how perception quality changes
behavior. It is NOT a production planner; docs/architecture/nextgen-adr.md
discusses what a real planner would change.

REUSE: all safety-region math comes from sensorflow.safety.ssam_ext —
projected_ttc (rectangle conflict-point projection), rect_gap (rotated-
rectangle separation), rects_overlap (collision detection),
collision_probability (TTC-threshold proxy). Open-loop metrics (per-frame
recall etc.) are computed alongside and attached to every assessment:
closed-loop NEVER replaces open-loop.

Perception model: detection sampling with modality-informed failure rates
mirroring sensorflow.bevfusion.sensors (occlusion/night/rain/distance), with
two engine profiles (camera-primary baseline vs BEV-fused candidate) plus
parameterized extended-condition effects (fog/glare/wet — conditions the
bevfusion simulators don't model; documented in
PERCEPTION_CONDITION_EFFECTS). A "corrected" mode injects ground truth
(perfect perception) — that is what causal.py diffs against. All randomness
is seeded: same scenario + engine + seed -> identical behavior.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from sensorflow.bevfusion.scenes import DT, EGO_SPEED
# REUSE: SSAM safety-region math from the safety package.
from sensorflow.safety.ssam_ext import (
    collision_probability, projected_ttc, rect_gap, rects_overlap,
)
from sensorflow.nextgen.models import (
    BehavioralAssessment, BehavioralMetrics, DataLabel,
)
from sensorflow.nextgen.worldmodel import ActorTrack

EGO_DIMS = (4.5, 1.9)          # l, w (matches ssam_ext synthetic ego)
CORRIDOR_HALF_WIDTH_M = 1.6    # lateral extent of the ego path
TTC_BRAKE_S = 4.0              # begin comfort braking below this projected TTC
TTC_EMERGENCY_S = 1.5          # FHWA SSAM conflict threshold (ssam_ext default)
A_MAX_ACCEL = 2.5              # m/s^2
A_COMFORT_BRAKE = 3.5
A_MAX_BRAKE = 8.0
IDM_T_HEADWAY = 1.5            # s
IDM_S0 = 2.0                   # m minimum gap
ACTUATION_TAU_S = 0.25         # first-order lag of the vehicle response
DETECT_RANGE_M = 75.0

ENGINE_PROFILES: Dict[str, Dict[str, float]] = {
    # Camera-primary baseline: heavy occlusion/night misses (cf.
    # bevfusion.sensors.simulate_camera failure model).
    "perception-v1-camera": {"base_miss": 0.06, "occlusion_miss": 0.80,
                             "night_miss": 0.40, "rain_miss": 0.06,
                             "range_miss_per_m": 0.0015, "pos_sigma": 0.55},
    # BEV-fused candidate: LiDAR recovers occlusion/night (cf.
    # bevfusion.engines.run_fused complementarity).
    "perception-v3-bevfusion": {"base_miss": 0.02, "occlusion_miss": 0.25,
                                "night_miss": 0.04, "rain_miss": 0.04,
                                "range_miss_per_m": 0.0008, "pos_sigma": 0.18},
}
DEFAULT_ENGINE = "perception-v3-bevfusion"

# Extended environment conditions produced by counterfactual transforms.
# These are OUR parameterization (the bevfusion sensor sims don't model them);
# additive miss probability, scaled by distance/DETECT_RANGE_M where noted.
PERCEPTION_CONDITION_EFFECTS: Dict[str, Dict[str, float]] = {
    "fog": {"add_miss_scaled_by_range": 0.45, "pos_sigma_mult": 1.6},
    "sunset_glare": {"add_miss": 0.12, "pos_sigma_mult": 1.2},
    "wet": {"add_miss": 0.03, "pos_sigma_mult": 1.1},
}


# ------------------------------------------------------------ perception


class PerceptionModel:
    """Deterministic seeded detection sampler over world-frame actors."""

    def __init__(self, engine: str, environment: Dict[str, str], seed: int,
                 corrected: bool = False,
                 faults: Optional[List[Dict]] = None):
        if engine not in ENGINE_PROFILES:
            raise ValueError(f"unknown engine {engine!r}; known: "
                             f"{sorted(ENGINE_PROFILES)}")
        self.engine = engine
        self.profile = ENGINE_PROFILES[engine]
        self.environment = environment
        self.corrected = corrected
        self.faults = faults or []
        self.rng = np.random.default_rng(
            [seed, sum(map(ord, engine)) % 100003, int(corrected)])
        self.tracked: Dict[str, int] = {}          # iid -> consecutive hits
        self.first_detection_s: Dict[str, float] = {}
        # open-loop accounting (complementary metrics)
        self.n_gt_visible = 0
        self.n_detected = 0
        self.n_misclassified = 0

    def _miss_probability(self, dist: float, occluded: bool) -> float:
        p = self.profile["base_miss"] + self.profile["range_miss_per_m"] * dist
        if occluded:
            p += self.profile["occlusion_miss"]
        if self.environment.get("time_of_day") == "night":
            p += self.profile["night_miss"]
        if self.environment.get("weather") == "rain":
            p += self.profile["rain_miss"]
        for tag in ("weather_extended", "glare", "road_surface"):
            eff = PERCEPTION_CONDITION_EFFECTS.get(self.environment.get(tag, ""))
            if eff:
                p += eff.get("add_miss", 0.0)
                p += eff.get("add_miss_scaled_by_range", 0.0) * min(1.0, dist / DETECT_RANGE_M)
        return min(p, 0.97)

    def _pos_sigma(self) -> float:
        s = self.profile["pos_sigma"]
        for tag in ("weather_extended", "glare", "road_surface"):
            eff = PERCEPTION_CONDITION_EFFECTS.get(self.environment.get(tag, ""))
            if eff:
                s *= eff.get("pos_sigma_mult", 1.0)
        return s

    def perceive(self, t: float, ego: Dict, actors: List[ActorTrack],
                 fi: int) -> List[Dict]:
        """Detections in world frame for the current step."""
        out: List[Dict] = []
        for a in actors:
            st = a.states[min(fi, len(a.states) - 1)]
            if st.get("absent"):
                continue
            rel_x, rel_y = st["x"] - ego["x"], st["y"] - ego["y"]
            dist = math.hypot(rel_x, rel_y)
            if rel_x < -5.0 or dist > DETECT_RANGE_M:
                continue
            self.n_gt_visible += 1

            fault = self._active_fault(a.instance_id, t)
            if self.corrected:
                detected = True
            elif fault and fault["type"] == "miss":
                detected = False
            else:
                p_miss = self._miss_probability(dist, bool(st.get("occluded")))
                if self.tracked.get(a.instance_id, 0) >= 2:
                    p_miss *= 0.15  # track continuity (masklet-style latching)
                detected = self.rng.random() >= p_miss

            if not detected:
                self.tracked[a.instance_id] = 0
                continue
            self.tracked[a.instance_id] = self.tracked.get(a.instance_id, 0) + 1
            self.first_detection_s.setdefault(a.instance_id, t)
            self.n_detected += 1

            sigma = 0.0 if self.corrected else self._pos_sigma()
            cls = a.class_name
            if fault and fault["type"] == "misclassify" and not self.corrected:
                cls = fault.get("as_class", cls)
                self.n_misclassified += 1
            dx_bias = fault.get("dx_m", 0.0) if fault and fault["type"] == "position_bias" \
                and not self.corrected else 0.0
            out.append({
                "instance_id": a.instance_id, "class_name": cls,
                "x": st["x"] + float(self.rng.normal(0, sigma)) + dx_bias,
                "y": st["y"] + float(self.rng.normal(0, sigma * 0.6)),
                "vx": st["vx"], "vy": st["vy"],  # tracker velocity estimate
                "dims": a.dims, "yaw": st["yaw"],
            })
        return out

    def _active_fault(self, iid: str, t: float) -> Optional[Dict]:
        for f in self.faults:
            if f.get("instance_id") != iid:
                continue
            if f["type"] == "miss":
                if float(f.get("from_s", 0.0)) <= t < float(f.get("until_s", 1e9)):
                    return f
            else:
                return f
        return None

    def open_loop_metrics(self) -> Dict:
        recall = self.n_detected / self.n_gt_visible if self.n_gt_visible else None
        return {
            "engine": self.engine,
            "frame_recall": None if recall is None else round(recall, 4),
            "n_gt_visible": self.n_gt_visible,
            "n_detected": self.n_detected,
            "n_misclassified": self.n_misclassified,
            "note": "open-loop metrics are complementary to the behavioral "
                    "metrics and are always reported alongside them",
        }


# ------------------------------------------------------------ planner


def plan_acceleration(ego: Dict, detections: List[Dict]) -> Dict:
    """IDM-style longitudinal command with TTC-based emergency reaction.

    The planner reacts to perceived GEOMETRY (position/velocity), not class:
    a cosmetic misclassification does not change the plan — which is exactly
    what causal replay demonstrates as METRIC_ONLY.
    """
    v = ego["v"]
    a_cmd = A_MAX_ACCEL * (1.0 - (v / max(EGO_SPEED, 0.1)) ** 4)
    reason = "free_road"
    intervention = False

    ego_state = {"x": ego["x"], "y": ego["y"], "speed": v, "heading": ego["heading"]}
    min_ttc: Optional[float] = None
    for det in detections:
        rel_x = det["x"] - ego["x"]
        obj_state = {"x": det["x"], "y": det["y"],
                     "speed": math.hypot(det["vx"], det["vy"]),
                     "heading": math.atan2(det["vy"], det["vx"])
                     if (det["vx"], det["vy"]) != (0.0, 0.0) else 0.0}
        # In-corridor lead vehicle: IDM following.
        if abs(det["y"] - ego["y"]) <= CORRIDOR_HALF_WIDTH_M and rel_x > 0:
            gap = rect_gap((ego["x"], ego["y"]), ego["heading"], *EGO_DIMS,
                           (det["x"], det["y"]), obj_state["heading"],
                           det["dims"][0], det["dims"][1])
            dv = v - det["vx"]  # closing speed along the lane
            s_star = IDM_S0 + max(0.0, v * IDM_T_HEADWAY
                                  + v * dv / (2 * math.sqrt(A_MAX_ACCEL * A_COMFORT_BRAKE)))
            a_idm = A_MAX_ACCEL * (1.0 - (v / max(EGO_SPEED, 0.1)) ** 4
                                   - (s_star / max(gap, 0.5)) ** 2)
            if a_idm < a_cmd:
                a_cmd, reason = a_idm, f"idm_follow:{det['instance_id']}"

        # Projected-TTC reaction (REUSED ssam_ext conflict-point projection):
        # covers crossing/merging actors that are not yet in the corridor.
        ttc = projected_ttc(ego_state, EGO_DIMS, obj_state,
                            (det["dims"][0], det["dims"][1]),
                            look_ahead=TTC_BRAKE_S, dt=0.1)
        if ttc is not None:
            min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
            if ttc <= TTC_EMERGENCY_S:
                a_cmd = -A_MAX_BRAKE
                reason = f"emergency_brake:{det['instance_id']}"
                intervention = True
            elif ttc <= TTC_BRAKE_S:
                a_brake = -A_COMFORT_BRAKE * (TTC_BRAKE_S - ttc) / (TTC_BRAKE_S - TTC_EMERGENCY_S)
                if a_brake < a_cmd:
                    a_cmd = a_brake
                    reason = f"ttc_brake:{det['instance_id']}"
                    intervention = True

    return {"a_cmd": float(np.clip(a_cmd, -A_MAX_BRAKE, A_MAX_ACCEL)),
            "reason": reason, "intervention": intervention,
            "perceived_min_ttc": min_ttc}


# ------------------------------------------------------------ simulation


def run_closed_loop(actors: List[ActorTrack], environment: Dict[str, str],
                    scenario_id: str, data_label: DataLabel,
                    engine: str = DEFAULT_ENGINE, seed: int = 0,
                    corrected: bool = False,
                    faults: Optional[List[Dict]] = None,
                    horizon_s: Optional[float] = None) -> BehavioralAssessment:
    """Simulate the full loop and measure behavioral metrics."""
    n_frames = max(len(a.states) for a in actors) if actors else 0
    n_steps = int(round((horizon_s or n_frames * DT) / DT))
    perception = PerceptionModel(engine, environment, seed,
                                 corrected=corrected, faults=faults)

    target_y = -1.6 if "construction_zone" in environment else 0.0
    ego = {"x": 0.0, "y": 0.0, "v": EGO_SPEED, "a": 0.0, "heading": 0.0}

    trajectory: List[Dict[str, float]] = []
    min_ttc: Optional[float] = None
    min_sep: Optional[float] = None
    min_sep_speed = 0.0
    max_decel = 0.0
    max_steer_rate = 0.0
    interventions = 0
    intervening = False
    collision = False
    max_col_prob = 0.0
    brake_start_x: Optional[float] = None
    stopping_distance: Optional[float] = None
    critical_iid: Optional[str] = None

    for step in range(n_steps):
        t = step * DT
        fi = min(step, n_frames - 1) if n_frames else 0
        detections = perception.perceive(t, ego, actors, fi)
        plan = plan_acceleration(ego, detections)

        if plan["intervention"] and not intervening:
            interventions += 1
            critical_iid = critical_iid or plan["reason"].split(":")[-1]
        intervening = plan["intervention"]

        # Vehicle response: first-order actuation lag + kinematic update.
        alpha = DT / (DT + ACTUATION_TAU_S)
        ego["a"] += alpha * (plan["a_cmd"] - ego["a"])
        prev_heading = ego["heading"]
        ego["v"] = max(0.0, ego["v"] + ego["a"] * DT)
        ego["x"] += ego["v"] * DT
        vy_cmd = float(np.clip(0.8 * (target_y - ego["y"]), -1.5, 1.5))
        ego["y"] += vy_cmd * DT
        ego["heading"] = math.atan2(vy_cmd, max(ego["v"], 0.5))
        max_steer_rate = max(max_steer_rate, abs(ego["heading"] - prev_heading) / DT)
        max_decel = max(max_decel, -ego["a"])

        if ego["a"] < -2.0 and brake_start_x is None:
            brake_start_x = ego["x"]
        if brake_start_x is not None and stopping_distance is None and ego["v"] < 0.1:
            stopping_distance = ego["x"] - brake_start_x

        # Ground-truth safety measures vs every actor (REUSED ssam_ext math).
        ego_state = {"x": ego["x"], "y": ego["y"], "speed": ego["v"],
                     "heading": ego["heading"]}
        for a in actors:
            st = a.states[fi]
            if st.get("absent"):
                continue
            if abs(st["x"] - ego["x"]) > 60 or abs(st["y"] - ego["y"]) > 25:
                continue
            obj_state = {"x": st["x"], "y": st["y"],
                         "speed": math.hypot(st["vx"], st["vy"]),
                         "heading": st["yaw"]}
            gap = rect_gap((ego["x"], ego["y"]), ego["heading"], *EGO_DIMS,
                           (st["x"], st["y"]), st["yaw"], a.dims[0], a.dims[1])
            if min_sep is None or gap < min_sep:
                min_sep, min_sep_speed = gap, ego["v"]
            ttc = projected_ttc(ego_state, EGO_DIMS, obj_state,
                                (a.dims[0], a.dims[1]), look_ahead=3.0, dt=0.1)
            if ttc is not None:
                min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
                max_col_prob = max(max_col_prob,
                                   collision_probability(ttc, TTC_EMERGENCY_S))
            if rects_overlap((ego["x"], ego["y"]), ego["heading"], *EGO_DIMS,
                             (st["x"], st["y"]), st["yaw"],
                             a.dims[0], a.dims[1]):
                collision = True
                max_col_prob = 1.0

        trajectory.append({"t": round(t, 2), "x": round(ego["x"], 3),
                           "y": round(ego["y"], 3), "v": round(ego["v"], 3),
                           "a": round(ego["a"], 3),
                           "n_detections": len(detections)})
        if collision:
            break

    # Detection latency of the intervention-critical actor: delay between it
    # becoming detectable (in range/FOV) and its first detection.
    detection_latency = time_to_detection = None
    if critical_iid:
        first_det = perception.first_detection_s.get(critical_iid)
        if first_det is not None:
            time_to_detection = first_det
            appear_t = _first_detectable_time(actors, critical_iid, n_steps)
            if appear_t is not None:
                detection_latency = max(0.0, first_det - appear_t)

    # Safety margin at closest approach: gap minus the reaction-time headway
    # distance at the speed the ego carried at that moment.
    safety_margin = None
    if min_sep is not None:
        safety_margin = min_sep - min_sep_speed * 0.6  # 0.6 s reaction headway

    metrics = BehavioralMetrics(
        detection_latency_s=_round(detection_latency),
        time_to_detection_s=_round(time_to_detection),
        min_ttc_s=_round(min_ttc),
        stopping_distance_m=_round(stopping_distance),
        max_deceleration_mps2=round(max_decel, 3),
        max_steering_rate_radps=round(max_steer_rate, 4),
        planner_interventions=interventions,
        collision=collision,
        collision_probability=round(max_col_prob, 4),
        min_separation_m=_round(min_sep),
        safety_margin_m=_round(safety_margin),
        final_speed_mps=round(ego["v"], 3))

    return BehavioralAssessment(
        scenario_id=scenario_id, data_label=data_label,
        perception_mode="corrected" if corrected else "actual",
        metrics=metrics,
        open_loop=perception.open_loop_metrics(),
        trajectory=trajectory)


def demo_emergence_scenario(conflict_t_s: float = 3.0,
                            t_emerge_s: float = 1.6) -> Dict:
    """Controlled occluded-emergence scenario for causal-replay demos/tests.

    Minimal actor set (occluder truck + emergent crossing pedestrian + one
    benign lead vehicle) so the causal experiment is not polluted by
    unrelated actors triggering the planner in both branches. The pedestrian
    is timed to enter the ego corridor when a non-reacting ego would reach
    its position at conflict_t_s. Data label COUNTERFACTUAL (constructed).
    """
    n_frames = 60
    x_conflict = EGO_SPEED * conflict_t_s
    y0 = 4.0
    vy = -1.9
    ramp_a = 6.0
    truck = ActorTrack(
        instance_id="demo-occluder-truck", class_name="truck",
        dims=[8.5, 2.5, 3.2],
        states=[{"x": x_conflict - 2.0, "y": y0 + 2.0, "vx": 0.0, "vy": 0.0,
                 "yaw": 0.0, "occluded": False} for _ in range(n_frames)])
    ped_states = []
    for fi in range(n_frames):
        t = fi * DT
        if t <= t_emerge_s:
            y, vy_i = y0, 0.0
        else:
            dt_m = t - t_emerge_s
            t_ramp = abs(vy) / ramp_a
            if dt_m <= t_ramp:
                vy_i = -ramp_a * dt_m
                y = y0 - 0.5 * ramp_a * dt_m * dt_m
            else:
                vy_i = vy
                y = y0 - 0.5 * ramp_a * t_ramp * t_ramp + vy * (dt_m - t_ramp)
        ped_states.append({"x": x_conflict + 3.5, "y": y, "vx": 0.0,
                           "vy": vy_i, "yaw": -math.pi / 2,
                           "occluded": t <= t_emerge_s})
    ped = ActorTrack(instance_id="demo-crossing-ped", class_name="pedestrian",
                     dims=[0.6, 0.6, 1.75], states=ped_states)
    lead = ActorTrack(
        instance_id="demo-lead-vehicle", class_name="vehicle",
        dims=[4.5, 1.9, 1.6],
        states=[{"x": 55.0 + 9.5 * fi * DT, "y": 0.0, "vx": 9.5, "vy": 0.0,
                 "yaw": 0.0, "occluded": False} for fi in range(n_frames)])
    return {"actors": [truck, ped, lead],
            "environment": {"time_of_day": "day", "weather": "clear"},
            "scenario_id": "demo-occluded-emergence",
            "critical_instance_id": ped.instance_id,
            "horizon_s": n_frames * DT}


def _first_detectable_time(actors: List[ActorTrack], iid: str,
                           n_steps: int) -> Optional[float]:
    actor = next((a for a in actors if a.instance_id == iid), None)
    if actor is None:
        return None
    # Approximate the moving ego with the nominal profile for detectability
    # onset (detectability begins before behavior diverges materially).
    for step in range(n_steps):
        fi = min(step, len(actor.states) - 1)
        st = actor.states[fi]
        ego_x = EGO_SPEED * DT * step
        rel_x = st["x"] - ego_x
        if -5.0 <= rel_x and math.hypot(rel_x, st["y"]) <= DETECT_RANGE_M:
            return step * DT
    return None


def _round(v: Optional[float], nd: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), nd)
