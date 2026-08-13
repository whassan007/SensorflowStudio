"""Safety-informed metrics: Safety-Critical Recall (SCR) and risk weighting.

Design principle: open-loop metrics (recall/precision/IoU) are NEVER replaced
— every report here carries them alongside the safety-informed view. The two
views answer different questions ("how good is perception?" vs "how good is
perception where it matters?") and can legitimately move in opposite
directions; `divergence_demo` constructs exactly that case.

Safety-critical region (explicitly defined, parameterized)
----------------------------------------------------------
NOT naive stopping distance. An object is safety-critical when ANY of:

1. Longitudinal criticality: it lies ahead within
       d_crit(v) = v * t_react + v^2 / (2 * a_eff) + buffer
   where a_eff = min(brake_capability, mu * g) — the friction-limited
   achievable deceleration — AND laterally within the encroachment band
       |y| <= corridor_half_width + v_lat_max(class) * t_stop(v)
   with t_stop(v) = t_react + v / a_eff: anything that can REACH the ego
   corridor before the ego can stop counts, scaled by how fast that class
   of actor can move laterally.
2. TTC criticality: projected TTC below ttc_critical_s (reuses the
   sensorflow.safety.ssam_ext conflict-point projection when velocity data
   is available).

Risk weight (per object, deterministic):
    w = vulnerability(class) * proximity * closing_speed_factor
        * ttc_factor * occlusion_factor * ego_speed_factor
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from sensorflow.safety.ssam_ext import projected_ttc

G = 9.81

VULNERABILITY = {"pedestrian": 1.0, "cyclist": 0.9, "motorcycle": 0.8,
                 "vehicle": 0.4, "truck": 0.35}
# How fast each class can move laterally toward the corridor (m/s).
LATERAL_SPEED_CAP = {"pedestrian": 2.5, "cyclist": 5.0, "motorcycle": 8.0,
                     "vehicle": 6.0, "truck": 4.0}


@dataclass
class SafetyRegionParams:
    """All region math is a pure function of these documented parameters."""

    reaction_time_s: float = 0.9        # perception->brake-onset latency budget
    brake_capability_mps2: float = 7.0  # vehicle hardware limit
    friction_mu: float = 0.9            # 0.9 dry asphalt; ~0.5-0.6 wet
    corridor_half_width_m: float = 1.6
    buffer_m: float = 2.0
    ttc_critical_s: float = 3.0

    @property
    def effective_decel(self) -> float:
        return min(self.brake_capability_mps2, self.friction_mu * G)

    def stopping_distance(self, v: float) -> float:
        return v * self.reaction_time_s + v ** 2 / (2 * self.effective_decel)

    def d_crit(self, v: float) -> float:
        return self.stopping_distance(v) + self.buffer_m

    def t_stop(self, v: float) -> float:
        return self.reaction_time_s + v / self.effective_decel


def in_safety_critical_region(obj: Dict, ego_speed: float,
                              params: Optional[SafetyRegionParams] = None) -> Dict:
    """Classify one object. obj: {x, y (ego frame), class_name, vx, vy,
    occluded}. Returns the verdict plus every intermediate quantity."""
    p = params or SafetyRegionParams()
    x, y = float(obj["x"]), float(obj["y"])
    cls = obj.get("class_name", "vehicle")
    d_crit = p.d_crit(ego_speed)
    lat_band = (p.corridor_half_width_m
                + LATERAL_SPEED_CAP.get(cls, 5.0) * p.t_stop(ego_speed))
    longitudinal = 0.0 <= x <= d_crit and abs(y) <= lat_band

    ttc = None
    if "vx" in obj and "vy" in obj:
        ego_state = {"x": 0.0, "y": 0.0, "speed": ego_speed, "heading": 0.0}
        obj_state = {"x": x, "y": y,
                     "speed": math.hypot(obj["vx"], obj["vy"]),
                     "heading": math.atan2(obj["vy"], obj["vx"])
                     if (obj["vx"], obj["vy"]) != (0.0, 0.0) else 0.0}
        dims = tuple(obj.get("dims", [1.0, 1.0])[:2])
        ttc = projected_ttc(ego_state, (4.5, 1.9), obj_state, dims,
                            look_ahead=p.ttc_critical_s + 1.0, dt=0.1)
    ttc_critical = ttc is not None and ttc <= p.ttc_critical_s

    return {"safety_critical": bool(longitudinal or ttc_critical),
            "longitudinal_critical": bool(longitudinal),
            "ttc_critical": bool(ttc_critical),
            "d_crit_m": round(d_crit, 3),
            "lateral_band_m": round(lat_band, 3),
            "effective_decel_mps2": round(p.effective_decel, 3),
            "ttc_s": None if ttc is None else round(ttc, 3)}


def risk_weight(obj: Dict, ego_speed: float,
                params: Optional[SafetyRegionParams] = None) -> float:
    """Deterministic risk weight: vulnerability x proximity x closing speed
    x TTC x occlusion x ego speed."""
    p = params or SafetyRegionParams()
    x, y = float(obj["x"]), float(obj["y"])
    dist = math.hypot(x, y)
    vuln = VULNERABILITY.get(obj.get("class_name", "vehicle"), 0.5)
    proximity = float(np.clip(1.0 - dist / 80.0, 0.05, 1.0))
    closing = ego_speed - float(obj.get("vx", 0.0)) if x > 0 else 0.0
    closing_f = float(np.clip(0.3 + closing / 20.0, 0.3, 1.5))
    region = in_safety_critical_region(obj, ego_speed, p)
    ttc = region["ttc_s"]
    ttc_f = 1.5 if ttc is None and region["safety_critical"] else (
        float(np.clip(2.0 - (ttc or 10.0) / p.ttc_critical_s, 0.5, 2.0)))
    occl_f = 1.25 if obj.get("occluded") else 1.0
    ego_f = float(np.clip(ego_speed / 15.0, 0.3, 1.5))
    return round(vuln * proximity * closing_f * ttc_f * occl_f * ego_f, 6)


# ------------------------------------------------------------ reporting


def safety_report(objects: List[Dict], ego_speed: float = 10.0,
                  params: Optional[SafetyRegionParams] = None,
                  data_label: str = "SIMULATED") -> Dict:
    """Full safety-informed report for a set of GT objects with `detected`
    flags. Open-loop recall is ALWAYS reported alongside SCR."""
    p = params or SafetyRegionParams()
    n = len(objects)
    detected = np.array([bool(o.get("detected")) for o in objects])
    critical = np.zeros(n, dtype=bool)
    weights = np.zeros(n)
    per_object = []
    for i, o in enumerate(objects):
        region = in_safety_critical_region(o, ego_speed, p)
        critical[i] = region["safety_critical"]
        weights[i] = risk_weight(o, ego_speed, p)
        per_object.append({**{k: o.get(k) for k in
                              ("x", "y", "class_name", "occluded", "detected")},
                           **region, "risk_weight": weights[i]})

    overall_recall = float(detected.mean()) if n else None
    n_crit = int(critical.sum())
    scr = float(detected[critical].mean()) if n_crit else None
    rw_recall = (float((weights * detected).sum() / weights.sum())
                 if weights.sum() > 0 else None)

    by_class: Dict[str, Dict] = {}
    for cls in sorted({o.get("class_name", "vehicle") for o in objects}):
        mask = np.array([o.get("class_name") == cls for o in objects])
        cmask = mask & critical
        by_class[cls] = {
            "n": int(mask.sum()),
            "recall": round(float(detected[mask].mean()), 4) if mask.any() else None,
            "n_safety_critical": int(cmask.sum()),
            "safety_critical_recall": (round(float(detected[cmask].mean()), 4)
                                       if cmask.any() else None),
            "risk_weighted_recall": (round(float((weights[mask] * detected[mask]).sum()
                                                 / weights[mask].sum()), 4)
                                     if weights[mask].sum() > 0 else None),
        }

    return {
        "data_label": data_label,
        "region_params": asdict(p),
        "region_definition": (
            "d_crit(v) = v*t_react + v^2/(2*min(brake, mu*g)) + buffer; "
            "lateral band = corridor + v_lat_max(class)*t_stop(v); "
            "OR projected TTC <= ttc_critical_s (ssam_ext conflict-point "
            "projection)"),
        "ego_speed_mps": ego_speed,
        "open_loop": {"n_objects": n, "recall": _r(overall_recall)},
        "safety_informed": {
            "n_safety_critical": n_crit,
            "safety_critical_recall": _r(scr),
            "risk_weighted_recall": _r(rw_recall),
        },
        "by_class": by_class,
        "per_object": per_object,
        "note": "open-loop and safety-informed metrics are complementary; "
                "reports always show both",
    }


# ------------------------------------------------------------ divergence demo


def divergence_demo(ego_speed: float = 12.0) -> Dict:
    """Constructed demonstration: candidate improves OVERALL recall while
    Safety-Critical Recall DEGRADES. Deterministic synthetic data:

    * 400 far-field vehicles (45-75 m, off-corridor): easy volume. The
      candidate detects far more of them (0.60 -> 0.85) — e.g. a model tuned
      for long-range benchmarks.
    * 60 near in-path pedestrians (inside d_crit, in corridor): the candidate
      regresses (0.92 -> 0.78) — e.g. lost close-range VRU sensitivity.

    Detected counts are exact (first k of each group), so the demonstration
    is fully deterministic and testable.
    """
    p = SafetyRegionParams()
    objects_base: List[Dict] = []
    objects_cand: List[Dict] = []

    def add_group(n: int, base_rate: float, cand_rate: float, maker) -> None:
        k_base, k_cand = int(round(n * base_rate)), int(round(n * cand_rate))
        for i in range(n):
            o = maker(i)
            objects_base.append({**o, "detected": i < k_base})
            objects_cand.append({**o, "detected": i < k_cand})

    add_group(400, 0.60, 0.85, lambda i: {
        "x": 45.0 + 30.0 * (i % 20) / 20.0, "y": 6.0 * (1 if i % 2 else -1),
        "class_name": "vehicle", "vx": 8.0, "vy": 0.0, "occluded": False})
    d_near = p.d_crit(ego_speed) * 0.6
    add_group(60, 0.92, 0.78, lambda i: {
        "x": 4.0 + d_near * (i % 12) / 12.0, "y": 0.9 * (1 if i % 2 else -1),
        "class_name": "pedestrian", "vx": 0.0,
        "vy": 1.2 * (-1 if i % 2 else 1), "occluded": i % 5 == 0})

    base = safety_report(objects_base, ego_speed, p)
    cand = safety_report(objects_cand, ego_speed, p)
    for rep in (base, cand):
        rep.pop("per_object", None)

    return {
        "baseline": base,
        "candidate": cand,
        "deltas": {
            "overall_recall": _delta(base["open_loop"]["recall"],
                                     cand["open_loop"]["recall"]),
            "safety_critical_recall": _delta(
                base["safety_informed"]["safety_critical_recall"],
                cand["safety_informed"]["safety_critical_recall"]),
            "risk_weighted_recall": _delta(
                base["safety_informed"]["risk_weighted_recall"],
                cand["safety_informed"]["risk_weighted_recall"]),
        },
        "headline": "overall recall IMPROVES while safety-critical recall "
                    "DEGRADES — why launch decisions must never rely on "
                    "aggregate recall alone",
        "construction": "400 far-field vehicles (0.60->0.85 detected) + 60 "
                        "near in-path pedestrians (0.92->0.78 detected); "
                        "exact counts, deterministic",
        "data_label": "SIMULATED",
    }


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 4)


def _delta(b: Optional[float], c: Optional[float]) -> Optional[float]:
    if b is None or c is None:
        return None
    return round(c - b, 4)
