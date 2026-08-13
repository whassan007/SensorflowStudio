"""Extended SSAM surrogate safety measures (FHWA Surrogate Safety Assessment Model).

Extends the platform's existing TTC/PET/severity SSAM logic (app_backend
/api/ssam/*, kept intact) into a shared analysis module operating on TRJ-like
trajectory data:

- TTC via conflict-point projection: each vehicle is a rectangle polygon
  projected along its trajectory (constant-velocity extrapolation) over a
  look-ahead interval; TTC is the first projected time at which the two
  rectangles' separation reaches zero.
- PET via a zone-grid proximity function: actual footprints are rasterized
  into a spatial grid; PET is the minimum positive time gap between one road
  user leaving a cell and another entering it (classic SSAM definition).
- DRAC (deceleration rate to avoid crash) = closing_speed^2 / (2 * gap).
- DeltaS = max relative speed differential during a conflict (severity proxy).
- MaxS   = max absolute speed of either participant during a conflict.
- Collision probability over the look-ahead horizon:
  p(t) = clip(1 - TTC(t)/TTC_threshold, 0, 1); the conflict-level probability
  is the max instantaneous value, with the time integral reported as exposure.
- CSI (Conflict Severity Index, time-weighted): per conflict,
  CSI = sum_t [ 0.5 * mu * DeltaS(t)^2 * p(t) * dt ]
  where mu is the reduced mass proxy of the pair (kinetic-energy proxy of the
  relative motion). Aggregate CSI for a scenario suite is the sum across
  conflicts; the release Safety Gate blocks candidates whose aggregate CSI
  rises vs baseline beyond tolerance.

Honest markers: trajectories are synthetic (generate_trajectories) and the
model-conditioned reaction-time mapping in csi_for_run is a deterministic
simulation of how perception quality affects downstream conflict severity —
it is not a physics-validated driver model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.metrics.perception_3d import bev_iou
from sensorflow.safety.store import read_json, write_json

DEFAULT_PARAMS = {
    "ttc_threshold_s": 1.5,      # FHWA SSAM default TTC conflict threshold
    "pet_threshold_s": 5.0,      # FHWA SSAM default PET conflict threshold
    "look_ahead_s": 3.0,
    "projection_dt_s": 0.05,
    "grid_cell_m": 0.5,
    "rear_end_angle_deg": 30.0,  # SSAM conflict-type angle conventions
    "crossing_angle_deg": 85.0,
}

# Unit-less mass proxies for the kinetic-energy severity term.
MASS_PROXY = {
    "car": 1.0, "vehicle": 1.0, "truck": 2.5, "bus": 2.2,
    "motorcycle": 0.25, "cyclist": 0.10, "pedestrian": 0.08,
}


# ------------------------------------------------------------------ geometry


def _velocity(state: Dict) -> Tuple[float, float]:
    return (state["speed"] * math.cos(state["heading"]),
            state["speed"] * math.sin(state["heading"]))


def _support_extent(length: float, width: float, heading: float,
                    line_angle: float) -> float:
    """Half-extent of a rectangle along a line direction (support function)."""
    phi = line_angle - heading
    return (length / 2) * abs(math.cos(phi)) + (width / 2) * abs(math.sin(phi))


def rect_gap(pa: Tuple[float, float], ha: float, la: float, wa: float,
             pb: Tuple[float, float], hb: float, lb: float, wb: float) -> float:
    """Separation between two rectangles along the center line (support-function
    approximation; exact for aligned rear-end geometry, conservative otherwise)."""
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return 0.0
    ang = math.atan2(dy, dx)
    gap = dist - _support_extent(la, wa, ha, ang) - _support_extent(lb, wb, hb, ang)
    return max(0.0, gap)


def rects_overlap(pa, ha, la, wa, pb, hb, lb, wb) -> bool:
    """Exact rotated-rectangle overlap via the shared BEV IoU clipper."""
    la, wa = max(la, 0.05), max(wa, 0.05)
    lb, wb = max(lb, 0.05), max(wb, 0.05)
    return bev_iou([pa[0], pa[1], 0, la, wa, 0, ha],
                   [pb[0], pb[1], 0, lb, wb, 0, hb]) > 1e-12


def projected_ttc(state_a: Dict, dims_a: Tuple[float, float],
                  state_b: Dict, dims_b: Tuple[float, float],
                  look_ahead: float, dt: float) -> Optional[float]:
    """Conflict-point projection TTC: first projected time (constant-velocity
    rectangle projection) at which the separation reaches zero.

    Scans the look-ahead interval and linearly interpolates the zero crossing
    of the gap, so linear closing geometry yields near-exact TTC values.
    """
    vax, vay = _velocity(state_a)
    vbx, vby = _velocity(state_b)
    la, wa = dims_a
    lb, wb = dims_b
    prev_tau, prev_gap = 0.0, None
    tau = 0.0
    while tau <= look_ahead + 1e-9:
        pa = (state_a["x"] + vax * tau, state_a["y"] + vay * tau)
        pb = (state_b["x"] + vbx * tau, state_b["y"] + vby * tau)
        gap = rect_gap(pa, state_a["heading"], la, wa, pb, state_b["heading"], lb, wb)
        if gap <= 1e-9 or rects_overlap(pa, state_a["heading"], la, wa,
                                        pb, state_b["heading"], lb, wb):
            if prev_gap is None or prev_gap <= 1e-9:
                return tau
            frac = prev_gap / max(prev_gap - gap, 1e-9)
            return prev_tau + frac * (tau - prev_tau)
        prev_tau, prev_gap = tau, gap
        tau += dt
    return None


def drac(state_a: Dict, dims_a: Tuple[float, float],
         state_b: Dict, dims_b: Tuple[float, float]) -> Optional[float]:
    """Deceleration Rate to Avoid Crash = closing_speed^2 / (2 * gap)."""
    pa, pb = (state_a["x"], state_a["y"]), (state_b["x"], state_b["y"])
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return None
    vax, vay = _velocity(state_a)
    vbx, vby = _velocity(state_b)
    # d|pb-pa|/dt ; negative when the pair is closing.
    range_rate = ((dx * (vbx - vax)) + (dy * (vby - vay))) / dist
    closing = -range_rate
    if closing <= 0:
        return None
    gap = rect_gap(pa, state_a["heading"], *dims_a, pb, state_b["heading"], *dims_b)
    return closing ** 2 / (2 * max(gap, 0.1))


def collision_probability(ttc: Optional[float], ttc_threshold: float) -> float:
    """Instantaneous collision probability proxy: 1 - TTC/threshold, clipped."""
    if ttc is None:
        return 0.0
    return float(np.clip(1.0 - ttc / ttc_threshold, 0.0, 1.0))


def _conflict_type(heading_a: float, heading_b: float, params: Dict) -> str:
    diff = abs(heading_a - heading_b) % (2 * math.pi)
    diff = min(diff, 2 * math.pi - diff)
    deg = math.degrees(diff)
    if deg < params["rear_end_angle_deg"]:
        return "rear_end"
    if deg > params["crossing_angle_deg"]:
        return "crossing"
    return "lane_change"


# ------------------------------------------------------------------ zone-grid PET


def _footprint_cells(state: Dict, length: float, width: float, cell: float) -> List[Tuple[int, int]]:
    """Grid cells whose centers fall inside the vehicle rectangle footprint."""
    x, y, h = state["x"], state["y"], state["heading"]
    half_diag = math.hypot(length, width) / 2 + cell
    cells = []
    i0 = int(math.floor((x - half_diag) / cell))
    i1 = int(math.ceil((x + half_diag) / cell))
    j0 = int(math.floor((y - half_diag) / cell))
    j1 = int(math.ceil((y + half_diag) / cell))
    c, s = math.cos(-h), math.sin(-h)
    hl, hw = max(length / 2, cell / 2), max(width / 2, cell / 2)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            cx, cy = (i + 0.5) * cell - x, (j + 0.5) * cell - y
            rx = cx * c - cy * s
            ry = cx * s + cy * c
            if abs(rx) <= hl and abs(ry) <= hw:
                cells.append((i, j))
    return cells


def zone_grid_pet(traj_a: Dict, traj_b: Dict, cell: float) -> Optional[float]:
    """Post-encroachment time via zone-grid occupancy intervals.

    PET = min over shared cells of the positive gap between one vehicle leaving
    the cell and the other entering it.
    """
    def occupancy(traj) -> Dict[Tuple[int, int], List[float]]:
        occ: Dict[Tuple[int, int], List[float]] = {}
        for st in traj["states"]:
            for c in _footprint_cells(st, traj.get("length", 0.0),
                                      traj.get("width", 0.0), cell):
                span = occ.setdefault(c, [st["t"], st["t"]])
                span[0] = min(span[0], st["t"])
                span[1] = max(span[1], st["t"])
        return occ

    occ_a, occ_b = occupancy(traj_a), occupancy(traj_b)
    pets = []
    for c in set(occ_a) & set(occ_b):
        (a0, a1), (b0, b1) = occ_a[c], occ_b[c]
        if b0 > a1:
            pets.append(b0 - a1)
        elif a0 > b1:
            pets.append(a0 - b1)
        # overlapping occupancy => simultaneous presence (PET 0 handled by TTC path)
        else:
            pets.append(0.0)
    return min(pets) if pets else None


# ------------------------------------------------------------------ analysis


def analyze_trajectories(trajectories: List[Dict], params: Optional[Dict] = None) -> Dict:
    """Full SSAM-extended analysis: conflicts + surrogate measures + CSI."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    conflicts: List[Dict] = []
    total_time = 0.0
    for traj in trajectories:
        ts = [s["t"] for s in traj["states"]]
        if len(ts) >= 2:
            total_time = max(total_time, ts[-1] - ts[0])

    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            conflicts.extend(_analyze_pair(trajectories[i], trajectories[j], p))

    agg = {
        "num_conflicts": len(conflicts),
        "by_type": {},
        "min_ttc_s": None,
        "min_pet_s": None,
        "max_drac_mps2": None,
        "max_delta_s_mps": None,
        "max_s_mps": None,
        "max_collision_probability": None,
        "aggregate_csi": round(sum(c["csi"] for c in conflicts), 6),
        "mean_csi_per_conflict": (round(sum(c["csi"] for c in conflicts) / len(conflicts), 6)
                                  if conflicts else 0.0),
        "observed_duration_s": round(total_time, 3),
    }
    for c in conflicts:
        agg["by_type"][c["conflict_type"]] = agg["by_type"].get(c["conflict_type"], 0) + 1
        for key, src, fn in (("min_ttc_s", "min_ttc_s", min), ("min_pet_s", "pet_s", min),
                             ("max_drac_mps2", "max_drac_mps2", max),
                             ("max_delta_s_mps", "delta_s_mps", max),
                             ("max_s_mps", "max_s_mps", max),
                             ("max_collision_probability", "collision_probability", max)):
            v = c.get(src)
            if v is None:
                continue
            agg[key] = v if agg[key] is None else fn(agg[key], v)

    return {"params": p, "conflicts": conflicts, "aggregate": agg,
            "measures_glossary": {
                "TTC": "time-to-collision via rectangle conflict-point projection",
                "PET": "post-encroachment time via zone-grid occupancy",
                "DRAC": "deceleration rate to avoid crash = closing^2/(2*gap)",
                "DeltaS": "max relative speed differential during conflict",
                "MaxS": "max absolute speed during conflict",
                "CSI": "time-weighted sum of 0.5*mu*DeltaS^2 * P(collision) * dt",
            }}


def _analyze_pair(traj_a: Dict, traj_b: Dict, p: Dict) -> List[Dict]:
    dims_a = (traj_a.get("length", 0.0), traj_a.get("width", 0.0))
    dims_b = (traj_b.get("length", 0.0), traj_b.get("width", 0.0))
    mass_a = MASS_PROXY.get(traj_a.get("vehicle_type", "car"), 1.0)
    mass_b = MASS_PROXY.get(traj_b.get("vehicle_type", "car"), 1.0)
    mu = (mass_a * mass_b) / max(mass_a + mass_b, 1e-9)  # reduced mass proxy

    by_t_b = {round(s["t"], 6): s for s in traj_b["states"]}
    samples = []
    for sa in traj_a["states"]:
        sb = by_t_b.get(round(sa["t"], 6))
        if sb is None:
            continue
        ttc = projected_ttc(sa, dims_a, sb, dims_b, p["look_ahead_s"], p["projection_dt_s"])
        samples.append((sa, sb, ttc))

    pet = zone_grid_pet(traj_a, traj_b, p["grid_cell_m"])

    # Contiguous samples with TTC below threshold form one conflict event.
    events: List[List[Tuple[Dict, Dict, float]]] = []
    current: List[Tuple[Dict, Dict, float]] = []
    for sa, sb, ttc in samples:
        active = ttc is not None and ttc <= p["ttc_threshold_s"]
        if active:
            current.append((sa, sb, ttc))
        elif current:
            events.append(current)
            current = []
    if current:
        events.append(current)

    ts = sorted(s["t"] for s in traj_a["states"])
    dt = (ts[1] - ts[0]) if len(ts) >= 2 else 0.1

    conflicts = []
    for ev in events:
        delta_s = max_s = 0.0
        max_drac_v: Optional[float] = None
        min_ttc = None
        max_p = 0.0
        csi = 0.0
        exposure = 0.0
        for sa, sb, ttc in ev:
            vax, vay = _velocity(sa)
            vbx, vby = _velocity(sb)
            ds = math.hypot(vax - vbx, vay - vby)
            delta_s = max(delta_s, ds)
            max_s = max(max_s, sa["speed"], sb["speed"])
            d = drac(sa, dims_a, sb, dims_b)
            if d is not None:
                max_drac_v = d if max_drac_v is None else max(max_drac_v, d)
            min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
            prob = collision_probability(ttc, p["ttc_threshold_s"])
            max_p = max(max_p, prob)
            csi += 0.5 * mu * ds ** 2 * prob * dt
            exposure += prob * dt
        sa0, sb0, ttc0 = ev[0]
        min_sample = min(ev, key=lambda e: e[2])
        sa_m, sb_m, ttc_m = min_sample
        vax, vay = _velocity(sa_m)
        vbx, vby = _velocity(sb_m)
        cp = ((sa_m["x"] + sb_m["x"]) / 2 + (vax + vbx) / 2 * ttc_m,
              (sa_m["y"] + sb_m["y"]) / 2 + (vay + vby) / 2 * ttc_m)
        conflicts.append({
            "vehicle_a": traj_a["vehicle_id"],
            "vehicle_b": traj_b["vehicle_id"],
            "conflict_type": _conflict_type(sa_m["heading"], sb_m["heading"], p),
            "t_start_s": round(sa0["t"], 3),
            "t_end_s": round(ev[-1][0]["t"], 3),
            "duration_s": round(ev[-1][0]["t"] - sa0["t"] + dt, 3),
            "min_ttc_s": round(min_ttc, 4),
            "pet_s": None if pet is None else round(pet, 4),
            "max_drac_mps2": None if max_drac_v is None else round(max_drac_v, 4),
            "delta_s_mps": round(delta_s, 4),
            "max_s_mps": round(max_s, 4),
            "collision_probability": round(max_p, 4),
            "probability_exposure_s": round(exposure, 4),
            "csi": round(csi, 6),
            "mass_proxy_reduced": round(mu, 4),
            "conflict_point": [round(cp[0], 2), round(cp[1], 2)],
        })
    return conflicts


# ------------------------------------------------------------------ synthetic TRJ data


def _states(xy_fn, speed_fn, heading: float, duration: float, dt: float) -> List[Dict]:
    out = []
    t = 0.0
    while t <= duration + 1e-9:
        x, y = xy_fn(t)
        out.append({"t": round(t, 4), "x": round(x, 4), "y": round(y, 4),
                    "speed": round(max(speed_fn(t), 0.0), 4), "heading": heading})
        t += dt
    return out


def _braking_profile(v0: float, t_brake: float, decel: float):
    """Position/speed along a 1-D line: constant v0, then braking at decel."""
    t_stop = t_brake + v0 / decel

    def pos(t: float) -> float:
        if t <= t_brake:
            return v0 * t
        tb = min(t, t_stop) - t_brake
        return v0 * t_brake + v0 * tb - 0.5 * decel * tb ** 2

    def speed(t: float) -> float:
        if t <= t_brake:
            return v0
        return max(0.0, v0 - decel * (t - t_brake))

    return pos, speed


def scenario_crossing(reaction_delay_s: float = 0.6, duration: float = 8.0,
                      dt: float = 0.1, seed: int = 0) -> List[Dict]:
    """Ego approaches a crossing pedestrian; brakes reaction_delay after t_detect=1s."""
    rng = np.random.default_rng(seed)
    v_ego = 12.5 + float(rng.uniform(-0.3, 0.3))
    ped_v = 1.9 + float(rng.uniform(-0.1, 0.1))
    pos, speed = _braking_profile(v_ego, 1.0 + reaction_delay_s, 5.0)
    ego = {"vehicle_id": "ego", "vehicle_type": "car", "length": 4.5, "width": 1.9,
           "states": _states(lambda t: (-42.0 + pos(t), 0.0), speed, 0.0, duration, dt)}
    ped = {"vehicle_id": "ped-1", "vehicle_type": "pedestrian", "length": 0.6, "width": 0.6,
           "states": _states(lambda t: (0.0, -6.5 + ped_v * t), lambda t: ped_v,
                             math.pi / 2, duration, dt)}
    return [ego, ped]


def scenario_rear_end(reaction_delay_s: float = 0.6, duration: float = 8.0,
                      dt: float = 0.1, seed: int = 0) -> List[Dict]:
    """Lead vehicle brakes hard at t=1s; ego brakes reaction_delay later."""
    rng = np.random.default_rng(seed + 100)
    v = 15.0 + float(rng.uniform(-0.4, 0.4))
    lead_pos, lead_speed = _braking_profile(v, 1.0, 6.0)
    ego_pos, ego_speed = _braking_profile(v, 1.0 + reaction_delay_s, 5.5)
    lead = {"vehicle_id": "lead", "vehicle_type": "car", "length": 4.5, "width": 1.9,
            "states": _states(lambda t: (18.0 + lead_pos(t), 0.0), lead_speed, 0.0, duration, dt)}
    ego = {"vehicle_id": "ego", "vehicle_type": "car", "length": 4.5, "width": 1.9,
           "states": _states(lambda t: (0.0 + ego_pos(t), 0.0), ego_speed, 0.0, duration, dt)}
    return [ego, lead]


def scenario_lane_change(reaction_delay_s: float = 0.6, duration: float = 8.0,
                         dt: float = 0.1, seed: int = 0) -> List[Dict]:
    """Cyclist merges toward the ego lane at a shallow (lane-change) angle."""
    rng = np.random.default_rng(seed + 200)
    v_ego = 11.0 + float(rng.uniform(-0.3, 0.3))
    v_cyc = 4.5
    heading = math.radians(35.0)  # oblique merge toward the ego lane
    pos, speed = _braking_profile(v_ego, 0.8 + reaction_delay_s, 4.5)
    ego = {"vehicle_id": "ego", "vehicle_type": "car", "length": 4.5, "width": 1.9,
           "states": _states(lambda t: (-14.0 + pos(t), 0.0), speed, 0.0, duration, dt)}
    cyc = {"vehicle_id": "cyclist-1", "vehicle_type": "cyclist", "length": 1.8, "width": 0.6,
           "states": _states(lambda t: (2.0 + v_cyc * math.cos(heading) * t,
                                        -3.0 + v_cyc * math.sin(heading) * t),
                             lambda t: v_cyc, heading, duration, dt)}
    return [ego, cyc]


_SCENARIOS = {
    "crossing": scenario_crossing,
    "rear_end": scenario_rear_end,
    "lane_change": scenario_lane_change,
}


def generate_trajectories(seed: int = 0, scenario: str = "mixed",
                          reaction_delay_s: float = 0.6,
                          duration: float = 8.0, dt: float = 0.1) -> List[Dict]:
    """Deterministic synthetic TRJ-like trajectory data to exercise the analyzer."""
    if scenario in _SCENARIOS:
        return _SCENARIOS[scenario](reaction_delay_s, duration, dt, seed)
    if scenario != "mixed":
        raise ValueError(f"unknown scenario {scenario!r}; expected one of "
                         f"{sorted(_SCENARIOS)} or 'mixed'")
    out: List[Dict] = []
    for name, fn in _SCENARIOS.items():
        for traj in fn(reaction_delay_s, duration, dt, seed):
            traj = dict(traj)
            traj["vehicle_id"] = f"{name}:{traj['vehicle_id']}"
            out.append(traj)
    return out


# ------------------------------------------------------------------ run-level CSI


def reaction_delay_for_model(model_version: str, overrides: Optional[Dict] = None) -> Dict:
    """Deterministic mapping: perception skill profile -> downstream reaction delay.

    SIMULATED: a weaker perception stack (higher miss penalties) is modeled as a
    longer effective detection-to-brake delay, which the surrogate-safety
    scenarios convert into more severe conflicts. Marked simulated; the mapping
    is monotonic and seeded by the same model_params used by megaeval runs.
    """
    from sensorflow.megaeval.runs import model_params
    params = model_params(model_version, overrides or {})
    perception_score = params["base_detect"] - (
        params["night_penalty"] + params["rain_penalty"]
        + params["occlusion_penalty"] + params["vru_penalty"])
    deficit = float(np.clip(1.0 - perception_score, 0.0, 1.0))
    delay = 0.45 + 1.6 * deficit
    return {"model_version": model_version, "perception_score": round(perception_score, 4),
            "perception_deficit": round(deficit, 4),
            "reaction_delay_s": round(delay, 4), "simulated": True}


def csi_for_run(run, seeds: Tuple[int, ...] = (0, 1, 2), force: bool = False) -> Dict:
    """Aggregate CSI for a megaeval evaluation run (cached under runs/safety/ssam/).

    Runs the three surrogate-safety scenarios (crossing / rear-end / lane-change)
    with the model-conditioned reaction delay and sums CSI across all conflicts.
    """
    cached = None if force else read_json("ssam", f"{run.run_id}.json")
    if cached is not None:
        return cached

    reaction = reaction_delay_for_model(run.model_version, run.overrides)
    scenario_rows = []
    total_csi = 0.0
    total_conflicts = 0
    min_ttc = None
    for name in _SCENARIOS:
        for seed in seeds:
            trajs = generate_trajectories(seed=seed, scenario=name,
                                          reaction_delay_s=reaction["reaction_delay_s"])
            res = analyze_trajectories(trajs)
            agg = res["aggregate"]
            total_csi += agg["aggregate_csi"]
            total_conflicts += agg["num_conflicts"]
            if agg["min_ttc_s"] is not None:
                min_ttc = agg["min_ttc_s"] if min_ttc is None else min(min_ttc, agg["min_ttc_s"])
            scenario_rows.append({"scenario": name, "seed": seed,
                                  "num_conflicts": agg["num_conflicts"],
                                  "min_ttc_s": agg["min_ttc_s"],
                                  "max_drac_mps2": agg["max_drac_mps2"],
                                  "aggregate_csi": agg["aggregate_csi"]})

    summary = {
        "run_id": run.run_id,
        "model_version": run.model_version,
        "reaction_model": reaction,
        "scenarios": scenario_rows,
        "total_conflicts": total_conflicts,
        "min_ttc_s": min_ttc,
        "aggregate_csi": round(total_csi, 6),
        "simulated": True,
        "method": "deterministic surrogate-safety scenario suite; CSI = "
                  "time-weighted kinetic-energy proxy x collision probability",
    }
    write_json(summary, "ssam", f"{run.run_id}.json")
    return summary
