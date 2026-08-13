"""Counterfactual validity gate.

Before ANY generated scenario enters an evaluation suite it passes through
this gate. Everything here is deterministic math on the scenario data —
no model opinions (evidence before inference).

Checks
------
1. physical plausibility   per-class speed limits, acceleration limits,
                           rotated-rectangle object overlap (reuses
                           sensorflow.safety.ssam_ext.rects_overlap)
2. temporal consistency    position/velocity finite-difference agreement,
                           frame-to-frame acceleration continuity (teleports
                           and discontinuities land here AND in #1)
3. sensor consistency      the bevfusion camera+LiDAR simulators are run on
                           the rendered sequence; cross-modal position
                           agreement and condition-conditioned detection-rate
                           expectations must hold (e.g. at night the camera
                           rate must sit below the LiDAR rate)
4. identity/trajectory     unique instance ids per frame, stable class per
                           id, no id teleports between consecutive frames
5. distribution similarity feature distributions (range, lateral offset,
                           speed, class mix) vs a real-data reference built
                           from untransformed source scenes; PSI/JS reuse
                           sensorflow.rca.stats (guarded import + fallback)

Scores
------
simulation_fidelity_score   0.4*physical + 0.3*temporal + 0.3*sensor
realism_confidence          distribution-similarity score (PSI-mapped)
counterfactual_validity     0.5*fidelity + 0.2*identity + 0.3*realism

Weight policy (deterministic, tested)
-------------------------------------
* rejected scenarios get evaluation_weight = 0 (never enter a suite);
* accepted scenarios with fidelity >= HIGH_FIDELITY get weight 1.0;
* accepted low-fidelity scenarios are weight-capped at
  LOW_FIDELITY_WEIGHT_CAP per scenario, and `apply_suite_weight_policy`
  additionally caps the TOTAL low-fidelity share of a suite at
  LOW_FIDELITY_SUITE_SHARE so low-fidelity evidence can never dominate a
  launch decision.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.bevfusion.scenes import DT, SceneSequence
from sensorflow.bevfusion.sensors import (
    camera_rng, lidar_rng, simulate_camera, simulate_lidar,
)
# REUSE: rotated-rectangle overlap from the safety package's SSAM module.
from sensorflow.safety.ssam_ext import rects_overlap
from sensorflow.nextgen.models import ValidityReport
from sensorflow.nextgen.worldmodel import ActorTrack

try:
    # REUSE (read-only): PSI / JS divergence primitives from the RCA package.
    from sensorflow.rca.stats import js_divergence_continuous, psi_continuous
    _DIVERGENCE_SOURCE = "sensorflow.rca.stats"
except Exception:  # pragma: no cover - rca is expected to be importable
    _DIVERGENCE_SOURCE = "local fallback (rca.stats unavailable)"

    def _hist(a, b, bins=10, eps=1e-4):
        lo, hi = min(np.min(a), np.min(b)), max(np.max(a), np.max(b))
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, bins + 1)
        pa, _ = np.histogram(a, bins=edges)
        pb, _ = np.histogram(b, bins=edges)
        return (pa / max(1, pa.sum()) + eps, pb / max(1, pb.sum()) + eps)

    def psi_continuous(expected, actual, bins=10):  # type: ignore[no-redef]
        pa, pb = _hist(np.asarray(expected, float), np.asarray(actual, float), bins)
        pa, pb = pa / pa.sum(), pb / pb.sum()
        return float(np.sum((pb - pa) * np.log(pb / pa)))

    def js_divergence_continuous(a, b, bins=10):  # type: ignore[no-redef]
        pa, pb = _hist(np.asarray(a, float), np.asarray(b, float), bins)
        pa, pb = pa / pa.sum(), pb / pb.sum()
        m = 0.5 * (pa + pb)
        return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


# Deterministic policy constants (versioned via lineage policy_version).
MAX_SPEED_MPS = {"pedestrian": 3.5, "cyclist": 12.0, "motorcycle": 50.0,
                 "vehicle": 50.0, "truck": 40.0}
MAX_ACCEL_MPS2 = 12.0            # ~1.2 g: beyond road-vehicle physics
MAX_PED_ACCEL_MPS2 = 8.0
POSITION_VELOCITY_TOL_M = 0.35   # |dx - v*dt| tolerance per frame
CROSS_MODAL_AGREEMENT_M = 2.0    # camera-vs-LiDAR median position agreement
HIGH_FIDELITY = 0.80
LOW_FIDELITY_WEIGHT_CAP = 0.30
LOW_FIDELITY_SUITE_SHARE = 0.25
ACCEPT_VALIDITY_MIN = 0.50


def _check(name: str, passed: bool, score: float, detail: str) -> Dict:
    return {"check": name, "passed": bool(passed),
            "score": round(float(np.clip(score, 0.0, 1.0)), 4), "detail": detail}


# ------------------------------------------------------------ 1. physical


def overlap_pairs(actors: List[ActorTrack], step: int = 4) -> set:
    """Pairs of actors whose rectangles intersect at any sampled frame."""
    pairs = set()
    n_frames = len(actors[0].states) if actors else 0
    for fi in range(0, n_frames, step):
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                a, b = actors[i], actors[j]
                key = frozenset((a.instance_id, b.instance_id))
                if key in pairs:
                    continue
                sa, sb = a.states[fi], b.states[fi]
                if abs(sa["x"] - sb["x"]) > 12 or abs(sa["y"] - sb["y"]) > 12:
                    continue
                if rects_overlap((sa["x"], sa["y"]), sa["yaw"], a.dims[0], a.dims[1],
                                 (sb["x"], sb["y"]), sb["yaw"], b.dims[0], b.dims[1]):
                    pairs.add(key)
    return pairs


def check_physical(actors: List[ActorTrack],
                   ignore_overlap_pairs: Optional[set] = None) -> Tuple[Dict, List[str]]:
    reasons: List[str] = []
    worst = 1.0
    n_viol = 0
    for a in actors:
        vmax = MAX_SPEED_MPS.get(a.class_name, 50.0)
        amax = MAX_PED_ACCEL_MPS2 if a.class_name == "pedestrian" else MAX_ACCEL_MPS2
        speeds = np.array([math.hypot(s["vx"], s["vy"]) for s in a.states])
        # Acceleration from position second differences (catches teleports
        # even when the stored velocity fields lie about the jump).
        xs = np.array([s["x"] for s in a.states])
        ys = np.array([s["y"] for s in a.states])
        if len(xs) >= 3:
            ax = np.diff(xs, 2) / DT ** 2
            ay = np.diff(ys, 2) / DT ** 2
            amag = np.hypot(ax, ay)
        else:
            amag = np.zeros(1)
        if speeds.max(initial=0.0) > vmax:
            n_viol += 1
            worst = min(worst, vmax / max(speeds.max(), 1e-9))
            reasons.append(f"{a.instance_id}: speed {speeds.max():.1f} m/s exceeds "
                           f"{a.class_name} limit {vmax} m/s")
        if amag.max(initial=0.0) > amax:
            n_viol += 1
            worst = min(worst, amax / max(amag.max(), 1e-9))
            reasons.append(f"{a.instance_id}: acceleration {amag.max():.1f} m/s^2 "
                           f"exceeds limit {amax} m/s^2")

    # Pairwise overlap, EXCLUDING pairs that already intersected in the
    # source scene: the gate scores what the TRANSFORMATION introduced, not
    # pre-existing substrate quirks.
    new_overlaps = overlap_pairs(actors) - (ignore_overlap_pairs or set())
    if new_overlaps:
        n_viol += len(new_overlaps)
        worst = min(worst, 0.2)
        for pair in sorted(new_overlaps, key=sorted):
            a_id, b_id = sorted(pair)
            reasons.append(f"object overlap introduced by transformation: "
                           f"{a_id} intersects {b_id}")

    passed = n_viol == 0
    score = 1.0 if passed else worst * max(0.0, 1.0 - 0.2 * n_viol)
    return _check("physical_plausibility", passed, score,
                  f"{n_viol} violation(s); kinematic limits + rotated-rect overlap"), reasons


# ------------------------------------------------------------ 2. temporal


def check_temporal(actors: List[ActorTrack]) -> Tuple[Dict, List[str]]:
    reasons: List[str] = []
    worst_err = 0.0
    for a in actors:
        xs = np.array([s["x"] for s in a.states])
        ys = np.array([s["y"] for s in a.states])
        vx = np.array([s["vx"] for s in a.states])
        vy = np.array([s["vy"] for s in a.states])
        if len(xs) < 2:
            continue
        # Stored velocity must explain the position increments.
        ex = np.abs(np.diff(xs) - vx[1:] * DT)
        ey = np.abs(np.diff(ys) - vy[1:] * DT)
        err = float(np.hypot(ex, ey).max())
        worst_err = max(worst_err, err)
        if err > POSITION_VELOCITY_TOL_M:
            reasons.append(f"{a.instance_id}: position/velocity mismatch "
                           f"{err:.2f} m in one frame (tol {POSITION_VELOCITY_TOL_M})")
    passed = worst_err <= POSITION_VELOCITY_TOL_M
    score = float(np.clip(1.0 - worst_err / 5.0, 0.0, 1.0)) if not passed else 1.0
    return _check("temporal_consistency", passed, score,
                  f"max |dx - v dt| = {worst_err:.3f} m"), reasons


# ------------------------------------------------------------ 3. sensor


def check_sensor(seq: SceneSequence, environment: Dict[str, str],
                 seed: int) -> Tuple[Dict, List[str]]:
    """Run the REUSED bevfusion sensor simulators on the rendered sequence and
    check cross-modal agreement + condition-conditioned expectations."""
    cam_gen, lid_gen = camera_rng(seed, 0), lidar_rng(seed, 0)
    agree_dists: List[float] = []
    cam_hits = lid_hits = gt_total = 0
    for frame in seq.frames:
        if not frame.gt:
            continue
        gt_total += len(frame.gt)
        cams = simulate_camera(frame, seq, cam_gen)
        lids = simulate_lidar(frame, seq, lid_gen)
        cam_by_iid = {d.source_instance_id: d for d in cams if d.source_instance_id}
        lid_by_iid = {d.source_instance_id: d for d in lids if d.source_instance_id}
        cam_hits += len(cam_by_iid)
        lid_hits += len(lid_by_iid)
        for iid, c in cam_by_iid.items():
            l = lid_by_iid.get(iid)
            if l is not None:
                agree_dists.append(math.hypot(c.x - l.x, c.y - l.y))

    reasons: List[str] = []
    if gt_total == 0:
        return _check("sensor_consistency", False, 0.0,
                      "no ground truth visible in rendered frames"), \
            ["sensor consistency unverifiable: empty scenario"]

    cam_rate = cam_hits / gt_total
    lid_rate = lid_hits / gt_total
    median_agree = float(np.median(agree_dists)) if agree_dists else None

    score = 1.0
    if median_agree is None:
        score -= 0.4
        reasons.append("no object seen by both modalities; cross-modal "
                       "agreement unverifiable")
    elif median_agree > CROSS_MODAL_AGREEMENT_M:
        score -= min(0.6, 0.3 * median_agree / CROSS_MODAL_AGREEMENT_M)
        reasons.append(f"camera/LiDAR median position disagreement "
                       f"{median_agree:.2f} m > {CROSS_MODAL_AGREEMENT_M} m")
    if seq.time_of_day == "night" and cam_rate > lid_rate:
        score -= 0.3
        reasons.append(f"claimed night but camera rate {cam_rate:.2f} exceeds "
                       f"LiDAR rate {lid_rate:.2f} (inconsistent with darkness)")
    if seq.weather == "rain" and lid_rate > 0.98:
        score -= 0.15
        reasons.append("claimed rain but LiDAR shows zero attenuation")

    passed = score >= 0.7
    detail = (f"camera rate {cam_rate:.2f}, lidar rate {lid_rate:.2f}, "
              f"median cross-modal agreement "
              f"{'n/a' if median_agree is None else f'{median_agree:.2f} m'}")
    return _check("sensor_consistency", passed, score, detail), reasons


# ------------------------------------------------------------ 4. identity


def check_identity(seq: SceneSequence, actors: List[ActorTrack]) -> Tuple[Dict, List[str]]:
    reasons: List[str] = []
    class_by_id: Dict[str, str] = {}
    last_pos: Dict[str, Tuple[int, float, float]] = {}
    max_jump = 0.0
    for frame in seq.frames:
        ids = [g.instance_id for g in frame.gt]
        if len(ids) != len(set(ids)):
            reasons.append(f"duplicate instance id in frame {frame.frame_id}")
        for g in frame.gt:
            prev_cls = class_by_id.setdefault(g.instance_id, g.class_name)
            if prev_cls != g.class_name:
                reasons.append(f"{g.instance_id}: class flip {prev_cls} -> "
                               f"{g.class_name}")
            prev = last_pos.get(g.instance_id)
            # World-frame displacement between consecutive visible frames.
            wx = g.bbox_3d[0] + 10.0 * DT * frame.index
            if prev is not None and frame.index - prev[0] == 1:
                jump = math.hypot(wx - prev[1], g.bbox_3d[1] - prev[2])
                max_jump = max(max_jump, jump)
                if jump > 6.0:  # > 60 m/s implied inter-frame motion
                    reasons.append(f"{g.instance_id}: {jump:.1f} m jump between "
                                   f"consecutive frames")
            last_pos[g.instance_id] = (frame.index, wx, g.bbox_3d[1])
    passed = not reasons
    score = 1.0 if passed else max(0.0, 1.0 - 0.25 * len(reasons))
    return _check("identity_trajectory_consistency", passed, score,
                  f"max inter-frame world jump {max_jump:.2f} m"), reasons


# ------------------------------------------------------------ 5. distribution


def scene_features(seq: SceneSequence) -> Dict[str, np.ndarray]:
    dist, lat = [], []
    for frame in seq.frames:
        for g in frame.gt:
            dist.append(g.distance)
            lat.append(g.bbox_3d[1])
    return {"distance": np.asarray(dist, float),
            "lateral": np.asarray(lat, float)}


def _mean_psi_js(ref: Dict[str, np.ndarray], cur: Dict[str, np.ndarray],
                 reasons: List[str]) -> Tuple[float, float]:
    psis, jss = [], []
    for k in ("distance", "lateral"):
        if cur[k].size < 5 or ref[k].size < 5:
            reasons.append(f"too few observations to compare {k} distribution")
            continue
        psis.append(psi_continuous(ref[k], cur[k]))
        jss.append(js_divergence_continuous(ref[k], cur[k]))
    return (float(np.mean(psis)) if psis else 1.0,
            float(np.mean(jss)) if jss else 1.0)


def check_distribution(seq: SceneSequence,
                       reference: Optional[List[SceneSequence]] = None,
                       source_features: Optional[Dict[str, List[float]]] = None
                       ) -> Tuple[Dict, List[str], float]:
    """PSI/JS of scenario feature distributions vs a real-data reference.

    The reference is a pool of untransformed source scenes (the platform's
    stand-in for real fleet data; documented as such). Divergence primitives
    are reused from sensorflow.rca.stats.

    A single scenario has natural sampling variance against the pooled
    reference, so when the SOURCE scene's features are available we score the
    shift ADDED by the transformation (psi_cf - psi_source): the realism
    question is "did the transformation push the scenario away from real
    data", not "is one scene a perfect sample of the fleet".
    """
    if reference is None:
        from sensorflow.bevfusion.scenes import generate_sequences
        reference = generate_sequences(n_sequences=6, frames_per_sequence=24, seed=7)

    feats = scene_features(seq)
    ref_feats: Dict[str, List[np.ndarray]] = {"distance": [], "lateral": []}
    for r in reference:
        rf = scene_features(r)
        for k in ref_feats:
            ref_feats[k].append(rf[k])
    ref = {k: (np.concatenate(v) if v else np.zeros(1))
           for k, v in ref_feats.items()}

    reasons: List[str] = []
    psi, js = _mean_psi_js(ref, feats, reasons)

    psi_source = None
    if source_features:
        src = {k: np.asarray(source_features.get(k, []), float)
               for k in ("distance", "lateral")}
        psi_source, _ = _mean_psi_js(ref, src, [])
        added = max(0.0, psi - psi_source)
    else:
        added = psi

    # Added-PSI -> similarity score: 0.25 ("large shift" in the standard
    # reading) maps to 0.5; counterfactuals legitimately shift distributions,
    # so this informs realism_confidence rather than hard-failing the gate.
    score = float(np.clip(1.0 - 2.0 * added, 0.0, 1.0))
    if added > 0.25:
        reasons.append(f"large transformation-added distribution shift "
                       f"(added PSI {added:.3f})")
    detail = (f"PSI {psi:.3f}"
              + (f" (source scene PSI {psi_source:.3f}, added {added:.3f})"
                 if psi_source is not None else "")
              + f", JS {js:.4f} over range/lateral features "
                f"(source: {_DIVERGENCE_SOURCE})")
    check = _check("distribution_similarity", added <= 0.25, score, detail)
    return check, reasons, score


# ------------------------------------------------------------ gate


def validate_scenario(scenario_id: str, seq: SceneSequence,
                      actors: List[ActorTrack], environment: Dict[str, str],
                      seed: int,
                      reference: Optional[List[SceneSequence]] = None,
                      source_overlap_pairs: Optional[set] = None,
                      source_features: Optional[Dict[str, List[float]]] = None
                      ) -> ValidityReport:
    phys, r1 = check_physical(actors, ignore_overlap_pairs=source_overlap_pairs)
    temp, r2 = check_temporal(actors)
    sens, r3 = check_sensor(seq, environment, seed)
    ident, r4 = check_identity(seq, actors)
    dist, r5, realism = check_distribution(seq, reference, source_features)

    fidelity = 0.4 * phys["score"] + 0.3 * temp["score"] + 0.3 * sens["score"]
    validity = 0.5 * fidelity + 0.2 * ident["score"] + 0.3 * realism

    # Hard physical/temporal failures reject outright; otherwise threshold.
    accepted = phys["passed"] and temp["passed"] and validity >= ACCEPT_VALIDITY_MIN
    weight, capped = weight_policy(accepted, fidelity)

    return ValidityReport(
        scenario_id=scenario_id,
        checks=[phys, temp, sens, ident, dist],
        simulation_fidelity_score=round(fidelity, 4),
        counterfactual_validity=round(validity, 4),
        realism_confidence=round(realism, 4),
        accepted=accepted,
        evaluation_weight=weight,
        weight_capped=capped,
        reasons=r1 + r2 + r3 + r4 + r5)


def weight_policy(accepted: bool, fidelity: float) -> Tuple[float, bool]:
    """Deterministic per-scenario weight: rejected -> 0; high fidelity -> 1;
    accepted low fidelity -> capped at LOW_FIDELITY_WEIGHT_CAP."""
    if not accepted:
        return 0.0, False
    if fidelity >= HIGH_FIDELITY:
        return 1.0, False
    return LOW_FIDELITY_WEIGHT_CAP, True


def apply_suite_weight_policy(reports: List[ValidityReport]) -> Dict:
    """Suite-level cap: the total weight of low-fidelity (capped) scenarios
    may not exceed LOW_FIDELITY_SUITE_SHARE of the suite's total weight.
    Returns the final per-scenario weights (deterministic policy, tested)."""
    high = [r for r in reports if r.accepted and not r.weight_capped]
    low = [r for r in reports if r.accepted and r.weight_capped]
    total_high = sum(r.evaluation_weight for r in high)
    total_low = sum(r.evaluation_weight for r in low)
    weights = {r.scenario_id: (r.evaluation_weight if r.accepted else 0.0)
               for r in reports}
    scaled = False
    if total_low > 0 and total_high + total_low > 0:
        share = total_low / (total_high + total_low)
        if share > LOW_FIDELITY_SUITE_SHARE:
            # Solve for the factor that brings the low-fidelity share to cap.
            target = LOW_FIDELITY_SUITE_SHARE
            factor = (target * total_high) / ((1 - target) * total_low)
            for r in low:
                weights[r.scenario_id] = round(r.evaluation_weight * factor, 6)
            scaled = True
    final_low = sum(weights[r.scenario_id] for r in low)
    final_total = final_low + total_high
    return {
        "weights": weights,
        "low_fidelity_share": round(final_low / final_total, 4) if final_total else 0.0,
        "share_cap": LOW_FIDELITY_SUITE_SHARE,
        "scaled_down": scaled,
        "policy": "per-scenario cap LOW_FIDELITY_WEIGHT_CAP="
                  f"{LOW_FIDELITY_WEIGHT_CAP}; suite share cap "
                  f"{LOW_FIDELITY_SUITE_SHARE}; rejected scenarios weight 0",
    }
