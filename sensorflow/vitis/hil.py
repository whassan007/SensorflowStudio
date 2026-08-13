"""HIL quantization-gap regression detection.

Runs the SAME image-space detection pipeline through the float32 reference
backend and through the Vitis emulated backend (fixed-point + streaming +
LUT constraints) on identical frames, then answers three questions:

1. How far apart are the two paths, per object? (confidence drift, position
   drift, IoU, dropped/spurious detections, class flips)
2. WHY are they apart? One-factor-at-a-time ablation isolates precision
   loss (bit-width), streaming-depth artifacts (XFCVDEPTH), and HLS LUT
   approximation error, and attributes the observed gap to each factor.
3. Is the gap a REGRESSION in the sequential-testing sense? Paired
   per-object correctness deltas feed sensorflow.seqeval's anytime-valid
   PairedSequentialTest when importable; otherwise a local paired-t
   confidence-interval fallback produces the same three-outcome verdict.

Frames come from the bevfusion synthetic scene generator (read-only import),
so day/night/rain cohorts and planted occlusions are inherited for free.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from sensorflow.bevfusion.scenes import SceneSequence, generate_sequences
from sensorflow.vitis.backend import PipelineConfig, VisionBackend, get_backend
from sensorflow.vitis.render import (
    BEV_H, BEV_W, ego_to_bev_px, render_bev_frame,
)
from sensorflow.vitis.store import new_id, save_run

# Detection pipeline constants (identical for every backend).
DETECT_THRESHOLD = 0.30
MIN_BLOB_AREA = 3
PROC_H, PROC_W = BEV_H // 2, BEV_W // 2  # detection runs at half resolution
IOU_CORRECT = 0.30                        # detection counts as correct vs GT

# Ablation sweep uses I=4 integer bits (signal is in [0,1] with small gains).
SWEEP_INT_BITS = 4
HIGH_PRECISION = (24, 8)   # "constraint off" precision for ablation legs


# --------------------------------------------------------------------------
# Sequential verdict: seqeval if available, local paired-t fallback otherwise
# --------------------------------------------------------------------------

def _try_seqeval_test(delta: float, alpha: float):
    try:
        from sensorflow.seqeval.sequential import PairedSequentialTest
        return PairedSequentialTest(delta=delta, alpha=alpha)
    except Exception:
        return None


def _fallback_verdict(frame_deltas: List[float], delta: float,
                      alpha: float) -> Dict:
    """Paired-t CI on per-frame mean correctness deltas (d = vitis - ref).

    REGRESSION if the CI sits entirely below -delta, PASS if entirely above,
    else INSUFFICIENT_EVIDENCE. Not anytime-valid (fixed-n), which is why
    seqeval is preferred when importable.
    """
    d = np.asarray(frame_deltas, dtype=np.float64)
    n = d.size
    if n < 3:
        return {"decision": "INSUFFICIENT_EVIDENCE", "mean_delta": float(d.mean()) if n else 0.0,
                "ci": [-1.0, 1.0], "n": int(n), "method": "paired_t_fallback"}
    from scipy import stats
    mean = float(d.mean())
    sem = float(d.std(ddof=1)) / np.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    lo, hi = mean - tcrit * max(sem, 1e-12), mean + tcrit * max(sem, 1e-12)
    if hi < -delta:
        decision = "REGRESSION"
    elif lo > -delta:
        decision = "PASS"
    else:
        decision = "INSUFFICIENT_EVIDENCE"
    return {"decision": decision, "mean_delta": mean, "ci": [lo, hi],
            "n": int(n), "method": "paired_t_fallback"}


def sequential_verdict(frame_deltas: List[float],
                       object_pairs: List[Tuple[bool, bool]],
                       delta: float = 0.02, alpha: float = 0.05) -> Dict:
    """Three-outcome verdict on paired ref-vs-vitis correctness deltas."""
    test = _try_seqeval_test(delta, alpha)
    if test is not None:
        try:
            test.update_clusters(frame_deltas)
            if object_pairs:
                b = np.array([p[0] for p in object_pairs], dtype=bool)
                c = np.array([p[1] for p in object_pairs], dtype=bool)
                test.record_objects(b, c)
            decision = test.evaluate()
            lo, hi = test.delta_interval()
            return {"decision": decision,
                    "mean_delta": float(np.mean(frame_deltas)) if frame_deltas else 0.0,
                    "ci": [float(lo), float(hi)], "n": len(frame_deltas),
                    "method": "seqeval_anytime_valid"}
        except Exception:
            pass  # seqeval mid-edit: degrade to the local fallback
    return _fallback_verdict(frame_deltas, delta, alpha)


# --------------------------------------------------------------------------
# The detection pipeline (identical logic; only the backend differs)
# --------------------------------------------------------------------------

def _classify_area(area_px: float) -> str:
    if area_px < 9:
        return "pedestrian"
    if area_px < 16:
        return "cyclist"
    if area_px < 30:
        return "vehicle"
    return "truck"


def detect_frame(img: np.ndarray, backend: VisionBackend,
                 gain: float = 1.4) -> List[Dict]:
    """Preprocess through backend ops, then extract blob detections.

    Every arithmetic step before thresholding runs through the backend, so
    fixed-point/streaming/LUT effects propagate into the detections.
    """
    x = backend.gain_exposure(img, gain=gain, offset=-0.05)
    x = backend.gaussian_filter(x, sigma=1.0)
    x = backend.resize(x, PROC_H, PROC_W, method="bilinear")
    mask = x > DETECT_THRESHOLD
    labels, n = ndimage.label(mask)
    dets: List[Dict] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        area = float(ys.size)
        if area < MIN_BLOB_AREA:
            continue
        vals = x[ys, xs]
        dets.append({
            "cy": float(ys.mean()), "cx": float(xs.mean()),
            "h": float(ys.max() - ys.min() + 1),
            "w": float(xs.max() - xs.min() + 1),
            "area": area,
            "confidence": float(np.clip(vals.mean(), 0.0, 1.0)),
            "class_name": _classify_area(area),
        })
    return dets


def _gt_in_proc_px(gt) -> Dict:
    x, y, _, l, w, _, _ = gt.bbox_3d
    r, c = ego_to_bev_px(x, y)
    scale_r, scale_c = PROC_H / BEV_H, PROC_W / BEV_W
    return {"cy": r * scale_r, "cx": c * scale_c,
            "h": max(1.0, l / (80.0 / BEV_H) * scale_r),
            "w": max(1.0, w / (50.0 / BEV_W) * scale_c),
            "class_name": gt.class_name, "instance_id": gt.instance_id,
            "occluded": gt.occluded}


def _iou(a: Dict, b: Dict) -> float:
    ax0, ax1 = a["cx"] - a["w"] / 2, a["cx"] + a["w"] / 2
    ay0, ay1 = a["cy"] - a["h"] / 2, a["cy"] + a["h"] / 2
    bx0, bx1 = b["cx"] - b["w"] / 2, b["cx"] + b["w"] / 2
    by0, by1 = b["cy"] - b["h"] / 2, b["cy"] + b["h"] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _match_to_gt(dets: List[Dict], gts: List[Dict],
                 max_dist: float = 6.0) -> Dict[str, Dict]:
    """Greedy nearest match detection -> GT. Returns {instance_id: det}."""
    matched: Dict[str, Dict] = {}
    used = set()
    for gt in gts:
        best, best_d = None, max_dist
        for j, det in enumerate(dets):
            if j in used:
                continue
            d = np.hypot(det["cy"] - gt["cy"], det["cx"] - gt["cx"])
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            matched[gt["instance_id"]] = dets[best]
    return matched


# --------------------------------------------------------------------------
# Paired comparison over a scene set
# --------------------------------------------------------------------------

def run_paired_comparison(sequences: List[SceneSequence], config: PipelineConfig,
                          seed: int) -> Dict:
    """Run reference + vitis_emulated on identical frames; pair per object."""
    ref = get_backend("reference")
    vit = get_backend("vitis_emulated", config)

    frame_deltas: List[float] = []
    object_pairs: List[Tuple[bool, bool]] = []
    per_object: List[Dict] = []
    dropped = spurious = class_flips = 0
    ref_detected = vit_detected = total_gt = 0
    conf_drifts: List[float] = []
    pos_drifts: List[float] = []
    ious: List[float] = []
    cohort_acc: Dict[str, List[float]] = {}

    for seq in sequences:
        cohort = f"{seq.time_of_day}/{seq.weather}"
        for frame in seq.frames:
            gts = [_gt_in_proc_px(g) for g in frame.gt]
            if not gts:
                continue
            img = render_bev_frame(frame, seq, seed)
            dets_ref = detect_frame(img, ref)
            dets_vit = detect_frame(img, vit)
            m_ref = _match_to_gt(dets_ref, gts)
            m_vit = _match_to_gt(dets_vit, gts)
            spurious += max(0, (len(dets_vit) - len(m_vit)) -
                            (len(dets_ref) - len(m_ref)))

            frame_b, frame_c = [], []
            for gt in gts:
                total_gt += 1
                iid = gt["instance_id"]
                dr, dv = m_ref.get(iid), m_vit.get(iid)
                b_ok = bool(dr and _iou(dr, gt) >= IOU_CORRECT)
                c_ok = bool(dv and _iou(dv, gt) >= IOU_CORRECT)
                ref_detected += int(dr is not None)
                vit_detected += int(dv is not None)
                if dr and not dv:
                    dropped += 1
                object_pairs.append((b_ok, c_ok))
                frame_b.append(b_ok)
                frame_c.append(c_ok)
                rec = {"instance_id": iid, "frame_id": frame.frame_id,
                       "cohort": cohort, "occluded": gt["occluded"],
                       "ref_detected": dr is not None,
                       "vitis_detected": dv is not None,
                       "ref_correct": b_ok, "vitis_correct": c_ok}
                if dr and dv:
                    cd = dv["confidence"] - dr["confidence"]
                    pd = float(np.hypot(dv["cy"] - dr["cy"], dv["cx"] - dr["cx"]))
                    pair_iou = _iou(dr, dv)
                    flip = dr["class_name"] != dv["class_name"]
                    class_flips += int(flip)
                    conf_drifts.append(cd)
                    pos_drifts.append(pd)
                    ious.append(pair_iou)
                    rec.update({"confidence_drift": round(cd, 4),
                                "position_drift_px": round(pd, 3),
                                "pair_iou": round(pair_iou, 3),
                                "class_flip": flip,
                                "ref_class": dr["class_name"],
                                "vitis_class": dv["class_name"]})
                per_object.append(rec)
            d = float(np.mean(frame_c)) - float(np.mean(frame_b))
            frame_deltas.append(d)
            cohort_acc.setdefault(cohort, []).append(d)

    n_paired = len(conf_drifts)
    gap_score = _gap_score(dropped, class_flips, total_gt, conf_drifts, ious)
    return {
        "config": config.to_dict(),
        "totals": {
            "gt_objects": total_gt,
            "ref_detected": ref_detected,
            "vitis_detected": vit_detected,
            "paired_detections": n_paired,
            "dropped_by_vitis": dropped,
            "spurious_in_vitis": spurious,
            "class_flips": class_flips,
        },
        "drift": {
            "mean_confidence_drift": round(float(np.mean(conf_drifts)), 5) if conf_drifts else 0.0,
            "mean_abs_confidence_drift": round(float(np.mean(np.abs(conf_drifts))), 5) if conf_drifts else 0.0,
            "mean_position_drift_px": round(float(np.mean(pos_drifts)), 4) if pos_drifts else 0.0,
            "mean_pair_iou": round(float(np.mean(ious)), 4) if ious else 1.0,
        },
        "gap_score": round(gap_score, 5),
        "frame_deltas": [round(d, 5) for d in frame_deltas],
        "object_pairs": object_pairs,
        "per_object": per_object,
        "cohort_delta": {k: round(float(np.mean(v)), 5)
                         for k, v in sorted(cohort_acc.items())},
    }


def _gap_score(dropped: int, flips: int, total: int, conf_drifts: List[float],
               ious: List[float]) -> float:
    """Scalar quantization-gap severity in [0, ~1]; higher is worse."""
    if total == 0:
        return 0.0
    drop_rate = dropped / total
    flip_rate = flips / total
    conf = float(np.mean(np.abs(conf_drifts))) if conf_drifts else 0.0
    iou_loss = 1.0 - (float(np.mean(ious)) if ious else 1.0)
    return 0.45 * drop_rate + 0.2 * flip_rate + 0.2 * conf + 0.15 * iou_loss


# --------------------------------------------------------------------------
# Ablation + sweep + full run
# --------------------------------------------------------------------------

def _ablation_configs(config: PipelineConfig) -> Dict[str, PipelineConfig]:
    """One-factor-at-a-time legs: each isolates a single constraint."""
    w, i = config.precision_for("default")
    big_depth = 1 << 16
    return {
        "precision_only": PipelineConfig(
            precision={"default": (w, i)}, max_line_buffer_depth=big_depth,
            use_lut_approx=False, device=config.device),
        "streaming_only": PipelineConfig(
            precision={"default": HIGH_PRECISION},
            max_line_buffer_depth=config.max_line_buffer_depth,
            use_lut_approx=False, device=config.device),
        "hls_approx_only": PipelineConfig(
            precision={"default": HIGH_PRECISION}, max_line_buffer_depth=big_depth,
            use_lut_approx=True, lut_bits=config.lut_bits, device=config.device),
    }


def run_hil(n_sequences: int = 4, frames_per_sequence: int = 14, seed: int = 7,
            width_bits: int = 10, int_bits: int = SWEEP_INT_BITS,
            max_line_buffer_depth: int = 2048, use_lut_approx: bool = True,
            lut_bits: int = 8, device: str = "versal-ai-edge",
            regression_delta: float = 0.02, alpha: float = 0.05,
            run_ablation: bool = True, persist: bool = True) -> Dict:
    """Full HIL run: paired comparison + ablation attribution + verdict."""
    from sensorflow.vitis.backend import DeviceConfig
    t0 = time.perf_counter()
    dev = DeviceConfig(name=device)
    config = PipelineConfig(precision={"default": (width_bits, int_bits)},
                            max_line_buffer_depth=max_line_buffer_depth,
                            use_lut_approx=use_lut_approx, lut_bits=lut_bits,
                            device=dev)
    sequences = generate_sequences(n_sequences, frames_per_sequence, seed)

    full = run_paired_comparison(sequences, config, seed)
    verdict = sequential_verdict(full["frame_deltas"], full["object_pairs"],
                                 delta=regression_delta, alpha=alpha)

    ablation = None
    if run_ablation:
        legs = {}
        for name, leg_cfg in _ablation_configs(config).items():
            leg = run_paired_comparison(sequences, leg_cfg, seed)
            legs[name] = {"gap_score": leg["gap_score"],
                          "totals": leg["totals"], "drift": leg["drift"]}
        total_leg_gap = sum(l["gap_score"] for l in legs.values())
        attribution = {name: (round(l["gap_score"] / total_leg_gap, 4)
                              if total_leg_gap > 0 else 0.0)
                       for name, l in legs.items()}
        ablation = {"legs": legs, "attribution": attribution,
                    "note": ("One-factor-at-a-time: each leg enables exactly "
                             "one constraint; attribution is that leg's gap "
                             "share of the summed single-factor gaps.")}

    run_id = new_id("hilrun")
    payload = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "hil",
        "params": {"n_sequences": n_sequences,
                   "frames_per_sequence": frames_per_sequence, "seed": seed,
                   "width_bits": width_bits, "int_bits": int_bits,
                   "max_line_buffer_depth": max_line_buffer_depth,
                   "use_lut_approx": use_lut_approx, "lut_bits": lut_bits,
                   "device": device, "regression_delta": regression_delta,
                   "alpha": alpha},
        "comparison": {k: v for k, v in full.items()
                       if k not in ("object_pairs", "per_object", "frame_deltas")},
        "per_object_sample": full["per_object"][:60],
        "verdict": verdict,
        "ablation": ablation,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": {
            "decision": verdict["decision"],
            "gap_score": full["gap_score"],
            "dropped": full["totals"]["dropped_by_vitis"],
            "width_bits": width_bits,
        },
        "emulation_note": ("vitis_emulated is a constraint-faithful CPU "
                           "emulator; no FPGA hardware was used."),
    }
    if persist:
        save_run("hil", run_id, payload)
    return payload


def run_bitwidth_sweep(n_sequences: int = 4, frames_per_sequence: int = 14,
                       seed: int = 7, widths: Optional[List[int]] = None,
                       int_bits: int = SWEEP_INT_BITS,
                       max_line_buffer_depth: int = 2048,
                       use_lut_approx: bool = True, lut_bits: int = 8,
                       device: str = "versal-ai-edge",
                       regression_delta: float = 0.02, alpha: float = 0.05,
                       persist: bool = True) -> Dict:
    """Sweep W, find the minimal bit-width whose verdict is not REGRESSION."""
    from sensorflow.vitis.backend import DeviceConfig
    t0 = time.perf_counter()
    widths = sorted(widths or [8, 10, 12, 14, 16], reverse=True)
    dev = DeviceConfig(name=device)
    sequences = generate_sequences(n_sequences, frames_per_sequence, seed)

    points = []
    minimal_passing = None
    for w in widths:
        cfg = PipelineConfig(precision={"default": (w, int_bits)},
                             max_line_buffer_depth=max_line_buffer_depth,
                             use_lut_approx=use_lut_approx, lut_bits=lut_bits,
                             device=dev)
        cmp_ = run_paired_comparison(sequences, cfg, seed)
        verdict = sequential_verdict(cmp_["frame_deltas"], cmp_["object_pairs"],
                                     delta=regression_delta, alpha=alpha)
        points.append({
            "width_bits": w, "int_bits": int_bits,
            "gap_score": cmp_["gap_score"],
            "dropped": cmp_["totals"]["dropped_by_vitis"],
            "class_flips": cmp_["totals"]["class_flips"],
            "mean_abs_confidence_drift": cmp_["drift"]["mean_abs_confidence_drift"],
            "mean_position_drift_px": cmp_["drift"]["mean_position_drift_px"],
            "mean_pair_iou": cmp_["drift"]["mean_pair_iou"],
            "decision": verdict["decision"],
            "mean_delta": verdict["mean_delta"],
        })
    by_width = sorted(points, key=lambda p: p["width_bits"])
    for p in by_width:  # prefer the smallest width with an affirmative PASS
        if p["decision"] == "PASS":
            minimal_passing = {"width_bits": p["width_bits"],
                               "int_bits": int_bits, "decision": "PASS"}
            break
    if minimal_passing is None:
        for p in by_width:
            if p["decision"] != "REGRESSION":
                minimal_passing = {"width_bits": p["width_bits"],
                                   "int_bits": int_bits,
                                   "decision": p["decision"],
                                   "note": ("No width reached an affirmative "
                                            "PASS; this is the smallest width "
                                            "without a detected regression.")}
                break

    run_id = new_id("hilsweep")
    payload = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "hil_sweep",
        "params": {"n_sequences": n_sequences,
                   "frames_per_sequence": frames_per_sequence, "seed": seed,
                   "widths": widths, "int_bits": int_bits,
                   "max_line_buffer_depth": max_line_buffer_depth,
                   "use_lut_approx": use_lut_approx,
                   "regression_delta": regression_delta, "alpha": alpha,
                   "device": device},
        "points": sorted(points, key=lambda p: p["width_bits"]),
        "minimal_passing_config": minimal_passing,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": {"minimal_passing_width": (minimal_passing or {}).get("width_bits"),
                    "widths": widths},
        "emulation_note": ("Sweep executed on the vitis_emulated CPU backend; "
                           "no FPGA hardware was used."),
    }
    if persist:
        save_run("hil", run_id, payload)
    return payload
