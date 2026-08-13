"""Temporal & stereo stability profiling on the acceleration backends.

Pyramidal dense optical flow over BEV rasters of the bevfusion synthetic
scenes gives a MODEL-INDEPENDENT motion baseline: the flow field predicts
where every ground-truth object should be next frame, without consulting
any detection engine. Engines (perception-v1-camera, perception-v3-bevfusion,
imported read-only from sensorflow.bevfusion) are then judged against that
baseline:

* flicker rate       — detection present -> absent -> present on a track the
                       flow says was continuously observable
* bbox jitter        — engine frame-to-frame displacement vs the
                       flow-predicted displacement, normalized
* fragmentation      — distinct engine track ids per GT instance over
                       flow-continuous spans
* ID-switch/flow     — fraction of engine ID switches that coincide with a
                       genuine flow discontinuity (excusable) vs those on
                       smooth flow (the engine's own fault)

A synthetic stereo pair rendered from the exact scene geometry closes the
loop on depth: block-matching disparity -> metric depth is compared against
ground-truth depth and against each engine's 3D positions.

Everything runs twice — once with reference (float32) flow and once with
vitis_emulated (fixed-point) flow — and the report states whether the
accelerated flow changes any stability verdict (the meta-check that the
accelerated metric itself is trustworthy).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.bevfusion.engines import (
    BASELINE_ENGINE, FUSED_ENGINE, run_baseline, run_fused,
)
from sensorflow.bevfusion.scenes import SceneSequence, generate_sequences
from sensorflow.vitis.backend import DeviceConfig, PipelineConfig, get_backend
from sensorflow.vitis.render import (
    BEV_M_PER_PX_X, BEV_M_PER_PX_Y, BASELINE_M, FOCAL_PX,
    ego_to_bev_px, render_bev_frame, render_stereo_pair,
)
from sensorflow.vitis.store import new_id, save_run

KNOWN_ENGINES = [BASELINE_ENGINE, FUSED_ENGINE]
MATCH_GATE_M = 2.5          # detection <-> GT association gate (meters)
FLOW_RESIDUAL_OK_M = 1.2    # flow residual below this => flow-continuous
STEREO_FRAMES_PER_SEQ = 3


# --------------------------------------------------------------------------
# Flow baseline
# --------------------------------------------------------------------------

def _flow_displacement_m(flow: np.ndarray, r: float, c: float) -> Tuple[float, float]:
    """Sample flow (dx, dy in px) near (r, c); return ego (dx_m, dy_m)."""
    h, w = flow.shape[:2]
    r0, r1 = int(max(0, r - 2)), int(min(h, r + 3))
    c0, c1 = int(max(0, c - 2)), int(min(w, c + 3))
    if r1 <= r0 or c1 <= c0:
        return 0.0, 0.0
    patch = flow[r0:r1, c0:c1]
    d_col = float(np.median(patch[..., 0]))
    d_row = float(np.median(patch[..., 1]))
    # row grows toward ego (-x), col grows toward +y.
    return -d_row * BEV_M_PER_PX_X, d_col * BEV_M_PER_PX_Y


def compute_flow_baseline(seq: SceneSequence, backend, seed: int) -> Dict:
    """Per-track flow records: for each GT instance and frame transition,
    the flow-predicted displacement vs the true GT displacement."""
    imgs = [render_bev_frame(f, seq, seed) for f in seq.frames]
    flows = [backend.optical_flow(imgs[i], imgs[i + 1], levels=3,
                                  window=9, iterations=3)
             for i in range(len(imgs) - 1)]

    gt_pos: Dict[str, Dict[int, Tuple[float, float, bool]]] = {}
    for frame in seq.frames:
        for g in frame.gt:
            gt_pos.setdefault(g.instance_id, {})[frame.index] = (
                g.bbox_3d[0], g.bbox_3d[1], g.occluded)

    records: Dict[str, Dict[int, Dict]] = {}
    for iid, per_frame in gt_pos.items():
        for fi, (x, y, occ) in per_frame.items():
            nxt = per_frame.get(fi + 1)
            if nxt is None or fi >= len(flows):
                continue
            r, c = ego_to_bev_px(x, y)
            fdx, fdy = _flow_displacement_m(flows[fi], r, c)
            true_dx, true_dy = nxt[0] - x, nxt[1] - y
            residual = float(np.hypot(fdx - true_dx, fdy - true_dy))
            records.setdefault(iid, {})[fi] = {
                "flow_dx_m": fdx, "flow_dy_m": fdy,
                "true_dx_m": true_dx, "true_dy_m": true_dy,
                "residual_m": residual,
                "continuous": residual <= FLOW_RESIDUAL_OK_M and not occ,
                "occluded": occ,
            }
    return records


# --------------------------------------------------------------------------
# Engine evaluation against the flow baseline
# --------------------------------------------------------------------------

def _run_engine(name: str, seq: SceneSequence, seed: int,
                seq_index: int) -> Dict[str, List[Dict]]:
    if name == BASELINE_ENGINE:
        return run_baseline(seq, seed, seq_index)
    if name == FUSED_ENGINE:
        out, _ = run_fused(seq, seed, seq_index)
        return out
    raise ValueError(f"Unknown engine {name!r}; known: {KNOWN_ENGINES}")


def _match_engine_to_gt(seq: SceneSequence,
                        engine_out: Dict[str, List[Dict]]) -> Dict[str, Dict[int, Dict]]:
    """{instance_id: {frame_index: {x, y, track_id} or absent}} via NN gate."""
    assoc: Dict[str, Dict[int, Dict]] = {}
    for frame in seq.frames:
        dets = list(engine_out.get(frame.frame_id, []))
        used = set()
        for g in sorted(frame.gt, key=lambda g: g.distance):
            best, best_d = None, MATCH_GATE_M
            for j, det in enumerate(dets):
                if j in used:
                    continue
                d = float(np.hypot(det["bbox_3d"][0] - g.bbox_3d[0],
                                   det["bbox_3d"][1] - g.bbox_3d[1]))
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                used.add(best)
                det = dets[best]
                assoc.setdefault(g.instance_id, {})[frame.index] = {
                    "x": det["bbox_3d"][0], "y": det["bbox_3d"][1],
                    "track_id": det.get("track_id"),
                }
    return assoc


def _engine_track_metrics(seq: SceneSequence, flow_records: Dict,
                          assoc: Dict[str, Dict[int, Dict]]) -> Dict:
    flicker_events = 0
    flicker_opportunities = 0
    jitter_vals: List[float] = []
    fragments = 0
    fragment_tracks = 0
    id_switches = 0
    id_switches_at_flow_break = 0
    per_track: List[Dict] = []

    gt_frames: Dict[str, List[int]] = {}
    for frame in seq.frames:
        for g in frame.gt:
            gt_frames.setdefault(g.instance_id, []).append(frame.index)

    for iid, findices in gt_frames.items():
        findices = sorted(findices)
        det_frames = assoc.get(iid, {})
        flow_rec = flow_records.get(iid, {})
        continuous_frames = [fi for fi in findices
                             if flow_rec.get(fi, {}).get("continuous")]

        # Flicker: present -> absent -> present inside flow-continuous span.
        presence = [(fi, fi in det_frames) for fi in findices]
        detected_indices = [fi for fi, p in presence if p]
        events_here = 0
        if len(detected_indices) >= 2:
            lo, hi = detected_indices[0], detected_indices[-1]
            gap_open = False
            for fi in findices:
                if fi < lo or fi > hi:
                    continue
                if fi not in det_frames:
                    if flow_rec.get(fi, {}).get("continuous") or \
                       flow_rec.get(fi - 1, {}).get("continuous"):
                        gap_open = True
                elif gap_open:
                    events_here += 1
                    gap_open = False
        flicker_events += events_here
        flicker_opportunities += max(0, len(continuous_frames) - 1)

        # Jitter: engine displacement vs flow-predicted displacement.
        for fi in findices:
            if fi in det_frames and (fi + 1) in det_frames and fi in flow_rec:
                ddx = det_frames[fi + 1]["x"] - det_frames[fi]["x"]
                ddy = det_frames[fi + 1]["y"] - det_frames[fi]["y"]
                fdx = flow_rec[fi]["flow_dx_m"]
                fdy = flow_rec[fi]["flow_dy_m"]
                mag = float(np.hypot(fdx, fdy))
                dev = float(np.hypot(ddx - fdx, ddy - fdy))
                jitter_vals.append(dev / (mag + 0.5))

        # Fragmentation over flow-continuous spans + ID-switch/flow overlap.
        ids_seen = []
        prev_fi = None
        for fi in sorted(det_frames):
            tid = det_frames[fi]["track_id"]
            if ids_seen and tid != ids_seen[-1]:
                id_switches += 1
                broke = not flow_rec.get(prev_fi, {}).get("continuous", True)
                id_switches_at_flow_break += int(broke)
            if not ids_seen or tid != ids_seen[-1]:
                ids_seen.append(tid)
            prev_fi = fi
        if continuous_frames:
            n_ids_continuous = len({det_frames[fi]["track_id"]
                                    for fi in continuous_frames
                                    if fi in det_frames})
            if n_ids_continuous > 0:
                fragment_tracks += 1
                fragments += n_ids_continuous - 1

        per_track.append({
            "instance_id": iid,
            "gt_frames": len(findices),
            "detected_frames": len(det_frames),
            "flow_continuous_frames": len(continuous_frames),
            "flicker_events": events_here,
            "distinct_track_ids": len(set(ids_seen)) if ids_seen else 0,
        })

    return {
        "flicker_events": flicker_events,
        "flicker_opportunities": flicker_opportunities,
        "jitter_vals": jitter_vals,
        "fragments": fragments,
        "fragment_tracks": fragment_tracks,
        "id_switches": id_switches,
        "id_switches_at_flow_break": id_switches_at_flow_break,
        "per_track": per_track,
    }


def _aggregate_engine(seq_metrics: List[Dict]) -> Dict:
    flicker_e = sum(m["flicker_events"] for m in seq_metrics)
    flicker_o = max(1, sum(m["flicker_opportunities"] for m in seq_metrics))
    jitter = [v for m in seq_metrics for v in m["jitter_vals"]]
    frags = sum(m["fragments"] for m in seq_metrics)
    frag_tracks = max(1, sum(m["fragment_tracks"] for m in seq_metrics))
    switches = sum(m["id_switches"] for m in seq_metrics)
    excusable = sum(m["id_switches_at_flow_break"] for m in seq_metrics)

    flicker_rate = flicker_e / flicker_o
    mean_jitter = float(np.mean(jitter)) if jitter else 0.0
    frag_rate = frags / frag_tracks
    unexcused = (switches - excusable) / max(1, switches) if switches else 0.0

    score = 100.0 * (
        0.35 * max(0.0, 1.0 - 4.0 * flicker_rate) +
        0.25 * float(np.exp(-1.5 * mean_jitter)) +
        0.25 * max(0.0, 1.0 - 0.6 * frag_rate) +
        0.15 * (1.0 - unexcused)
    )
    return {
        "flicker_rate": round(flicker_rate, 4),
        "mean_jitter": round(mean_jitter, 4),
        "fragmentation_per_track": round(frag_rate, 4),
        "id_switches": switches,
        "id_switches_at_flow_break": excusable,
        "unexcused_id_switch_fraction": round(unexcused, 4),
        "stability_score": round(score, 2),
    }


# --------------------------------------------------------------------------
# Stereo consistency
# --------------------------------------------------------------------------

def _stereo_check(sequences: List[SceneSequence], backend, seed: int,
                  max_frames_per_seq: int = STEREO_FRAMES_PER_SEQ) -> Dict:
    depth_errors: List[float] = []
    rel_errors: List[float] = []
    near_errors: List[float] = []  # objects closer than 30 m
    disparity_errors: List[float] = []
    n_objects = 0
    for seq in sequences:
        step = max(1, len(seq.frames) // max_frames_per_seq)
        for frame in seq.frames[::step][:max_frames_per_seq]:
            left, right, objs = render_stereo_pair(frame, seq, seed)
            if not objs:
                continue
            disp = backend.stereo_block_match(left, right, max_disparity=48,
                                              block=9)
            for o in objs:
                u, v = int(round(o["u"])), int(round(o["v"]))
                if not (2 <= v < disp.shape[0] - 2 and 2 <= u < disp.shape[1] - 2):
                    continue
                d_meas = float(np.median(disp[v - 2:v + 3, u - 2:u + 3]))
                if d_meas < 0.5:
                    continue
                depth = FOCAL_PX * BASELINE_M / d_meas
                err = abs(depth - o["depth_m"])
                depth_errors.append(err)
                rel_errors.append(err / o["depth_m"])
                if o["depth_m"] < 30.0:
                    near_errors.append(err)
                disparity_errors.append(abs(d_meas - o["disparity_px"]))
                n_objects += 1
    return {
        "objects_checked": n_objects,
        "median_abs_depth_error_m": round(float(np.median(depth_errors)), 3) if depth_errors else None,
        "median_rel_depth_error": round(float(np.median(rel_errors)), 4) if rel_errors else None,
        "median_abs_depth_error_near_m": round(float(np.median(near_errors)), 3) if near_errors else None,
        "p90_abs_depth_error_m": round(float(np.percentile(depth_errors, 90)), 3) if depth_errors else None,
        "median_abs_disparity_error_px": round(float(np.median(disparity_errors)), 3) if disparity_errors else None,
        "note": ("Stereo depth error grows quadratically with range at fixed "
                 "disparity error; near-field (<30 m) stats are the "
                 "geometry-limited figure of merit."),
    }


# --------------------------------------------------------------------------
# Full profile
# --------------------------------------------------------------------------

def run_temporal_profile(engines: Optional[List[str]] = None,
                         n_sequences: int = 3, frames_per_sequence: int = 18,
                         seed: int = 7, width_bits: int = 12, int_bits: int = 6,
                         device: str = "versal-ai-edge",
                         persist: bool = True) -> Dict:
    """Profile engines against reference AND vitis_emulated flow baselines."""
    t0 = time.perf_counter()
    engines = engines or list(KNOWN_ENGINES)
    for e in engines:
        if e not in KNOWN_ENGINES:
            raise ValueError(f"Unknown engine {e!r}; known: {KNOWN_ENGINES}")
    sequences = generate_sequences(n_sequences, frames_per_sequence, seed)

    backends = {
        "reference": get_backend("reference"),
        "vitis_emulated": get_backend(
            "vitis_emulated",
            PipelineConfig(precision={"default": (width_bits, int_bits)},
                           device=DeviceConfig(name=device))),
    }

    # Engine outputs are backend-independent (computed once).
    engine_assoc: Dict[str, List[Tuple[SceneSequence, Dict]]] = {e: [] for e in engines}
    for si, seq in enumerate(sequences):
        for e in engines:
            out = _run_engine(e, seq, seed, si)
            engine_assoc[e].append((seq, _match_engine_to_gt(seq, out)))

    results: Dict[str, Dict] = {}
    timeline_sample = None
    for be_name, backend in backends.items():
        flow_by_seq = [compute_flow_baseline(seq, backend, seed)
                       for seq in sequences]
        engines_report = {}
        for e in engines:
            seq_metrics, cohort_metrics = [], {}
            for (seq, assoc), flow_records in zip(engine_assoc[e], flow_by_seq):
                m = _engine_track_metrics(seq, flow_records, assoc)
                seq_metrics.append(m)
                cohort = f"{seq.time_of_day}/{seq.weather}"
                cohort_metrics.setdefault(cohort, []).append(m)
                occluded = _occluded_subset(seq, flow_records, assoc)
                if occluded is not None:
                    cohort_metrics.setdefault("occluded", []).append(occluded)
            engines_report[e] = _aggregate_engine(seq_metrics)
            engines_report[e]["cohorts"] = {
                c: _aggregate_engine(ms) for c, ms in sorted(cohort_metrics.items())}
            if timeline_sample is None and be_name == "reference":
                timeline_sample = _timeline_sample(
                    engine_assoc[e][0][0], flow_by_seq[0],
                    engine_assoc[e][0][1], engine=e)
        stereo = _stereo_check(sequences, backend, seed)
        results[be_name] = {"engines": engines_report, "stereo": stereo}

    # Meta-check: does fixed-point flow change any verdict?
    ranking = {}
    for be_name, res in results.items():
        ranking[be_name] = sorted(
            engines, key=lambda e: -res["engines"][e]["stability_score"])
    score_deltas = {
        e: round(results["vitis_emulated"]["engines"][e]["stability_score"] -
                 results["reference"]["engines"][e]["stability_score"], 2)
        for e in engines}
    meta = {
        "ranking_agrees": ranking["reference"] == ranking["vitis_emulated"],
        "ranking_reference": ranking["reference"],
        "ranking_vitis_emulated": ranking["vitis_emulated"],
        "stability_score_delta_by_engine": score_deltas,
        "max_abs_score_delta": max(abs(v) for v in score_deltas.values()),
        "note": ("Meta-check that the accelerated (fixed-point) flow metric "
                 "is itself trustworthy: engine stability verdicts should "
                 "not change between reference and emulated flow."),
    }

    run_id = new_id("temprun")
    best = ranking["reference"][0]
    payload = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "temporal",
        "params": {"engines": engines, "n_sequences": n_sequences,
                   "frames_per_sequence": frames_per_sequence, "seed": seed,
                   "width_bits": width_bits, "int_bits": int_bits,
                   "device": device},
        "results": results,
        "backend_agreement": meta,
        "timeline_sample": timeline_sample,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": {
            "best_engine": best,
            "best_score": results["reference"]["engines"][best]["stability_score"],
            "ranking_agrees": meta["ranking_agrees"],
        },
        "emulation_note": ("vitis_emulated flow runs on a constraint-faithful "
                           "CPU emulator; no FPGA hardware was used."),
    }
    if persist:
        save_run("temporal", run_id, payload)
    return payload


def _occluded_subset(seq: SceneSequence, flow_records: Dict,
                     assoc: Dict) -> Optional[Dict]:
    """Metrics restricted to instances that carry a planted occlusion window."""
    occluded_iids = {g.instance_id for f in seq.frames for g in f.gt if g.occluded}
    if not occluded_iids:
        return None
    sub_seq = SceneSequence(sequence_id=seq.sequence_id,
                            time_of_day=seq.time_of_day, weather=seq.weather)
    for frame in seq.frames:
        sub = frame.model_copy(deep=True)
        sub.gt = [g for g in sub.gt if g.instance_id in occluded_iids]
        sub_seq.frames.append(sub)
    sub_assoc = {k: v for k, v in assoc.items() if k in occluded_iids}
    sub_flow = {k: v for k, v in flow_records.items() if k in occluded_iids}
    return _engine_track_metrics(sub_seq, sub_flow, sub_assoc)


def _timeline_sample(seq: SceneSequence, flow_records: Dict, assoc: Dict,
                     engine: str) -> Optional[Dict]:
    """Flow-vs-detection timeline for the most interesting track (the one
    with the most detection gaps while flow was continuous)."""
    best_iid, best_gaps = None, -1
    gt_frames: Dict[str, List[int]] = {}
    for frame in seq.frames:
        for g in frame.gt:
            gt_frames.setdefault(g.instance_id, []).append(frame.index)
    for iid, findices in gt_frames.items():
        det = assoc.get(iid, {})
        gaps = sum(1 for fi in findices
                   if fi not in det and
                   flow_records.get(iid, {}).get(fi, {}).get("continuous"))
        if gaps > best_gaps and len(findices) >= 6:
            best_iid, best_gaps = iid, gaps
    if best_iid is None:
        return None
    det = assoc.get(best_iid, {})
    flow_rec = flow_records.get(best_iid, {})
    frames = []
    for fi in sorted(gt_frames[best_iid]):
        frames.append({
            "frame_index": fi,
            "detected": fi in det,
            "track_id": det.get(fi, {}).get("track_id"),
            "flow_continuous": bool(flow_rec.get(fi, {}).get("continuous")),
            "flow_residual_m": round(flow_rec.get(fi, {}).get("residual_m", 0.0), 3)
            if fi in flow_rec else None,
            "occluded": bool(flow_rec.get(fi, {}).get("occluded", False)),
        })
    return {"instance_id": best_iid, "engine": engine,
            "sequence_id": seq.sequence_id, "frames": frames}
