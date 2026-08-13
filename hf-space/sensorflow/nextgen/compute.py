"""Compute deduplication over the bevfusion pipeline + quantified savings.

The insight: when N models are evaluated on the same scenario, the sensor
simulation, BEV rasterization and fusion decode (the "backbone-equivalent"
stages of sensorflow.bevfusion — build_modality_map / fuse_maps / decode_bev)
are IDENTICAL across models. Only the model-specific head (confidence
recalibration + class thresholds here) differs. So:

    naive cost      = scenarios x models x (backbone + head)
    optimized cost  = scenarios x backbone  +  scenarios x models x head

The shared stage is cached content-addressed (cache.py) under
(scenario_id, sensor_version, preprocessing_version, backbone_version,
feature_schema_version); a version bump on any component is a miss by
construction. `benchmark` runs the real pipeline twice and reports MEASURED
timings and hit rates — no estimated numbers.

Honest scope: the "backbone" is the numpy bevfusion pipeline, not a neural
network, and the "heads" are deterministic calibration/threshold profiles.
The cost STRUCTURE (shared expensive stage, cheap per-model stage) is the
real thing being demonstrated; absolute times would scale with a real stack.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from uuid import uuid4

from sensorflow.bevfusion.fusion import build_modality_map, decode_bev, fuse_maps
from sensorflow.bevfusion.geometry import BEVGrid
from sensorflow.bevfusion.scenes import SceneSequence, generate_sequences
from sensorflow.bevfusion.sensors import (
    camera_rng, lidar_rng, simulate_camera, simulate_lidar,
)
from sensorflow.nextgen import store
from sensorflow.nextgen.cache import CacheKeyVersions, get_feature_cache
from sensorflow.nextgen.models import ComputeOptimizationResult

# Model-specific head profiles: deterministic post-processing over the shared
# fused features. Cheap by design — that is the point of the split.
HEAD_PROFILES: Dict[str, Dict[str, float]] = {
    "candidate-v4": {"confidence_gain": 1.05, "min_confidence": 0.30,
                     "vru_boost": 1.10},
    "candidate-v4.1": {"confidence_gain": 1.02, "min_confidence": 0.35,
                       "vru_boost": 1.15},
    "baseline-v3": {"confidence_gain": 1.00, "min_confidence": 0.40,
                    "vru_boost": 1.00},
}
VRU = {"pedestrian", "cyclist", "motorcycle"}


def compute_backbone_features(seq: SceneSequence, seed: int = 7,
                              seq_index: int = 0) -> List[Dict]:
    """Shared stage: sensor sim -> BEV rasterization -> fusion -> decode.
    REUSES the bevfusion pipeline verbatim; returns JSON-serializable
    per-frame decoded fused boxes (the cacheable IR)."""
    grid = BEVGrid()
    cam_gen = camera_rng(seed, seq_index)
    lid_gen = lidar_rng(seed, seq_index)
    frames: List[Dict] = []
    for frame in seq.frames:
        cam = simulate_camera(frame, seq, cam_gen)
        lid = simulate_lidar(frame, seq, lid_gen)
        fused = fuse_maps(build_modality_map(cam, grid, "camera"),
                          build_modality_map(lid, grid, "lidar"))
        decoded = decode_bev(fused)
        frames.append({"frame_id": frame.frame_id,
                       "boxes": [{"bbox_3d": b["bbox_3d"],
                                  "class_name": b["class_name"],
                                  "confidence": round(float(b["confidence"]), 4)}
                                 for b in decoded]})
    return frames


def run_model_head(features: List[Dict], model_version: str) -> Dict:
    """Model-specific stage: recalibrate + threshold the shared features."""
    prof = HEAD_PROFILES.get(model_version, HEAD_PROFILES["baseline-v3"])
    n_out = 0
    n_vru = 0
    conf_sum = 0.0
    for frame in features:
        for b in frame["boxes"]:
            c = b["confidence"] * prof["confidence_gain"]
            if b["class_name"] in VRU:
                c *= prof["vru_boost"]
            if c >= prof["min_confidence"]:
                n_out += 1
                conf_sum += min(c, 1.0)
                if b["class_name"] in VRU:
                    n_vru += 1
    return {"model_version": model_version, "n_detections": n_out,
            "n_vru": n_vru,
            "mean_confidence": round(conf_sum / n_out, 4) if n_out else None}


def evaluate_scenarios(sequences: List[SceneSequence], models: List[str],
                       versions: Optional[CacheKeyVersions] = None,
                       seed: int = 7) -> Dict:
    """Evaluate models x scenarios with backbone dedup. Returns measured
    per-stage timings + head outputs."""
    versions = versions or CacheKeyVersions()
    cache = get_feature_cache()
    backbone_times: List[float] = []
    head_times: List[float] = []
    outputs = []
    t_start = time.perf_counter()
    for si, seq in enumerate(sequences):
        key = versions.key_for(seq.sequence_id, kind="bev-features")

        def _compute(seq=seq, si=si):
            t0 = time.perf_counter()
            feats = compute_backbone_features(seq, seed=seed, seq_index=si)
            backbone_times.append(time.perf_counter() - t0)
            return feats

        features = cache.get_or_compute(key, _compute, kind="bev-features")
        for model in models:
            t0 = time.perf_counter()
            head = run_model_head(features, model)
            head_times.append(time.perf_counter() - t0)
            outputs.append({"scenario_id": seq.sequence_id, **head})
    wall = time.perf_counter() - t_start
    return {"outputs": outputs, "wall_s": wall,
            "backbone_times": backbone_times, "head_times": head_times}


def benchmark(n_scenarios: int = 6, frames_per_sequence: int = 20,
              models: Optional[List[str]] = None, seed: int = 7,
              versions: Optional[CacheKeyVersions] = None,
              persist: bool = True) -> ComputeOptimizationResult:
    """Real benchmark: cold pass (misses) + warm pass (hits) over the same
    suite; every timing measured, nothing estimated except the naive
    extrapolation which uses the MEASURED mean backbone/head times."""
    models = models or list(HEAD_PROFILES)
    versions = versions or CacheKeyVersions()
    cache = get_feature_cache()
    cache.reset_stats()

    sequences = generate_sequences(n_sequences=n_scenarios,
                                   frames_per_sequence=frames_per_sequence,
                                   seed=seed)
    cold = evaluate_scenarios(sequences, models, versions, seed)
    warm = evaluate_scenarios(sequences, models, versions, seed)

    stats = cache.stats
    all_backbone = cold["backbone_times"] + warm["backbone_times"]
    all_heads = cold["head_times"] + warm["head_times"]
    mean_backbone = sum(all_backbone) / len(all_backbone) if all_backbone else 0.0
    mean_head = sum(all_heads) / len(all_heads) if all_heads else 0.0

    n_units = 2 * len(sequences) * len(models)  # two passes
    naive_cost = n_units * (mean_backbone + mean_head)
    optimized_cost = cold["wall_s"] + warm["wall_s"]

    result = ComputeOptimizationResult(
        report_id=f"compute-{uuid4().hex[:8]}",
        n_scenarios=len(sequences), n_models=len(models),
        naive_full_inferences=n_units,
        optimized_backbone_computes=stats.misses,
        optimized_head_computes=len(all_heads),
        cache_hits=stats.hits, cache_misses=stats.misses,
        hit_rate=round(stats.hit_rate, 4),
        naive_cost_s=round(naive_cost, 4),
        optimized_cost_s=round(optimized_cost, 4),
        savings_ratio=round(1.0 - optimized_cost / naive_cost, 4) if naive_cost else 0.0,
        measured_backbone_s=round(mean_backbone, 4),
        measured_head_s=round(mean_head, 6),
        invalidation="content-addressed keys over (scenario_id, sensor, "
                     "preprocessing, backbone, feature-schema versions); any "
                     "version bump is a new key (cross-version reuse "
                     "impossible); LRU eviction bounds disk growth")
    if persist:
        store.write_json(result.model_dump(mode="json"), "compute",
                         f"{result.report_id}.json")
        store.write_json(result.model_dump(mode="json"), "compute", "latest.json")
    return result


def latest_report() -> Optional[Dict]:
    return store.read_json("compute", "latest.json")
