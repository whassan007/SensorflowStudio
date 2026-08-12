"""Immutable synthetic populations, generated vectorized and stored partitioned.

Storage layout (single-node stand-in for an Iceberg/parquet lakehouse — no parquet
engine is installed in this venv, so partitions are compressed npz column files):

    runs/megaeval/populations/{pop_id}/
        meta.json          dataset card: dims vocab, counts (exact), train mix, lineage
        part-0000.npz ...  column arrays partitioned by container_id % NUM_PARTITIONS

Every object row carries:
  - 9 low-cardinality dimension codes (the cube dimensions)
  - a model-independent `difficulty` in [0,1] (occlusion/distance/lighting/weather/size)
  - `safety_critical` flag (VRUs in high-risk scenarios)

The "training mix" is a deliberately shifted joint distribution over
(class, weather, lighting) used by shift detection: training saw less
night/rain/fog than this evaluation population.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

MEGA_ROOT = os.path.join("runs", "megaeval")
NUM_PARTITIONS = 32

DIMENSIONS: Dict[str, List[str]] = {
    "class": ["vehicle", "pedestrian", "cyclist", "motorcycle", "truck", "bus"],
    "weather": ["clear", "rain", "fog", "snow"],
    "lighting": ["day", "night", "dusk"],
    "road_type": ["highway", "urban", "rural", "intersection"],
    "scenario": ["nominal", "near_miss", "occlusion_heavy", "night_glare",
                 "extreme_ttc", "sensor_degraded"],
    "sensor": ["lidar", "camera", "fused"],
    "distance_band": ["0-20m", "20-50m", "50-100m", "100m+"],
    "speed_band": ["stationary", "slow", "medium", "fast"],
    "occlusion": ["none", "partial", "heavy"],
}
DIM_NAMES = list(DIMENSIONS.keys())
CONTAINER_DIMS = ["weather", "lighting", "road_type", "scenario"]
OBJECT_DIMS = ["class", "sensor", "distance_band", "speed_band", "occlusion"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pop_dir(pop_id: str) -> str:
    return os.path.join(MEGA_ROOT, "populations", pop_id)


def set_mega_root(path: str) -> None:
    """Test hook: relocate all megaeval storage (and clear partition caches)."""
    global MEGA_ROOT
    MEGA_ROOT = str(path)
    with _CACHE_LOCK:
        _PART_CACHE.clear()


# ------------------------------------------------------------------ generation


def _sample_container_dims(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    """Correlated container-level context (weather/lighting/road/scenario)."""
    weather = rng.choice(4, size=n, p=[0.62, 0.20, 0.11, 0.07]).astype(np.int8)
    lighting = rng.choice(3, size=n, p=[0.60, 0.28, 0.12]).astype(np.int8)
    road = rng.choice(4, size=n, p=[0.30, 0.38, 0.14, 0.18]).astype(np.int8)
    scenario = rng.choice(6, size=n, p=[0.66, 0.09, 0.09, 0.06, 0.05, 0.05]).astype(np.int8)
    # correlations: night_glare scenario forces night; fog/snow raise sensor_degraded odds
    scenario_names = DIMENSIONS["scenario"]
    lighting[scenario == scenario_names.index("night_glare")] = 1  # night
    bad_wx = (weather >= 2)
    flip = bad_wx & (rng.random(n) < 0.18)
    scenario[flip] = scenario_names.index("sensor_degraded")
    return {"weather": weather, "lighting": lighting, "road_type": road, "scenario": scenario}


_CLASS_MIX_BY_ROAD = np.array([
    # vehicle, pedestrian, cyclist, motorcycle, truck, bus
    [0.62, 0.02, 0.01, 0.05, 0.24, 0.06],   # highway
    [0.44, 0.26, 0.12, 0.06, 0.07, 0.05],   # urban
    [0.55, 0.08, 0.06, 0.08, 0.18, 0.05],   # rural
    [0.40, 0.30, 0.14, 0.05, 0.06, 0.05],   # intersection
])

_SPEED_MIX_BY_ROAD = np.array([
    [0.02, 0.08, 0.35, 0.55],   # highway
    [0.22, 0.44, 0.28, 0.06],   # urban
    [0.10, 0.30, 0.40, 0.20],   # rural
    [0.30, 0.45, 0.20, 0.05],   # intersection
])


def _object_difficulty(codes: Dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Model-independent difficulty in [0,1] — the physical hardness of each object."""
    n = codes["class"].shape[0]
    d = np.full(n, 0.12, dtype=np.float32)
    d += np.array([0.00, 0.14, 0.12, 0.10, 0.02, 0.02], dtype=np.float32)[codes["class"]]
    d += np.array([0.00, 0.08, 0.12, 0.10], dtype=np.float32)[codes["weather"]]
    d += np.array([0.00, 0.12, 0.05], dtype=np.float32)[codes["lighting"]]
    d += np.array([0.00, 0.10, 0.24], dtype=np.float32)[codes["occlusion"]]
    d += np.array([0.00, 0.05, 0.14, 0.26], dtype=np.float32)[codes["distance_band"]]
    d += np.array([0.00, 0.04, 0.08, 0.06, 0.07, 0.12], dtype=np.float32)[codes["scenario"]]
    d += rng.normal(0.0, 0.05, size=n).astype(np.float32)
    return np.clip(d, 0.01, 0.99)


def generate_population(
    name: str = "mega-perception",
    num_objects: int = 320_000,
    seed: int = 42,
    avg_objects_per_container: float = 12.0,
) -> Dict:
    """Generate + persist a partitioned population. Vectorized; ~seconds for 320k."""
    rng = np.random.default_rng(seed)
    num_containers = max(64, int(num_objects / avg_objects_per_container))

    cdims = _sample_container_dims(rng, num_containers)
    per_container = np.clip(rng.poisson(avg_objects_per_container, size=num_containers),
                            3, 48).astype(np.int64)
    total = int(per_container.sum())
    container_id = np.repeat(np.arange(num_containers, dtype=np.int64), per_container)

    codes: Dict[str, np.ndarray] = {k: cdims[k][container_id] for k in CONTAINER_DIMS}

    road_obj = codes["road_type"].astype(np.int64)
    u = rng.random(total)
    cls_cdf = np.cumsum(_CLASS_MIX_BY_ROAD, axis=1)[road_obj]
    codes["class"] = (u[:, None] > cls_cdf).sum(axis=1).astype(np.int8)
    u2 = rng.random(total)
    spd_cdf = np.cumsum(_SPEED_MIX_BY_ROAD, axis=1)[road_obj]
    codes["speed_band"] = (u2[:, None] > spd_cdf).sum(axis=1).astype(np.int8)
    codes["sensor"] = rng.choice(3, size=total, p=[0.38, 0.22, 0.40]).astype(np.int8)
    codes["distance_band"] = rng.choice(4, size=total, p=[0.24, 0.40, 0.25, 0.11]).astype(np.int8)

    occ_p = np.tile(np.array([0.62, 0.28, 0.10]), (total, 1))
    heavy_scene = codes["scenario"] == DIMENSIONS["scenario"].index("occlusion_heavy")
    occ_p[heavy_scene] = [0.20, 0.42, 0.38]
    occ_cdf = np.cumsum(occ_p, axis=1)
    codes["occlusion"] = (rng.random(total)[:, None] > occ_cdf).sum(axis=1).astype(np.int8)

    difficulty = _object_difficulty(codes, rng)
    vru = np.isin(codes["class"], [1, 2, 3])
    risky_scene = np.isin(codes["scenario"], [1, 4])  # near_miss, extreme_ttc
    near = codes["distance_band"] <= 1
    safety_critical = (vru & (risky_scene | (near & (rng.random(total) < 0.15)))).astype(np.uint8)

    object_id = np.arange(total, dtype=np.int64)
    pop_id = f"pop-{uuid.uuid4().hex[:10]}"
    d = pop_dir(pop_id)
    os.makedirs(d, exist_ok=True)

    part_of = (container_id % NUM_PARTITIONS).astype(np.int64)
    order = np.argsort(part_of, kind="stable")
    part_sorted = part_of[order]
    bounds = np.searchsorted(part_sorted, np.arange(NUM_PARTITIONS + 1))
    partition_meta = []
    for p in range(NUM_PARTITIONS):
        sl = order[bounds[p]:bounds[p + 1]]
        cols = {k: codes[k][sl] for k in DIM_NAMES}
        cols["object_id"] = object_id[sl]
        cols["container_id"] = container_id[sl]
        cols["difficulty"] = difficulty[sl]
        cols["safety_critical"] = safety_critical[sl]
        np.savez_compressed(os.path.join(d, f"part-{p:04d}.npz"), **cols)
        partition_meta.append({"partition": p, "rows": int(sl.size)})

    # exact per-dimension counts (dataset card)
    dim_counts = {
        k: {DIMENSIONS[k][i]: int(c) for i, c in enumerate(np.bincount(codes[k], minlength=len(DIMENSIONS[k])))}
        for k in DIM_NAMES
    }

    train_mix = _training_mix(codes, rng)

    meta = {
        "population_id": pop_id,
        "name": name,
        "created_at": _now(),
        "seed": seed,
        "num_objects": total,
        "num_containers": num_containers,
        "num_partitions": NUM_PARTITIONS,
        "avg_objects_per_container": round(total / num_containers, 2),
        "dimensions": DIMENSIONS,
        "dim_counts": dim_counts,
        "partitions": partition_meta,
        "safety_critical_count": int(safety_critical.sum()),
        "train_mix": train_mix,
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _training_mix(codes: Dict[str, np.ndarray], rng: np.random.Generator) -> List[Dict]:
    """Joint (class, weather, lighting) proportions the *training* set had.

    Derived from the eval mix with deliberate, deterministic under-representation of
    hard conditions (night, rain/fog/snow, VRU at night) so shift detection has
    honest signal: shift = eval share vs train share of the same cohort.
    """
    n = codes["class"].shape[0]
    key = (codes["class"].astype(np.int64) * 100
           + codes["weather"].astype(np.int64) * 10
           + codes["lighting"].astype(np.int64))
    uniq, counts = np.unique(key, return_counts=True)
    eval_share = counts / n

    cls, wx, lt = uniq // 100, (uniq // 10) % 10, uniq % 10
    factor = np.ones(uniq.shape)
    factor[lt == 1] *= 0.55                       # trained on less night
    factor[wx >= 1] *= 0.70                       # less rain/fog/snow
    factor[(np.isin(cls, [1, 2])) & (lt == 1)] *= 0.75  # far fewer night VRUs
    train_share = eval_share * factor
    train_share = train_share / train_share.sum()

    out = []
    for i in range(uniq.size):
        out.append({
            "class": DIMENSIONS["class"][int(cls[i])],
            "weather": DIMENSIONS["weather"][int(wx[i])],
            "lighting": DIMENSIONS["lighting"][int(lt[i])],
            "train_share": float(train_share[i]),
            "eval_share": float(eval_share[i]),
            "eval_count": int(counts[i]),
        })
    return out


# ------------------------------------------------------------------ loading


_PART_CACHE: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
_CACHE_LOCK = threading.Lock()


def load_meta(pop_id: str) -> Optional[Dict]:
    path = os.path.join(pop_dir(pop_id), "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_populations() -> List[Dict]:
    base = os.path.join(MEGA_ROOT, "populations")
    if not os.path.isdir(base):
        return []
    metas = []
    for pid in sorted(os.listdir(base)):
        m = load_meta(pid)
        if m:
            metas.append(m)
    return metas


def load_partition(pop_id: str, partition: int) -> Dict[str, np.ndarray]:
    key = (pop_id, partition)
    with _CACHE_LOCK:
        if key in _PART_CACHE:
            return _PART_CACHE[key]
    path = os.path.join(pop_dir(pop_id), f"part-{partition:04d}.npz")
    with np.load(path) as z:
        cols = {k: z[k] for k in z.files}
    with _CACHE_LOCK:
        if len(_PART_CACHE) > 96:  # bounded partition cache
            _PART_CACHE.clear()
        _PART_CACHE[key] = cols
    return cols


def partition_of_container(container_id: int) -> int:
    return int(container_id) % NUM_PARTITIONS
