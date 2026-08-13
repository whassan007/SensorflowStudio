"""Paired evaluation harness with a fingerprint-keyed prediction cache.

Core design: the SAME sampled units are run through both models. The paired
difference d_i = c_i - b_i removes the between-unit difficulty variance that
dominates two-independent-sample comparisons (Var(d) ~ discordance rate, an
order of magnitude below p(1-p) for adjacent model versions), which is where
most of the sample-size savings in this system come from.

Models are simulated through the megaeval synthetic machinery's population
(dimension codes + difficulty + container structure) with configurable
per-stratum performance effects, so a "2pp pedestrian-night regression"
candidate is constructible deterministically:

    SimulatedModel("cand-v2", effects={"pedestrian|night": -0.02, "__global__": 0.004})

Determinism/pairing:
  * a per-OBJECT pairing latent u_i drawn once per dataset (seeded by the
    population seed) is shared by every model -> outcomes are paired;
  * each model adds a small independent "flip" noise stream seeded by its own
    fingerprint -> realistic discordance even at Delta = 0;
  * a container-level scene effect (shared by both models) induces honest
    intra-cluster correlation for the clustering machinery to handle.

Prediction cache: full-population success vectors keyed by
(dataset fingerprint, model fingerprint) and persisted under
runs/seqeval/cache/. Baseline predictions are computed once and reused across
candidate updates; COMPUTE_COUNTS lets tests assert no recompute happened.
Invalidation is by construction: a new dataset or new checkpoint changes the
fingerprint and therefore the key; stale entries never collide.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from sensorflow.megaeval import population as pop_mod
from sensorflow.seqeval import ledger as ledger_mod

# Simulation profile (baseline physics of the synthetic detector).
BASE_DETECT = 0.945
DIFFICULTY_WEIGHT = 0.16
NIGHT_PENALTY = 0.030
VRU_PENALTY = 0.020
SCENE_EFFECT_SD = 0.030
FLIP_NOISE = 0.04          # per-model chance of drawing a fresh latent
VRU_CLASSES = (1, 2, 3)    # pedestrian, cyclist, motorcycle

GLOBAL_EFFECT_KEY = "__global__"

# fingerprint -> number of actual (non-cache) prediction computations
COMPUTE_COUNTS: Dict[str, int] = {}
_COUNT_LOCK = threading.Lock()


class SimulatedModel:
    """A model version plus per-stratum performance effects.

    effects keys: "class|condition" (e.g. "pedestrian|night") in absolute
    detection-probability points, or "__global__" applied everywhere.
    """

    def __init__(self, model_version: str, effects: Optional[Dict[str, float]] = None):
        self.model_version = model_version
        self.effects = dict(effects or {})

    def fingerprint(self) -> str:
        blob = json.dumps({"v": self.model_version,
                           "e": {k: round(float(v), 6)
                                 for k, v in sorted(self.effects.items())}},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {"model_version": self.model_version, "effects": self.effects,
                "fingerprint": self.fingerprint()}


def dataset_fingerprint(meta: Dict) -> str:
    blob = json.dumps({"pop": meta["population_id"], "seed": meta["seed"],
                       "n": meta["num_objects"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ frame


_FRAME_CACHE: Dict[str, Dict[str, np.ndarray]] = {}
_FRAME_LOCK = threading.Lock()


def load_frame(meta: Dict) -> Dict[str, np.ndarray]:
    """Full-population sampling frame (dimension codes only), sorted by
    object_id so that object_id == row index. Cached in memory per population."""
    pid = meta["population_id"]
    with _FRAME_LOCK:
        if pid in _FRAME_CACHE:
            return _FRAME_CACHE[pid]
    cols_needed = ["object_id", "container_id", "class", "lighting", "scenario",
                   "weather", "difficulty", "safety_critical"]
    parts = {k: [] for k in cols_needed}
    for p in range(meta["num_partitions"]):
        part = pop_mod.load_partition(pid, p)
        for k in cols_needed:
            parts[k].append(part[k])
    frame = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(frame["object_id"], kind="stable")
    frame = {k: v[order] for k, v in frame.items()}
    with _FRAME_LOCK:
        if len(_FRAME_CACHE) > 4:
            _FRAME_CACHE.clear()
        _FRAME_CACHE[pid] = frame
    return frame


# ------------------------------------------------------------------ latents


def _pair_latents(meta: Dict) -> np.ndarray:
    """Per-object pairing latent, shared by all models on this dataset."""
    rng = np.random.default_rng([int(meta["seed"]), 0x5E9, 424242])
    return rng.random(int(meta["num_objects"]))


def _scene_effects(meta: Dict) -> np.ndarray:
    """Container-level 'scene hardness' shared by all models (drives ICC)."""
    rng = np.random.default_rng([int(meta["seed"]), 777])
    return rng.normal(0.0, SCENE_EFFECT_SD, size=int(meta["num_containers"]))


_LATENT_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _latents(meta: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pid = meta["population_id"]
    if pid not in _LATENT_CACHE:
        if len(_LATENT_CACHE) > 4:
            _LATENT_CACHE.clear()
        _LATENT_CACHE[pid] = (_pair_latents(meta), _scene_effects(meta))
    return _LATENT_CACHE[pid]


# ------------------------------------------------------------------ prediction


def _detect_probability(frame: Dict[str, np.ndarray], meta: Dict,
                        model: SimulatedModel, condition_dim: str) -> np.ndarray:
    difficulty = frame["difficulty"].astype(np.float64)
    p = np.full(difficulty.shape, BASE_DETECT)
    p -= DIFFICULTY_WEIGHT * difficulty
    p -= NIGHT_PENALTY * (frame["lighting"] == 1)
    p -= VRU_PENALTY * np.isin(frame["class"], VRU_CLASSES)
    _, scene = _latents(meta)
    p += scene[frame["container_id"]]
    p += float(model.effects.get(GLOBAL_EFFECT_KEY, 0.0))
    cond_vocab = pop_mod.DIMENSIONS[condition_dim]
    cls_vocab = pop_mod.DIMENSIONS["class"]
    for key, eff in model.effects.items():
        if key == GLOBAL_EFFECT_KEY or "|" not in key:
            continue
        cls_name, cond_name = key.split("|", 1)
        if cls_name not in cls_vocab or cond_name not in cond_vocab:
            continue
        mask = ((frame["class"] == cls_vocab.index(cls_name))
                & (frame[condition_dim] == cond_vocab.index(cond_name)))
        p[mask] += float(eff)
    return np.clip(p, 0.01, 0.995)


def _compute_predictions(meta: Dict, model: SimulatedModel,
                         condition_dim: str) -> np.ndarray:
    frame = load_frame(meta)
    p = _detect_probability(frame, meta, model, condition_dim)
    u_pair, _ = _latents(meta)
    fp_int = int(model.fingerprint(), 16)
    rng = np.random.default_rng([fp_int % (2**32), int(meta["seed"]), 7])
    n = u_pair.size
    swap = rng.random(n) < FLIP_NOISE
    fresh = rng.random(n)
    u = np.where(swap, fresh, u_pair)
    return (u < p).astype(np.uint8)


class PredictionCache:
    """Disk + memory cache of full-population success vectors.

    Key = (dataset fingerprint, model fingerprint). Both fingerprints cover
    everything the predictions depend on, so invalidation is automatic: any
    change to the dataset (new population/seed/size) or the model (version or
    effects) produces a new key.
    """

    def __init__(self):
        self._mem: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _path(self, key: str) -> str:
        return os.path.join(ledger_mod.seq_root(), "cache", f"{key}.npz")

    def get_or_compute(self, meta: Dict, model: SimulatedModel,
                       condition_dim: str = "lighting") -> np.ndarray:
        key = f"{dataset_fingerprint(meta)}-{model.fingerprint()}"
        with self._lock:
            if key in self._mem:
                return self._mem[key]
        path = self._path(key)
        if os.path.exists(path):
            with np.load(path) as z:
                success = z["success"]
        else:
            success = _compute_predictions(meta, model, condition_dim)
            with _COUNT_LOCK:
                COMPUTE_COUNTS[key] = COMPUTE_COUNTS.get(key, 0) + 1
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, success=success,
                                model_version=np.array([model.model_version]))
        with self._lock:
            if len(self._mem) > 16:
                self._mem.clear()
            self._mem[key] = success
        return success

    def clear_memory(self) -> None:
        with self._lock:
            self._mem.clear()


_CACHE: Optional[PredictionCache] = None
_CACHE_LOCK = threading.Lock()


def get_prediction_cache() -> PredictionCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = PredictionCache()
        return _CACHE


def reset_prediction_cache() -> None:
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None
    with _FRAME_LOCK:
        _FRAME_CACHE.clear()
    _LATENT_CACHE.clear()
    with _COUNT_LOCK:
        COMPUTE_COUNTS.clear()


def paired_outcomes(meta: Dict, baseline: SimulatedModel, candidate: SimulatedModel,
                    object_ids: np.ndarray, condition_dim: str = "lighting"
                    ) -> Dict[str, np.ndarray]:
    """Per-unit paired outcomes for the given objects (baseline served from
    cache; the same sampled units go through both models)."""
    cache = get_prediction_cache()
    b_full = cache.get_or_compute(meta, baseline, condition_dim)
    c_full = cache.get_or_compute(meta, candidate, condition_dim)
    idx = np.asarray(object_ids, dtype=np.int64)
    b = b_full[idx].astype(np.int8)
    c = c_full[idx].astype(np.int8)
    return {"baseline": b, "candidate": c, "d": (c - b).astype(np.float64)}
