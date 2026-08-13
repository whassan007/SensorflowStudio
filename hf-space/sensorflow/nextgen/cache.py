"""Content-addressed intermediate-representation cache.

Keying: sha256 over the canonical JSON of
    (scenario_id, sensor_version, preprocessing_version,
     backbone_version, feature_schema_version)
plus an artifact kind. ANY version bump changes the key, so a version
mismatch is a MISS by construction — cross-version reuse is impossible, not
merely discouraged (tested in tests/test_nextgen/test_cache.py).

Invalidation strategy (documented, deliberate):
* No explicit invalidation exists or is needed: keys are content-addressed
  over everything the artifact depends on. Producing new sensor data, a new
  preprocessing pipeline, a new backbone or a new feature schema produces
  new keys; stale entries simply stop being referenced.
* Disk growth is bounded by an LRU-style entry cap (`max_entries`): oldest
  by mtime are evicted when the cap is exceeded.
* A deploy that changes semantics WITHOUT bumping a version is the one
  hazard; lineage records (lineage.py COMPONENT_VERSIONS) make the versions
  part of every evaluation run so that mistake is visible in audits.

REUSE NOTE: sensorflow.seqeval.paired.PredictionCache uses the same
fingerprint-keyed design for full-population success vectors. It is bound to
megaeval populations and npz success arrays, while this cache stores
arbitrary JSON-serializable feature artifacts for the bevfusion pipeline —
shared design, different artifact domain (documented reuse decision:
pattern reused, code not force-fit).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sensorflow.nextgen import store


@dataclass
class CacheKeyVersions:
    """Everything a shared intermediate representation depends on."""

    sensor_version: str = "bevfusion-sensors-1.0"
    preprocessing_version: str = "nextgen-preproc-1.0"
    backbone_version: str = "bevfusion-fusion-1.0"
    feature_schema_version: str = "nextgen-features-1.0"

    def key_for(self, scenario_id: str, kind: str = "features") -> str:
        blob = json.dumps({
            "scenario_id": scenario_id, "kind": kind,
            "sensor": self.sensor_version,
            "preprocessing": self.preprocessing_version,
            "backbone": self.backbone_version,
            "schema": self.feature_schema_version,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    compute_time_s: float = 0.0     # time spent on misses (real compute)
    lookup_time_s: float = 0.0      # time spent serving hits
    per_kind: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> Dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 4),
                "compute_time_s": round(self.compute_time_s, 4),
                "lookup_time_s": round(self.lookup_time_s, 4),
                "per_kind": self.per_kind}


class FeatureCache:
    """Memory + disk cache of shared intermediate representations."""

    def __init__(self, subdir: str = "cache", max_entries: int = 4096,
                 max_memory_entries: int = 256):
        self.subdir = subdir
        self.max_entries = max_entries
        self.max_memory_entries = max_memory_entries
        self._mem: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def _dir(self) -> Path:
        return Path(store.nextgen_root()) / self.subdir

    def _path(self, key: str) -> Path:
        return self._dir() / f"{key}.json"

    def get_or_compute(self, key: str, compute: Callable[[], Any],
                       kind: str = "features") -> Any:
        pk = self.stats.per_kind.setdefault(kind, {"hits": 0, "misses": 0})
        with self._lock:
            if key in self._mem:
                t0 = time.perf_counter()
                value = self._mem[key]
                self.stats.hits += 1
                pk["hits"] += 1
                self.stats.lookup_time_s += time.perf_counter() - t0
                return value
        path = self._path(key)
        if path.exists():
            t0 = time.perf_counter()
            with open(path) as f:
                value = json.load(f)
            os.utime(path)  # LRU recency
            with self._lock:
                self.stats.hits += 1
                pk["hits"] += 1
                self.stats.lookup_time_s += time.perf_counter() - t0
                self._remember(key, value)
            return value

        t0 = time.perf_counter()
        value = compute()
        elapsed = time.perf_counter() - t0
        self._dir().mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, path)
        with self._lock:
            self.stats.misses += 1
            pk["misses"] += 1
            self.stats.compute_time_s += elapsed
            self._remember(key, value)
        self._evict_if_needed()
        return value

    def _remember(self, key: str, value: Any) -> None:
        if len(self._mem) >= self.max_memory_entries:
            self._mem.clear()
        self._mem[key] = value

    def _evict_if_needed(self) -> None:
        d = self._dir()
        if not d.exists():
            return
        entries = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in entries[:max(0, len(entries) - self.max_entries)]:
            try:
                p.unlink()
            except OSError:
                pass

    def contains(self, key: str) -> bool:
        with self._lock:
            if key in self._mem:
                return True
        return self._path(key).exists()

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            self.stats = CacheStats()
        d = self._dir()
        if d.exists():
            for p in d.glob("*.json"):
                try:
                    p.unlink()
                except OSError:
                    pass

    def reset_stats(self) -> None:
        with self._lock:
            self.stats = CacheStats()


_CACHE: Optional[FeatureCache] = None
_CACHE_LOCK = threading.Lock()


def get_feature_cache() -> FeatureCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = FeatureCache()
        return _CACHE


def reset_feature_cache() -> None:
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None
