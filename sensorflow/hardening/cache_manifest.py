"""Reproducible feature-cache keys, manifests, integrity and eviction.

Audit findings this design corrects:
- F-006 (megaeval query cache): key omitted run identity -> collisions
  between a run and its baseline. Fixed surgically there by keying on
  run_id; this module is the general design every cache should migrate to.
- F-027 (megaeval artifact cache): insertion-order eviction, no checksums.
- vitis feature reuse: keyed on config but without dependency versions.

Principles:
1. The key is a content hash over a COMPLETE manifest of everything the
   cached value depends on. If it isn't in the manifest, it can't be allowed
   to affect the value.
2. Integrity: every payload stores a checksum, verified on read; corrupt
   entries are treated as misses, never returned.
3. Invalidation is free: bump any dependency version and the key changes.
   Explicit invalidation exists only for "the manifest lied" incidents.
4. Eviction: LRU by last access (not insertion order), with a byte budget.

Migration note for the audited caches:
- megaeval QueryCache: adopt CacheManifest with dependencies {run_id,
  population_id, model_version, label_version, threshold_config, seed,
  query(filters, group_by, metrics)}. The surgical run_id fix already pins
  lineage; the manifest form adds integrity + persistence.
- megaeval RunRegistry._artifacts: keep in-memory map but move eviction to
  LRU and add npz checksums (this module's LocalDiskCache can back it).
- vitis feature reuse: manifest must include IP core version, precision
  flags (quant mode), calibration id and input data hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

from pydantic import BaseModel, Field


class CacheManifest(BaseModel):
    """Complete dependency record for one cached value.

    `dependencies` must include EVERY version/config/precision input:
    e.g. {"data_hash": ..., "model_version": ..., "label_version": ...,
    "evaluator_version": ..., "config_hash": ..., "precision": "fp32",
    "seed": "7"}. Keys and values are strings for stable hashing.
    """

    namespace: str                       # e.g. "vitis.features", "megaeval.query"
    dependencies: Dict[str, str] = Field(default_factory=dict)
    schema_version: str = "manifest-v1"

    def cache_key(self) -> str:
        blob = json.dumps({
            "ns": self.namespace,
            "deps": dict(sorted(self.dependencies.items())),
            "schema": self.schema_version,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def payload_checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FeatureCacheStorage(Protocol):
    """Storage seam for the cache.

    Local implementation below writes to disk. The PRODUCTION option is an
    object store (S3/GCS: key = f"{namespace}/{cache_key}", payload +
    manifest JSON as separate objects, checksum in object metadata, lifecycle
    rules for eviction) fronted by a small metadata DB for LRU bookkeeping.
    """

    def read(self, key: str) -> Optional[Tuple[bytes, Dict]]: ...
    def write(self, key: str, payload: bytes, manifest: Dict) -> None: ...
    def delete(self, key: str) -> None: ...


class LocalDiskCache:
    """LOCAL implementation: content-addressed files with sidecar manifests.

    Layout: root/<key>.bin (payload), root/<key>.json (manifest + checksum +
    access time). Reads verify the checksum; a mismatch deletes the entry and
    reports a miss (corruption never propagates). Eviction is LRU by
    last-access when the byte budget is exceeded.
    """

    def __init__(self, root: Path, max_bytes: int = 512 * 1024 * 1024):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def _paths(self, key: str) -> Tuple[Path, Path]:
        return self.root / f"{key}.bin", self.root / f"{key}.json"

    def read(self, key: str) -> Optional[Tuple[bytes, Dict]]:
        bin_p, meta_p = self._paths(key)
        with self._lock:
            if not bin_p.exists() or not meta_p.exists():
                return None
            payload = bin_p.read_bytes()
            meta = json.loads(meta_p.read_text())
            if payload_checksum(payload) != meta.get("checksum"):
                # Integrity failure: purge and miss.
                bin_p.unlink(missing_ok=True)
                meta_p.unlink(missing_ok=True)
                return None
            meta["last_accessed_at"] = time.time()
            meta_p.write_text(json.dumps(meta))
            return payload, meta

    def write(self, key: str, payload: bytes, manifest: Dict) -> None:
        bin_p, meta_p = self._paths(key)
        with self._lock:
            bin_p.write_bytes(payload)
            meta_p.write_text(json.dumps({
                "manifest": manifest,
                "checksum": payload_checksum(payload),
                "size_bytes": len(payload),
                "created_at": time.time(),
                "last_accessed_at": time.time(),
            }))
            self._evict_if_needed()

    def delete(self, key: str) -> None:
        bin_p, meta_p = self._paths(key)
        with self._lock:
            bin_p.unlink(missing_ok=True)
            meta_p.unlink(missing_ok=True)

    def _evict_if_needed(self) -> None:
        entries = []
        total = 0
        for meta_p in self.root.glob("*.json"):
            try:
                meta = json.loads(meta_p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            total += meta.get("size_bytes", 0)
            entries.append((meta.get("last_accessed_at", 0), meta.get("size_bytes", 0), meta_p))
        if total <= self.max_bytes:
            return
        entries.sort()  # oldest access first
        for _, size, meta_p in entries:
            if total <= self.max_bytes:
                break
            key = meta_p.stem
            (self.root / f"{key}.bin").unlink(missing_ok=True)
            meta_p.unlink(missing_ok=True)
            total -= size


class FeatureCache:
    """Manifest-keyed cache facade over any FeatureCacheStorage."""

    def __init__(self, storage: FeatureCacheStorage):
        self.storage = storage
        self.hits = 0
        self.misses = 0

    def get(self, manifest: CacheManifest) -> Optional[bytes]:
        result = self.storage.read(manifest.cache_key())
        if result is None:
            self.misses += 1
            return None
        self.hits += 1
        return result[0]

    def put(self, manifest: CacheManifest, payload: bytes) -> str:
        key = manifest.cache_key()
        self.storage.write(key, payload, manifest.model_dump())
        return key

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else None}
