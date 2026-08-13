"""Production interface protocols with explicitly-labeled LOCAL implementations.

Audit F-026: the platform's stores are single-node, in-process dicts and JSON
files under runs/, while the narrative talks exabyte scale. The honest fix is
a seam: code depends on these Protocols; the CURRENT implementations are
registered as LOCAL/MOCK so nobody can mistake them for the production
substrate; production options are documented per protocol.

Registry usage:
    impl = get_implementation("VectorDB")   # -> registered LOCAL impl
    describe_implementations()              # -> label + production options
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ protocols


class VectorDB(Protocol):
    """Similarity search over embeddings.

    Production options: Milvus/Qdrant/Vertex Matching Engine with HNSW or
    IVF-PQ indexes; embeddings versioned alongside the encoder model.
    """

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]: ...


class ObjectStorage(Protocol):
    """Immutable blob storage.

    Production options: S3/GCS with bucket versioning, lifecycle policies,
    SSE; content-addressed keys from cache_manifest.CacheManifest.
    """

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> Optional[bytes]: ...
    def exists(self, key: str) -> bool: ...


class DistributedCompute(Protocol):
    """Partitioned map + reduce over large datasets.

    Production options: Spark/Ray/Dask. megaeval's partial_stats/
    reduce_partials pair is already shaped for map-side combine + shuffle
    reduce and should port directly.
    """

    def map_partitions(self, partitions: Iterable[Any],
                       fn: Callable[[Any], Any]) -> List[Any]: ...
    def reduce(self, partials: List[Any], fn: Callable[[List[Any]], Any]) -> Any: ...


class GPUInference(Protocol):
    """Batched model inference.

    Production options: Triton Inference Server / TensorRT-LLM / Ray Serve
    with explicit model_version pinning per request.
    """

    def infer(self, model_version: str, batch: np.ndarray) -> np.ndarray: ...


class FeatureCacheProtocol(Protocol):
    """Manifest-keyed feature cache (see cache_manifest.py).

    Production options: object store payloads + Redis/DB metadata for LRU.
    """

    def get(self, manifest: Any) -> Optional[bytes]: ...
    def put(self, manifest: Any, payload: bytes) -> str: ...


class MetadataStore(Protocol):
    """Transactional metadata (datasets, runs, lineage).

    Production options: Postgres with migrations; DataHub/OpenLineage for
    lineage graphs. NOT JSON files under runs/.
    """

    def put(self, kind: str, record_id: str, record: Dict) -> None: ...
    def get(self, kind: str, record_id: str) -> Optional[Dict]: ...
    def query(self, kind: str, **filters: Any) -> List[Dict]: ...


class ExperimentTracking(Protocol):
    """Run/metric/artifact tracking.

    Production options: MLflow / W&B; every EvaluationResult logged with
    dataset_version, model_version, seed and config hash.
    """

    def log_run(self, run_id: str, params: Dict, metrics: Dict) -> None: ...
    def get_run(self, run_id: str) -> Optional[Dict]: ...


class Observability(Protocol):
    """Metrics/traces/alerts.

    Production options: Prometheus + Grafana + OpenTelemetry traces; alert
    rules on funnel-stage drop rates and cache integrity failures.
    """

    def emit_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None: ...
    def counters(self) -> Dict[str, float]: ...


# ------------------------------------------------------------------ LOCAL implementations
# Every one of these is a single-process, in-memory or local-disk stand-in.
# They exist so the seam is real and testable — NOT to serve production load.


class LocalBruteForceVectorDB:
    """LOCAL/MOCK: exact brute-force search in memory. O(n) per query;
    fine below ~10^5 vectors, nowhere near production."""

    LABEL = "LOCAL"

    def __init__(self):
        self._ids: List[str] = []
        self._vecs: Optional[np.ndarray] = None

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        self._ids.extend(ids)
        self._vecs = vectors if self._vecs is None else np.vstack([self._vecs, vectors])

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self._vecs is None or not len(self._ids):
            return []
        d = np.linalg.norm(self._vecs - query[None, :], axis=1)
        order = np.argsort(d)[:k]
        return [(self._ids[i], float(d[i])) for i in order]


class LocalDiskObjectStorage:
    """LOCAL/MOCK: files under a root dir. No versioning, no durability."""

    LABEL = "LOCAL"

    def __init__(self, root: Path = Path("runs/hardening/objects")):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / safe

    def put(self, key: str, data: bytes) -> None:
        self._p(key).write_bytes(data)

    def get(self, key: str) -> Optional[bytes]:
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def exists(self, key: str) -> bool:
        return self._p(key).exists()


class LocalSerialCompute:
    """LOCAL/MOCK: serial in-process execution posing as distributed compute.
    Preserves the map/reduce shape so megaeval-style partials port unchanged."""

    LABEL = "LOCAL"

    def map_partitions(self, partitions: Iterable[Any],
                       fn: Callable[[Any], Any]) -> List[Any]:
        return [fn(p) for p in partitions]

    def reduce(self, partials: List[Any], fn: Callable[[List[Any]], Any]) -> Any:
        return fn(partials)


class MockGPUInference:
    """MOCK: deterministic hash-seeded outputs; NO real model runs here.
    Outputs are simulated and must never be presented as model predictions."""

    LABEL = "MOCK"

    def infer(self, model_version: str, batch: np.ndarray) -> np.ndarray:
        seed = abs(hash(model_version)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random((len(batch), 8))


class LocalJsonMetadataStore:
    """LOCAL/MOCK: JSON files, no transactions, no concurrent writers."""

    LABEL = "LOCAL"

    def __init__(self, root: Path = Path("runs/hardening/metadata")):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, kind: str, record_id: str) -> Path:
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{record_id}.json"

    def put(self, kind: str, record_id: str, record: Dict) -> None:
        self._p(kind, record_id).write_text(json.dumps(record, default=str))

    def get(self, kind: str, record_id: str) -> Optional[Dict]:
        p = self._p(kind, record_id)
        return json.loads(p.read_text()) if p.exists() else None

    def query(self, kind: str, **filters: Any) -> List[Dict]:
        d = self.root / kind
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text())
            if all(rec.get(k) == v for k, v in filters.items()):
                out.append(rec)
        return out


class LocalExperimentTracking:
    """LOCAL/MOCK: in-memory run log, lost on restart."""

    LABEL = "LOCAL"

    def __init__(self):
        self._runs: Dict[str, Dict] = {}

    def log_run(self, run_id: str, params: Dict, metrics: Dict) -> None:
        self._runs[run_id] = {"params": params, "metrics": metrics,
                              "logged_at": time.time()}

    def get_run(self, run_id: str) -> Optional[Dict]:
        return self._runs.get(run_id)


class LocalObservability:
    """LOCAL/MOCK: in-memory counters; no alerting, no retention."""

    LABEL = "LOCAL"

    def __init__(self):
        self._counters: Dict[str, float] = {}

    def emit_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        tag_key = "" if not tags else "|" + ",".join(
            f"{k}={v}" for k, v in sorted(tags.items()))
        self._counters[name + tag_key] = self._counters.get(name + tag_key, 0.0) + value

    def counters(self) -> Dict[str, float]:
        return dict(self._counters)


# ------------------------------------------------------------------ registry


@dataclass
class Registration:
    protocol: str
    implementation: Any
    label: str                    # LOCAL | MOCK | PRODUCTION
    production_options: str


_REGISTRY: Dict[str, Registration] = {}


def register(protocol: str, implementation: Any, label: str,
             production_options: str) -> None:
    _REGISTRY[protocol] = Registration(protocol, implementation, label,
                                       production_options)


def get_implementation(protocol: str) -> Any:
    return _REGISTRY[protocol].implementation


def describe_implementations() -> List[Dict]:
    return [{
        "protocol": r.protocol,
        "implementation": type(r.implementation).__name__,
        "label": r.label,
        "production_options": r.production_options,
    } for r in _REGISTRY.values()]


def _register_defaults() -> None:
    from sensorflow.hardening.cache_manifest import FeatureCache, LocalDiskCache
    register("VectorDB", LocalBruteForceVectorDB(), "LOCAL",
             "Milvus / Qdrant / Vertex Matching Engine (HNSW, IVF-PQ)")
    register("ObjectStorage", LocalDiskObjectStorage(), "LOCAL",
             "S3 / GCS with versioning, lifecycle policies, SSE")
    register("DistributedCompute", LocalSerialCompute(), "LOCAL",
             "Spark / Ray / Dask (megaeval partials port directly)")
    register("GPUInference", MockGPUInference(), "MOCK",
             "Triton Inference Server / TensorRT / Ray Serve")
    register("FeatureCache",
             FeatureCache(LocalDiskCache(Path("runs/hardening/feature_cache"))),
             "LOCAL", "Object-store payloads + Redis/DB metadata for LRU")
    register("MetadataStore", LocalJsonMetadataStore(), "LOCAL",
             "Postgres + migrations; DataHub/OpenLineage for lineage")
    register("ExperimentTracking", LocalExperimentTracking(), "LOCAL",
             "MLflow / Weights & Biases")
    register("Observability", LocalObservability(), "LOCAL",
             "Prometheus + Grafana + OpenTelemetry")


_register_defaults()
