"""Vector store with two interchangeable implementations.

- ChromaStore: chromadb (its bundled embedder), when installed.
- NumpyCosineStore: deterministic hashed-TF embeddings + cosine similarity.

Both expose the exact same interface: add(ids, texts, metadatas) and
query(text, k) -> List[StoreHit]. build_store() picks chroma when available
unless RETRO_FORCE_FALLBACK_STORE=1.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol

import numpy as np
from pydantic import BaseModel

from sensorflow.retro.rag.embedder import HashedTfEmbedder, chromadb_available


class StoreHit(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, str]
    relevance_score: float  # cosine similarity in [0, 1] (clipped)


class VectorStore(Protocol):
    backend_name: str

    def add(self, ids: List[str], texts: List[str],
            metadatas: List[Dict[str, str]]) -> None: ...

    def query(self, text: str, k: int = 5) -> List[StoreHit]: ...


class NumpyCosineStore:
    """Deterministic fallback store: hashed-TF embeddings, cosine ranking."""

    backend_name = "numpy-cosine-fallback"

    def __init__(self, embedder: Optional[HashedTfEmbedder] = None):
        self.embedder = embedder or HashedTfEmbedder()
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict[str, str]] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, ids: List[str], texts: List[str],
            metadatas: List[Dict[str, str]]) -> None:
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("ids/texts/metadatas length mismatch")
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metas.extend(metadatas)
        embs = self.embedder.embed(texts)
        self._matrix = embs if self._matrix is None else np.vstack([self._matrix, embs])

    def query(self, text: str, k: int = 5) -> List[StoreHit]:
        if self._matrix is None or not self._ids:
            return []
        q = self.embedder.embed([text])[0]
        sims = self._matrix @ q  # rows are L2-normalized -> dot = cosine
        order = np.argsort(-sims)[:k]
        return [StoreHit(chunk_id=self._ids[i], text=self._texts[i],
                         metadata=self._metas[i],
                         relevance_score=float(np.clip(sims[i], 0.0, 1.0)))
                for i in order]


class ChromaStore:
    """chromadb-backed store using its default bundled embedder."""

    backend_name = "chromadb"

    def __init__(self, collection_name: str = "retro_safety_case"):
        import chromadb
        self._client = chromadb.EphemeralClient()
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, ids: List[str], texts: List[str],
            metadatas: List[Dict[str, str]]) -> None:
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, text: str, k: int = 5) -> List[StoreHit]:
        res = self._collection.query(query_texts=[text], n_results=k)
        hits: List[StoreHit] = []
        for i, cid in enumerate(res["ids"][0]):
            distance = float(res["distances"][0][i])  # cosine distance
            hits.append(StoreHit(
                chunk_id=cid,
                text=res["documents"][0][i],
                metadata={k2: str(v) for k2, v in (res["metadatas"][0][i] or {}).items()},
                relevance_score=float(np.clip(1.0 - distance, 0.0, 1.0))))
        return hits


def build_store() -> VectorStore:
    if chromadb_available() and os.environ.get("RETRO_FORCE_FALLBACK_STORE") != "1":
        try:
            return ChromaStore()
        except Exception:
            # chromadb installed but not functional (e.g. onnxruntime issue):
            # fall back rather than fail — the interface is identical.
            pass
    return NumpyCosineStore()
