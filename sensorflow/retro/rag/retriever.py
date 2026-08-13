"""Retriever over the safety-case index.

Every hit is returned as a RetrievedStandard carrying
{source, document, version, section, retrieved_text, relevance_score} —
ALWAYS, with no optional escape hatch. The scorecard layer refuses citations
that are not instances of this model (the no-citation-without-retrieval
rule, enforced and tested).
"""

from __future__ import annotations

import threading
from typing import List, Optional

from sensorflow.retro.rag.chunking import chunk_document
from sensorflow.retro.rag.corpus import SEED_DOCUMENTS, corpus_metadata
from sensorflow.retro.rag.store import VectorStore, build_store
from sensorflow.retro.scorecard import RetrievedStandard


class SafetyCaseIndex:
    def __init__(self, store: Optional[VectorStore] = None,
                 chunk_size: int = 160, overlap: int = 40):
        self.store = store or build_store()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._loaded_docs = 0
        self._load_seed_corpus()

    @property
    def backend_name(self) -> str:
        return self.store.backend_name

    def _load_seed_corpus(self) -> None:
        ids: List[str] = []
        texts: List[str] = []
        metas: List[dict] = []
        for doc in SEED_DOCUMENTS:
            base_meta = corpus_metadata(doc)
            for chunk in chunk_document(doc["doc_id"], doc["text"], base_meta,
                                        self.chunk_size, self.overlap):
                ids.append(chunk.chunk_id)
                texts.append(chunk.text)
                metas.append(chunk.metadata)
        self.store.add(ids, texts, metas)
        self._loaded_docs = len(SEED_DOCUMENTS)

    def search(self, query: str, k: int = 4) -> List[RetrievedStandard]:
        hits = self.store.query(query, k=k)
        out: List[RetrievedStandard] = []
        for h in hits:
            meta = h.metadata
            out.append(RetrievedStandard(
                source=meta.get("source", "unknown"),
                document=meta.get("document", meta.get("doc_id", "unknown")),
                version=meta.get("version", "unknown"),
                section=meta.get("section", "unknown"),
                retrieved_text=h.text,
                relevance_score=round(h.relevance_score, 4),
                doc_id=meta.get("doc_id", "unknown"),
                doc_type=meta.get("doc_type", "unknown"),
                jurisdiction=meta.get("jurisdiction", "unknown"),
                effective_date=meta.get("effective_date", "unknown"),
                synthetic=meta.get("synthetic", "true") == "true",
                label=meta.get("label", ""),
                chunk_id=h.chunk_id,
            ))
        return out


_INDEX: Optional[SafetyCaseIndex] = None
_INDEX_LOCK = threading.Lock()


def get_index() -> SafetyCaseIndex:
    """Process-wide lazily-built index (seed corpus is static)."""
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = SafetyCaseIndex()
        return _INDEX
