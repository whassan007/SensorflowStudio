"""Embedding layer with an honest, documented tradeoff.

Preferred path: chromadb's bundled embedder (all-MiniLM-L6-v2 via ONNX) when
chromadb is installed — real semantic embeddings.

Fallback path (always available): a deterministic hashed TF vector embedder.
Tradeoff: the hashing embedder is purely LEXICAL — it captures word overlap
(with sublinear term-frequency weighting and stable feature hashing) but has
zero semantic generalization ("pedestrian" and "person on foot" do not
match). For this package's seed corpus, which uses consistent engineering
vocabulary, lexical retrieval is adequate and — critically — fully
deterministic and dependency-free, which the test suite relies on.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words that would otherwise dominate hashed-TF cosine similarity.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with within must shall should not no".split())


class HashedTfEmbedder:
    """Deterministic hashed term-frequency embedder (numpy only)."""

    name = "hashed-tf-fallback"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % self.dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                if tok in _STOPWORDS:
                    continue
                out[i, self._bucket(tok)] += 1.0
            # sublinear tf then L2 normalize
            nz = out[i] > 0
            out[i, nz] = 1.0 + np.log(out[i, nz])
            norm = math.sqrt(float(np.dot(out[i], out[i])))
            if norm > 0:
                out[i] /= norm
        return out


def chromadb_available() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False
