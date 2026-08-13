"""Lightweight approximate data structures for population statistics.

All numpy-based, mergeable (so they can be maintained per partition and reduced),
and each clearly reports whether an answer is exact or approximate.

- QuantileHistogram: fixed-bin histogram over a known range; percentile queries by
  linear interpolation inside the winning bin. Stand-in for a t-digest.
- HyperLogLog: classic HLL with 2^p registers and numpy vectorized updates.
- Reservoir: uniform reservoir sample of exemplar ids (Algorithm R, vectorized-ish).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------- hashing

_SPLITMIX_1 = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_2 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_3 = np.uint64(0x94D049BB133111EB)


def _hash64(values: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit mix (splitmix64) of integer identifiers."""
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + _SPLITMIX_1
        z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_2
        z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_3
        return z ^ (z >> np.uint64(31))


# ---------------------------------------------------------------- quantile histogram


class QuantileHistogram:
    """Fixed-bin histogram over [lo, hi]; approximate percentiles via interpolation.

    Mergeable: histograms with identical bin config add counts element-wise.
    """

    def __init__(self, lo: float = 0.0, hi: float = 1.0, bins: int = 64,
                 counts: Optional[np.ndarray] = None):
        self.lo, self.hi, self.bins = float(lo), float(hi), int(bins)
        self.counts = counts if counts is not None else np.zeros(bins, dtype=np.int64)

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        clipped = np.clip(values, self.lo, self.hi)
        idx = np.minimum(
            ((clipped - self.lo) / (self.hi - self.lo) * self.bins).astype(np.int64),
            self.bins - 1)
        self.counts += np.bincount(idx, minlength=self.bins)

    def merge(self, other: "QuantileHistogram") -> "QuantileHistogram":
        assert (self.lo, self.hi, self.bins) == (other.lo, other.hi, other.bins)
        return QuantileHistogram(self.lo, self.hi, self.bins, self.counts + other.counts)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def percentile(self, q: float) -> Optional[float]:
        """Approximate q-th percentile (q in [0, 100])."""
        total = self.total
        if total == 0:
            return None
        target = total * (q / 100.0)
        cum = np.cumsum(self.counts)
        bin_idx = int(np.searchsorted(cum, target, side="left"))
        bin_idx = min(bin_idx, self.bins - 1)
        prev_cum = cum[bin_idx - 1] if bin_idx > 0 else 0
        in_bin = self.counts[bin_idx]
        frac = 0.5 if in_bin == 0 else float(np.clip((target - prev_cum) / in_bin, 0.0, 1.0))
        width = (self.hi - self.lo) / self.bins
        return self.lo + (bin_idx + frac) * width

    def to_dict(self) -> Dict:
        return {"lo": self.lo, "hi": self.hi, "bins": self.bins,
                "counts": self.counts.tolist(), "exact": False}

    @staticmethod
    def from_arrays(lo: float, hi: float, counts: np.ndarray) -> "QuantileHistogram":
        return QuantileHistogram(lo, hi, len(counts), counts.astype(np.int64))


# ---------------------------------------------------------------- HyperLogLog


class HyperLogLog:
    """HLL cardinality estimator with 2^p registers (approximate).

    Mergeable: element-wise max of registers.
    """

    def __init__(self, p: int = 12, registers: Optional[np.ndarray] = None):
        self.p = int(p)
        self.m = 1 << self.p
        self.registers = registers if registers is not None else np.zeros(self.m, dtype=np.uint8)

    def add_ids(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        h = _hash64(np.asarray(ids))
        reg_idx = (h >> np.uint64(64 - self.p)).astype(np.int64)
        rest = (h << np.uint64(self.p)) | np.uint64((1 << self.p) - 1)
        # rank = leading zeros of the remaining bits + 1
        ranks = np.zeros(ids.shape, dtype=np.uint8)
        remaining = rest.copy()
        # 64-bit leading-zero count via float log2 is unsafe; loop over 64 bits max via vector ops
        lz = np.full(ids.shape, 64, dtype=np.int64)
        nonzero = remaining != 0
        # bit_length via log2 on uint64 split into hi/lo 32-bit halves (exact for ints < 2^53 per half)
        hi = (remaining >> np.uint64(32)).astype(np.float64)
        lo = (remaining & np.uint64(0xFFFFFFFF)).astype(np.float64)
        bl = np.zeros(ids.shape, dtype=np.int64)
        hi_nz = hi > 0
        bl[hi_nz] = 32 + np.floor(np.log2(hi[hi_nz])).astype(np.int64) + 1
        lo_only = (~hi_nz) & (lo > 0)
        bl[lo_only] = np.floor(np.log2(lo[lo_only])).astype(np.int64) + 1
        lz[nonzero] = 64 - bl[nonzero]
        ranks = np.minimum(lz + 1, 64 - self.p + 1).astype(np.uint8)
        np.maximum.at(self.registers, reg_idx, ranks)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        assert self.p == other.p
        return HyperLogLog(self.p, np.maximum(self.registers, other.registers))

    def estimate(self) -> float:
        m = float(self.m)
        alpha = 0.7213 / (1.0 + 1.079 / m)
        inv_sum = np.sum(np.exp2(-self.registers.astype(np.float64)))
        raw = alpha * m * m / inv_sum
        zeros = int(np.sum(self.registers == 0))
        if raw <= 2.5 * m and zeros > 0:  # small-range correction
            return m * np.log(m / zeros)
        return float(raw)


# ---------------------------------------------------------------- reservoir sample


class Reservoir:
    """Uniform reservoir sample of ids (exemplars for drill-down)."""

    def __init__(self, k: int = 200, seed: int = 0):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.items: List[int] = []
        self.seen = 0

    def add_ids(self, ids: np.ndarray) -> None:
        for v in np.asarray(ids).tolist():
            self.seen += 1
            if len(self.items) < self.k:
                self.items.append(int(v))
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.k:
                    self.items[j] = int(v)

    def merge(self, other: "Reservoir") -> "Reservoir":
        merged = Reservoir(self.k, seed=self.seen + other.seen + 1)
        pool = self.items + other.items
        merged.seen = self.seen + other.seen
        if len(pool) <= self.k:
            merged.items = pool
        else:
            idx = merged.rng.choice(len(pool), size=self.k, replace=False)
            merged.items = [pool[i] for i in idx]
        return merged
