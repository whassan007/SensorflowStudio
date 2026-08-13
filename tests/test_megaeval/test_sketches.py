"""Sketch accuracy: quantile histograms, HyperLogLog, reservoir, Wilson CIs."""

from __future__ import annotations

import numpy as np
import pytest

from sensorflow.megaeval.sketches import HyperLogLog, QuantileHistogram, Reservoir
from sensorflow.megaeval.sampling import wilson_interval


def test_quantile_histogram_percentiles_close_to_exact():
    rng = np.random.default_rng(3)
    data = np.clip(rng.beta(5, 2, size=50_000), 0, 1)
    h = QuantileHistogram(0, 1, 64)
    h.add(data)
    bin_width = 1 / 64
    for q in (10, 25, 50, 75, 90, 99):
        approx = h.percentile(q)
        exact = float(np.percentile(data, q))
        assert abs(approx - exact) <= bin_width + 1e-9, f"p{q}: {approx} vs {exact}"


def test_quantile_histogram_merge_equals_combined():
    rng = np.random.default_rng(4)
    a, b = rng.random(10_000), rng.random(10_000)
    ha, hb, hc = QuantileHistogram(), QuantileHistogram(), QuantileHistogram()
    ha.add(a)
    hb.add(b)
    hc.add(np.concatenate([a, b]))
    merged = ha.merge(hb)
    assert np.array_equal(merged.counts, hc.counts)


def test_hll_estimate_within_five_percent():
    hll = HyperLogLog(12)
    n = 20_000
    hll.add_ids(np.arange(n, dtype=np.int64))
    est = hll.estimate()
    assert abs(est - n) / n < 0.05, est


def test_hll_merge_is_union():
    h1, h2 = HyperLogLog(12), HyperLogLog(12)
    h1.add_ids(np.arange(0, 8_000, dtype=np.int64))
    h2.add_ids(np.arange(4_000, 12_000, dtype=np.int64))
    merged = h1.merge(h2)
    est = merged.estimate()
    assert abs(est - 12_000) / 12_000 < 0.06, est


def test_reservoir_bounded_and_uniformish():
    r = Reservoir(k=100, seed=1)
    r.add_ids(np.arange(10_000))
    assert len(r.items) == 100
    assert r.seen == 10_000
    # sampled ids should span the range, not just the head
    assert max(r.items) > 5_000 and min(r.items) < 5_000


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=0.01)
    assert hi == pytest.approx(0.943, abs=0.01)
    lo0, hi0 = wilson_interval(0, 0)
    assert (lo0, hi0) == (0.0, 1.0)
    # Wilson upper bound at p=1 is mathematically < 1 (that's the point of Wilson)
    lo1, hi1 = wilson_interval(50, 50)
    assert 0.9 < lo1 < hi1 <= 1.0
    assert hi1 == pytest.approx(0.9928, abs=0.01)
