"""Content-addressed cache: hits, version-bump misses, measured speedup."""

from __future__ import annotations

import time

from sensorflow.nextgen.cache import (
    CacheKeyVersions, FeatureCache, get_feature_cache,
)
from sensorflow.nextgen import compute as compute_mod


def test_hit_on_identical_versions_miss_on_any_bump():
    v = CacheKeyVersions()
    key = v.key_for("scene-1")
    assert key == CacheKeyVersions().key_for("scene-1")  # content-addressed
    # ANY single component bump changes the key -> structural miss.
    for field in ("sensor_version", "preprocessing_version",
                  "backbone_version", "feature_schema_version"):
        bumped = CacheKeyVersions(**{field: "2.0"})
        assert bumped.key_for("scene-1") != key
    assert v.key_for("scene-2") != key


def test_cache_serves_hits_without_recompute():
    cache = FeatureCache(subdir="cache-test")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"payload": list(range(100))}

    key = CacheKeyVersions().key_for("scene-x")
    a = cache.get_or_compute(key, compute)
    b = cache.get_or_compute(key, compute)
    assert a == b
    assert calls["n"] == 1
    assert cache.stats.hits == 1 and cache.stats.misses == 1

    # Version bump -> different key -> recompute (never cross-version reuse).
    key2 = CacheKeyVersions(backbone_version="bev-2.0").key_for("scene-x")
    cache.get_or_compute(key2, compute)
    assert calls["n"] == 2


def test_benchmark_measures_real_speedup_and_hit_rate():
    result = compute_mod.benchmark(n_scenarios=3, frames_per_sequence=10,
                                   persist=False)
    # cold pass misses (one per scenario), warm pass hits everything
    assert result.cache_misses == 3
    assert result.cache_hits == 3
    assert result.hit_rate == 0.5
    # dedup must beat naive scenario x model full inference substantially
    assert result.savings_ratio > 0.4
    assert result.optimized_cost_s < result.naive_cost_s
    assert result.measured_backbone_s > result.measured_head_s
    assert "version bump" in result.invalidation


def test_warm_rerun_measured_faster_than_cold():
    cache = get_feature_cache()
    from sensorflow.bevfusion.scenes import generate_sequences
    seqs = generate_sequences(n_sequences=2, frames_per_sequence=10, seed=7)
    models = ["baseline-v3", "candidate-v4"]

    t0 = time.perf_counter()
    compute_mod.evaluate_scenarios(seqs, models)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    compute_mod.evaluate_scenarios(seqs, models)
    warm = time.perf_counter() - t0
    assert warm < cold / 2  # measured speedup threshold
    assert cache.stats.hits >= 2
