"""Property-style tests for the sensorflow.hardening package."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from sensorflow.hardening import hitl, power, quality, readiness, safety_config, sampling
from sensorflow.hardening.cache_manifest import (
    CacheManifest,
    FeatureCache,
    LocalDiskCache,
    payload_checksum,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ determinism


class TestDeterminism:
    def _population(self):
        return {
            sampling.Stratum.REPRESENTATIVE: [f"rep-{i}" for i in range(100)],
            sampling.Stratum.RARE: [f"rare-{i}" for i in range(20)],
            sampling.Stratum.SAFETY_CRITICAL: [f"safe-{i}" for i in range(10)],
        }

    def test_same_seed_same_sample(self):
        a = sampling.stratified_sample(self._population(), budget=30, seed=42)
        b = sampling.stratified_sample(self._population(), budget=30, seed=42)
        assert [i.item_id for i in a.items] == [i.item_id for i in b.items]

    def test_different_seed_different_sample(self):
        a = sampling.stratified_sample(self._population(), budget=30, seed=1)
        b = sampling.stratified_sample(self._population(), budget=30, seed=2)
        assert [i.item_id for i in a.items] != [i.item_id for i in b.items]

    def test_ranking_deterministic_and_order_independent(self):
        comp = {"risk": {"s1": 0.5, "s2": 0.9, "s3": 0.9},
                "novelty": {"s1": 0.2, "s2": 0.1, "s3": 0.1}}
        r1 = sampling.rank_scenarios(["s1", "s2", "s3"], comp)
        r2 = sampling.rank_scenarios(["s3", "s2", "s1"], comp)
        assert [r.scenario_id for r in r1] == [r.scenario_id for r in r2]
        # Tie between s2 and s3 breaks on id, deterministically.
        assert [r.scenario_id for r in r1] == ["s2", "s3", "s1"]
        # Components stay attached (decomposed, explainable).
        assert r1[0].components == {"risk": 0.9, "novelty": 0.1}

    def test_no_unseeded_randomness_in_fixed_modules(self):
        """The audit's fixed production modules must stay free of unseeded
        numpy/stdlib randomness. Guards the F-008 fix and the new package."""
        banned = re.compile(
            r"np\.random\.(randn|rand\b|random|choice|permutation|randint)"
            r"|(?<!\.)random\.(random|randint|choice|shuffle|uniform)\(")
        fixed_modules = [
            REPO_ROOT / "sensorflow" / "perception_automator.py",
            REPO_ROOT / "sensorflow" / "temporal_tracker.py",
            REPO_ROOT / "sensorflow" / "quality_gate.py",
            REPO_ROOT / "sensorflow" / "metrics" / "perception_3d.py",
            REPO_ROOT / "sensorflow" / "metrics" / "temporal_mot.py",
            *sorted((REPO_ROOT / "sensorflow" / "hardening").glob("*.py")),
        ]
        offenders = []
        for path in fixed_modules:
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if banned.search(line):
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
        assert not offenders, f"unseeded randomness reintroduced: {offenders}"


# ------------------------------------------------------------------ stratification


class TestStratificationAndReweighting:
    def test_quotas_respected(self):
        pop = {
            sampling.Stratum.REPRESENTATIVE: [f"r{i}" for i in range(1000)],
            sampling.Stratum.RARE: [f"x{i}" for i in range(100)],
        }
        s = sampling.stratified_sample(
            pop, budget=100,
            quotas={sampling.Stratum.REPRESENTATIVE: 0.7, sampling.Stratum.RARE: 0.3},
            seed=0)
        assert s.quotas[sampling.Stratum.REPRESENTATIVE] == 70
        assert s.quotas[sampling.Stratum.RARE] == 30
        assert len(s.items) == 100

    def test_reweighting_recovers_population_rate_under_oversampling(self):
        """Rare stratum is oversampled 10x; the HT estimate must still recover
        the true population rate while the naive mean is badly inflated."""
        rng = np.random.default_rng(123)
        pop_sizes = {sampling.Stratum.NOMINAL: 2000, sampling.Stratum.RARE: 100}
        rates = {sampling.Stratum.NOMINAL: 0.02, sampling.Stratum.RARE: 0.5}
        pop, outcome = {}, {}
        for stratum, n in pop_sizes.items():
            ids = [f"{stratum.value}-{i}" for i in range(n)]
            pop[stratum] = ids
            fails = rng.random(n) < rates[stratum]
            outcome.update({i: float(f) for i, f in zip(ids, fails)})

        true_rate = float(np.mean([outcome[i] for ids in pop.values() for i in ids]))

        s = sampling.stratified_sample(
            pop, budget=200,
            quotas={sampling.Stratum.NOMINAL: 0.5, sampling.Stratum.RARE: 0.5},
            seed=7)
        measured = {it.item_id: outcome[it.item_id] for it in s.items}
        ht = s.estimate_population_rate(measured)
        naive = float(np.mean(list(measured.values())))

        assert abs(ht - true_rate) < 0.03, (ht, true_rate)
        assert naive > true_rate + 0.10  # naive is visibly biased upward

    def test_small_stratum_fully_taken_with_probability_one(self):
        pop = {sampling.Stratum.KNOWN_FAILURES: ["k1", "k2"],
               sampling.Stratum.NOMINAL: [f"n{i}" for i in range(100)]}
        s = sampling.stratified_sample(
            pop, budget=50,
            quotas={sampling.Stratum.KNOWN_FAILURES: 0.5, sampling.Stratum.NOMINAL: 0.5},
            seed=0)
        kf = [i for i in s.items if i.stratum == sampling.Stratum.KNOWN_FAILURES]
        assert len(kf) == 2
        assert all(i.inclusion_probability == 1.0 and i.weight == 1.0 for i in kf)
        # Leftover budget redistributed: total stays at 50.
        assert len(s.items) == 50

    def test_refuses_estimate_with_unmeasured_stratum(self):
        pop = {sampling.Stratum.NOMINAL: ["a", "b"], sampling.Stratum.RARE: ["c"]}
        s = sampling.stratified_sample(pop, budget=3, seed=0)
        # Measurements missing for the entire RARE stratum -> refuse (None).
        only_nominal = {i.item_id: 1.0 for i in s.items
                        if i.stratum == sampling.Stratum.NOMINAL}
        assert s.estimate_population_rate(only_nominal) is None


# ------------------------------------------------------------------ novelty


class TestNovelty:
    def test_knn_flags_outlier(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, (200, 4))
        queries = np.vstack([np.zeros((1, 4)), np.full((1, 4), 8.0)])
        scores = sampling.knn_novelty(queries, reference, k=5)
        assert scores[1] > scores[0] * 3

    def test_centroid_misses_between_modes_knn_does_not(self):
        """Documents the trade-off: a point between two far modes is novel,
        centroid novelty scores it LOW (near global centroid), kNN scores it
        higher than in-mode points."""
        rng = np.random.default_rng(1)
        mode_a = rng.normal(-10, 0.5, (100, 2))   # centered at (-10, -10)
        mode_b = rng.normal(10, 0.5, (100, 2))    # centered at (10, 10)
        reference = np.vstack([mode_a, mode_b])
        between = np.zeros((1, 2))       # novel: no mass here
        in_mode = np.array([[-10.0, -10.0]])
        knn_b = sampling.knn_novelty(between, reference, k=5)[0]
        knn_m = sampling.knn_novelty(in_mode, reference, k=5)[0]
        cen_b = sampling.centroid_novelty(between, reference)[0]
        cen_m = sampling.centroid_novelty(in_mode, reference)[0]
        assert knn_b > knn_m * 5          # kNN catches it
        assert cen_b < cen_m              # centroid is fooled


# ------------------------------------------------------------------ routing


class TestRouting:
    def _route(self, ev):
        return quality.route(ev)

    def test_clean_item_accepts_with_explanation(self):
        r = self._route(quality.QualityEvidence(
            geometric_validity=0.95, sensor_agreement=0.9, temporal_consistency=0.92,
            semantic_agreement=0.9, confidence=0.88, uncertainty=0.1,
            occlusion=0.1, safety_relevance=0.1))
        assert r["decision"] == quality.RoutingDecision.ACCEPT
        assert r["explanation"]

    def test_vru_miss_in_boundary_never_accepted(self):
        r = self._route(quality.QualityEvidence(
            geometric_validity=0.6, sensor_agreement=0.25, temporal_consistency=0.5,
            semantic_agreement=0.4, confidence=0.3, uncertainty=0.8,
            occlusion=0.6, safety_relevance=0.95))
        assert r["decision"] in (quality.RoutingDecision.QUARANTINE,
                                 quality.RoutingDecision.HITL)
        # Safety escalation must appear in the explanation.
        assert any("safety" in e.lower() for e in r["explanation"])

    def test_phantom_obstacle_quarantined_for_sensor_contradiction(self):
        r = self._route(quality.QualityEvidence(
            geometric_validity=0.8, sensor_agreement=0.15, temporal_consistency=0.3,
            semantic_agreement=0.5, confidence=0.7, uncertainty=0.6,
            occlusion=0.2, safety_relevance=0.7))
        assert r["decision"] == quality.RoutingDecision.QUARANTINE
        assert any("sensor_agreement" in e for e in r["explanation"])

    def test_incomplete_telemetry_never_silent_accept(self):
        r = self._route(quality.QualityEvidence(
            geometric_validity=0.9, sensor_agreement=None, temporal_consistency=None,
            semantic_agreement=0.85, confidence=0.9, uncertainty=0.2,
            occlusion=0.1, safety_relevance=0.2))
        assert r["decision"] != quality.RoutingDecision.ACCEPT
        assert any("incomplete evidence" in e for e in r["explanation"])

    def test_routing_deterministic(self):
        ev = quality.QualityEvidence(
            geometric_validity=0.5, sensor_agreement=0.5, temporal_consistency=0.5,
            semantic_agreement=0.5, confidence=0.5, uncertainty=0.5,
            occlusion=0.5, safety_relevance=0.5)
        results = {self._route(ev)["decision"] for _ in range(10)}
        assert len(results) == 1


# ------------------------------------------------------------------ grader dependence


class TestGraderDependence:
    def test_correlated_graders_confidence_below_naive(self):
        votes = {"g1": "ped", "g2": "ped", "g3": "ped"}
        correlated = [
            quality.GraderProfile(grader_id=g, backbone="vit", training_data="web",
                                  architecture="det")
            for g in votes]
        independent = [
            quality.GraderProfile(grader_id=g, backbone=f"b{g}",
                                  training_data=f"d{g}", architecture=f"a{g}")
            for g in votes]
        c = quality.consensus_with_dependence(votes, correlated)
        i = quality.consensus_with_dependence(votes, independent)
        assert c["naive_consensus"] == i["naive_consensus"] == 1.0
        assert c["adjusted_confidence"] < c["naive_confidence"]
        assert c["adjusted_confidence"] < i["adjusted_confidence"]
        assert c["effective_graders"] < 3 <= i["effective_graders"] + 1e-9

    def test_effective_count_bounds(self):
        clones = [quality.GraderProfile(grader_id=f"g{i}", backbone="x",
                                        training_data="y", architecture="z")
                  for i in range(5)]
        k_eff = quality.effective_grader_count(clones)
        assert 1.0 <= k_eff < 5.0


# ------------------------------------------------------------------ HITL


class TestHitl:
    def _cands(self):
        return [
            hitl.ReviewCandidate(item_id="extreme_risk", risk=1.0, uncertainty=0.05,
                                 novelty=0.05, training_value=0.05),
            hitl.ReviewCandidate(item_id="balanced", risk=0.6, uncertainty=0.6,
                                 novelty=0.6, training_value=0.6),
            hitl.ReviewCandidate(item_id="routine", risk=0.1, uncertainty=0.1,
                                 novelty=0.1, training_value=0.1),
        ]

    def test_pareto_keeps_single_axis_extreme_on_first_front(self):
        fronts = hitl.pareto_rank(self._cands())
        assert fronts["extreme_risk"] == 0
        assert fronts["balanced"] == 0
        assert fronts["routine"] > 0

    def test_product_score_buries_extreme_risk_pareto_rescues_it(self):
        cands = self._cands()
        by_product = sorted(cands, key=lambda c: -hitl.information_gain_score(c))
        assert by_product[0].item_id == "balanced"       # product prefers balance
        prioritized = hitl.prioritize(cands, budget=2)
        assert {r["item_id"] for r in prioritized} == {"balanced", "extreme_risk"}

    def test_prioritize_deterministic_under_input_order(self):
        a = hitl.prioritize(self._cands())
        b = hitl.prioritize(list(reversed(self._cands())))
        assert [r["item_id"] for r in a] == [r["item_id"] for r in b]

    def test_acceptance_metrics(self):
        m = hitl.acceptance_metrics(
            routed_ids=["a", "b", "c"],
            true_problem_ids=["a", "b", "d"],
            critical_ids=["a", "d"],
            total_items=10)
        assert m["hitl_precision"] == pytest.approx(2 / 3)
        assert m["hitl_recall"] == pytest.approx(2 / 3)
        assert m["workload"] == pytest.approx(0.3)
        assert m["critical_miss_rate"] == pytest.approx(0.5)   # 'd' missed
        assert m["false_routing_rate"] == pytest.approx(1 / 7)


# ------------------------------------------------------------------ cache keys


class TestCacheManifest:
    BASE_DEPS = {
        "data_hash": "abc123", "model_version": "m-v3", "label_version": "l-v2",
        "evaluator_version": "e-v1", "config_hash": "cfg9", "precision": "fp32",
        "seed": "7",
    }

    def test_same_inputs_same_key(self):
        m1 = CacheManifest(namespace="vitis.features", dependencies=dict(self.BASE_DEPS))
        m2 = CacheManifest(namespace="vitis.features",
                           dependencies=dict(reversed(list(self.BASE_DEPS.items()))))
        assert m1.cache_key() == m2.cache_key()

    def test_every_dependency_bump_changes_key(self):
        base = CacheManifest(namespace="vitis.features",
                             dependencies=dict(self.BASE_DEPS)).cache_key()
        for dep in self.BASE_DEPS:
            bumped = dict(self.BASE_DEPS)
            bumped[dep] = bumped[dep] + "-bumped"
            key = CacheManifest(namespace="vitis.features",
                                dependencies=bumped).cache_key()
            assert key != base, f"key insensitive to dependency {dep!r}"

    def test_namespace_changes_key(self):
        a = CacheManifest(namespace="a", dependencies=dict(self.BASE_DEPS)).cache_key()
        b = CacheManifest(namespace="b", dependencies=dict(self.BASE_DEPS)).cache_key()
        assert a != b

    def test_roundtrip_and_integrity(self, tmp_path):
        cache = FeatureCache(LocalDiskCache(tmp_path, max_bytes=10_000))
        manifest = CacheManifest(namespace="t", dependencies=dict(self.BASE_DEPS))
        assert cache.get(manifest) is None
        key = cache.put(manifest, b"payload-bytes")
        assert cache.get(manifest) == b"payload-bytes"
        # Corrupt the payload on disk: read must MISS, not return garbage.
        (tmp_path / f"{key}.bin").write_bytes(b"tampered")
        assert cache.get(manifest) is None

    def test_lru_eviction_by_access_not_insertion(self, tmp_path):
        cache = FeatureCache(LocalDiskCache(tmp_path, max_bytes=250))
        deps = lambda i: {**self.BASE_DEPS, "data_hash": f"h{i}"}
        m0 = CacheManifest(namespace="t", dependencies=deps(0))
        m1 = CacheManifest(namespace="t", dependencies=deps(1))
        cache.put(m0, b"x" * 100)
        cache.put(m1, b"y" * 100)
        cache.get(m0)  # refresh m0: m1 is now least-recently-used
        m2 = CacheManifest(namespace="t", dependencies=deps(2))
        cache.put(m2, b"z" * 100)  # exceeds budget -> evict LRU (m1)
        assert cache.get(m0) is not None
        assert cache.get(m1) is None


# ------------------------------------------------------------------ power


class TestPower:
    def test_larger_mde_needs_smaller_n(self):
        n_small = power.required_n_two_proportions(0.9, mde=0.01)
        n_large = power.required_n_two_proportions(0.9, mde=0.05)
        assert n_large < n_small

    def test_pairing_correlation_reduces_n(self):
        n0 = power.required_n_paired(0.9, mde=0.02, pairing_correlation=0.0)
        n5 = power.required_n_paired(0.9, mde=0.02, pairing_correlation=0.5)
        assert n5 < n0
        assert n5 == pytest.approx(n0 * 0.5, rel=0.05)

    def test_higher_power_and_lower_alpha_need_more_n(self):
        base = power.required_n_two_proportions(0.9, 0.03, alpha=0.05, power=0.8)
        assert power.required_n_two_proportions(0.9, 0.03, alpha=0.01, power=0.8) > base
        assert power.required_n_two_proportions(0.9, 0.03, alpha=0.05, power=0.95) > base

    def test_rare_prevalence_inflates_stream_n(self):
        r = power.required_events_rare(prevalence=0.01, baseline_rate=0.95, mde=0.02)
        assert r["n_stream"] == pytest.approx(r["n_slice"] * 100, rel=0.01)

    def test_cluster_design_effect(self):
        assert power.cluster_design_effect(1, 0.5) == 1.0
        assert power.cluster_design_effect(20, 0.1) == pytest.approx(2.9)

    def test_tier_sizing_no_fixed_counts(self):
        tiers = [power.tier_sizing(s) for s in power.default_tiers().values()]
        # Sizes strictly increase with tier stringency, derived not fixed.
        ns = [t["n_stream_items"] for t in tiers]
        assert ns == sorted(ns)
        assert ns[0] < ns[-1]
        for t in tiers:
            assert t["inputs"]["mde"] > 0
            assert "seqeval" in t["sequential_note"]


# ------------------------------------------------------------------ readiness


class TestReadiness:
    def test_scorecard_never_ready_with_open_critical(self):
        audit = {"findings": [
            {"id": "X-1", "area": "Mock-presented-as-real", "severity": "Critical",
             "disposition": "follow_up"},
        ], "summary": {}}
        sc = readiness.scorecard(audit)
        assert sc["overall_status"] == "NOT_PRODUCTION_READY"

    def test_scorecard_from_real_audit(self):
        sc = readiness.scorecard()
        assert sc["overall_status"] == "NOT_PRODUCTION_READY"
        assert any(c["open_critical_ids"] for c in sc["categories"])
        # Every finding category has the 4 required scorecard columns.
        for c in sc["categories"]:
            assert c["prototype"] and c["production_requirement"]
            assert "gap_count" in c and "status" in c

    def test_all_audit_findings_categorized(self):
        audit = readiness.load_audit()
        buckets = readiness._categorize(audit["findings"])
        categorized = sum(len(v) for v in buckets.values())
        assert categorized == len(audit["findings"])


# ------------------------------------------------------------------ safety config


class TestSafetyConfig:
    def test_every_threshold_has_provenance_and_source(self):
        for spec in safety_config.THRESHOLDS.values():
            assert spec.provenance in (safety_config.FHWA_SSAM_DEFAULT,
                                       safety_config.ILLUSTRATIVE_THRESHOLD)
            assert spec.source

    def test_severity_matches_legacy_formula_on_full_inputs(self):
        """Same functional form as app_backend._compute_severity: keep parity
        so wiring it in later is behavior-preserving for full inputs."""
        legacy = 1.0 - (min(0.8, 1.5) / 1.5) * 0.5 - (min(1.2, 5.0) / 5.0) * 0.2 \
            - (1.0 - min(12.5, 18.0) / 18.0) * 0.3
        got = safety_config.compute_severity(0.8, 1.2, 12.5)
        assert got["severity_index"] == pytest.approx(round(max(0, min(1, legacy)), 3))
        assert got["config_version"] == safety_config.CONFIG_VERSION

    def test_missing_inputs_reported_not_defaulted(self):
        got = safety_config.compute_severity(0.8, None, None)
        assert got["inputs_missing"] == ["pet", "speed"]
