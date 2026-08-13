"""Statistical machinery: clustering handling, confidence sequences,
three-outcome logic, e-process validity behaviors."""

from __future__ import annotations

import numpy as np

from sensorflow.seqeval import units
from sensorflow.seqeval.sequential import (DECISION_INSUFFICIENT,
                                           DECISION_PASS,
                                           DECISION_REGRESSION,
                                           EmpiricalBernsteinCS,
                                           PairedSequentialTest)


def _clustered_values(rng, n_clusters, cluster_size, effect_sd=0.45, noise_sd=0.1):
    z = rng.normal(0.0, effect_sd, size=n_clusters)
    d = np.repeat(z, cluster_size) + rng.normal(0.0, noise_sd, size=n_clusters * cluster_size)
    d = np.tanh(d)
    cid = np.repeat(np.arange(n_clusters), cluster_size)
    return d, cid


class TestClustering:
    def test_effective_n_reduced_by_correlation(self):
        rng = np.random.default_rng(3)
        d, cid = _clustered_values(rng, n_clusters=60, cluster_size=20)
        summary = units.cluster_summary(d, cid)
        assert summary["n"] == 1200
        assert summary["icc"] > 0.4
        assert summary["design_effect"] > 5
        assert summary["n_effective"] < 0.5 * summary["n"]

    def test_iid_data_has_no_design_effect(self):
        rng = np.random.default_rng(4)
        d = rng.normal(0, 0.3, size=1200)
        cid = np.repeat(np.arange(60), 20)  # arbitrary grouping of iid data
        summary = units.cluster_summary(d, cid)
        assert summary["design_effect"] < 1.5

    def test_cluster_aware_sequence_wider_than_naive(self):
        """Feeding correlated objects as if iid produces an (invalidly) tighter
        interval than the correct cluster-mean treatment."""
        rng = np.random.default_rng(5)
        d, cid = _clustered_values(rng, n_clusters=50, cluster_size=24)
        naive = PairedSequentialTest(delta=0.01, alpha=0.05)
        naive.update_clusters(d)                       # WRONG: objects as units
        means, _ = units.cluster_units(d, cid)
        correct = PairedSequentialTest(delta=0.01, alpha=0.05)
        correct.update_clusters(means)                 # RIGHT: clusters as units
        naive_width = naive.delta_interval()[1] - naive.delta_interval()[0]
        correct_width = correct.delta_interval()[1] - correct.delta_interval()[0]
        assert correct_width > naive_width * 1.5

    def test_cluster_means_stay_bounded(self):
        rng = np.random.default_rng(6)
        d = rng.choice([-1.0, 0.0, 1.0], size=500)
        cid = rng.integers(0, 40, size=500)
        w = rng.uniform(0.5, 5.0, size=500)
        means, sizes = units.cluster_units(d, cid, weights=w)
        assert np.all(means <= 1.0) and np.all(means >= -1.0)
        assert sizes.sum() == 500


class TestConfidenceSequence:
    def test_cs_covers_true_mean_and_shrinks(self):
        rng = np.random.default_rng(7)
        cs = EmpiricalBernsteinCS(alpha=0.05)
        widths = []
        for chunk in range(20):
            for x in rng.binomial(1, 0.7, size=100):
                cs.update(float(x))
            widths.append(cs.upper - cs.lower)
        assert cs.lower <= 0.7 <= cs.upper
        assert widths[-1] < widths[0]
        assert widths[-1] < 0.12

    def test_running_intersection_never_widens(self):
        rng = np.random.default_rng(8)
        cs = EmpiricalBernsteinCS(alpha=0.05)
        prev = (0.0, 1.0)
        for x in rng.random(2000):
            cs.update(x)
            assert cs.lower >= prev[0] - 1e-12
            assert cs.upper <= prev[1] + 1e-12
            prev = (cs.lower, cs.upper)


class TestThreeOutcome:
    def test_regression_detected(self):
        rng = np.random.default_rng(9)
        t = PairedSequentialTest(delta=0.005, alpha=0.01)
        while t.n_clusters < 5000 and t.decision == DECISION_INSUFFICIENT:
            d = rng.choice([-1.0, 0.0, 1.0], p=[0.05, 0.93, 0.02], size=200)  # Delta=-3pp
            t.update_clusters(d)
            t.evaluate()
        assert t.decision == DECISION_REGRESSION

    def test_pass_on_true_equivalence(self):
        rng = np.random.default_rng(10)
        t = PairedSequentialTest(delta=0.02, alpha=0.05)
        while t.n_clusters < 8000 and t.decision == DECISION_INSUFFICIENT:
            d = rng.choice([-1.0, 0.0, 1.0], p=[0.01, 0.98, 0.01], size=200)  # Delta=0
            t.update_clusters(d)
            t.evaluate()
        assert t.decision == DECISION_PASS

    def test_tiny_sample_is_insufficient_never_pass(self):
        rng = np.random.default_rng(11)
        t = PairedSequentialTest(delta=0.005, alpha=0.05)
        t.update_clusters(rng.choice([-1.0, 0.0, 1.0], p=[0.02, 0.96, 0.02], size=30))
        t.evaluate()
        assert t.decision == DECISION_INSUFFICIENT

    def test_bayes_posterior_tracks_regression(self):
        t = PairedSequentialTest(delta=0.005)
        b = np.ones(2000, dtype=bool)
        c = np.ones(2000, dtype=bool)
        c[:80] = False  # 4pp regression pairs, no improvements
        t.record_objects(b, c)
        assert t.bayes_p_regression() > 0.99

    def test_no_false_regression_under_null_monte_carlo(self):
        """Anytime-valid type-I control at the single-test level: run many null
        streams with continuous monitoring; false REGRESSION rate <= alpha."""
        alpha = 0.05
        false_rejections = 0
        reps = 60
        for rep in range(reps):
            rng = np.random.default_rng(1000 + rep)
            t = PairedSequentialTest(delta=0.005, alpha=alpha)
            for _ in range(40):  # 40 sequential looks
                d = rng.choice([-1.0, 0.0, 1.0], p=[0.015, 0.97, 0.015], size=50)
                t.update_clusters(d)
                if t.evaluate() == DECISION_REGRESSION:
                    false_rejections += 1
                    break
        assert false_rejections / reps <= alpha + 0.02
