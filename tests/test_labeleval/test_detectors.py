"""Anomaly detector + ensemble tests."""

import numpy as np
import pytest

from sensorflow.evaluation.detectors import (
    AnomalyEnsemble,
    AutoencoderDetector,
    DBSCANDetector,
    FewShotDetector,
    GANDetector,
    IsolationForestDetector,
    KNNDetector,
    LOFDetector,
    OCSVMDetector,
    VAEDetector,
    normalize_scores,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    normal = rng.normal(0, 1, size=(300, 6))
    # Scattered isolated outliers (a tight far-away cluster would legitimately
    # defeat purely local detectors like KNN/DBSCAN).
    outliers = rng.uniform(5, 10, size=(12, 6)) * rng.choice([-1, 1], size=(12, 6))
    X = np.vstack([normal, outliers])
    y = np.array([0] * 300 + [1] * 12)
    return X, y


ALL_DETECTORS = [
    KNNDetector(k=8),
    LOFDetector(n_neighbors=15),
    IsolationForestDetector(n_estimators=50),
    OCSVMDetector(nu=0.05),
    DBSCANDetector(eps=1.5, min_samples=5),
    AutoencoderDetector(latent_dim=3, epochs=40),
    VAEDetector(latent_dim=3),
    GANDetector(),
    FewShotDetector(support_per_class=15),
]


@pytest.mark.parametrize("det", ALL_DETECTORS, ids=lambda d: d.name)
def test_each_detector_ranks_outliers_higher(det, data):
    X, y = data
    det.fit(X)
    scores = det.score(X)
    assert scores.shape == (len(X),)
    # Mean score of true outliers should exceed mean score of normals.
    assert scores[y == 1].mean() > scores[y == 0].mean()
    # Top-12 scores should be dominated by the injected outliers.
    top = np.argsort(-scores)[:12]
    assert (y[top] == 1).sum() >= 8


def test_normalize_scores_is_rank_based():
    s = np.array([10.0, -5.0, 3.0, 100.0])
    n = normalize_scores(s)
    assert n.min() == 0.0 and n.max() == 1.0
    assert n[np.argmax(s)] == 1.0 and n[np.argmin(s)] == 0.0


@pytest.mark.parametrize("strategy", ["majority_vote", "weighted_average", "meta_classifier"])
def test_ensemble_strategies(strategy, data):
    X, y = data
    cfg = {"advanced": {"ensemble_strategy": strategy, "decision_threshold": 0.9,
                        "few_shot": {"enabled": True, "support_per_class": 15}},
           "deep": {"gan": {"enabled": True}}}
    ens = AnomalyEnsemble(cfg, seed=5)
    scores, raw, norm = ens.run(X, supervision=y.astype(float))
    assert len(scores) == len(X)
    assert set(raw) == set(norm)
    assert len(raw) >= 8  # all default detectors + gan
    for k in norm:
        assert 0.0 <= norm[k].min() and norm[k].max() <= 1.0
    # ensemble should separate outliers from normals
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_ensemble_respects_disabled_detectors(data):
    X, _ = data
    cfg = {
        "detectors": {
            "knn": {"enabled": True, "k": 5},
            "lof": {"enabled": False},
            "isolation_forest": {"enabled": False},
            "ocsvm": {"enabled": False},
            "dbscan": {"enabled": False},
        },
        "deep": {"autoencoder": {"enabled": False}, "vae": {"enabled": False}, "gan": {"enabled": False}},
        "advanced": {"few_shot": {"enabled": False}, "ensemble_strategy": "weighted_average",
                     "decision_threshold": 0.9},
    }
    ens = AnomalyEnsemble(cfg)
    _, raw, _ = ens.run(X)
    assert list(raw) == ["knn"]
