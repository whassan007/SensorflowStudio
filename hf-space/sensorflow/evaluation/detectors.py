"""ML anomaly detection engine: multiple detector families + score ensemble.

Families (spec §13):
- density:      KNN distance, Local Outlier Factor
- isolation:    Isolation Forest, One-Class SVM, DBSCAN noise/centroid distance
- deep:         Autoencoder reconstruction error (lightweight numpy MLP)
- generative:   VAE (numpy, PCA-latent Gaussian likelihood), GAN (numpy
                Mahalanobis "discriminator" surrogate)
- metric:       few-shot prototype embeddings (support/query distance)

The deep/generative detectors are intentionally lightweight numpy
implementations (no torch in this environment): clearly marked, deterministic,
and still produce meaningful scores on the synthetic features.

All detectors implement Detector.fit(X) / .score(X) where HIGHER = more
anomalous. Ensemble normalizes scores before combining and records
detector_scores, normalized_scores, ensemble_strategy, ensemble_score and the
decision_threshold on every decision (spec §14).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class Detector:
    name = "detector"
    family = "generic"

    def fit(self, X: np.ndarray) -> "Detector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        raise NotImplementedError


class KNNDetector(Detector):
    name = "knn"
    family = "density"

    def __init__(self, k: int = 10):
        self.k = k
        self.nn: Optional[NearestNeighbors] = None

    def fit(self, X: np.ndarray) -> "KNNDetector":
        self.nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(X)))
        self.nn.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        dists, _ = self.nn.kneighbors(X)
        return dists[:, 1:].mean(axis=1)  # skip self-distance


class LOFDetector(Detector):
    name = "lof"
    family = "density"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self._scores: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        lof = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, max(2, len(X) - 1)))
        lof.fit(X)
        self._scores = -lof.negative_outlier_factor_
        self._X = X
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._X is not None and X.shape == self._X.shape and np.allclose(X, self._X):
            return self._scores
        # Novel points (e.g. a re-scored corrected label): approximate LOF by
        # the ratio of the query's kNN distance to the training population's.
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self._X)))
        nn.fit(self._X)
        d_query, _ = nn.kneighbors(X)
        d_train, _ = nn.kneighbors(self._X)
        ref = max(float(d_train[:, 1:].mean()), 1e-9)
        return d_query.mean(axis=1) / ref


class IsolationForestDetector(Detector):
    name = "isolation_forest"
    family = "isolation"

    def __init__(self, n_estimators: int = 100, seed: int = 7):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class OCSVMDetector(Detector):
    name = "ocsvm"
    family = "isolation"

    def __init__(self, nu: float = 0.05):
        self.model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")

    def fit(self, X: np.ndarray) -> "OCSVMDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class DBSCANDetector(Detector):
    name = "dbscan"
    family = "isolation"

    def __init__(self, eps: float = 1.2, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.centroids: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(X)
        cents = [X[labels == c].mean(axis=0) for c in sorted(set(labels)) if c != -1]
        self.centroids = np.array(cents) if cents else X.mean(axis=0, keepdims=True)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return d.min(axis=1)


class AutoencoderDetector(Detector):
    """Lightweight numpy MLP autoencoder (in-repo substitute for a torch AE).

    Single hidden bottleneck, tanh activation, plain SGD; deterministic seed.
    Score = per-sample reconstruction MSE.
    """

    name = "autoencoder"
    family = "deep"

    def __init__(self, latent_dim: int = 4, epochs: int = 60, lr: float = 0.05, seed: int = 7):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.W1 = self.b1 = self.W2 = self.b2 = None

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = min(self.latent_dim, d)
        self.W1 = rng.normal(0, 0.1, (d, k))
        self.b1 = np.zeros(k)
        self.W2 = rng.normal(0, 0.1, (k, d))
        self.b2 = np.zeros(d)
        for _ in range(self.epochs):
            idx = rng.permutation(n)[: min(n, 512)]
            xb = X[idx]
            z = np.tanh(xb @ self.W1 + self.b1)
            xh = z @ self.W2 + self.b2
            err = xh - xb  # (B, d)
            gW2 = z.T @ err / len(xb)
            gb2 = err.mean(axis=0)
            dz = (err @ self.W2.T) * (1 - z ** 2)
            gW1 = xb.T @ dz / len(xb)
            gb1 = dz.mean(axis=0)
            self.W2 -= self.lr * gW2
            self.b2 -= self.lr * gb2
            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * gb1
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z = np.tanh(X @ self.W1 + self.b1)
        xh = z @ self.W2 + self.b2
        return ((xh - X) ** 2).mean(axis=1)


class VAEDetector(Detector):
    """Lightweight generative surrogate: Gaussian likelihood in a PCA latent
    space (stands in for a torch VAE ELBO). Score = negative log-likelihood.
    """

    name = "vae"
    family = "generative"

    def __init__(self, latent_dim: int = 5):
        self.latent_dim = latent_dim
        self.mean = self.components = self.latent_var = self.noise_var = None

    def fit(self, X: np.ndarray) -> "VAEDetector":
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.latent_dim, vt.shape[0])
        self.components = vt[:k]
        z = Xc @ self.components.T
        self.latent_var = z.var(axis=0) + 1e-6
        recon = z @ self.components
        self.noise_var = max(float(((Xc - recon) ** 2).mean()), 1e-6)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        z = Xc @ self.components.T
        recon = z @ self.components
        nll_latent = 0.5 * ((z ** 2) / self.latent_var).sum(axis=1)
        nll_recon = 0.5 * ((Xc - recon) ** 2).sum(axis=1) / self.noise_var
        return nll_latent + nll_recon


class GANDetector(Detector):
    """Lightweight generative-adversarial surrogate: discriminator score
    approximated by Mahalanobis distance under a shrunk covariance of the
    "real" data distribution (stands in for a torch GAN discriminator).
    """

    name = "gan"
    family = "generative"

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage
        self.mean = self.prec = None

    def fit(self, X: np.ndarray) -> "GANDetector":
        self.mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.eye(X.shape[1]) * np.trace(cov) / X.shape[1]
        self.prec = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mean
        return np.sqrt(np.maximum((d @ self.prec * d).sum(axis=1), 0))


class FewShotDetector(Detector):
    """Metric-learning detector: class prototypes from a small support set of
    presumed-normal samples; query score = distance to nearest prototype.
    """

    name = "few_shot"
    family = "metric"

    def __init__(self, support_per_class: int = 20, n_prototypes: int = 8, seed: int = 7):
        self.support_per_class = support_per_class
        self.n_prototypes = n_prototypes
        self.seed = seed
        self.prototypes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FewShotDetector":
        rng = np.random.default_rng(self.seed)
        # Support set: samples nearest to the data median (most "typical").
        med = np.median(X, axis=0)
        d = np.linalg.norm(X - med, axis=1)
        order = np.argsort(d)
        support = X[order[: max(self.n_prototypes * self.support_per_class, 10)]]
        # Simple k-means style prototypes over the support set.
        k = min(self.n_prototypes, len(support))
        protos = support[rng.choice(len(support), k, replace=False)]
        for _ in range(10):
            assign = np.argmin(np.linalg.norm(support[:, None] - protos[None], axis=2), axis=1)
            for j in range(k):
                pts = support[assign == j]
                if len(pts):
                    protos[j] = pts.mean(axis=0)
        self.prototypes = protos
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self.prototypes[None, :, :], axis=2)
        return d.min(axis=1)


# ------------------------------------------------------------------ ensemble


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0, 1]; robust to scale differences."""
    order = scores.argsort().argsort().astype(np.float64)
    denom = max(len(scores) - 1, 1)
    return order / denom


DEFAULT_WEIGHTS = {
    "knn": 1.0,
    "lof": 1.0,
    "isolation_forest": 1.2,
    "ocsvm": 0.8,
    "dbscan": 0.8,
    "autoencoder": 1.0,
    "vae": 0.9,
    "gan": 0.7,
    "few_shot": 0.9,
}


class AnomalyEnsemble:
    """Runs enabled detectors on standardized features and combines them."""

    def __init__(self, config: Optional[Dict] = None, seed: int = 7):
        self.config = config or {}
        self.seed = seed
        self.detectors = self._build()
        self.scaler = StandardScaler()
        # detector name -> error string for detectors that failed in the last
        # run() and were excluded from the ensemble (never silently zeroed).
        self.last_failures: Dict[str, str] = {}

    def _cfg(self, *path, default=None):
        node = self.config
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def _build(self) -> List[Detector]:
        dets: List[Detector] = []
        if self._cfg("detectors", "knn", "enabled", default=True):
            dets.append(KNNDetector(k=int(self._cfg("detectors", "knn", "k", default=10))))
        if self._cfg("detectors", "lof", "enabled", default=True):
            dets.append(LOFDetector(n_neighbors=int(self._cfg("detectors", "lof", "n_neighbors", default=20))))
        if self._cfg("detectors", "isolation_forest", "enabled", default=True):
            dets.append(IsolationForestDetector(
                n_estimators=int(self._cfg("detectors", "isolation_forest", "n_estimators", default=100)),
                seed=self.seed))
        if self._cfg("detectors", "ocsvm", "enabled", default=True):
            dets.append(OCSVMDetector(nu=float(self._cfg("detectors", "ocsvm", "nu", default=0.05))))
        if self._cfg("detectors", "dbscan", "enabled", default=True):
            dets.append(DBSCANDetector(
                eps=float(self._cfg("detectors", "dbscan", "eps", default=1.2)),
                min_samples=int(self._cfg("detectors", "dbscan", "min_samples", default=5))))
        if self._cfg("deep", "autoencoder", "enabled", default=True):
            dets.append(AutoencoderDetector(
                latent_dim=int(self._cfg("deep", "autoencoder", "latent_dim", default=4)),
                epochs=int(self._cfg("deep", "autoencoder", "epochs", default=60)),
                seed=self.seed))
        if self._cfg("deep", "vae", "enabled", default=True):
            dets.append(VAEDetector(latent_dim=int(self._cfg("deep", "vae", "latent_dim", default=5))))
        if self._cfg("deep", "gan", "enabled", default=False):
            dets.append(GANDetector())
        if self._cfg("advanced", "few_shot", "enabled", default=True):
            dets.append(FewShotDetector(
                support_per_class=int(self._cfg("advanced", "few_shot", "support_per_class", default=20)),
                seed=self.seed))
        return dets

    @property
    def strategy(self) -> str:
        return str(self._cfg("advanced", "ensemble_strategy", default="weighted_average"))

    @property
    def threshold(self) -> float:
        return float(self._cfg("advanced", "decision_threshold", default=0.9))

    def run(
        self,
        X: np.ndarray,
        supervision: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns (ensemble_scores, raw_scores_by_detector, normalized_by_detector).

        supervision (optional 0/1 labels) is used ONLY by the meta_classifier
        strategy, trained on a held-out half to avoid trivially memorizing.
        """
        Xs = self.scaler.fit_transform(X)
        raw: Dict[str, np.ndarray] = {}
        norm: Dict[str, np.ndarray] = {}
        self.last_failures: Dict[str, str] = {}
        for det in self.detectors:
            try:
                det.fit(Xs)
                s = det.score(Xs)
            except Exception as exc:
                # A failed detector is recorded and EXCLUDED from the ensemble.
                # Substituting zeros (the previous behavior) silently diluted
                # every other detector's signal and masked anomalies.
                self.last_failures[det.name] = f"{type(exc).__name__}: {exc}"
                continue
            raw[det.name] = s
            norm[det.name] = normalize_scores(s)

        if not norm:
            return np.zeros(len(X)), raw, norm

        M = np.stack([norm[k] for k in norm], axis=1)
        strategy = self.strategy
        if strategy == "majority_vote":
            votes = (M >= self.threshold).astype(float)
            ensemble = votes.mean(axis=1)
        elif strategy == "meta_classifier" and supervision is not None and supervision.sum() >= 4:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(M))
            half = len(M) // 2
            tr, te = idx[:half], idx[half:]
            clf = LogisticRegression(max_iter=500)
            ensemble = np.zeros(len(M))
            try:
                clf.fit(M[tr], supervision[tr])
                ensemble[te] = clf.predict_proba(M[te])[:, 1]
                clf2 = LogisticRegression(max_iter=500).fit(M[te], supervision[te])
                ensemble[tr] = clf2.predict_proba(M[tr])[:, 1]
            except Exception:
                ensemble = M.mean(axis=1)
        else:
            weights = np.array([DEFAULT_WEIGHTS.get(k, 1.0) for k in norm])
            ensemble = (M * weights).sum(axis=1) / weights.sum()

        return ensemble, raw, norm
