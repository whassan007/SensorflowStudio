"""Statistical primitives for the RCA diagnostics.

Local implementations of Wilson CIs and cluster-aware paired deltas. When the
sequential-testing package (sensorflow.seqeval: anytime-valid confidence
sequences + paired evaluation) lands, the significance stage should prefer it;
see diagnostics.significance_engine().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

Z95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ------------------------------------------------------------- divergence


def _hist_probs(a: np.ndarray, b: np.ndarray, bins: int = 10,
                eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Common-bin probability vectors for two continuous samples."""
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / max(1, pa.sum())
    pb = pb / max(1, pb.sum())
    return pa + eps, pb + eps


def psi_continuous(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    pa, pb = _hist_probs(np.asarray(expected, float), np.asarray(actual, float), bins)
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def psi_categorical(expected: Sequence, actual: Sequence, eps: float = 1e-4) -> float:
    cats = sorted(set(expected) | set(actual))
    e = pd.Series(list(expected)).value_counts(normalize=True)
    a = pd.Series(list(actual)).value_counts(normalize=True)
    pe = np.array([e.get(c, 0.0) + eps for c in cats])
    pa = np.array([a.get(c, 0.0) + eps for c in cats])
    pe, pa = pe / pe.sum(), pa / pa.sum()
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def js_divergence_continuous(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    pa, pb = _hist_probs(np.asarray(a, float), np.asarray(b, float), bins)
    pa, pb = pa / pa.sum(), pb / pb.sum()
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def js_divergence_categorical(a: Sequence, b: Sequence, eps: float = 1e-4) -> float:
    cats = sorted(set(a) | set(b))
    sa = pd.Series(list(a)).value_counts(normalize=True)
    sb = pd.Series(list(b)).value_counts(normalize=True)
    pa = np.array([sa.get(c, 0.0) + eps for c in cats])
    pb = np.array([sb.get(c, 0.0) + eps for c in cats])
    pa, pb = pa / pa.sum(), pb / pb.sum()
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def ks_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    res = sps.ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def chi2_test(a: Sequence, b: Sequence) -> Tuple[float, float]:
    cats = sorted(set(a) | set(b))
    ca = pd.Series(list(a)).value_counts()
    cb = pd.Series(list(b)).value_counts()
    table = np.array([[ca.get(c, 0) for c in cats], [cb.get(c, 0) for c in cats]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0
    stat, p, _, _ = sps.chi2_contingency(table)
    return float(stat), float(p)


def shift_magnitude_label(psi: float) -> str:
    """Practical-effect label for a PSI value (independent of p-values)."""
    if psi < 0.02:
        return "negligible"
    if psi < 0.1:
        return "small"
    if psi < 0.25:
        return "moderate"
    return "large"


# ---------------------------------------------------- cluster-aware paired delta


def paired_delta_cluster(df: pd.DataFrame, col_a: str = "a_correct",
                         col_b: str = "b_correct",
                         cluster_col: str = "entity_id") -> Dict:
    """Paired B-A delta with a cluster-robust CI and effective sample size.

    The per-unit paired differences d_i = b_i - a_i are aggregated per cluster
    (drive segment); the SE of the overall delta uses between-cluster variance
    of cluster sums (standard cluster-robust estimator for a mean). The
    effective sample size is n / design_effect with DE = 1 + (m_bar - 1) * ICC.
    """
    d = (df[col_b].astype(float) - df[col_a].astype(float)).to_numpy()
    n = len(d)
    if n == 0:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "effective_n": 0, "icc": 0.0, "design_effect": 1.0, "se": 0.0}
    delta = float(d.mean())

    groups = df.groupby(cluster_col).indices
    cluster_sums = np.array([d[idx].sum() for idx in groups.values()])
    cluster_sizes = np.array([len(idx) for idx in groups.values()])
    g = len(cluster_sums)
    if g > 1:
        # Cluster-robust variance of the mean of d.
        resid = cluster_sums - cluster_sizes * delta
        var = float(np.sum(resid ** 2)) * g / (g - 1) / (n ** 2)
        se = math.sqrt(max(var, 1e-12))
    else:
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    icc = _icc(d, df[cluster_col].to_numpy())
    m_bar = float(cluster_sizes.mean()) if g else 1.0
    design_effect = max(1.0, 1.0 + (m_bar - 1.0) * max(0.0, icc))
    eff_n = int(n / design_effect)

    return {
        "n": int(n),
        "n_clusters": int(g),
        "delta": delta,
        "se": se,
        "ci_low": delta - Z95 * se,
        "ci_high": delta + Z95 * se,
        "icc": icc,
        "design_effect": design_effect,
        "effective_n": eff_n,
    }


def _icc(values: np.ndarray, clusters: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation estimate."""
    dfr = pd.DataFrame({"v": values, "c": clusters})
    grp = dfr.groupby("c")["v"]
    k = grp.ngroups
    n = len(dfr)
    if k < 2 or n <= k:
        return 0.0
    grand = dfr["v"].mean()
    m_i = grp.mean()
    n_i = grp.size()
    ss_between = float((n_i * (m_i - grand) ** 2).sum())
    ss_within = float(((dfr["v"] - grp.transform("mean")) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / max(1, n - k)
    n_bar = n / k
    denom = ms_between + (n_bar - 1) * ms_within
    if denom <= 0:
        return 0.0
    return float(max(0.0, (ms_between - ms_within) / denom))


def three_way_outcome(delta: float, ci_low: float, ci_high: float,
                      practical_margin: float) -> str:
    """Classify a paired delta CI against a minimum practically-important
    regression margin (expressed as a positive fraction, e.g. 0.01 = 1pp)."""
    if ci_high < -practical_margin:
        return "significant_regression"
    if ci_low > -practical_margin:
        # The CI excludes a practically-important regression entirely.
        return "no_significant_difference"
    return "insufficient_evidence"
