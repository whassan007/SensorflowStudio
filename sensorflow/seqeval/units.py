"""Statistical-unit handling: clustering, intra-cluster correlation, design effect.

Frames/objects are NOT independent observations: objects in the same container
(scene/track) share weather, lighting, scenario, scene-level "hardness", and the
same model failure modes. Treating them as iid understates variance and makes
any test anti-conservative.

The seqeval convention, used by every module downstream:

  * The inference unit is the CLUSTER: all sampled objects of one stratum that
    live in the same container. Per-cluster mean paired differences (bounded in
    [-1, 1]) are what feed the confidence sequences and e-processes. This is
    the clustered analogue of a paired analysis and is valid regardless of the
    within-cluster correlation structure.
  * Object-level counts are still tracked for reporting; the *effective* sample
    size n_eff = n / DEFF with DEFF = 1 + (m_bar - 1) * ICC is attached to every
    evidence record so readers can see how much clustering cost us.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def cluster_units(values: np.ndarray, cluster_ids: np.ndarray,
                  weights: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse object-level values into per-cluster (weight-normalized) means.

    Returns (cluster_means, cluster_sizes). Means are convex combinations of the
    inputs, so they stay inside the input range — a requirement for the bounded
    confidence sequences downstream.
    """
    values = np.asarray(values, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(cluster_ids, kind="stable")
    cid, v, w = cluster_ids[order], values[order], weights[order]
    uniq, start = np.unique(cid, return_index=True)
    bounds = np.append(start, cid.size)
    wsum = np.add.reduceat(w, start)
    vsum = np.add.reduceat(v * w, start)
    sizes = np.diff(bounds)
    means = vsum / np.where(wsum > 0, wsum, 1.0)
    return means, sizes


def icc_anova(values: np.ndarray, cluster_ids: np.ndarray) -> float:
    """One-way ANOVA (method-of-moments) estimator of the intra-cluster
    correlation. Clipped to [0, 1]; returns 0.0 when it cannot be estimated."""
    values = np.asarray(values, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    order = np.argsort(cluster_ids, kind="stable")
    v, cid = values[order], cluster_ids[order]
    uniq, start = np.unique(cid, return_index=True)
    k = uniq.size
    n = v.size
    if k < 2 or n <= k:
        return 0.0
    sizes = np.diff(np.append(start, n)).astype(np.float64)
    sums = np.add.reduceat(v, start)
    means = sums / sizes
    grand = v.mean()
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(np.sum((v - np.repeat(means, sizes.astype(int))) ** 2))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    # average cluster size adjusted for imbalance (n0 in Donner & Klar)
    n0 = (n - float(np.sum(sizes ** 2)) / n) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    if denom <= 0:
        return 0.0
    return float(np.clip((ms_between - ms_within) / denom, 0.0, 1.0))


def design_effect(cluster_sizes: np.ndarray, icc: float) -> float:
    """Kish design effect DEFF = 1 + (m_bar - 1) * ICC, with m_bar the
    size-weighted mean cluster size (sum m^2 / sum m)."""
    sizes = np.asarray(cluster_sizes, dtype=np.float64)
    if sizes.size == 0 or sizes.sum() <= 0:
        return 1.0
    m_bar = float(np.sum(sizes ** 2) / np.sum(sizes))
    return max(1.0, 1.0 + (m_bar - 1.0) * icc)


def effective_sample_size(n: int, deff: float) -> float:
    return float(n) / max(deff, 1.0)


def cluster_summary(values: np.ndarray, cluster_ids: np.ndarray) -> Dict:
    """Full clustering diagnostics for one node/stratum (for the ledger)."""
    values = np.asarray(values, dtype=np.float64)
    _, sizes = cluster_units(values, cluster_ids)
    icc = icc_anova(values, cluster_ids)
    deff = design_effect(sizes, icc)
    return {
        "n": int(values.size),
        "n_clusters": int(sizes.size),
        "mean_cluster_size": round(float(sizes.mean()), 3) if sizes.size else 0.0,
        "icc": round(icc, 4),
        "design_effect": round(deff, 3),
        "n_effective": round(effective_sample_size(values.size, deff), 1),
    }
