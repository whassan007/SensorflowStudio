"""Corrected scenario ranking and stratified sampling with reweighting.

Fixes the audit's sampling findings for the legacy paths (F-020: arbitrary
rarity coefficients and confidence-from-rarity; "most interesting first"
ordering biasing P(failure) estimates):

1. Ranking is deterministic and evidence-decomposed (no unseeded randomness,
   no collapsed opaque score without its components).
2. Evaluation sets are drawn by STRATIFIED sampling with explicit quotas, and
   every drawn item carries an inverse-inclusion-probability weight so
   fleet-level rates can be estimated without oversampling bias
   (Horvitz-Thompson, same estimator megaeval.sampling already uses — that
   implementation was verified correct, F-029).
3. Novelty is real geometry over embeddings: kNN distance and centroid
   distance, with the trade-offs documented on each scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Stratum(str, Enum):
    REPRESENTATIVE = "Representative"
    NOMINAL = "Nominal"
    RARE = "Rare"
    SAFETY_CRITICAL = "SafetyCritical"
    KNOWN_FAILURES = "KnownFailures"
    NOVEL = "Novel"
    DISTRIBUTION_SHIFTED = "DistributionShifted"


#: Default sampling quotas (fraction of the evaluation budget per stratum).
#: These are POLICY parameters — explicit, versioned, and overridable — not
#: buried magic numbers. They intentionally oversample rare/safety strata;
#: the HT weights below undo that oversampling for population estimates.
DEFAULT_QUOTAS: Dict[Stratum, float] = {
    Stratum.REPRESENTATIVE: 0.30,
    Stratum.NOMINAL: 0.15,
    Stratum.RARE: 0.15,
    Stratum.SAFETY_CRITICAL: 0.15,
    Stratum.KNOWN_FAILURES: 0.10,
    Stratum.NOVEL: 0.10,
    Stratum.DISTRIBUTION_SHIFTED: 0.05,
}


@dataclass
class SampledItem:
    item_id: str
    stratum: Stratum
    inclusion_probability: float
    weight: float                # 1 / inclusion_probability (Horvitz-Thompson)


@dataclass
class StratifiedSample:
    items: List[SampledItem]
    quotas: Dict[Stratum, int]
    population_sizes: Dict[Stratum, int]
    seed: int

    def estimate_population_rate(self, indicator: Dict[str, float]) -> Optional[float]:
        """HT estimate of a population mean/rate from the stratified sample.

        indicator: item_id -> 0/1 (or real-valued) outcome measured on the
        sample. Returns sum(w_i * y_i) / sum over population size — i.e. the
        stratum-weighted mean that recovers the true population rate under
        oversampling (each stratum's sample mean weighted by its true share).
        """
        total_pop = sum(self.population_sizes.values())
        if total_pop == 0:
            return None
        est = 0.0
        for stratum, pop in self.population_sizes.items():
            in_stratum = [it for it in self.items if it.stratum == stratum
                          and it.item_id in indicator]
            if not in_stratum:
                if pop > 0:
                    return None  # a non-empty stratum with no measurements: refuse
                continue
            stratum_mean = float(np.mean([indicator[it.item_id] for it in in_stratum]))
            est += (pop / total_pop) * stratum_mean
        return est


def stratified_sample(
    population: Dict[Stratum, Sequence[str]],
    budget: int,
    quotas: Optional[Dict[Stratum, float]] = None,
    seed: int = 0,
) -> StratifiedSample:
    """Draw a stratified sample with explicit quotas and HT weights.

    - Quota fractions are converted to integer counts by largest-remainder
      allocation (fixes the naive-rounding caveat noted in the audit, F-029).
    - Sampling within a stratum is uniform without replacement, from a seeded
      generator: same seed -> same sample, always.
    - If a stratum has fewer items than its quota, the whole stratum is taken
      (inclusion probability 1) and the leftover budget is redistributed to
      the remaining strata proportionally.
    """
    quotas = quotas or DEFAULT_QUOTAS
    rng = np.random.default_rng(seed)

    # Largest-remainder integer allocation of the budget.
    fracs = {s: quotas.get(s, 0.0) for s in population}
    total_frac = sum(fracs.values()) or 1.0
    raw = {s: budget * f / total_frac for s, f in fracs.items()}
    alloc = {s: int(v) for s, v in raw.items()}
    remainders = sorted(raw, key=lambda s: raw[s] - alloc[s], reverse=True)
    for s in remainders[: budget - sum(alloc.values())]:
        alloc[s] += 1

    # Cap at stratum size; redistribute leftovers deterministically.
    leftover = 0
    for s in sorted(alloc, key=lambda s: s.value):
        n_avail = len(population[s])
        if alloc[s] > n_avail:
            leftover += alloc[s] - n_avail
            alloc[s] = n_avail
    while leftover > 0:
        expandable = [s for s in sorted(alloc, key=lambda s: s.value)
                      if alloc[s] < len(population[s])]
        if not expandable:
            break
        for s in expandable:
            if leftover == 0:
                break
            alloc[s] += 1
            leftover -= 1

    items: List[SampledItem] = []
    for s in sorted(population, key=lambda s: s.value):
        ids = sorted(population[s])  # order-independence: canonical order first
        k = alloc.get(s, 0)
        if k == 0 or not ids:
            continue
        chosen = rng.choice(len(ids), size=k, replace=False)
        p_inc = k / len(ids)
        for idx in sorted(chosen.tolist()):
            items.append(SampledItem(
                item_id=ids[idx], stratum=s,
                inclusion_probability=p_inc, weight=1.0 / p_inc,
            ))

    return StratifiedSample(
        items=items,
        quotas=alloc,
        population_sizes={s: len(v) for s, v in population.items()},
        seed=seed,
    )


# ------------------------------------------------------------------ novelty


def knn_novelty(embeddings: np.ndarray, reference: np.ndarray, k: int = 10) -> np.ndarray:
    """Novelty = mean Euclidean distance to the k nearest reference embeddings.

    Trade-offs: sensitive to LOCAL density, so it finds points in sparse
    pockets even inside the reference hull (good for multi-modal fleets);
    cost is O(n_query * n_reference) without an index — use the VectorDB
    interface (interfaces.py) with an ANN index at production scale. k too
    small -> noisy; k too large -> smooths away small novel clusters.
    """
    if len(reference) == 0:
        return np.full(len(embeddings), np.inf)
    k = min(k, len(reference))
    # Chunked exact distances; replace with ANN via VectorDB in production.
    out = np.empty(len(embeddings))
    for i in range(0, len(embeddings), 1024):
        chunk = embeddings[i:i + 1024]
        d = np.linalg.norm(chunk[:, None, :] - reference[None, :, :], axis=2)
        d.sort(axis=1)
        out[i:i + 1024] = d[:, :k].mean(axis=1)
    return out


def centroid_novelty(embeddings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Novelty = distance to the reference centroid, scaled by mean spread.

    Trade-offs: O(n) and index-free, but UNIMODAL — it misses novel points
    that sit near the global centroid of a multi-modal distribution and
    over-flags legitimate points on the far side of a wide mode. Use as a
    cheap first pass; confirm with knn_novelty before routing on it.
    """
    if len(reference) == 0:
        return np.full(len(embeddings), np.inf)
    centroid = reference.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(reference - centroid, axis=1))) or 1.0
    return np.linalg.norm(embeddings - centroid, axis=1) / spread


# ------------------------------------------------------------------ ranking


@dataclass
class RankedScenario:
    scenario_id: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


def rank_scenarios(
    scenario_ids: Sequence[str],
    component_scores: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> List[RankedScenario]:
    """Deterministic, decomposed scenario ranking for REVIEW ordering only.

    component_scores: component name -> {scenario_id -> score in [0, 1]}.
    The returned score keeps its components attached so every ranking is
    explainable. Ties break on scenario_id, so the order is fully
    deterministic regardless of dict insertion order.

    IMPORTANT: this ordering must never be used to SELECT an evaluation set —
    "most interesting first" selection biases P(failure) estimates upward.
    Selection goes through stratified_sample(); ranking only orders items
    already selected (or a human review queue).
    """
    weights = weights or {name: 1.0 for name in component_scores}
    ranked = []
    for sid in scenario_ids:
        comps = {name: float(scores.get(sid, 0.0))
                 for name, scores in component_scores.items()}
        total_w = sum(weights.get(n, 0.0) for n in comps) or 1.0
        score = sum(comps[n] * weights.get(n, 0.0) for n in comps) / total_w
        ranked.append(RankedScenario(scenario_id=sid, score=round(score, 9),
                                     components=comps))
    ranked.sort(key=lambda r: (-r.score, r.scenario_id))
    return ranked
