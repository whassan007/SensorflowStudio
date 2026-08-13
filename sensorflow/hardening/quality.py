"""Explicit quality evidence model, explainable routing, grader dependence.

Addresses audit F-015 (opaque HITL routing), F-012 (grader consensus treated
as independent evidence) and the platform-wide requirement that routing
decisions be evidence-decomposed and explainable.

Design rule enforced throughout: the DECOMPOSED evidence vector is the
authoritative object. The aggregate scalar exists only for sorting; every
decision cites the specific evidence dimensions that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


class QualityEvidence(BaseModel):
    """Decomposed evidence about one label/proposal. All dimensions in [0, 1]
    where 1 is "good/confident" — except uncertainty, occlusion and
    safety_relevance where 1 means "high" (direction noted per field).
    None means "not measured" and is treated conservatively (never as 1.0)."""

    geometric_validity: Optional[float] = None    # 1 = geometrically sound
    sensor_agreement: Optional[float] = None      # 1 = modalities agree
    temporal_consistency: Optional[float] = None  # 1 = stable across frames
    semantic_agreement: Optional[float] = None    # 1 = graders/models agree on class
    confidence: Optional[float] = None            # model confidence
    uncertainty: Optional[float] = None           # 1 = high epistemic uncertainty
    occlusion: Optional[float] = None             # 1 = heavily occluded
    safety_relevance: Optional[float] = None      # 1 = VRU/near-path/high severity

    def missing_dimensions(self) -> List[str]:
        return [k for k, v in self.model_dump().items() if v is None]


class RoutingDecision(str, Enum):
    ACCEPT = "ACCEPT"
    HITL = "HITL"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class RoutingPolicy:
    """Explicit, versioned routing thresholds (policy parameters, not magic).

    Semantics:
    - QUARANTINE: evidence of an actively dangerous or unusable label
      (bad geometry, contradictory sensors) — block it from any downstream
      use until resolved.
    - HITL: not provably bad, but not trustworthy enough to auto-accept —
      low agreement, high uncertainty, safety-relevant, or unmeasured
      dimensions on a safety-relevant item.
    - ACCEPT: every measured dimension is healthy AND nothing
      safety-relevant is unmeasured. Incomplete telemetry can never be
      silently accepted.
    """

    version: str = "routing-policy-v1"
    geometric_quarantine_below: float = 0.3
    sensor_quarantine_below: float = 0.3
    accept_min_good: float = 0.7          # every measured "good" dim must exceed
    accept_max_uncertainty: float = 0.3
    safety_hitl_at_or_above: float = 0.5  # safety-relevant items always get eyes
    max_missing_for_accept: int = 0       # ACCEPT requires complete evidence


GOOD_DIMS = ("geometric_validity", "sensor_agreement", "temporal_consistency",
             "semantic_agreement", "confidence")
BAD_DIMS = ("uncertainty", "occlusion")


def route(evidence: QualityEvidence,
          policy: RoutingPolicy = RoutingPolicy()) -> Dict:
    """Deterministic, explainable routing. Returns decision + per-decision
    explanation listing the exact evidence dimensions responsible."""
    ev = evidence.model_dump()
    reasons: List[str] = []

    # 1) Quarantine conditions: provably bad.
    if ev["geometric_validity"] is not None and \
            ev["geometric_validity"] < policy.geometric_quarantine_below:
        reasons.append(
            f"geometric_validity={ev['geometric_validity']:.2f} < "
            f"{policy.geometric_quarantine_below} (physically implausible box)")
    if ev["sensor_agreement"] is not None and \
            ev["sensor_agreement"] < policy.sensor_quarantine_below:
        reasons.append(
            f"sensor_agreement={ev['sensor_agreement']:.2f} < "
            f"{policy.sensor_quarantine_below} (modalities contradict — possible "
            f"phantom object or sensor fault)")
    if reasons:
        # Safety-relevant quarantines also demand human review of the quarantine.
        if (ev["safety_relevance"] or 0.0) >= policy.safety_hitl_at_or_above:
            reasons.append("safety_relevance high: quarantine escalated to human review")
        return {"decision": RoutingDecision.QUARANTINE, "explanation": reasons,
                "policy_version": policy.version, "evidence": ev}

    # 2) HITL conditions: not trustworthy enough to auto-accept.
    missing = evidence.missing_dimensions()
    if (ev["safety_relevance"] or 0.0) >= policy.safety_hitl_at_or_above:
        reasons.append(f"safety_relevance={ev['safety_relevance']:.2f} >= "
                       f"{policy.safety_hitl_at_or_above} (safety-relevant items always reviewed)")
    if ev["uncertainty"] is not None and ev["uncertainty"] > policy.accept_max_uncertainty:
        reasons.append(f"uncertainty={ev['uncertainty']:.2f} > {policy.accept_max_uncertainty}")
    for dim in GOOD_DIMS:
        if ev[dim] is not None and ev[dim] < policy.accept_min_good:
            reasons.append(f"{dim}={ev[dim]:.2f} < {policy.accept_min_good}")
    if len(missing) > policy.max_missing_for_accept:
        reasons.append(f"incomplete evidence ({', '.join(missing)} unmeasured) — "
                       f"never silently accepted")
    if reasons:
        return {"decision": RoutingDecision.HITL, "explanation": reasons,
                "policy_version": policy.version, "evidence": ev}

    # 3) Accept: complete evidence, everything healthy.
    return {"decision": RoutingDecision.ACCEPT,
            "explanation": ["all measured dimensions healthy; evidence complete"],
            "policy_version": policy.version, "evidence": ev}


def aggregate_score(evidence: QualityEvidence) -> float:
    """Sorting-only scalar (never a decision input): mean of good dims minus
    mean of bad dims, missing treated as 0 contribution and penalized."""
    ev = evidence.model_dump()
    goods = [ev[d] for d in GOOD_DIMS if ev[d] is not None]
    bads = [ev[d] for d in BAD_DIMS if ev[d] is not None]
    missing_penalty = 0.05 * len(evidence.missing_dimensions())
    g = sum(goods) / len(goods) if goods else 0.0
    b = sum(bads) / len(bads) if bads else 0.0
    return max(0.0, min(1.0, g - 0.5 * b - missing_penalty))


# ------------------------------------------------------------------ grader dependence
#
# Layered on evaluation/graders.py WITHOUT modifying it (audit F-012). The
# existing weighted-majority consensus treats each grader as independent
# evidence; when graders share a backbone, training data, or architecture,
# their errors correlate and k agreeing graders are worth fewer than k
# independent opinions.


class GraderProfile(BaseModel):
    grader_id: str
    backbone: str = ""          # e.g. "vit-l14"
    training_data: str = ""     # e.g. "laion-2b"
    architecture: str = ""      # e.g. "two-stage-detector"


#: How much shared infrastructure correlates two graders' ERRORS. These are
#: priors to be replaced by measured error correlations once per-grader audit
#: data exists (the honest path: estimate rho from disagreement-with-GT logs).
SHARED_COMPONENT_RHO = {
    "backbone": 0.5,
    "training_data": 0.3,
    "architecture": 0.2,
}


def dependence_matrix(profiles: Sequence[GraderProfile]) -> List[List[float]]:
    """Pairwise error-correlation prior from shared components (capped at 0.9)."""
    n = len(profiles)
    rho = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = 0.0
            if profiles[i].backbone and profiles[i].backbone == profiles[j].backbone:
                r += SHARED_COMPONENT_RHO["backbone"]
            if profiles[i].training_data and \
                    profiles[i].training_data == profiles[j].training_data:
                r += SHARED_COMPONENT_RHO["training_data"]
            if profiles[i].architecture and \
                    profiles[i].architecture == profiles[j].architecture:
                r += SHARED_COMPONENT_RHO["architecture"]
            rho[i][j] = rho[j][i] = min(r, 0.9)
    return rho


def effective_grader_count(profiles: Sequence[GraderProfile]) -> float:
    """Effective number of independent graders under equicorrelated votes.

    For k unit-variance votes with average pairwise correlation rho_bar, the
    variance of the mean is (1 + (k-1)*rho_bar)/k, so the equivalent number
    of independent votes is  k_eff = k / (1 + (k-1) * rho_bar).
    k_eff == k when all graders are independent; k_eff -> 1 as rho_bar -> 1.
    """
    k = len(profiles)
    if k <= 1:
        return float(k)
    rho = dependence_matrix(profiles)
    off_diag = [rho[i][j] for i in range(k) for j in range(k) if i != j]
    rho_bar = sum(off_diag) / len(off_diag)
    return k / (1.0 + (k - 1) * rho_bar)


def consensus_with_dependence(
    votes: Dict[str, str],
    profiles: Sequence[GraderProfile],
) -> Dict:
    """Compute naive consensus AND independence-adjusted confidence, separately.

    - naive_consensus: fraction of graders voting for the majority class
      (what evaluation/graders.py effectively reports).
    - adjusted_confidence: the same agreement level, but evaluated as if it
      came from k_eff independent graders instead of k. Using a normal
      approximation to the binomial vote margin, the z-score of the majority
      margin scales with sqrt(k_eff) rather than sqrt(k); we report
      Phi(margin * sqrt(k_eff)) which is strictly below the naive
      Phi(margin * sqrt(k)) whenever graders are correlated.
    """
    if not votes:
        return {"majority_class": None, "naive_consensus": 0.0,
                "adjusted_confidence": 0.0, "effective_graders": 0.0}
    by_grader = {p.grader_id: p for p in profiles}
    counts: Dict[str, int] = {}
    for cls in votes.values():
        counts[cls] = counts.get(cls, 0) + 1
    majority = max(sorted(counts), key=lambda c: counts[c])
    k = len(votes)
    naive = counts[majority] / k
    k_eff = effective_grader_count(
        [by_grader.get(g, GraderProfile(grader_id=g)) for g in votes])

    margin = 2 * naive - 1.0  # majority margin in [-1, 1]
    z_adj = margin * math.sqrt(max(k_eff, 1e-9))
    z_naive = margin * math.sqrt(k)
    phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return {
        "majority_class": majority,
        "naive_consensus": round(naive, 6),
        "naive_confidence": round(phi(z_naive), 6),
        "adjusted_confidence": round(phi(z_adj), 6),
        "effective_graders": round(k_eff, 4),
        "total_graders": k,
        "dependence_note": (
            "adjusted_confidence < naive_confidence indicates correlated "
            "graders; priors from shared backbone/data/architecture, to be "
            "replaced by measured error correlations."),
    }
