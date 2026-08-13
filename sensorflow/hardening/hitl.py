"""Information-gain HITL prioritization (audit F-015).

The legacy review queue (evaluation/pipeline.py review batches,
mitl_copilot's flat FIFO with `pred_tracks[:3]` as "evidence") spends human
attention in arrival order. This module prioritizes by expected information
value, two ways, with a documented comparison:

1. `information_gain_score`: multiplicative Risk x Uncertainty x Novelty x
   TrainingValue. One scalar, easy to sort and budget. Weakness: the product
   collapses the axes — a maximally risky but certain item scores the same
   as a moderately risky, moderately uncertain one, and a zero on any axis
   zeroes the whole item even when another axis is extreme.
2. `pareto_rank`: non-dominated sorting over the same four axes. Keeps items
   that are extreme on ANY axis on the first fronts (nothing with the highest
   risk can be starved by a low novelty score). Weakness: partial order —
   within a front you still need a tiebreaker (we use the product score),
   and front sizes vary so budgeting is less direct.

Recommendation encoded in `prioritize`: rank by Pareto front first (so no
single-axis-extreme item is starved), break ties inside a front with the
multiplicative score (so budget-limited review still favors balanced value).

Acceptance metrics for the ROUTING SYSTEM itself (not the labels) are in
`acceptance_metrics`: HITL precision, recall, workload, critical-miss-rate
and false-routing rate. "99.99% of routed items contain anomalies" is NOT a
valid acceptance criterion — a router that routes almost nothing achieves it
while missing almost everything; the pair that matters is precision AND
critical-miss-rate at a given workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel


class ReviewCandidate(BaseModel):
    item_id: str
    risk: float            # [0,1] harm if this label is wrong (safety relevance x exposure)
    uncertainty: float     # [0,1] epistemic uncertainty / disagreement
    novelty: float         # [0,1] distance from training distribution (see sampling.py)
    training_value: float  # [0,1] expected model improvement if corrected


def information_gain_score(c: ReviewCandidate) -> float:
    """Risk x Uncertainty x Novelty x TrainingValue, floored at 0.01 per axis.

    The floor keeps a zero on one axis from erasing extreme value on another
    (partial mitigation of the product's main weakness; the Pareto ranking is
    the full mitigation).
    """
    floor = 0.01
    score = 1.0
    for v in (c.risk, c.uncertainty, c.novelty, c.training_value):
        score *= max(floor, min(1.0, v))
    return score


def _dominates(a: ReviewCandidate, b: ReviewCandidate) -> bool:
    axes = ("risk", "uncertainty", "novelty", "training_value")
    ge = all(getattr(a, x) >= getattr(b, x) for x in axes)
    gt = any(getattr(a, x) > getattr(b, x) for x in axes)
    return ge and gt


def pareto_rank(candidates: Sequence[ReviewCandidate]) -> Dict[str, int]:
    """Non-dominated sorting: front 0 = not dominated by anyone, etc.
    O(n^2) per front — fine for review queues (thousands); use NSGA-II style
    sorting if queues grow beyond ~10^5."""
    remaining = list(candidates)
    fronts: Dict[str, int] = {}
    front = 0
    while remaining:
        nondominated = [c for c in remaining
                        if not any(_dominates(o, c) for o in remaining if o is not c)]
        if not nondominated:   # all mutually dominated cycles impossible, but guard
            nondominated = remaining[:]
        for c in nondominated:
            fronts[c.item_id] = front
        remaining = [c for c in remaining if c.item_id not in fronts]
        front += 1
    return fronts


def prioritize(candidates: Sequence[ReviewCandidate],
               budget: Optional[int] = None) -> List[Dict]:
    """Pareto front first, multiplicative score as tiebreaker, id as final tie.

    Deterministic: same candidates -> same order regardless of input order.
    """
    fronts = pareto_rank(candidates)
    rows = [{
        "item_id": c.item_id,
        "pareto_front": fronts[c.item_id],
        "info_gain": round(information_gain_score(c), 9),
        "components": {"risk": c.risk, "uncertainty": c.uncertainty,
                       "novelty": c.novelty, "training_value": c.training_value},
    } for c in candidates]
    rows.sort(key=lambda r: (r["pareto_front"], -r["info_gain"], r["item_id"]))
    return rows[:budget] if budget else rows


# ------------------------------------------------------------------ acceptance metrics


def acceptance_metrics(
    routed_ids: Sequence[str],
    true_problem_ids: Sequence[str],
    critical_ids: Sequence[str],
    total_items: int,
) -> Dict[str, Optional[float]]:
    """Metrics for evaluating the HITL ROUTER itself.

    - hitl_precision: fraction of routed items that were true problems
      (wasted-review complement).
    - hitl_recall: fraction of true problems that got routed.
    - workload: fraction of all items sent to humans.
    - critical_miss_rate: fraction of CRITICAL problems NOT routed — the
      metric that must gate deployment of any router change.
    - false_routing_rate: fraction of clean items routed (reviewer burn).
    """
    routed = set(routed_ids)
    problems = set(true_problem_ids)
    critical = set(critical_ids)
    clean = total_items - len(problems)
    return {
        "hitl_precision": len(routed & problems) / len(routed) if routed else None,
        "hitl_recall": len(routed & problems) / len(problems) if problems else None,
        "workload": len(routed) / total_items if total_items else None,
        "critical_miss_rate": len(critical - routed) / len(critical) if critical else None,
        "false_routing_rate": len(routed - problems) / clean if clean > 0 else None,
    }
