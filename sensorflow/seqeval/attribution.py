"""Regression attribution: turn node states into a decision-first regression map.

On detection the question changes from "is there a regression?" to "WHERE and
HOW BIG?". Every node with data yields a row carrying the affected class /
condition / difficulty band, baseline and candidate values, absolute and
relative deltas, the anytime-valid CI, raw and effective sample sizes, the
e-values and the decision. The regression map is the REGRESSION-decided subset
sorted by magnitude — the payload a release manager (or the safety Regression
Gate) acts on.
"""

from __future__ import annotations

from typing import Dict, List

from sensorflow.seqeval.hierarchy import LEVELS, HierarchyController
from sensorflow.seqeval.sequential import DECISION_REGRESSION


def node_row(node, alpha_allocated: float) -> Dict:
    t = node.test
    clustering = node.clustering()
    base_rate = t.sum_baseline / t.n_objects if t.n_objects else None
    cand_rate = t.sum_candidate / t.n_objects if t.n_objects else None
    abs_delta = None if base_rate is None else cand_rate - base_rate
    rel_delta = (None if not base_rate else abs_delta / base_rate)
    lo, hi = t.delta_interval()
    return {
        "node": node.key,
        "level": node.level,
        "level_name": LEVELS[node.level],
        "safety_primary": node.safety_primary,
        "suspect": node.suspect,
        "metric": "recall",
        "n": t.n_objects,
        "n_clusters": clustering["n_clusters"],
        "n_effective": clustering["n_effective"],
        "icc": clustering["icc"],
        "design_effect": clustering["design_effect"],
        "baseline_value": None if base_rate is None else round(base_rate, 5),
        "candidate_value": None if cand_rate is None else round(cand_rate, 5),
        "abs_delta": None if abs_delta is None else round(abs_delta, 5),
        "rel_delta": None if rel_delta is None else round(rel_delta, 5),
        "delta_ci": [round(lo, 5), round(hi, 5)],
        "e_regression": round(t.e_reg.e_value, 4),
        "e_pass": round(t.e_pass.e_value, 4),
        "bayes_p_regression": t.bayes_p_regression(),
        "alpha_allocated": round(alpha_allocated, 6),
        "decision": t.decision,
        "decided_at_n": t.decided_at_n,
        "discordant_pairs": {"regressions_n01": t.n01, "improvements_n10": t.n10},
    }


def build_attribution(hier: HierarchyController) -> Dict:
    rows: List[Dict] = []
    for node in hier.nodes.values():
        if node.test.n_objects == 0:
            continue
        rows.append(node_row(node, hier.alpha_for(node)))
    regressions = [r for r in rows if r["decision"] == DECISION_REGRESSION]
    regressions.sort(key=lambda r: r["abs_delta"] if r["abs_delta"] is not None else 0.0)
    affected_strata = [r["node"] for r in regressions if r["level"] == 3]
    return {
        "regression_map": regressions,
        "affected_strata": affected_strata,
        "all_nodes": sorted(rows, key=lambda r: (r["level"], r["node"])),
        "note": ("regression_map lists nodes with a confirmed anytime-valid "
                 "regression, most severe first; deltas are candidate minus "
                 "baseline in absolute metric points"),
    }
