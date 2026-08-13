"""Baseline-vs-candidate regression analysis across strata, with statistical
and safety significance kept STRICTLY separate.

Strata: global / class / scenario-type / safety / ODD. For each stratum we
report absolute + relative delta, a cluster-robust confidence interval,
sample size, and two independent verdicts:

* statistical significance — is the delta distinguishable from noise?
  Delegated to sensorflow.seqeval (anytime-valid PairedSequentialTest on
  cluster means) — REUSE, not reimplementation. The descriptive CI uses
  seqeval's cluster machinery (units.py) as well.
* safety significance — does the delta matter for safety? A deterministic
  policy question: the stratum's safety margin (tighter for safety strata)
  and the direction of the effect. A tiny-but-certain regression can be
  statistically significant yet safety-insignificant; a large regression on
  a small safety stratum can be safety-significant (blocking further
  evaluation, demanding more data) while statistically unresolved. The two
  are NEVER combined into one number.

Launch recommendation policy (deterministic):
* lineage invalid                        -> INVALID (never launchable)
* any safety-significant regression that
  is also statistically confirmed       -> DO_NOT_LAUNCH
* safety-significant but statistically
  unresolved                            -> INSUFFICIENT_EVIDENCE
* otherwise                             -> LAUNCH
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# REUSE: all sequential statistics from seqeval.
from sensorflow.seqeval.sequential import (
    DECISION_INSUFFICIENT, DECISION_PASS, DECISION_REGRESSION,
    PairedSequentialTest,
)
from sensorflow.seqeval.units import cluster_summary, cluster_units

from sensorflow.nextgen.lineage import validate_lineage
from sensorflow.nextgen.models import LaunchRecommendation, LineageRecord

# Safety margins per stratum kind (absolute metric points).
SAFETY_MARGINS = {"safety": 0.005, "odd": 0.010, "scenario": 0.010,
                  "class": 0.015, "global": 0.010}
STATISTICAL_MARGIN = 0.005   # practical-significance margin for the e-process
ALPHA = 0.05


def analyze_stratum(name: str, kind: str,
                    baseline: np.ndarray, candidate: np.ndarray,
                    cluster_ids: np.ndarray,
                    data_label: str = "SIMULATED") -> Dict:
    """Full per-stratum record from paired per-unit outcomes."""
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    d = c - b

    test = PairedSequentialTest(delta=STATISTICAL_MARGIN, alpha=ALPHA)
    means, _sizes = cluster_units(d, cluster_ids)
    test.update_clusters(means)
    test.record_objects(b.astype(bool), c.astype(bool))
    decision = test.evaluate()
    lo, hi = test.delta_interval()

    base_rate = float(b.mean()) if b.size else None
    cand_rate = float(c.mean()) if c.size else None
    delta_abs = None if base_rate is None else cand_rate - base_rate
    delta_rel = (None if not base_rate else delta_abs / base_rate)

    margin = SAFETY_MARGINS.get(kind, 0.010)
    # Safety significance: the point estimate breaches the stratum's safety
    # margin in the harmful direction. Deliberately a POLICY statement about
    # magnitude, independent of the statistical verdict.
    safety_significant = delta_abs is not None and delta_abs < -margin
    statistically_significant = decision == DECISION_REGRESSION

    return {
        "stratum": name, "kind": kind, "data_label": data_label,
        "n": int(b.size),
        "clustering": cluster_summary(d, cluster_ids),
        "baseline_rate": _r(base_rate), "candidate_rate": _r(cand_rate),
        "delta_abs": _r(delta_abs), "delta_rel": _r(delta_rel),
        "delta_ci": [_r(lo), _r(hi)],
        "ci_method": "anytime-valid empirical-Bernstein confidence sequence "
                     "on cluster means (seqeval)",
        "statistical": {
            "decision": decision,
            "significant_regression": statistically_significant,
            "e_regression": round(test.e_reg.e_value, 4),
            "e_pass": round(test.e_pass.e_value, 4),
            "margin": STATISTICAL_MARGIN, "alpha": ALPHA,
        },
        "safety": {
            "margin": margin,
            "significant": bool(safety_significant),
            "note": "safety significance is a policy magnitude judgment; it "
                    "is never derived from, nor equated with, the "
                    "statistical verdict",
        },
    }


def launch_recommendation(run_id: str, strata: List[Dict],
                          lineage: Optional[LineageRecord],
                          data_labels: List[str]) -> LaunchRecommendation:
    lineage_ok, lineage_reasons = validate_lineage(lineage)

    blockers: List[str] = []
    stat_summary: Dict = {"regressions": [], "passes": [], "unresolved": []}
    safety_summary: Dict = {"significant_regressions": [], "margins": SAFETY_MARGINS}

    for s in strata:
        dec = s["statistical"]["decision"]
        if dec == DECISION_REGRESSION:
            stat_summary["regressions"].append(s["stratum"])
        elif dec == DECISION_PASS:
            stat_summary["passes"].append(s["stratum"])
        else:
            stat_summary["unresolved"].append(s["stratum"])
        if s["safety"]["significant"]:
            safety_summary["significant_regressions"].append(
                {"stratum": s["stratum"], "delta_abs": s["delta_abs"],
                 "statistical_decision": dec})

    if not lineage_ok:
        recommendation = "INVALID"
        blockers.extend(lineage_reasons)
        blockers.append("run is INVALID for launch purposes: incomplete lineage")
    else:
        confirmed = [r for r in safety_summary["significant_regressions"]
                     if r["statistical_decision"] == DECISION_REGRESSION]
        unresolved = [r for r in safety_summary["significant_regressions"]
                      if r["statistical_decision"] == DECISION_INSUFFICIENT]
        if confirmed:
            recommendation = "DO_NOT_LAUNCH"
            for r in confirmed:
                blockers.append(
                    f"safety-significant regression in {r['stratum']} "
                    f"(delta {r['delta_abs']:+.4f}), statistically confirmed")
        elif unresolved:
            recommendation = "INSUFFICIENT_EVIDENCE"
            for r in unresolved:
                blockers.append(
                    f"safety-significant point estimate in {r['stratum']} "
                    f"(delta {r['delta_abs']:+.4f}) not yet statistically "
                    f"resolved — needs more evidence, not a launch")
        else:
            recommendation = "LAUNCH"

    return LaunchRecommendation(
        run_id=run_id, recommendation=recommendation, blockers=blockers,
        statistical_significance=stat_summary,
        safety_significance=safety_summary,
        lineage_valid=lineage_ok,
        data_labels=list(dict.fromkeys(data_labels)))  # type: ignore[arg-type]


def _r(v: Optional[float], nd: int = 5) -> Optional[float]:
    return None if v is None else round(float(v), nd)
