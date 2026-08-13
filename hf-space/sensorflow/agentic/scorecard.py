"""AgenticSafetyScorecard — leadership-facing retrospective scorecard.

Every field is a ScorecardField carrying an OBSERVED / PREDICTED /
HYPOTHETICAL / REQUIRED_EVIDENCE tag plus an evidence reference, so a reader
can always tell a measured number from a model's guess (an improvement over a
plain key/value scorecard).
"""

from __future__ import annotations

from typing import Dict, Optional

from sensorflow.agentic import store as store_mod
from sensorflow.agentic.models import (AgenticSafetyScorecard,
                                       ConcentrationAnalysis, FailureEvent,
                                       ScorecardField, StatisticalAssessment,
                                       new_id)


def _f(value, tag: str, ref: str = "") -> ScorecardField:
    return ScorecardField(value=value, tag=tag, evidence_ref=ref)


def build_scorecard(failure: FailureEvent,
                    statistical: Optional[StatisticalAssessment],
                    concentration: Optional[ConcentrationAnalysis],
                    policy_evaluation: Optional[Dict],
                    agent_results: Dict) -> AgenticSafetyScorecard:
    basis = failure.detection_basis

    if statistical is not None:
        freq = _f({"rate": statistical.candidate.rate,
                   "wilson_ci": statistical.candidate.wilson_ci,
                   "events": statistical.candidate.events,
                   "denominator": statistical.candidate.denominator,
                   "baseline_rate": statistical.baseline.rate,
                   "significant": statistical.significant},
                  "OBSERVED", "statistical_assessment")
    else:
        freq = _f("statistical assessment not yet run", "REQUIRED_EVIDENCE",
                  "FAILURE_ANALYSIS stage")

    if concentration is not None:
        conc = _f({"determination": concentration.determination,
                   "concentrated_dimensions": concentration.concentrated_dimensions,
                   "top_strata": [s.model_dump() for s in concentration.strata
                                  if s.relative_risk and s.relative_risk >= 3.0][:3]},
                  "OBSERVED", "concentration_analysis")
    else:
        conc = _f("concentration analysis not yet run", "REQUIRED_EVIDENCE",
                  "FAILURE_ANALYSIS stage")

    # "escalated" is a successful analysis that requests human review (e.g.
    # unsafe replay behavior) — only "failed" means the evidence is missing.
    safety = agent_results.get("safety_impact")
    if safety is not None and safety.status in ("ok", "escalated"):
        chain = safety.output.get("chain", [])
        if safety.output.get("behavioral_evidence") == "observed":
            downstream = _f(safety.output, "OBSERVED", "safety_impact_agent (replay)")
        else:
            downstream = _f({"assessment": "safety assessment UNCERTAIN — no "
                                            "behavioral evidence",
                             "chain": chain},
                            "REQUIRED_EVIDENCE", "safety_impact_agent")
    else:
        downstream = _f("safety impact not assessed", "REQUIRED_EVIDENCE",
                        "safety_impact_agent")

    mining = agent_results.get("scenario_mining")
    novelty_val = (mining.output.get("novelty", "unknown")
                   if mining is not None and mining.status in ("ok", "escalated")
                   else "unknown")

    pol = policy_evaluation or {}
    severity_field = _f(
        {"severity": pol.get("severity"),
         "criteria": (pol.get("severity_assignment") or {}).get("taxonomy_description")},
        "OBSERVED" if pol else "REQUIRED_EVIDENCE",
        "policy_engine.assign_severity")

    resid = None
    for row in pol.get("expected_loss_table", []):
        if row["option"] == pol.get("recommended_option"):
            resid = row
            break
    residual = (_f({"option": resid["option"],
                    "residual_failure_rate": resid["residual_failure_rate"],
                    "expected_loss": resid["expected_loss"],
                    "feasible": resid["feasible"]},
                   "PREDICTED", "policy_engine.expected_loss_table")
                if resid else _f("no policy evaluation yet", "REQUIRED_EVIDENCE",
                                 "LAUNCH_DECISION stage"))

    ec = "INSUFFICIENT_EVIDENCE"
    if statistical is not None:
        if statistical.significant and not statistical.small_sample_flags:
            ec = "CONFIRMED"
        elif statistical.significant:
            ec = "LIKELY"
        elif statistical.candidate.events > 0:
            ec = "POSSIBLE"

    card = AgenticSafetyScorecard(
        scorecard_id=new_id("card"),
        failure_id=failure.failure_id,
        title=failure.title,
        failure_summary=_f(
            {"kind": failure.kind, "gt_class": failure.gt_class,
             "predicted_class": failure.predicted_class,
             "candidate_model": failure.candidate_model,
             "baseline_model": failure.baseline_model,
             "instances_captured": len(failure.instances)},
            "OBSERVED", "failure_detection_agent"),
        frequency=freq,
        exposure=_f({"denominator": basis.denominator,
                     "population": failure.population_id,
                     "construction_zone_exposure_share":
                         next((s.exposure_share for s in
                               (concentration.strata if concentration else [])
                               if s.stratum == "construction"), None)},
                    "OBSERVED", "rate population telemetry"),
        severity=severity_field,
        confidence=_f(ec, "OBSERVED", "statistical_assessment quality rules"),
        novelty=_f(novelty_val,
                   "OBSERVED" if novelty_val != "unknown" else "REQUIRED_EVIDENCE",
                   "scenario_mining_agent (deterministic retrieval)"),
        concentration=conc,
        downstream_impact=downstream,
        mitigations=_f(
            {"validated_mitigation": bool((pol.get("input") or {}).get("mitigation_validated")),
             "description": (pol.get("input") or {}).get("mitigation_description") or "none"},
            "OBSERVED", "policy input"),
        residual_risk=residual,
        evidence_quality=ec,
        policy_outcome=pol.get("outcome"),
        recommended_option=pol.get("recommended_option"),
        policy_version=pol.get("policy_version", ""),
        notes=[
            "Field tags: OBSERVED = measured on the synthetic campaign; "
            "PREDICTED = deterministic model output (e.g. expected loss); "
            "HYPOTHETICAL = agent hypothesis; REQUIRED_EVIDENCE = missing.",
        ],
    )
    store_mod.write_json(card, "scorecards", f"{card.scorecard_id}.json")
    store_mod.audit("scorecard_generated", failure.failure_id, "scorecard",
                    detail=f"scorecard {card.scorecard_id}")
    return card


def load_scorecard(scorecard_id: str) -> Optional[Dict]:
    return store_mod.read_json("scorecards", f"{scorecard_id}.json")
