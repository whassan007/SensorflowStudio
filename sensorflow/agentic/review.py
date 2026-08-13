"""Human review + governance.

Mandatory-review triggers (per spec): S3+ severity, disputed ground truth,
modality disagreement, statistically insufficient evidence, ODD reduction,
behavior-affecting changes, stop-ship outcomes, policy conflicts.

HumanReviewDecision records are append-only (JSONL per failure) and every
decision is audited. A failure only becomes `validated` through an explicit
`confirm_failure` decision — agents cannot validate their own findings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.agentic import store as store_mod
from sensorflow.agentic.models import (FailureEvent, HumanReviewDecision,
                                       new_id)

MANDATORY_TRIGGER_RULES = [
    ("severity_s3_plus", "severity S3 or higher"),
    ("disputed_ground_truth", "ground truth disputed or unavailable"),
    ("modality_disagreement", "sensor modality disagreement"),
    ("insufficient_statistics", "statistically insufficient evidence"),
    ("odd_reduction_proposed", "ODD reduction (Option C) proposed"),
    ("behavior_change", "change affects vehicle behavior"),
    ("stop_ship", "stop-ship outcome"),
    ("policy_conflict", "conflicting policy/agent verdicts"),
]


def mandatory_review_triggers(policy_evaluation: Optional[Dict],
                              fusion_verdict: Optional[str],
                              gt_available: bool,
                              small_sample: bool,
                              behavioral_evidence: str) -> List[Dict]:
    """Deterministic evaluation of every mandatory-review trigger."""
    pol = policy_evaluation or {}
    severity = pol.get("severity") or "S0"
    outcome = pol.get("outcome")
    option = pol.get("recommended_option")
    fired = {
        "severity_s3_plus": severity in ("S3", "S4", "S5"),
        "disputed_ground_truth": not gt_available,
        "modality_disagreement": fusion_verdict == "modality_conflict",
        "insufficient_statistics": small_sample,
        "odd_reduction_proposed": option == "OPTION_C_REDUCED_ODD",
        "behavior_change": behavioral_evidence in ("observed_unsafe",
                                                   "observed_contained"),
        "stop_ship": outcome == "AUTOMATIC_STOP_SHIP" or option == "STOP_SHIP",
        "policy_conflict": outcome == "INDETERMINATE",
    }
    return [{"trigger": key, "description": desc, "fired": fired[key]}
            for key, desc in MANDATORY_TRIGGER_RULES]


def record_decision(failure: FailureEvent, reviewer: str, decision: str,
                    rationale: str, evidence_reviewed: List[str],
                    policy_version: str,
                    approved_option: Optional[str] = None,
                    override_reason: Optional[str] = None) -> HumanReviewDecision:
    rec = HumanReviewDecision(
        review_id=new_id("rev"),
        failure_id=failure.failure_id,
        reviewer=reviewer,
        decision=decision,
        approved_option=approved_option,
        evidence_reviewed=evidence_reviewed,
        policy_version=policy_version,
        rationale=rationale,
        override_reason=override_reason,
    )
    store_mod.append_jsonl(rec.model_dump(), "reviews", f"{failure.failure_id}.jsonl")
    store_mod.audit("human_review_decision", failure.failure_id, reviewer,
                    detail=f"{decision}"
                           + (f" (option {approved_option})" if approved_option else ""),
                    payload={"review_id": rec.review_id,
                             "policy_version": policy_version,
                             "override_reason": override_reason})
    return rec


def decisions_for(failure_id: str) -> List[Dict]:
    return store_mod.read_jsonl("reviews", f"{failure_id}.jsonl")
