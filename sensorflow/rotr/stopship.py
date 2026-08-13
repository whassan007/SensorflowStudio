"""Deterministic catastrophic-case stop-ship trigger.

Explicitly NOT LLM-driven: the trigger is a versioned, hash-addressed
CONJUNCTION of independent evidence — a vulnerable road user was involved
AND the perception layer has positive missed-detection evidence AND the
counterfactual consequence is SAFETY_CRITICAL. Any two of the three do not
fire (tested). Surrogate measures alone can never fire it.

Integration: when the agentic release machinery is importable (in-progress
workstream — guarded import), each NO_GO event is FORWARDED to
agentic.policy.evaluate as an advisory PolicyInput and the policy engine's
outcome is attached; the local gate record is always written either way,
so the gate never depends on the concurrent workstream landing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr.models import Provenance, ReleaseGateResult

STOPSHIP_POLICY: Dict = {
    "policy_name": "rotr-catastrophic-conjunction",
    "policy_semver": "1.0.0",
    "conjunction": {
        "vru_involved": True,
        "perception_missed_detection_supported": True,
        "consequence_class": "SAFETY_CRITICAL",
    },
    "outcome_on_match": "NO_GO",
    "notes": "Deterministic conjunction; NOT LLM-driven. Each leg is an "
             "independent evidence stream (taxonomy, attribution, "
             "counterfactual replay). Surrogate thresholds alone never fire.",
}


def policy_version(doc: Optional[Dict] = None) -> str:
    blob = json.dumps(doc or STOPSHIP_POLICY, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _leg_vru(signature: Dict[str, str]) -> bool:
    return signature.get("vulnerability") == "VRU"


def _leg_perception_missed(attribution: Dict) -> bool:
    le = (attribution.get("layers") or {}).get("perception") or {}
    return le.get("status") == "SUPPORTED" and \
        "missed detection" in (le.get("evidence") or "")


def _leg_safety_critical(consequence: Dict) -> bool:
    return consequence.get("consequence_class") == "SAFETY_CRITICAL"


def evaluate_gate(run_id: str, items: List[Dict],
                  forward_to_agentic: bool = True) -> ReleaseGateResult:
    """items: [{violation_id, signature, attribution, consequence}]."""
    version = policy_version()
    events: List[Dict] = []
    for it in items:
        legs = {
            "vru_involved": _leg_vru(it.get("signature", {})),
            "perception_missed_detection_supported":
                _leg_perception_missed(it.get("attribution", {})),
            "consequence_safety_critical":
                _leg_safety_critical(it.get("consequence", {})),
        }
        fired = all(legs.values())
        if fired or sum(legs.values()) >= 2:
            events.append({"violation_id": it["violation_id"],
                           "legs": legs, "fired": fired,
                           "note": "conjunction fired -> NO_GO" if fired
                           else "partial match only — does NOT fire "
                                "(conjunction is strict)"})
    outcome = "NO_GO" if any(e["fired"] for e in events) else "GO"

    advisory = None
    if forward_to_agentic and outcome == "NO_GO":
        advisory = _forward_to_agentic(run_id, events)

    return ReleaseGateResult(
        gate_id=f"{run_id}-gate", run_id=run_id, policy_version=version,
        outcome=outcome, events=events, agentic_advisory=advisory,
        provenance=Provenance(
            software_version=f"{SOFTWARE_VERSION}/stopship-{version}",
            source="SYNTHETIC"))


def _forward_to_agentic(run_id: str, events: List[Dict]) -> Optional[Dict]:
    """Advisory forwarding to the platform's deterministic policy engine."""
    try:
        from sensorflow.agentic.policy import PolicyInput, evaluate

        fired = next(e for e in events if e["fired"])
        inp = PolicyInput(
            failure_id=f"rotr-{fired['violation_id']}",
            safety_critical_class=True,
            behavioral_evidence="observed_unsafe",
            functional_impact=True,
            significant=True,
            evidence_confidence="CONFIRMED",
            downstream_consequence="unsafe",
            novelty="novel",
        )
        result = evaluate(inp, actor="rotr-stopship")
        return {"engine": "sensorflow.agentic.policy",
                "outcome": result.get("outcome"),
                "severity": result.get("severity"),
                "policy_version": result.get("policy_version"),
                "recommended_option": result.get("recommended_option")}
    except Exception as e:
        return {"engine": "unavailable",
                "note": f"agentic policy engine not importable/usable ({e}); "
                        "local gate record stands alone"}
