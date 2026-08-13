"""EvalFlywheelAgent — converts validated failures into evaluation-suite
recommendations.

The agent proposes the taxonomy, sampling policy and membership; the
deterministic flywheel service (sensorflow.agentic.flywheel) owns suite
creation, governance fields and the contamination guard. Unvalidated failures
are refused here AND in the service.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sensorflow.agentic.agents.base import BaseAgent, compact, escalate, no_escalation
from sensorflow.agentic.models import AgentEscalation, FailureEvent


class EvalFlywheelAgent(BaseAgent):
    name = "eval_flywheel"
    version = "1.0.0"
    epistemic_status = "DERIVED"

    def _analyze(self, failure_id: str, failure: FailureEvent = None,
                 concentration=None,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assert failure is not None
        if not failure.validated:
            return ({"proposal": None,
                     "reason": "failure is not human-validated; the flywheel "
                               "only learns from confirmed failures"},
                    0.2, "validation gate not met",
                    escalate(["flywheel requested on unvalidated failure"],
                             ["policy_conflict"]))

        tags = [f"{failure.gt_class}->{failure.predicted_class}",
                failure.kind]
        construction = any(i.construction_zone for i in failure.instances)
        if construction:
            tags.append("construction_zone")
        conc_dims = (concentration.concentrated_dimensions
                     if concentration is not None else [])

        proposal = {
            "suite_name": (f"construction-zone-{failure.gt_class}-suite"
                           if construction else
                           f"{failure.gt_class}-{failure.predicted_class}-suite"),
            "taxonomy_tags": tags,
            "creation_reason": (f"validated failure {failure.failure_id}: "
                                f"{failure.title}"),
            "sampling_policy": ("diversity-aware: at most 2 frames per "
                                "(sequence, object) pair; no duplicate-frame "
                                "stuffing"),
            "concentrated_dimensions": conc_dims,
            "recommend_construction_zone_suite": construction,
            "known_limitations": [
                "members come from a synthetic campaign, not fleet data",
                "temporal context limited to the captured windows",
            ],
        }
        return ({"proposal": proposal}, 0.8,
                "deterministic mapping from validated failure to suite proposal",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Describe why this evaluation suite proposal closes the loop "
                "on the failure: " + compact(output))
