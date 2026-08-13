"""LaunchDecisionAgent — drafts the recommendation NARRATIVE only.

Structural guarantee of the AI/deterministic separation: this agent receives
the policy engine's already-computed evaluation and turns it into prose for
humans. It cannot change the outcome, the option, or any number — the
deterministic PolicyEngine (sensorflow.agentic.policy) decides, and human
review authorizes.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sensorflow.agentic.agents.base import BaseAgent, compact, no_escalation
from sensorflow.agentic.models import AgentEscalation, FailureEvent

OPTION_TEXT = {
    "STOP_SHIP": "stop the release",
    "OPTION_A_DELAY": "Option A — delay launch until the failure is fixed",
    "OPTION_B_MITIGATION": "Option B — ship with the validated mitigation",
    "OPTION_C_REDUCED_ODD": "Option C — ship with a reduced ODD excluding the "
                            "concentrated region",
    "EXPAND_EVALUATION": "expand evaluation before any launch claim",
    "HUMAN_SAFETY_REVIEW": "route to human safety review",
    "PROCEED": "proceed (no launch impact established)",
}


class LaunchDecisionAgent(BaseAgent):
    name = "launch_decision"
    version = "1.0.0"
    epistemic_status = "DERIVED"

    def _analyze(self, failure_id: str, failure: FailureEvent = None,
                 policy_evaluation: Dict = None,
                 **inputs) -> Tuple[Dict[str, Any], float, str, AgentEscalation]:
        assert failure is not None and policy_evaluation is not None
        pol = policy_evaluation
        outcome = pol.get("outcome")
        option = pol.get("recommended_option")
        row = pol.get("matrix_row_fired") or {}
        sev = pol.get("severity")
        inp = pol.get("input") or {}

        lines = [
            f"POLICY OUTCOME (deterministic): {outcome} under policy "
            f"{pol.get('policy_version')}.",
            f"Severity {sev}: {(pol.get('severity_assignment') or {}).get('taxonomy_description')}",
            f"Recommended option (matrix row {row.get('row')}, condition "
            f"{row.get('condition')}): {OPTION_TEXT.get(option, option)}.",
        ]
        if inp.get("rate") is not None:
            lines.append(
                f"Measured candidate rate {inp['rate']:.6f} "
                f"(CI {inp.get('rate_ci')}) over denominator "
                f"{inp.get('denominator')}; significant={inp.get('significant')}.")
        if pol.get("indeterminate_reasons"):
            lines.append("INDETERMINATE reasons: "
                         + "; ".join(pol["indeterminate_reasons"]))
        infeasible = [r for r in pol.get("expected_loss_table", [])
                      if not r["feasible"]]
        if infeasible:
            lines.append(
                "Hard safety constraint excluded from selection: "
                + ", ".join(f"{r['option']} (residual rate "
                            f"{r['residual_failure_rate']:.2e})"
                            for r in infeasible)
                + " — regardless of expected loss.")
        lines.append(
            "This narrative is ADVISORY. Final authorization requires the "
            "deterministic policy outcome above plus a recorded human review "
            "decision; no agent output authorizes a launch.")

        output = {
            "narrative": "\n".join(lines),
            "restated_outcome": outcome,
            "restated_option": option,
            "decision_authority": "policy engine + human approval (never this agent)",
        }
        return (output, 0.85,
                "narrative restates deterministic results; it introduces no "
                "new claims",
                no_escalation())

    def _llm_prompt(self, failure_id, output, **inputs):
        return ("Polish this launch-decision narrative without changing any "
                "fact, number, outcome or option: " + compact(output))
