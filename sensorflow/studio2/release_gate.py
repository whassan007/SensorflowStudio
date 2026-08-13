"""Deterministic release gate: composes the real verdicts into one decision.

ReleaseGate.evaluate(safety_metrics, regression_results, distribution_shift,
policy) -> ReleaseDecision, where the inputs are the actual outputs of the
existing engines:

    safety_metrics       sensorflow.safety.gates.evaluate_gates(...) result
                         (decision RELEASE_READY/BLOCKED + layered gates)
    regression_results   sensorflow.seqeval.evaluate_regression(...) verdict
                         (decision PASS/REGRESSION/INCONCLUSIVE + regression map)
    distribution_shift   sensorflow.megaeval.analysis.distribution_shift(...)
                         report ({"shifts": [...]})
    agentic_outcome      sensorflow.agentic.policy.evaluate(...) result (optional)
    closed_loop          sensorflow.nextgen gauntlet result or causal-replay
                         verdict (optional)
    hardware_matrix      sensorflow.studio2.hardware.evaluate_matrix(...) report
                         (optional)

Hard rules (all covered by tests):
  - GO never authorizes deployment. A ReleaseDecision with status GO still has
    deployment_authorized=False; authorization is a separate recorded human
    approval action (Registry 'approvals' + audit).
  - A missing or failed required subsystem degrades the decision to REVIEW
    with the gap named — never a silent GO.
  - The policy is content-hash versioned, and every decision records the full
    evidence tuple it was computed from.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.studio2 import store
from sensorflow.studio2.registry import Registry, get_registry, new_id

GO, REVIEW, NO_GO = "GO", "REVIEW", "NO_GO"

DEFAULT_RELEASE_POLICY: Dict = {
    "policy_name": "studio2-release-v1",
    # subsystems that MUST contribute evidence; absence -> REVIEW, named
    "required_inputs": ["safety_gates", "sequential_regression",
                        "distribution_shift"],
    # consulted when supplied; never silently skipped once configured here
    "optional_inputs": ["agentic_policy", "closed_loop", "hardware_matrix"],
    # a cohort whose eval share shifted AND whose recall trails overall by
    # more than this gap forces REVIEW (megaeval shift report semantics)
    "shift_review_recall_gap": -0.05,
    "inconclusive_outcome": REVIEW,          # never GO on absence of evidence
    "critical_hardware_failure_outcome": NO_GO,
    "confidence_penalty_per_gap": 0.25,
    "confidence_penalty_per_review_condition": 0.10,
}


class ReleaseGate:
    def __init__(self, registry: Optional[Registry] = None,
                 policy: Optional[Dict] = None):
        self.registry = registry or get_registry()
        doc = {**DEFAULT_RELEASE_POLICY, **(policy or {})}
        self.policy_entity = self.registry.register_policy(
            name=doc.get("policy_name", "studio2-release"), doc=doc,
            provenance={"source_package": "studio2"})
        self.policy = doc

    # ---------------------------------------------------------------- signals

    def _signal_safety(self, payload: Optional[Dict]) -> Dict:
        if not payload or "decision" not in payload:
            return {"present": False, "gap": "safety gate results unavailable "
                    "(sensorflow.safety evaluate_gates output missing)"}
        blocking = payload.get("blocking_gates") or []
        if payload["decision"] == "RELEASE_READY" and not blocking:
            return {"present": True, "verdict": "pass",
                    "evidence": {"safety_gate_run": payload.get("candidate_run_id"),
                                 "evidence_package_id": payload.get("evidence_package_id")}}
        return {"present": True, "verdict": "block",
                "conditions": [f"safety gate blocked: {g}" for g in blocking]
                or ["safety layer decision BLOCKED"],
                "evidence": {"safety_gate_run": payload.get("candidate_run_id"),
                             "evidence_package_id": payload.get("evidence_package_id")}}

    def _signal_seqeval(self, payload: Optional[Dict]) -> Dict:
        if not payload or "decision" not in payload:
            return {"present": False, "gap": "sequential regression verdict "
                    "unavailable (sensorflow.seqeval evaluate_regression missing)"}
        decision = payload["decision"]
        evidence = {"seqeval_run_id": payload.get("run_id"),
                    "samples_used": payload.get("samples_used")}
        if decision == "PASS":
            return {"present": True, "verdict": "pass", "evidence": evidence}
        if decision == "REGRESSION":
            strata = payload.get("affected_strata") or []
            return {"present": True, "verdict": "block",
                    "conditions": [
                        "sequential regression confirmed"
                        + (f" in strata: {', '.join(map(str, strata))}" if strata else "")],
                    "evidence": evidence}
        # INCONCLUSIVE / INSUFFICIENT: absence of evidence is never a pass
        reason = payload.get("stopping_reason") or "budget exhausted without a decision"
        return {"present": True, "verdict": "review",
                "questions": [f"sequential regression verdict {decision}: {reason}"],
                "evidence": evidence}

    def _signal_shift(self, payload: Optional[Dict]) -> Dict:
        if not payload or "shifts" not in payload:
            return {"present": False, "gap": "distribution shift report "
                    "unavailable (megaeval shift monitor missing)"}
        gap_limit = self.policy["shift_review_recall_gap"]
        flagged = [s for s in payload["shifts"]
                   if s.get("recall_gap") is not None and s["recall_gap"] <= gap_limit]
        evidence = {"shift_run_id": payload.get("run_id"),
                    "shifts_reported": len(payload["shifts"])}
        if flagged:
            return {"present": True, "verdict": "review",
                    "questions": [
                        f"distribution shift with recall impact on {s['cohort']}: "
                        f"eval share {s.get('relative_change', 0)*100:+.0f}% vs train, "
                        f"cohort recall gap {s['recall_gap']:+.3f} — open an RCA "
                        "investigation (offline-vs-shadow 4-way root cause)"
                        for s in flagged[:5]],
                    "evidence": evidence}
        return {"present": True, "verdict": "pass", "evidence": evidence}

    def _signal_agentic(self, payload: Optional[Dict]) -> Dict:
        if not payload or "outcome" not in payload:
            return {"present": False, "gap": "agentic policy outcome unavailable "
                    "(sensorflow.agentic not importable or not evaluated)"}
        outcome = payload["outcome"]
        evidence = {"agentic_policy_version": payload.get("policy_version"),
                    "failure_id": payload.get("failure_id"),
                    "outcome": outcome}
        if outcome == "AUTOMATIC_STOP_SHIP":
            return {"present": True, "verdict": "block",
                    "conditions": ["agentic policy fired AUTOMATIC_STOP_SHIP "
                                   f"(severity {payload.get('severity')})"],
                    "evidence": evidence}
        if outcome == "INDETERMINATE":
            return {"present": True, "verdict": "review",
                    "questions": ["agentic policy INDETERMINATE (fail-safe): "
                                  + "; ".join(payload.get("indeterminate_reasons") or
                                              ["integrity failure"])],
                    "evidence": evidence}
        return {"present": True, "verdict": "pass", "evidence": evidence}

    def _signal_closed_loop(self, payload: Optional[Dict]) -> Dict:
        if not payload:
            return {"present": False, "gap": "closed-loop verdict unavailable "
                    "(sensorflow.nextgen not importable or no replay/gauntlet run)"}
        # accept either a gauntlet result or a causal replay result
        evidence = {"closed_loop_run_id": payload.get("run_id")
                    or payload.get("scenario_id")}
        if payload.get("halted") or payload.get("recommendation") == "NO_GO":
            return {"present": True, "verdict": "block",
                    "conditions": ["closed-loop gauntlet halted: "
                                   + str(payload.get("halt_reason")
                                         or payload.get("recommendation"))],
                    "evidence": evidence}
        if payload.get("verdict") == "BEHAVIORALLY_CONSEQUENTIAL":
            return {"present": True, "verdict": "review",
                    "questions": ["closed-loop replay shows behaviorally "
                                  "consequential perception divergence on "
                                  f"{payload.get('scenario_id')}"],
                    "evidence": evidence}
        return {"present": True, "verdict": "pass", "evidence": evidence}

    def _signal_hardware(self, payload: Optional[Dict]) -> Dict:
        if not payload or "status" not in payload:
            return {"present": False, "gap": "hardware gate matrix unavailable"}
        evidence = {"hardware_matrix_id": payload.get("matrix_id"),
                    "combinations": payload.get("n_combinations")}
        questions = [
            f"insufficient evidence for {c['combination_label']}: {c['reason']}"
            for c in (payload.get("insufficient") or [])[:5]]
        if payload["status"] == "FAIL_CRITICAL":
            outcome = self.policy["critical_hardware_failure_outcome"]
            return {"present": True,
                    "verdict": "block" if outcome == NO_GO else "review",
                    "conditions": [
                        f"critical hardware combination failing: {c['combination_label']}"
                        f" ({', '.join(c['failed_checks'])})"
                        for c in (payload.get("critical_failures") or [])[:5]],
                    "questions": questions, "evidence": evidence}
        if payload["status"] == "FAIL":
            return {"present": True, "verdict": "review",
                    "questions": questions + [
                        f"non-critical hardware combination failing: {c['combination_label']}"
                        for c in (payload.get("failures") or [])[:5]],
                    "evidence": evidence}
        return {"present": True,
                "verdict": "review" if questions else "pass",
                "questions": questions, "evidence": evidence}

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, safety_metrics: Optional[Dict],
                 regression_results: Optional[Dict],
                 distribution_shift: Optional[Dict],
                 policy: Optional[Dict] = None,
                 agentic_outcome: Optional[Dict] = None,
                 closed_loop: Optional[Dict] = None,
                 hardware_matrix: Optional[Dict] = None,
                 context: Optional[Dict] = None,
                 actor: str = "studio2") -> Dict:
        """Pure function of inputs + versioned policy; persists and returns a
        ReleaseDecision."""
        if policy:
            gate = ReleaseGate(self.registry, policy={**self.policy, **policy})
            return gate.evaluate(safety_metrics, regression_results,
                                 distribution_shift, agentic_outcome=agentic_outcome,
                                 closed_loop=closed_loop,
                                 hardware_matrix=hardware_matrix,
                                 context=context, actor=actor)

        signals = {
            "safety_gates": self._signal_safety(safety_metrics),
            "sequential_regression": self._signal_seqeval(regression_results),
            "distribution_shift": self._signal_shift(distribution_shift),
            "agentic_policy": self._signal_agentic(agentic_outcome),
            "closed_loop": self._signal_closed_loop(closed_loop),
            "hardware_matrix": self._signal_hardware(hardware_matrix),
        }

        required = self.policy["required_inputs"]
        optional = self.policy["optional_inputs"]
        considered = list(required) + list(optional)

        blocking: List[str] = []
        questions: List[str] = []
        gaps: List[str] = []
        evidence_tuple: Dict = {"release_policy_version":
                                self.policy_entity["policy_version"]}
        for name in considered:
            sig = signals[name]
            if not sig["present"]:
                if name in required:
                    gaps.append(sig["gap"])
                else:
                    # optional-but-configured subsystems: absence is surfaced
                    # as an open question, never silently ignored
                    questions.append(f"[{name}] {sig['gap']}")
                evidence_tuple[name] = None
                continue
            evidence_tuple[name] = sig.get("evidence")
            blocking.extend(sig.get("conditions") or [])
            questions.extend(sig.get("questions") or [])
        if context:
            evidence_tuple["context"] = context

        if blocking:
            status = NO_GO
        elif gaps or questions:
            status = REVIEW
        else:
            status = GO

        n_present = sum(1 for n in considered if signals[n]["present"])
        evidence_completeness = round(n_present / len(considered), 3)
        confidence = 1.0
        confidence -= self.policy["confidence_penalty_per_gap"] * len(gaps)
        confidence -= (self.policy["confidence_penalty_per_review_condition"]
                       * len(questions))
        confidence = round(max(0.05, min(confidence, evidence_completeness
                                         if status != NO_GO else 1.0)), 3)

        decision = {
            "entity_id": new_id("rd"),
            "status": status,
            "confidence": confidence,
            "evidence_completeness": evidence_completeness,
            "blocking_conditions": blocking,
            "unresolved_questions": questions,
            "degraded_inputs": gaps,
            # GO is a *recommendation*; deployment always needs a human
            "human_approval_required": True,
            "deployment_authorized": False,
            "approval": None,
            "policy_version": self.policy_entity["policy_version"],
            "evidence_tuple": evidence_tuple,
            "signals": {k: {kk: vv for kk, vv in v.items() if kk != "evidence"}
                        for k, v in signals.items()},
            "evaluated_at": store.now_iso(),
        }
        self.registry.put("decisions", decision)
        store.audit("release_decision", "decisions", decision["entity_id"],
                    actor=actor, detail=f"status={status} "
                    f"completeness={evidence_completeness}",
                    payload={"blocking": blocking[:5], "gaps": gaps[:5]})
        return decision

    # ---------------------------------------------------------------- approval

    def approve(self, decision_id: str, approver: str, rationale: str) -> Dict:
        """The separate human action that authorizes deployment. Only a GO
        decision can be approved; REVIEW/NO_GO must be re-evaluated first."""
        if not approver.strip() or not rationale.strip():
            raise ValueError("human approval requires a named approver and a rationale")
        decision = self.registry.get("decisions", decision_id)
        if decision is None:
            raise KeyError(f"unknown release decision {decision_id}")
        if decision["status"] != GO:
            raise ValueError(
                f"decision {decision_id} has status {decision['status']}; only GO "
                "decisions can be approved for deployment — resolve the blocking "
                "conditions / unresolved questions and re-evaluate")
        approval = {
            "entity_id": new_id("app"),
            "decision_id": decision_id,
            "approver": approver,
            "rationale": rationale,
            "approved_at": store.now_iso(),
        }
        self.registry.put("approvals", approval)
        decision["deployment_authorized"] = True
        decision["approval"] = approval
        self.registry.put("decisions", decision)
        store.audit("human_approval", "approvals", approval["entity_id"],
                    actor=approver, detail=f"authorized deployment for {decision_id}",
                    payload={"rationale": rationale})
        return decision
