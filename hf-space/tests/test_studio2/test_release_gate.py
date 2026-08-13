"""Release gate composition on planted inputs shaped like the real engines'
outputs. The hard governance rules live here."""

from __future__ import annotations

import pytest

from sensorflow.studio2.release_gate import GO, NO_GO, REVIEW, ReleaseGate


def make_gate(registry):
    return ReleaseGate(registry)


def test_all_pass_is_go_pending_approval(registry, passing_safety,
                                         passing_seqeval, clean_shift):
    gate = make_gate(registry)
    d = gate.evaluate(passing_safety, passing_seqeval, clean_shift,
                      agentic_outcome={"outcome": "CONTINUE_INVESTIGATION",
                                       "policy_version": "p", "failure_id": "f",
                                       "severity": "S1"},
                      closed_loop={"scenario_id": "s", "verdict": "METRIC_ONLY"},
                      hardware_matrix={"status": "PASS", "matrix_id": "m",
                                       "n_combinations": 27, "insufficient": []})
    assert d["status"] == GO
    assert d["blocking_conditions"] == []
    # GO is a recommendation, never a deployment
    assert d["human_approval_required"] is True
    assert d["deployment_authorized"] is False
    assert d["evidence_completeness"] == 1.0


def test_regression_verdict_forces_no_go(registry, passing_safety,
                                         regressed_seqeval, clean_shift):
    d = make_gate(registry).evaluate(passing_safety, regressed_seqeval, clean_shift)
    assert d["status"] == NO_GO
    assert any("sequential regression confirmed" in c
               for c in d["blocking_conditions"])
    assert "pedestrian|night" in " ".join(d["blocking_conditions"])


def test_blocked_safety_forces_no_go(registry, blocked_safety,
                                     passing_seqeval, clean_shift):
    d = make_gate(registry).evaluate(blocked_safety, passing_seqeval, clean_shift)
    assert d["status"] == NO_GO
    assert any("safety gate blocked: safety" in c for c in d["blocking_conditions"])


def test_missing_subsystem_degrades_to_review_with_named_gap(
        registry, passing_safety, passing_seqeval):
    """A missing required input can never produce a silent GO."""
    d = make_gate(registry).evaluate(passing_safety, passing_seqeval, None)
    assert d["status"] == REVIEW
    assert any("distribution shift report unavailable" in g
               for g in d["degraded_inputs"])
    assert d["evidence_completeness"] < 1.0


def test_all_missing_is_review_never_go(registry):
    d = make_gate(registry).evaluate(None, None, None)
    assert d["status"] == REVIEW
    assert len(d["degraded_inputs"]) == 3
    assert d["evidence_completeness"] == 0.0
    assert d["confidence"] <= 0.3


def test_optional_subsystem_absence_is_surfaced_not_silent(
        registry, passing_safety, passing_seqeval, clean_shift):
    """agentic / closed loop / hardware unavailable -> named unresolved
    question and no GO (graceful degradation, never a quiet pass)."""
    d = make_gate(registry).evaluate(passing_safety, passing_seqeval, clean_shift)
    assert d["status"] == REVIEW
    joined = " ".join(d["unresolved_questions"])
    assert "agentic policy outcome unavailable" in joined
    assert "closed-loop verdict unavailable" in joined
    assert "hardware gate matrix unavailable" in joined


def test_inconclusive_seqeval_maps_to_review(registry, passing_safety, clean_shift):
    inconclusive = {"run_id": "seq-x", "decision": "INSUFFICIENT_EVIDENCE",
                    "stopping_reason": "escalation_exhausted",
                    "samples_used": 9000, "affected_strata": []}
    d = make_gate(registry).evaluate(passing_safety, inconclusive, clean_shift)
    assert d["status"] == REVIEW
    assert any("INSUFFICIENT_EVIDENCE" in q for q in d["unresolved_questions"])


def test_shift_with_recall_impact_forces_review(registry, passing_safety,
                                                passing_seqeval, bad_shift):
    d = make_gate(registry).evaluate(passing_safety, passing_seqeval, bad_shift)
    assert d["status"] == REVIEW
    assert any("RCA" in q for q in d["unresolved_questions"])


def test_agentic_stop_ship_forces_no_go(registry, passing_safety,
                                        passing_seqeval, clean_shift):
    d = make_gate(registry).evaluate(
        passing_safety, passing_seqeval, clean_shift,
        agentic_outcome={"outcome": "AUTOMATIC_STOP_SHIP", "severity": "S4",
                         "policy_version": "p", "failure_id": "f"})
    assert d["status"] == NO_GO


def test_decision_records_full_evidence_tuple_and_policy_version(
        registry, passing_safety, passing_seqeval, clean_shift):
    d = make_gate(registry).evaluate(passing_safety, passing_seqeval, clean_shift,
                                     context={"candidate": "model-v9"})
    t = d["evidence_tuple"]
    assert t["release_policy_version"]
    assert t["safety_gates"]["evidence_package_id"] == "sep-eval-test"
    assert t["sequential_regression"]["seqeval_run_id"] == "seq-test"
    assert t["distribution_shift"]["shift_run_id"] == "eval-test"
    assert t["context"] == {"candidate": "model-v9"}
    # policy is content-hash versioned and registered
    pols = registry.list("policies")
    assert any(p["policy_version"] == d["policy_version"] for p in pols)
    # decision persisted
    assert registry.get("decisions", d["entity_id"])["status"] == d["status"]


# ------------------------------------------------------------------ approval


def _go_decision(registry, passing_safety, passing_seqeval, clean_shift):
    return make_gate(registry).evaluate(
        passing_safety, passing_seqeval, clean_shift,
        agentic_outcome={"outcome": "CONTINUE_INVESTIGATION",
                         "policy_version": "p", "failure_id": "f",
                         "severity": "S1"},
        closed_loop={"scenario_id": "s", "verdict": "METRIC_ONLY"},
        hardware_matrix={"status": "PASS", "matrix_id": "m",
                         "n_combinations": 27, "insufficient": []})


def test_go_is_not_deployment_until_separately_approved(
        registry, passing_safety, passing_seqeval, clean_shift):
    gate = make_gate(registry)
    d = _go_decision(registry, passing_safety, passing_seqeval, clean_shift)
    assert d["status"] == GO and d["deployment_authorized"] is False
    approved = gate.approve(d["entity_id"], "warda", "reviewed evidence package")
    assert approved["deployment_authorized"] is True
    assert approved["approval"]["approver"] == "warda"
    # the approval is a first-class recorded entity
    assert len(registry.list("approvals")) == 1


def test_only_go_decisions_can_be_approved(registry, passing_safety,
                                           regressed_seqeval, clean_shift):
    gate = make_gate(registry)
    d = gate.evaluate(passing_safety, regressed_seqeval, clean_shift)
    assert d["status"] == NO_GO
    with pytest.raises(ValueError):
        gate.approve(d["entity_id"], "warda", "trying anyway")


def test_approval_requires_named_approver_and_rationale(
        registry, passing_safety, passing_seqeval, clean_shift):
    gate = make_gate(registry)
    d = _go_decision(registry, passing_safety, passing_seqeval, clean_shift)
    with pytest.raises(ValueError):
        gate.approve(d["entity_id"], "  ", "reason")
    with pytest.raises(ValueError):
        gate.approve(d["entity_id"], "warda", "")


def test_hardware_critical_failure_blocks_despite_everything_else_passing(
        registry, passing_safety, passing_seqeval, clean_shift):
    matrix = {"status": "FAIL_CRITICAL", "matrix_id": "m", "n_combinations": 27,
              "critical_failures": [
                  {"combination_label": "san_francisco × versal-ai-edge × LiDAR-Gen2",
                   "failed_checks": ["recall 0.71 < 0.8"]}],
              "insufficient": []}
    d = make_gate(registry).evaluate(passing_safety, passing_seqeval, clean_shift,
                                     hardware_matrix=matrix)
    assert d["status"] == NO_GO
    assert any("critical hardware combination" in c
               for c in d["blocking_conditions"])
