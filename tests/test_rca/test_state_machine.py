"""Stage state machine: enforced ordering + unknown acknowledgment."""

from __future__ import annotations

import pytest

from sensorflow.rca.models import (Investigation, StageOrderError,
                                   UnknownsNotAcknowledgedError, make_finding,
                                   STAGES)


def _inv() -> Investigation:
    return Investigation.new("t", "a", "b", "STATISTICAL_NOISE", seed=1)


def test_initial_state():
    inv = _inv()
    assert len(inv.stages) == 13
    assert inv.stages[0].status == "in_progress"
    assert all(s.status == "pending" for s in inv.stages[1:])


def test_cannot_skip_ahead():
    inv = _inv()
    with pytest.raises(StageOrderError):
        inv.complete_stage(5)
    # Even the immediately-next stage is blocked until stage 0 completes.
    with pytest.raises(StageOrderError):
        inv.complete_stage(1)


def test_ordered_completion_advances():
    inv = _inv()
    inv.complete_stage(0)
    assert inv.stages[0].status == "complete"
    assert inv.stages[1].status == "in_progress"
    inv.complete_stage(1)
    assert inv.stages[2].status == "in_progress"


def test_critical_unknowns_block_without_ack():
    inv = _inv()
    inv.upsert_auto_findings("comparison_validity", [
        make_finding("comparison_validity", "CV_UNKNOWN:x", "x unknown",
                     "UNKNOWN", "CRITICAL")])
    with pytest.raises(UnknownsNotAcknowledgedError):
        inv.complete_stage(0)
    stage = inv.complete_stage(0, acknowledge_unknowns=True,
                               ack_note="shipping anyway, tracked in JIRA-123")
    assert stage.status == "complete_with_unknowns"
    assert stage.acknowledged_unknowns
    assert stage.ack_note == "shipping anyway, tracked in JIRA-123"
    # The acknowledgment is recorded in the event log ("proceeding with
    # unknowns" is auditable).
    kinds = [e["kind"] for e in inv.events]
    assert "unknowns_acknowledged" in kinds


def test_non_critical_unknowns_do_not_block():
    inv = _inv()
    inv.upsert_auto_findings("comparison_validity", [
        make_finding("comparison_validity", "CV_X", "warn unknown",
                     "UNKNOWN", "WARN")])
    stage = inv.complete_stage(0)
    assert stage.status == "complete"


def test_reopen():
    inv = _inv()
    inv.complete_stage(0)
    inv.reopen_stage(0)
    assert inv.stages[0].status == "in_progress"
    with pytest.raises(StageOrderError):
        inv.complete_stage(1)


def test_serialization_roundtrip():
    inv = _inv()
    inv.upsert_auto_findings("comparison_validity", [
        make_finding("comparison_validity", "CV_UNKNOWN:x", "x", "UNKNOWN",
                     "CRITICAL")])
    inv.complete_stage(0, acknowledge_unknowns=True)
    d = inv.to_json_dict()
    inv2 = Investigation.from_json_dict(d)
    assert inv2.stages[0].status == "complete_with_unknowns"
    assert inv2.findings[0].code == "CV_UNKNOWN:x"
    assert inv2.scenario_cause == "STATISTICAL_NOISE"


def test_training_mode_hides_cause():
    inv = Investigation.new("t", "a", "b", "FEATURE_SKEW", seed=1,
                            training_mode=True)
    assert "scenario_cause" not in inv.to_dict()
    inv.revealed = True
    assert inv.to_dict()["scenario_cause"] == "FEATURE_SKEW"


def test_stage_keys_match_spec_order():
    keys = [s["key"] for s in STAGES]
    assert keys[0] == "comparison_validity"
    assert keys[6] == "statistical_significance"
    assert keys[11] == "root_cause_scoring"
    assert keys[12] == "recommendations_report"
