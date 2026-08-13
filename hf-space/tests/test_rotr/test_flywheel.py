"""Flywheel governance: HITL -> regression artifact with an immutable
protected role; the contamination guard refuses training promotion without
a recorded override."""

from __future__ import annotations

import pytest

from sensorflow.rotr import flywheel
from sensorflow.rotr.flywheel import ContaminationError


def _seed_queue():
    violations = [{"violation_id": "v1", "scenario_id": "s1"},
                  {"violation_id": "v2", "scenario_id": "s2"}]
    clusters = [{"cluster_id": "cl-1", "member_violation_ids": ["v1", "v2"],
                 "exemplar_violation_id": "v1"}]
    return flywheel.build_queue("run-t", violations, clusters)


class TestHITL:
    def test_queue_carries_cluster_context(self):
        reviews = _seed_queue()
        assert all(r.status == "PENDING" for r in reviews)
        assert all(r.cluster_id == "cl-1" for r in reviews)

    def test_validate_creates_protected_regression_artifact(self):
        _seed_queue()
        out = flywheel.act("run-t", "rev-v1", "VALIDATE", actor="reviewer-a",
                           notes="confirmed failure to yield")
        cand = out["candidate"]
        assert cand["dataset_role"] == "REGRESSION"
        assert cand["training_eligible"] is False
        assert cand["guard_state"] == "PROTECTED"
        suite = flywheel.get_suite()
        assert any(m["violation_id"] == "v1" for m in suite["members"])
        assert suite["members"][0]["role"] == "REGRESSION"

    def test_reject_creates_no_artifact(self):
        _seed_queue()
        out = flywheel.act("run-t", "rev-v2", "REJECT", actor="reviewer-a")
        assert out["candidate"] is None
        assert out["review"]["status"] == "REJECTED"

    def test_double_action_refused(self):
        _seed_queue()
        flywheel.act("run-t", "rev-v1", "VALIDATE", actor="a")
        with pytest.raises(ValueError):
            flywheel.act("run-t", "rev-v1", "REJECT", actor="b")


class TestContaminationGuard:
    def test_protected_member_never_training_eligible_without_override(self):
        _seed_queue()
        cand = flywheel.act("run-t", "rev-v1", "VALIDATE", actor="a")["candidate"]
        with pytest.raises(ContaminationError):
            flywheel.promote_to_training(cand["candidate_id"], actor="sneaky")
        # still protected after the refused attempt
        doc = flywheel.get_candidate(cand["candidate_id"])
        assert doc["training_eligible"] is False

    def test_recorded_override_unlocks_promotion_with_audit(self):
        _seed_queue()
        cand = flywheel.act("run-t", "rev-v1", "VALIDATE", actor="a")["candidate"]
        with pytest.raises(ValueError):
            flywheel.governance_override(cand["candidate_id"], "lead", "  ")
        flywheel.governance_override(cand["candidate_id"], "safety-lead",
                                     "rare-event class needed in training; "
                                     "duplicate coverage retained in suite")
        doc = flywheel.promote_to_training(cand["candidate_id"], actor="lead")
        assert doc["training_eligible"] is True
        assert doc["guard_state"] == "OVERRIDDEN"
        assert doc["override"]["actor"] == "safety-lead"

    def test_guard_exception_interops_with_raremine_semantics(self):
        try:
            from sensorflow.raremine.lineage import LeakageError
        except Exception:
            pytest.skip("raremine unavailable")
        assert issubclass(ContaminationError, LeakageError)
