"""Flywheel: suite governance fields, diversity sampling, contamination
guard, and the seqeval-delegated regression hook."""

from __future__ import annotations

import pytest

from sensorflow.agentic import flywheel as flywheel_mod
from sensorflow.agentic.models import (DetectionBasis, FailureEvent,
                                       FailureInstance, new_id)

GOVERNANCE_FIELDS = ["suite_id", "version", "creation_reason",
                     "source_failures", "sampling_policy", "coverage",
                     "known_limitations", "approval_status"]


def _validated_failure(n_instances=6) -> FailureEvent:
    instances = []
    for k in range(n_instances):
        # two objects, three frames each -> diversity sampling must cap at 2
        obj = f"obj-{k % 2}"
        instances.append(FailureInstance(
            instance_id=new_id("inst"), sequence_id="seq-1",
            frame_id=f"seq-1-f{k:04d}", frame_index=k,
            object_instance_id=obj, gt_class="pedestrian",
            predicted_class="construction_cone", confidence=0.6,
            distance_m=20.0, construction_zone=True))
    return FailureEvent(
        failure_id=new_id("fail"), kind="classification_flip",
        title="test flip", gt_class="pedestrian",
        predicted_class="construction_cone",
        detection_basis=DetectionBasis(
            method="test", candidate_events=n_instances, baseline_events=0,
            denominator=1000, candidate_rate=n_instances / 1000,
            baseline_rate=0.0),
        instances=instances, validated=True)


def test_suite_refused_for_unvalidated_failure():
    failure = _validated_failure()
    failure.validated = False
    with pytest.raises(flywheel_mod.LeakageError):
        flywheel_mod.create_or_update_suite(
            failure, {"suite_name": "x", "creation_reason": "y"})


def test_suite_governance_fields_and_diversity_sampling():
    failure = _validated_failure(n_instances=6)
    suite = flywheel_mod.create_or_update_suite(
        failure, {"suite_name": f"suite-{failure.failure_id}",
                  "creation_reason": "validated test failure",
                  "taxonomy_tags": ["pedestrian->construction_cone"],
                  "sampling_policy": "diversity-aware",
                  "known_limitations": ["synthetic"]})
    doc = suite.model_dump()
    for field in GOVERNANCE_FIELDS:
        assert doc.get(field) not in (None, "", []), f"missing {field}"
    assert doc["approval_status"] == "draft"
    # 6 instances over 2 objects -> at most 2 frames per object kept
    assert len(suite.members) == 4
    per_object = {}
    for m in suite.members:
        per_object[m.object_instance_id] = per_object.get(m.object_instance_id, 0) + 1
    assert all(v <= flywheel_mod.MAX_FRAMES_PER_OBJECT
               for v in per_object.values())
    # version bump on re-ingest, no duplicate members
    suite2 = flywheel_mod.create_or_update_suite(
        failure, {"suite_name": suite.name, "creation_reason": "re-ingest"})
    assert suite2.suite_id == suite.suite_id
    assert suite2.version == suite.version + 1
    assert len(suite2.members) == len(suite.members)


def test_contamination_guard_blocks_training_promotion():
    failure = _validated_failure()
    suite = flywheel_mod.create_or_update_suite(
        failure, {"suite_name": f"guard-{failure.failure_id}",
                  "creation_reason": "guard test"})
    member = suite.members[0]
    assert member.training_eligible is False  # unconditional default

    with pytest.raises(flywheel_mod.LeakageError):
        flywheel_mod.promote_member_to_training(suite.suite_id,
                                                member.member_id)
    with pytest.raises(flywheel_mod.LeakageError):
        flywheel_mod.promote_member_to_training(
            suite.suite_id, member.member_id, actor="someone")  # no reason

    # explicit override (who + why) succeeds and is recorded
    result = flywheel_mod.promote_member_to_training(
        suite.suite_id, member.member_id, actor="data-lead",
        override_reason="approved augmentation experiment #42")
    assert result["training_eligible"] is True
    reloaded = flywheel_mod.get_suite(suite.suite_id)
    assert reloaded.governance_overrides
    assert reloaded.governance_overrides[-1]["actor"] == "data-lead"


def test_construction_zone_suite_auto_created_by_worked_example(walkthrough):
    layer5 = walkthrough["layers"]["5_learning_flywheel"]
    suite_ids = layer5["suites_created"]["value"]
    assert suite_ids
    suite = flywheel_mod.get_suite(suite_ids[0])
    assert suite is not None
    assert "construction-zone" in suite.name
    assert suite.coverage["construction_zone_share"] > 0
    assert all(m.training_eligible is False for m in suite.members)


def test_regression_hook_delegates_to_seqeval():
    report = flywheel_mod.regression_evaluate()
    names = {r["suite"] for r in report["suites"]}
    assert {"general", "historical-regression", "rare-event",
            "safety-critical"} <= names
    evaluated = [r for r in report["suites"] if r["n"] > 0]
    assert evaluated
    for r in evaluated:
        assert r["stats_delegated_to"] == \
            "sensorflow.seqeval.sequential.PairedSequentialTest"
        assert r["decision"] in ("PASS", "REGRESSION", "INSUFFICIENT_EVIDENCE")
        assert "delta_ci" in r and "delta" in r
