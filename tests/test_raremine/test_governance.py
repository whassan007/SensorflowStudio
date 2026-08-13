"""Lifecycle + leakage guard: silent eval->training promotion is impossible."""

from __future__ import annotations

import pytest

from sensorflow.raremine import lineage as lineage_mod
from sensorflow.raremine import pipeline, scenes
from sensorflow.raremine.lineage import LeakageError
from sensorflow.raremine.models import reset_store


@pytest.fixture()
def env(tmp_path):
    store = reset_store(tmp_path)
    bank = scenes.generate_scene_bank(store, n_scenes=60, seed=7)
    run = pipeline.run_full_pipeline(store, bank.bank_id)
    yield {"store": store, "bank": bank, "run": run}
    reset_store()


def _detected(store, bank):
    return [t for t in pipeline.review_queue(store, bank.bank_id)
            if t.representative.edge_case_detected]


def test_stage_lifecycle(env):
    store, bank = env["store"], env["bank"]
    tc = _detected(store, bank)[0]
    assert tc.stage == "AUTO_VALIDATED"  # mined -> dedup -> auto-validated
    assert tc.auto_validation is not None
    pipeline.human_review(store, tc.track_candidate_id, "approve",
                          note="confirmed costumed pedestrian", reviewer="qa-1")
    tc = store.get("track_candidates", tc.track_candidate_id)
    assert tc.stage == "CURATED"
    rec = lineage_mod.get_lineage(store, tc.track_candidate_id)
    assert rec.validation_status == "APPROVED"
    assert rec.curator == "qa-1"

    other = _detected(store, bank)[0]
    pipeline.human_review(store, other.track_candidate_id, "reject",
                          note="statue on second look", reviewer="qa-1")
    other = store.get("track_candidates", other.track_candidate_id)
    assert other.stage == "ARCHIVED"
    rec = lineage_mod.get_lineage(store, other.track_candidate_id)
    assert rec.training_eligible is False and rec.evaluation_eligible is False


def test_lineage_fields_complete(env):
    store, bank = env["store"], env["bank"]
    tc = _detected(store, bank)[0]
    rec = lineage_mod.get_lineage(store, tc.track_candidate_id)
    assert rec.source_frame_id == tc.representative.scene_id
    assert rec.source_sequence_id == tc.sequence_id
    assert rec.dataset_version
    assert rec.curation_timestamp
    assert rec.curator
    assert rec.validation_status in ("AUTO_COHERENT", "AUTO_INCOHERENT", "NO_GT")


def test_unvalidated_candidate_cannot_go_to_training(env):
    store, bank = env["store"], env["bank"]
    tc = _detected(store, bank)[0]
    with pytest.raises(LeakageError, match="unverified"):
        lineage_mod.promote_to_training(store, tc.track_candidate_id)


def test_protected_eval_example_blocks_silent_training_promotion(env):
    store, bank = env["store"], env["bank"]
    tc = _detected(store, bank)[0]
    pipeline.human_review(store, tc.track_candidate_id, "approve",
                          destination="SAFETY_CRITICAL_EVALUATION_SET", reviewer="qa-1")
    rec = lineage_mod.get_lineage(store, tc.track_candidate_id)
    assert rec.protected_evaluation is True
    assert rec.training_eligible is False, "protected eval examples are never training-eligible"
    assert rec.evaluation_eligible is True

    # silent promotion is impossible
    with pytest.raises(LeakageError, match="protected evaluation set"):
        lineage_mod.promote_to_training(store, tc.track_candidate_id)
    rec = lineage_mod.get_lineage(store, tc.track_candidate_id)
    assert rec.training_eligible is False


def test_governance_override_requires_actor_and_reason(env):
    store, bank = env["store"], env["bank"]
    tc = _detected(store, bank)[0]
    pipeline.human_review(store, tc.track_candidate_id, "approve",
                          destination="REGRESSION_EVALUATION_SET", reviewer="qa-1")
    with pytest.raises(ValueError, match="actor and a reason"):
        lineage_mod.governance_override(store, tc.track_candidate_id, actor="", reason="")


def test_governance_override_unlocks_training_and_is_audited(env):
    store, bank = env["store"], env["bank"]
    tc = _detected(store, bank)[0]
    pipeline.human_review(store, tc.track_candidate_id, "approve",
                          destination="SAFETY_CRITICAL_EVALUATION_SET", reviewer="qa-1")
    lineage_mod.governance_override(
        store, tc.track_candidate_id, actor="safety-lead",
        reason="eval set rotated out in v3; example released for training")
    rec = lineage_mod.promote_to_training(store, tc.track_candidate_id, curator="safety-lead")
    assert rec.training_eligible is True
    assert rec.governance_overrides and rec.governance_overrides[0]["actor"] == "safety-lead"
    audits = [a for a in store.all("audit_events") if a.action == "governance_override"]
    assert audits and audits[-1].actor == "safety-lead"
