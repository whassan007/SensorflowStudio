"""Miner discipline tests: modality enforcement, separated confidences,
temporal honesty, confounder handling, observed-vs-predicted separation."""

from __future__ import annotations

import pytest

from sensorflow.raremine import miner
from sensorflow.raremine.models import EVIDENCE_MODALITIES


def _frame_candidates(store, bank):
    return store.where("candidates", bank_id=bank.bank_id)


def _scene(store, scene_id):
    return store.get("scenes", scene_id)


def _truth_object(store, cand):
    scene = _scene(store, cand.scene_id)
    return next(o for o in scene.objects if o.object_id == cand.object_id)


# ------------------------------------------------------------ modality discipline


def test_no_evidence_from_absent_modalities(store, bank):
    """No candidate may cite a modality its scene does not provide."""
    cands = _frame_candidates(store, bank)
    assert cands
    for c in cands:
        scene = _scene(store, c.scene_id)
        available = {m for m, on in scene.modalities.items() if on}
        for ev in c.visual_evidence + c.human_identity_evidence:
            assert ev.modality in EVIDENCE_MODALITIES
            assert ev.modality in available, (
                f"{c.candidate_id} cites {ev.modality} but scene provides {sorted(available)}")


def test_no_lidar_claims_without_lidar(store, bank):
    """Scenes generated without LiDAR must yield zero LiDAR-based evidence."""
    lidar_free = [s for s in store.where("scenes", bank_id=bank.bank_id)
                  if not s.modalities.get("point_cloud")]
    assert lidar_free, "bank must contain LiDAR-free scenes"
    lidar_modalities = {"point_cloud", "lidar_projection", "lidar_intensity", "fusion_view"}
    for scene in lidar_free:
        for c in store.where("candidates", scene_id=scene.scene_id):
            cited = {e.modality for e in c.visual_evidence + c.human_identity_evidence}
            assert not (cited & lidar_modalities)


def test_evidence_collector_raises_on_absent_modality(store, bank):
    scene = next(s for s in store.where("scenes", bank_id=bank.bank_id)
                 if not s.modalities.get("point_cloud"))
    collector = miner._EvidenceCollector(scene)
    with pytest.raises(ValueError, match="modality discipline"):
        collector.add("point_cloud", "phantom LiDAR cluster")


# ------------------------------------------------------------ three confidences


def test_three_confidences_present_and_not_collapsed(store, bank):
    cands = _frame_candidates(store, bank)
    for c in cands:
        for field in ("confidence_human_identity", "confidence_costume", "confidence_rare_event"):
            v = getattr(c, field)
            assert 0.0 <= v <= 1.0
    # the axes must be able to diverge: e.g. a confounder can look costumed
    # (high costume) while human identity stays low
    diverging = [c for c in cands
                 if abs(c.confidence_costume - c.confidence_human_identity) > 0.3]
    assert diverging, "confidences appear collapsed into a single judgment"
    # and rare-event confidence is not simply either axis
    differing = [c for c in cands
                 if abs(c.confidence_rare_event - c.confidence_human_identity) > 0.05
                 and abs(c.confidence_rare_event - c.confidence_costume) > 0.05]
    assert differing


# ------------------------------------------------------------ temporal honesty


def test_single_frame_temporal_not_available(store, bank):
    """Single-frame scenes must never claim motion evidence."""
    singles = [s for s in store.where("scenes", bank_id=bank.bank_id)
               if not s.modalities.get("temporal_sequence")]
    assert singles, "bank must contain single-frame scenes"
    for scene in singles:
        for c in store.where("candidates", scene_id=scene.scene_id):
            assert c.temporal_validation.available is False
            assert c.temporal_validation.status == "NOT_AVAILABLE"
            assert c.temporal_validation.evidence == []
            cited = {e.modality for e in c.visual_evidence + c.human_identity_evidence}
            assert "temporal_sequence" not in cited


# ------------------------------------------------------------ confounders


def test_confounders_rejected_or_low_confidence_with_alternatives(store, bank):
    cands = _frame_candidates(store, bank)
    confounder_cands = [c for c in cands
                        if _truth_object(store, c).truth_kind == "confounder"]
    assert confounder_cands, "bank must contain confounder objects"
    for c in confounder_cands:
        if c.edge_case_detected:
            assert c.confidence_rare_event < 0.5, (
                f"confounder {c.candidate_id} became a confident candidate")
        retained = [a for a in c.alternative_hypotheses if a.status == "RETAINED"]
        assert retained, "confounders must keep alternative hypotheses alive"


def test_static_context_includes_static_object_hypotheses(store, bank):
    """Statue/mannequin/decoration hypotheses actively included when the
    context suggests them (event areas, stationary objects)."""
    cands = _frame_candidates(store, bank)
    event_area = [c for c in cands if c.location.get("context") == "event_area"]
    assert event_area
    static_names = {"mascot_statue", "mannequin", "inflatable_decoration"}
    for c in event_area:
        names = {a.hypothesis for a in c.alternative_hypotheses}
        assert names & static_names, (
            f"event-area candidate {c.candidate_id} lacks static-object hypotheses")


def test_normal_pedestrians_do_not_become_candidates(store, bank):
    cands = _frame_candidates(store, bank)
    normals = [c for c in cands
               if _truth_object(store, c).truth_kind == "normal_pedestrian"]
    assert normals
    detected = [c for c in normals if c.edge_case_detected]
    assert len(detected) / len(normals) < 0.1


def test_insufficient_evidence_is_explicit(store, bank):
    """Insufficient-evidence results must say so and never claim detection."""
    cands = _frame_candidates(store, bank)
    for c in cands:
        if c.insufficient_visual_evidence:
            assert c.edge_case_detected is False
            assert c.event_type == "none"


# ------------------------------------------------------------ observed vs predicted


def test_observed_behavior_only_with_predictions(store, bank):
    cands = _frame_candidates(store, bank)
    for c in cands:
        scene = _scene(store, c.scene_id)
        if scene.modalities.get("baseline_predictions"):
            assert c.observed_model_behavior is not None
        else:
            assert c.observed_model_behavior is None, (
                "observed model behavior invented without model outputs")


def test_observed_failures_match_planted_failures(store, bank):
    """The observed channel reflects what the baseline actually did."""
    checked = 0
    for c in _frame_candidates(store, bank):
        if not (c.observed_model_behavior and c.observed_model_behavior.failure_observed):
            continue
        scene = _scene(store, c.scene_id)
        pred = next((p for p in scene.baseline_predictions if p.object_id == c.object_id), None)
        if pred is None:
            assert "FALSE_NEGATIVE" in c.observed_model_behavior.failure_modes
            checked += 1
        elif pred.planted_failure and pred.planted_failure.startswith("MISCLASSIFICATION"):
            assert any(m.startswith("MISCLASSIFICATION") for m
                       in c.observed_model_behavior.failure_modes)
            checked += 1
    assert checked > 0


def test_predicted_failure_is_independent_of_observations(store, bank):
    """The miner forecasts failures even where no model outputs exist."""
    no_model = [c for c in _frame_candidates(store, bank)
                if c.edge_case_detected and c.observed_model_behavior is None]
    assert no_model
    assert any(c.predicted_failure_mode for c in no_model)


# ------------------------------------------------------------ routing rules


def test_unverified_candidates_never_recommend_training(store, bank):
    for c in _frame_candidates(store, bank):
        assert c.recommended_dataset_destination != "TRAINING_CANDIDATE"


def test_detected_candidates_require_human_validation(store, bank):
    for c in _frame_candidates(store, bank):
        if c.edge_case_detected:
            assert c.requires_human_validation
            assert c.human_validation_reason


def test_per_scene_ranking(store, bank):
    """Multi-candidate scenes rank by priority then rare-event confidence."""
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    by_scene = {}
    for c in _frame_candidates(store, bank):
        by_scene.setdefault(c.scene_id, []).append(c)
    multi = [v for v in by_scene.values() if len(v) > 1]
    assert multi
    for cands in multi:
        cands.sort(key=lambda c: c.rank_in_scene)
        keys = [(order.index(c.curation_priority), c.confidence_rare_event) for c in cands]
        assert keys == sorted(keys, reverse=True)
        assert [c.rank_in_scene for c in cands] == list(range(1, len(cands) + 1))
