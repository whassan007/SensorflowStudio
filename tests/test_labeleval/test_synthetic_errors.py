"""Every injected synthetic error must be caught by the appropriate subsystem."""

import numpy as np
import pytest

from sensorflow.evaluation import reporting, synthetic
from sensorflow.evaluation.pipeline import reset_pipeline
from sensorflow.evaluation.records import reset_store


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    store = reset_store(tmp_path_factory.mktemp("labeleval"))
    pipe = reset_pipeline(store)
    ds = synthetic.generate_dataset(store, num_sequences=4, frames_per_sequence=25, seed=7)
    pipe.run(ds.dataset_id, background=False)
    anns = store.where("annotations", dataset_id=ds.dataset_id)
    return store, pipe, ds, anns


def _with_error(anns, err):
    return [a for a in anns if err in a.injected_errors]


def _reasons(store, ann):
    d = store.get("triage_decisions", ann.annotation_id)
    return set(d.failure_reasons) if d else set()


def _detection_rate(store, anns, err, expected_reasons):
    subset = _with_error(anns, err)
    assert subset, f"generator injected no {err}"
    caught = [a for a in subset
              if a.status == "FLAGGED" and (_reasons(store, a) & expected_reasons)]
    return len(caught) / len(subset), subset


# ------------------------------------------------------------------ per-error assertions


def test_all_injected_errors_are_flagged(env):
    store, _, _, anns = env
    injected = [a for a in anns if a.injected_errors]
    flagged = [a for a in injected if a.status == "FLAGGED"]
    assert len(flagged) / len(injected) >= 0.95


def test_false_positive_caught(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "FALSE_POSITIVE", {"ANOMALY", "LOW_IOU", "INSUFFICIENT_POINT_SUPPORT"})
    assert rate >= 0.9


def test_false_negative_visible_as_missed_gt(env):
    store, _, ds, anns = env
    n_injected_fn = ds.generation_params["injected"]["FALSE_NEGATIVE"]
    assert n_injected_fn > 0
    # Missed objects surface as false_negative points in the haystack.
    points = reporting.haystack(store, ds.dataset_id)
    fn_points = [p for p in points if p["category"] == "false_negative"]
    assert len(fn_points) >= n_injected_fn * 0.9
    # And recall < 1 because references exist.
    m = reporting.quality_metrics(store, ds.dataset_id)["global"]
    assert m["recall"] is not None and m["recall"] < 1.0


def test_bad_3d_box_caught_by_validation(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "BAD_3D_BOX", {"LOW_IOU", "INSUFFICIENT_POINT_SUPPORT"})
    assert rate >= 0.9


def test_wrong_orientation_caught(env):
    store, _, _, anns = env
    rate, subset = _detection_rate(store, anns, "WRONG_ORIENTATION", {"ORIENTATION_ERROR", "LOW_IOU"})
    assert rate >= 0.9
    # Orientation error must be measured, not inferred.
    v = store.get("validations", subset[0].annotation_id)
    assert v.orientation_error_deg is None or v.orientation_error_deg > 10


def test_wrong_position_caught(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "WRONG_POSITION",
                              {"POSITION_ERROR", "LOW_IOU", "INSUFFICIENT_POINT_SUPPORT"})
    assert rate >= 0.9


def test_id_switch_caught_by_tracking(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "ID_SWITCH", {"ID_SWITCH"})
    assert rate >= 0.9


def test_track_fragmentation_caught(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "TRACK_FRAGMENTATION", {"TRACK_FRAGMENTATION", "ID_SWITCH"})
    assert rate >= 0.9


def test_low_point_density_caught(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "LOW_POINT_DENSITY", {"INSUFFICIENT_POINT_SUPPORT"})
    assert rate >= 0.9


def test_sensor_disagreement_caught(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "SENSOR_DISAGREEMENT", {"SENSOR_DISAGREEMENT"})
    assert rate >= 0.9


def test_grader_disagreement_caught(env):
    store, _, _, anns = env
    rate, subset = _detection_rate(store, anns, "GRADER_DISAGREEMENT", {"GRADER_DISAGREEMENT"})
    assert rate >= 0.9
    g = store.get("grader_comparisons", subset[0].annotation_id)
    assert g.consensus < 0.9 and "classification" in g.disagreement_types


def test_low_confidence_caught(env):
    store, _, _, anns = env
    rate, _ = _detection_rate(store, anns, "LOW_CONFIDENCE", {"LOW_CONFIDENCE"})
    assert rate >= 0.9


def test_clean_labels_mostly_auto_verified(env):
    store, _, _, anns = env
    clean = [a for a in anns if not a.injected_errors]
    verified = [a for a in clean if a.status == "VERIFIED"]
    assert len(verified) / len(clean) >= 0.8


# ------------------------------------------------------------------ rare events


def test_rare_events_detected_with_evidence(env):
    store, _, ds, _ = env
    events = store.where("rare_events", dataset_id=ds.dataset_id)
    types = {e.scenario_type for e in events}
    assert "near_collision" in types
    assert "sensor_failure" in types
    assert "vru_interaction" in types
    for e in events:
        assert e.evidence_frames and e.sensor_evidence
        assert 0 <= e.rarity_score <= 1


# ------------------------------------------------------------------ anomaly evidence retention


def test_anomaly_decisions_retain_full_evidence(env):
    store, _, _, anns = env
    an = store.get("anomalies", anns[0].annotation_id)
    assert an.detector_scores and an.normalized_scores
    assert set(an.detector_scores) == set(an.normalized_scores)
    assert an.ensemble_strategy in ("majority_vote", "weighted_average", "meta_classifier")
    assert 0 < an.decision_threshold <= 1


# ------------------------------------------------------------------ triage explainability


def test_every_decision_has_policy_and_gate_lines(env):
    store, _, _, anns = env
    for a in anns[:50]:
        d = store.get("triage_decisions", a.annotation_id)
        assert d is not None
        assert d.policy_id and d.policy_values
        assert d.gate_lines, "explainable per-gate lines required"
        if d.status == "FLAGGED":
            assert d.primary_failure_reason is not None
            failing = [l for l in d.gate_lines if l.applicable and not l.passed]
            assert failing, "flagged decisions must show a failing gate"


# ------------------------------------------------------------------ model regression


def test_model_regression_detected_on_degraded_second_run(tmp_path):
    store = reset_store(tmp_path)
    pipe = reset_pipeline(store)

    ds1 = synthetic.generate_dataset(store, num_sequences=3, frames_per_sequence=20, seed=21)
    pipe.run(ds1.dataset_id, background=False)
    first = [r for r in store.all("regressions")]
    assert first and not first[-1].regression_detected  # no baseline yet

    # Second dataset labeled by a degraded model: pedestrian recall collapses.
    ds2 = synthetic.generate_dataset(store, name="degraded", num_sequences=3,
                                     frames_per_sequence=20, seed=22)
    synthetic.generate_labels(store, ds2, model_version="model-v2-degraded",
                              degrade_classes=["pedestrian", "vehicle"], seed_offset=5)
    pipe.run(ds2.dataset_id, background=False)

    regs = sorted(store.all("regressions"), key=lambda r: r.date)
    latest = regs[-1]
    assert latest.regression_detected
    assert "performance" in latest.kinds
    regressed_metrics = {d.metric for d in latest.deltas if d.regressed}
    assert "recall" in regressed_metrics or "safety_critical_recall" in regressed_metrics
    assert pipe.regression_alert
