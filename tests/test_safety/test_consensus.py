"""Additive consensus extensions: consensus score vector, Kendall's tau, and
Minimum Bayes Risk selection — exact values on small fixtures, plus wiring into
the grader comparison records."""

from __future__ import annotations

import pytest

from sensorflow.evaluation.graders import (
    consensus_score_vector,
    dataset_grader_statistics,
    grade_annotation,
    kendalls_tau,
    mbr_consensus_select,
    mbr_utility,
)


def test_kendalls_tau_exact():
    assert kendalls_tau([1, 2, 3], [1, 2, 3]) == (1.0, pytest.approx(0.333333, abs=1e-3))
    tau, _ = kendalls_tau([1, 2, 3], [3, 2, 1])
    assert tau == -1.0
    # one swapped pair of three: tau-b = (2-1)/3 = 1/3
    tau, _ = kendalls_tau([1, 2, 3], [1, 3, 2])
    assert tau == pytest.approx(1 / 3, abs=1e-4)
    # degenerate (constant) input handled
    assert kendalls_tau([1, 1], [2, 2]) == (0.0, 1.0)


def test_consensus_score_vector_exact():
    votes = {"a": "vehicle", "b": "vehicle", "c": "vehicle", "d": "vehicle",
             "e": "truck"}
    detail = {"iou": 0.5, "center_distance": 1.0, "dimension_diff": 0.2,
              "orientation_diff_deg": 45.0}
    v = consensus_score_vector(votes, detail, temporal=1.0)
    assert v["class_majority_share"] == pytest.approx(0.8)
    assert v["class_weighted_agreement"] == pytest.approx(0.8)  # uniform weights
    assert v["spatial_iou"] == pytest.approx(0.5)
    assert v["spatial_center"] == pytest.approx(0.5)       # 1 - 1.0/2.0
    assert v["spatial_dims"] == pytest.approx(0.8)         # 1 - 0.2
    assert v["spatial_orientation"] == pytest.approx(0.75)  # 1 - 45/180
    assert v["temporal"] == 1.0


BOX = [0.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0]
FAR_BOX = [50.0, 50.0, 0.8, 4.5, 1.9, 1.6, 0.0]


def test_mbr_utility_exact():
    a = {"class_name": "vehicle", "bbox_3d": BOX}
    b = {"class_name": "vehicle", "bbox_3d": list(BOX)}
    c = {"class_name": "pedestrian", "bbox_3d": FAR_BOX}
    assert mbr_utility(a, b) == pytest.approx(1.0)   # class match + IoU 1
    assert mbr_utility(a, c) == pytest.approx(0.0)   # no match, no overlap
    d = {"class_name": "vehicle", "bbox_3d": FAR_BOX}
    assert mbr_utility(a, d) == pytest.approx(0.5)   # class only


def test_mbr_selects_minimum_expected_risk():
    candidates = [
        {"source": "auto_label", "class_name": "vehicle", "bbox_3d": BOX},
        {"source": "vendor_gt", "class_name": "vehicle", "bbox_3d": list(BOX)},
        {"source": "vlm_label", "class_name": "pedestrian", "bbox_3d": FAR_BOX},
    ]
    res = mbr_consensus_select(candidates)
    # risks: agreeing pair 1 - mean(1, 0) = 0.5 each; outlier 1 - mean(0, 0) = 1
    risks = [c["expected_risk"] for c in res["candidates"]]
    assert risks == [0.5, 0.5, 1.0]
    assert res["selected_index"] == 0  # first of the tied minimum
    assert res["selected"]["source"] == "auto_label"

    with pytest.raises(ValueError):
        mbr_consensus_select([])


def test_grade_annotation_carries_extension_evidence(eval_env):
    store, ds = eval_env
    frames = {f.frame_id: f for f in store.where("frames", dataset_id=ds.dataset_id)}
    anns = [a for a in store.where("annotations", dataset_id=ds.dataset_id)
            if a.bbox_3d][:6]
    assert len(anns) >= 2
    for ann in anns:
        cmp = grade_annotation(store, ann, frames[ann.frame_id])
        vector_keys = {k for k in cmp.kappa_stats if k.startswith("csv_")}
        assert {"csv_class_majority_share", "csv_spatial_iou",
                "csv_temporal"} <= vector_keys
        assert "mbr_selected_idx" in cmp.kappa_stats
        assert "mbr_risk_auto_label" in cmp.kappa_stats
        # canonical scalar consensus still present and bounded
        assert cmp.consensus is not None and 0.0 <= cmp.consensus <= 1.0

    stats = dataset_grader_statistics(store, ds.dataset_id)
    assert "cohen_kappa" in stats
    assert "kendall_tau_confidence_vs_consensus" in stats
    assert -1.0 <= stats["kendall_tau_confidence_vs_consensus"] <= 1.0
