"""Test quality gate and metrics."""

import pytest

from sensorflow.metrics.perception_3d import bev_iou, compute_map_mar, mean_orientation_error
from sensorflow.quality_gate import QualityGate
from sensorflow.adapters.alpamayo_adapter import AlpamayoAdapter, DEFAULT_ALPAMAYO_SAMPLES


def test_bev_iou_identical():
    box = [10.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.0]
    assert bev_iou(box, box) == pytest.approx(1.0, abs=0.01)


def test_bev_iou_no_overlap():
    a = [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]
    b = [20.0, 20.0, 0.0, 2.0, 1.0, 1.0, 0.0]
    assert bev_iou(a, b) == 0.0


def test_compute_map_mar():
    preds = [[10.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.0]]
    gts = [[10.1, 2.1, 0.5, 4.0, 1.8, 1.5, 0.0]]
    metrics = compute_map_mar(preds, gts)
    assert metrics["map_3d"] > 0
    assert metrics["mar_3d"] > 0


def test_orientation_error():
    pairs = [([10.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.1], [10.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.0])]
    err = mean_orientation_error(pairs)
    assert err > 0


def test_id_swap_rate():
    from sensorflow.metrics.temporal_mot import compute_id_swap_rate
    pred_tracks = [
        {"track_id": 1, "frames": [{"frame_id": "f0", "bbox_3d": [10, 2, 0.5, 4, 1.8, 1.5, 0]}]},
        {"track_id": 2, "frames": [{"frame_id": "f1", "bbox_3d": [10.5, 2, 0.5, 4, 1.8, 1.5, 0]}]},
    ]
    gt_tracks = [
        {"instance_id": "gt1", "frames": [
            {"frame_id": "f0", "bbox_3d": [10, 2, 0.5, 4, 1.8, 1.5, 0]},
            {"frame_id": "f1", "bbox_3d": [10.5, 2, 0.5, 4, 1.8, 1.5, 0]},
        ]},
    ]
    rate = compute_id_swap_rate(pred_tracks, gt_tracks)
    assert 0 <= rate <= 1


def test_quality_gate_evaluate():
    adapter = AlpamayoAdapter()
    seq = adapter.load(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"], "qg_test")
    pred_tracks = [
        {"track_id": 1, "frames": [
            {"frame_id": f.frame_id, "bbox_3d": gt.bbox_3d, "confidence": 0.9}
            for gt in f.ground_truth[:1]
        ]}
        for f in seq.frames if f.ground_truth
    ]
    gate = QualityGate()
    results = gate.evaluate(seq, pred_tracks)
    assert "metric_card" in results
    assert "process_units" in results["metric_card"]
    assert results["metric_card"]["compute_cycles"] > 0
