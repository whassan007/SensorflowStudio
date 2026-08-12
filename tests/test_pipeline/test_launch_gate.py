"""Test launch gate evaluator."""

from pathlib import Path

import pytest

from sensorflow.launch_gate_evaluator import LaunchGateEvaluator, DEFAULT_THRESHOLDS


def test_launch_gate_no_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = LaunchGateEvaluator()
    result = evaluator.evaluate("nonexistent")
    assert result["passed"] is False
    assert "export" in result["blocked_stages"]


def test_launch_gate_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seq_id = "pass_test"
    bench_dir = Path("runs/pipeline") / seq_id / "benchmark"
    bench_dir.mkdir(parents=True)
    metric_card = {
        "map_3d": 0.8,
        "orientation_error_deg": 2.0,
        "id_swap_rate": 0.01,
        "track_fragmentation_rate": 0.02,
        "position_error_m": 0.5,
    }
    import json
    with open(bench_dir / "metric_card.json", "w") as f:
        json.dump(metric_card, f)

    evaluator = LaunchGateEvaluator()
    result = evaluator.evaluate(seq_id)
    assert result["passed"] is True
    assert evaluator.is_export_allowed(seq_id) is True


def test_launch_gate_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seq_id = "fail_test"
    bench_dir = Path("runs/pipeline") / seq_id / "benchmark"
    bench_dir.mkdir(parents=True)
    metric_card = {
        "map_3d": 0.3,
        "orientation_error_deg": 15.0,
        "id_swap_rate": 0.1,
        "track_fragmentation_rate": 0.2,
    }
    import json
    with open(bench_dir / "metric_card.json", "w") as f:
        json.dump(metric_card, f)

    evaluator = LaunchGateEvaluator()
    result = evaluator.evaluate(seq_id)
    assert result["passed"] is False
    assert evaluator.is_export_allowed(seq_id) is False
