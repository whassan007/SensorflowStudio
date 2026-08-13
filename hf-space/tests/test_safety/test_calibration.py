"""Multi-sensor calibration validation: clean scenes pass, an injected 2-degree
extrinsic rotation must be flagged as MISCALIBRATED (systematic bias), tampered
2D boxes must be flagged as PERCEPTION_FAILURE (random disagreement) mapped to
the platform's SENSOR_DISAGREEMENT failure reason."""

from __future__ import annotations

import pytest

from sensorflow.safety import calibration as cal


def test_clean_scene_passes(fresh_safety_root):
    res = cal.run_validation(mode="clean", seed=3)
    assert res["status"] == "CALIBRATED"
    assert res["passed"] is True
    assert all(c["passed"] for c in res["checks"])
    assert not any(o["flagged"] for o in res["per_object"])
    assert res["metrics"]["bias_px"] <= res["thresholds"]["bias_px"]
    assert res["simulated"] is True


def test_two_degree_rotation_flagged_as_miscalibration(fresh_safety_root):
    res = cal.run_validation(mode="miscalibrated", rotation_offset_deg=2.0, seed=3)
    assert res["status"] == "MISCALIBRATED"
    assert res["passed"] is False
    # systematic: every object shares the bias
    assert all(o["flagged"] for o in res["per_object"])
    assert all(o["failure_reason"] == "MISCALIBRATION" for o in res["per_object"])
    # recovered rotation magnitude is in the right ballpark
    assert res["metrics"]["estimated_rotation_offset_deg"] == pytest.approx(2.0, abs=0.8)
    bias_check = next(c for c in res["checks"] if c["name"] == "systematic_bias_px")
    assert not bias_check["passed"]


def test_perception_failure_distinguished_from_miscalibration(fresh_safety_root):
    res = cal.run_validation(mode="perception_failure", tamper_fraction=0.3, seed=3)
    assert res["status"] == "PERCEPTION_FAILURE"
    flagged = [o for o in res["per_object"] if o["flagged"]]
    assert flagged and len(flagged) < len(res["per_object"])  # minority, not systematic
    # maps onto the existing platform failure reason
    assert all(o["failure_reason"] == "SENSOR_DISAGREEMENT" for o in flagged)


def test_status_persisted_and_alert_emitted(fresh_safety_root, tmp_path):
    from sensorflow.evaluation.records import reset_store
    store = reset_store(tmp_path / "eval")

    assert cal.latest_status() is None
    cal.run_validation(mode="miscalibrated", rotation_offset_deg=2.0, seed=3)
    status = cal.latest_status()
    assert status is not None and status["status"] == "MISCALIBRATED"
    # failure surfaced in the existing labeleval alert store
    alerts = [a for a in store.all("alerts") if a.kind == "calibration"]
    assert alerts and alerts[0].severity == "critical"


def test_extrinsic_projection_roundtrip():
    import numpy as np
    ext = cal.make_extrinsic()
    scene = cal.generate_scene(num_objects=6, seed=5)
    for obj in scene["objects"]:
        uv = cal.project_points(np.asarray(obj["points_lidar"]), ext)
        assert uv.shape[1] == 2
        assert np.isfinite(uv).all()
