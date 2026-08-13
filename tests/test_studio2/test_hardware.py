"""Hardware gate matrix: pure gate logic on planted combination rows."""

from __future__ import annotations

from sensorflow.studio2.hardware import (
    DEFAULT_HARDWARE_POLICY,
    evaluate_matrix,
    min_support,
)


def _row(platform="cpu-reference", sensor="lidar", region="san_francisco",
         metrics=None, n=10_000, evidence="PRESENT", reason=None):
    combo = {"compute_platform": platform, "sensor_config": sensor,
             "sensor_generation": "LiDAR-Gen2", "calibration_version": "calib-ok",
             "firmware": "host-fp32", "region": region}
    row = {"combination": combo,
           "combination_label": f"{region} × {platform} × {sensor}",
           "metrics": metrics if metrics is not None else
           {"recall": 0.9, "precision": 0.95},
           "n": n, "evidence": evidence,
           "derivation": {"metrics_source": "planted"}}
    if evidence != "PRESENT":
        row["metrics"] = None
        row["evidence_reason"] = reason or "planted absence"
    return row


def test_min_support_is_sane():
    s = min_support(0.10)
    # exact value depends on whether hardening.power is importable (power
    # calc) or the Wilson fallback runs; both live in this ballpark
    assert 100 <= s["n"] <= 1000
    assert "fallback" in s["method"] or "hardening" in s["method"]
    # tighter width -> more samples, regardless of method
    assert min_support(0.05)["n"] > s["n"]


def test_all_pass(registry):
    report = evaluate_matrix([_row(), _row(region="phoenix")], persist=False)
    assert report["status"] == "PASS"
    assert report["n_pass"] == 2 and report["n_fail"] == 0


def test_failing_critical_combination_blocks_despite_global_pass():
    rows = [
        _row(region="phoenix"),  # passing non-critical-ish row
        _row(region="san_francisco", sensor="lidar",
             metrics={"recall": 0.62, "precision": 0.95}),  # critical, fails
    ]
    report = evaluate_matrix(rows, global_metrics={"recall": 0.88,
                                                   "precision": 0.95},
                             persist=False)
    assert report["global_pass"] is True
    assert report["status"] == "FAIL_CRITICAL"
    assert report["critical_failures"][0]["combination"]["region"] == "san_francisco"
    assert report["global_vs_matrix_note"] is not None


def test_non_critical_failure_is_fail_not_fail_critical():
    policy = {**DEFAULT_HARDWARE_POLICY,
              "critical_combinations": [{"region": "san_francisco",
                                         "sensor_config": "fused"}]}
    rows = [_row(),
            _row(region="rural_pilot", sensor="camera",
                 metrics={"recall": 0.60, "precision": 0.95})]
    # the camera row's combination is not critical under this policy
    rows[1]["combination"]["sensor_config"] = "camera"
    report = evaluate_matrix(rows, policy=policy, persist=False)
    assert report["status"] == "FAIL"
    assert report["critical_failures"] == []


def test_insufficient_evidence_combinations_are_surfaced_never_passed():
    rows = [
        _row(),
        _row(platform="versal-ai-edge", evidence="MISSING",
             reason="no vitis HIL run for this device"),
        _row(region="rural_pilot", n=25),  # support below the Wilson minimum
    ]
    report = evaluate_matrix(rows, persist=False)
    assert report["n_insufficient"] == 2
    labels = [i["combination_label"] for i in report["insufficient"]]
    assert any("versal-ai-edge" in l for l in labels)
    reasons = [i["reason"] for i in report["insufficient"]]
    assert any("no vitis HIL run" in r for r in reasons)
    assert any("below minimum" in r for r in reasons)
    # insufficient rows never count as passing
    statuses = {r["combination_label"]: r["status"] for r in report["rows"]}
    assert statuses["rural_pilot × cpu-reference × lidar"] == "INSUFFICIENT"


def test_all_insufficient_reports_insufficient_status():
    rows = [_row(evidence="MISSING", reason="nothing measured")]
    report = evaluate_matrix(rows, persist=False)
    assert report["status"] == "INSUFFICIENT"


def test_missing_metric_fails_check():
    rows = [_row(metrics={"recall": 0.9})]  # precision missing entirely
    report = evaluate_matrix(rows, persist=False)
    assert report["rows"][0]["status"] == "FAIL"
    assert "precision missing" in report["rows"][0]["failed_checks"]
