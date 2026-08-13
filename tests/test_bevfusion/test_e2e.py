"""End-to-end self-evaluation: fused engine must measurably beat the baseline
on identical scenes, deterministically — plus the API surface."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.bevfusion import evaluate as evaluate_mod
from sensorflow.bevfusion.api import router as bevfusion_router
from sensorflow.bevfusion.evaluate import run_comparison


@pytest.fixture(scope="module")
def report():
    return run_comparison(n_sequences=6, frames_per_sequence=24, seed=7, persist=False)


def _headline(report, metric, who):
    row = next(h for h in report["headline_deltas"] if h["metric"] == metric)
    return row[who]


def _cohort(report, name):
    return next(c for c in report["per_cohort"] if c["cohort"] == name)


def test_fused_beats_baseline_on_recall_and_geometry(report):
    assert _headline(report, "recall", "delta") >= 0.20
    assert _headline(report, "mean_iou", "delta") >= 0.10
    # Fused position error must be at most 60% of the monocular baseline.
    assert _headline(report, "position_error_m", "candidate") <= \
        0.6 * _headline(report, "position_error_m", "baseline")
    assert _headline(report, "safety_recall", "delta") >= 0.10


def test_night_and_occluded_cohorts_recover_substantially(report):
    night = _cohort(report, "night")
    assert night["recall_delta"] >= 0.30, night
    occluded = _cohort(report, "occluded")
    assert occluded["recall_delta"] >= 0.30, occluded
    assert night["explanation"] and occluded["explanation"]


def test_tracking_id_switches_drop(report):
    base = _headline(report, "id_switch_rate", "baseline")
    cand = _headline(report, "id_switch_rate", "candidate")
    assert cand <= 0.5 * base, (base, cand)
    assert _headline(report, "idf1", "delta") >= 0.20
    assert _headline(report, "fragmentation_rate", "candidate") <= \
        _headline(report, "fragmentation_rate", "baseline")


def test_recommendation_is_promote_with_no_blockers(report):
    assert report["recommendation"] == "PROMOTE"
    assert report["blockers"] == []
    assert len(report["improvements"]) >= 5
    assert report["engines"] == {"baseline": "perception-v1-camera",
                                 "candidate": "perception-v3-bevfusion"}


def test_comparison_is_deterministic(report):
    again = run_comparison(n_sequences=6, frames_per_sequence=24, seed=7, persist=False)
    assert again["headline_deltas"] == report["headline_deltas"]
    assert again["per_cohort"] == report["per_cohort"]


def test_persistence_writes_latest_report(tmp_path):
    rep = run_comparison(n_sequences=3, frames_per_sequence=12, seed=3,
                         out_dir=tmp_path, persist=True)
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / f"{rep['run_id']}.json").exists()


# ------------------------------------------------------------------ API


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_mod, "RUNS_DIR", tmp_path)
    app = FastAPI()
    app.include_router(bevfusion_router)
    return TestClient(app)


def test_api_run_report_status_lifecycle(client):
    r = client.get("/api/bevfusion/report")
    assert r.status_code == 404

    r = client.post("/api/bevfusion/run",
                    json={"n_sequences": 3, "frames_per_sequence": 12, "seed": 11})
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"] in ("PROMOTE", "DO_NOT_PROMOTE")
    assert body["headline_deltas"] and body["per_cohort"]

    r = client.get("/api/bevfusion/report")
    assert r.status_code == 200
    assert r.json()["run_id"] == body["run_id"]

    r = client.get("/api/bevfusion/status")
    assert r.status_code == 200
    status = r.json()
    assert status["ready"] is True
    assert status["n_runs"] == 1
    assert status["engines"]["candidate"] == "perception-v3-bevfusion"


def test_api_rejects_out_of_range_params(client):
    r = client.post("/api/bevfusion/run", json={"n_sequences": 0})
    assert r.status_code == 422
