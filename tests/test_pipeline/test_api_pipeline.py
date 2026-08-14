"""Test FastAPI pipeline routes."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app_backend import app
    return TestClient(app)


def test_pipeline_status(client):
    res = client.get("/api/pipeline/status?sequence_id=seq_001")
    assert res.status_code == 200
    data = res.json()
    assert data["sequence_id"] == "seq_001"
    assert "ingest_complete" in data
    # Not-run stages must be null (UI shows blank), never false FAIL.
    assert data["tracking_complete"] is None
    assert data["launch_gate_passed"] is None


def test_ingest_endpoint(client):
    res = client.post("/api/dataset/ingest", json={
        "vendors": ["alpamayo"],
        "sequence_id": "api_test",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert Path(data["manifest"]).exists()
    assert data["frames"] >= 1
    assert data["demo_stub"] is True


def test_ingest_a2d2_endpoint(client):
    res = client.post("/api/dataset/ingest", json={
        "vendors": ["a2d2"],
        "sequence_id": "api_a2d2",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert Path(data["manifest"]).exists()
    assert data["frames"] >= 1
    assert data.get("demo_stub") is True


def test_full_pipeline(client):
    seq_id = "full_pipeline"
    ingest = client.post(
        "/api/dataset/ingest",
        json={"vendors": ["alpamayo", "waymo"], "sequence_id": seq_id, "allow_mix": True},
    )
    assert ingest.status_code == 200
    assert ingest.json()["frames"] == 6
    assert ingest.json()["demo_stub"] is True
    label = client.post("/api/perception/auto-label", json={"sequence_id": seq_id, "no_sam": True})
    assert label.json()["frames_processed"] == 6
    assert label.json()["frames_expected"] == 6
    client.post("/api/perception/track", json={"sequence_id": seq_id})
    res = client.post("/api/gates/quality", json={"sequence_id": seq_id})
    assert res.status_code == 200
    data = res.json()
    assert "metric_card" in data or "quality_score" in data

    launch_res = client.post("/api/gates/launch", json={"sequence_id": seq_id})
    assert launch_res.status_code == 200

    status = client.get(f"/api/pipeline/status?sequence_id={seq_id}").json()
    assert status["ingest_complete"] is True
    assert status["tracking_complete"] is True
    assert status["frames_ingested"] == 6
    assert status["demo_stub"] is True


def test_ingest_rejects_mix_without_flag(client):
    res = client.post(
        "/api/dataset/ingest",
        json={"vendors": ["alpamayo", "waymo"], "sequence_id": "no_mix"},
    )
    assert res.status_code == 400
    assert "allow_mix" in str(res.json()["detail"]).lower() or "Multiple vendors" in str(res.json()["detail"])
