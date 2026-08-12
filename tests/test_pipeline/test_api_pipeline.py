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


def test_ingest_endpoint(client):
    res = client.post("/api/dataset/ingest", json={
        "vendors": ["alpamayo"],
        "sequence_id": "api_test",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert Path(data["manifest"]).exists()


def test_full_pipeline(client):
    seq_id = "full_pipeline"
    client.post("/api/dataset/ingest", json={"vendors": ["alpamayo", "waymo"], "sequence_id": seq_id})
    client.post("/api/perception/auto-label", json={"sequence_id": seq_id, "no_sam": True})
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
