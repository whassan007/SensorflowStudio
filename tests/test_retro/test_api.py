"""API lifecycle tests for /api/retro via TestClient."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.retro.api import router


@pytest.fixture()
def client(retro_root):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_env_and_compat_endpoints(client):
    env = client.get("/api/retro/env").json()["environment"]
    assert "os_name" in env and "gpus" in env
    compat = client.get("/api/retro/compat").json()
    assert "vllm_supported" in compat["report"]
    assert "[" in compat["formatted"]  # formatted chain output


def test_backends_endpoint(client):
    body = client.get("/api/retro/backends").json()
    names = {b["backend"] for b in body["backends"]}
    assert names == {"mock", "ollama", "vllm"}


def test_rag_search_and_eval_endpoints(client):
    hits = client.get("/api/retro/rag/search",
                      params={"q": "phantom braking plastic bag", "k": 3}).json()
    assert hits["hits"] and all("retrieved_text" in h for h in hits["hits"])
    ev = client.get("/api/retro/rag/eval", params={"k": 4}).json()["report"]
    assert ev["precision_at_k"] >= 0.85


def test_fixtures_listing(client):
    fixtures = client.get("/api/retro/fixtures").json()["fixtures"]
    ids = {f["fixture_id"] for f in fixtures}
    assert {"phantom_brake_plastic_bag", "missed_pedestrian_rain"} <= ids


def test_analyze_lifecycle(client):
    res = client.post("/api/retro/analyze",
                      params={"fixture_id": "missed_pedestrian_rain",
                              "backend": "mock"})
    assert res.status_code == 200
    body = res.json()
    sc = body["scorecard"]
    assert sc["failure_type"] == "FALSE_NEGATIVE"
    assert sc["launch_recommendation"] == "FAIL"
    assert "# Retrospective Scorecard" in body["markdown"]

    eval_id = sc["evaluation_id"]
    listing = client.get("/api/retro/analyses").json()["analyses"]
    assert any(a["evaluation_id"] == eval_id for a in listing)

    detail = client.get(f"/api/retro/analyses/{eval_id}").json()
    assert detail["scorecard"]["severity"] == "CRITICAL"

    audit = client.get(f"/api/retro/analyses/{eval_id}/audit").json()
    assert audit["records"]
    assert all({"tool", "timestamp", "result_hash", "status"} <= set(r)
               for r in audit["records"])


def test_analyze_uploaded_log(client):
    from sensorflow.retro.tools.builtin import FIXTURES_DIR
    log = json.loads((FIXTURES_DIR / "benign_fp_static_sign.json").read_text())
    log["evaluation_id"] = "EVAL-UPLOADED-77"
    res = client.post("/api/retro/analyze", params={"backend": "mock"}, json=log)
    assert res.status_code == 200
    assert res.json()["scorecard"]["evaluation_id"] == "EVAL-UPLOADED-77"


def test_analyze_error_paths(client):
    assert client.post("/api/retro/analyze").status_code == 400
    assert client.post("/api/retro/analyze",
                       params={"fixture_id": "nope"}).status_code == 404
    assert client.post("/api/retro/analyze",
                       params={"fixture_id": "x", "backend": "gpt9"}
                       ).status_code == 422
    missing = client.get("/api/retro/analyses/NOPE")
    assert missing.status_code == 404


def test_tools_listing(client):
    tools = client.get("/api/retro/tools").json()["tools"]
    by_name = {t["name"]: t for t in tools}
    assert by_name["log_reader"]["read_only"] is True
    assert by_name["create_evaluation_case"]["read_only"] is False
    assert all("input_schema" in t and "output_schema" in t for t in tools)


def test_vllm_backend_honestly_unavailable_locally(client):
    """On this macOS machine an analysis with backend=vllm must fail with a
    clear 503 unless a remote CUDA/ROCm endpoint is configured."""
    import platform
    if platform.system() != "Darwin":
        pytest.skip("assertion set is for the macOS dev host")
    res = client.post("/api/retro/analyze",
                      params={"fixture_id": "benign_fp_static_sign",
                              "backend": "vllm"})
    assert res.status_code == 503
    assert "unavailable" in res.json()["detail"]
