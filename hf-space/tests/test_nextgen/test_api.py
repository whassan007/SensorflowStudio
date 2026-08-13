"""End-to-end API flows through the nextgen router (TestClient)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.nextgen.api import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_status(client):
    r = client.get("/api/nextgen/status")
    assert r.status_code == 200
    body = r.json()
    assert "COUNTERFACTUAL" in body["data_labels"]
    assert "actors.sudden_brake" in body["transformations"]


def test_generate_validate_flow_carries_provenance(client):
    r = client.post("/api/nextgen/counterfactuals/generate", json={
        "recipe": [{"kind": "actors.occluded_emergence", "params": {}}],
        "n_scenarios": 1, "seed": 7})
    assert r.status_code == 200
    scenario = r.json()["scenarios"][0]
    assert scenario["provenance"]["data_label"] == "COUNTERFACTUAL"
    sid = scenario["scenario_id"]

    r = client.post(f"/api/nextgen/counterfactuals/{sid}/validate")
    assert r.status_code == 200
    assert r.json()["accepted"] is True

    listed = client.get("/api/nextgen/counterfactuals").json()["scenarios"]
    assert listed[0]["validity"]["accepted"] is True

    weights = client.get("/api/nextgen/counterfactuals/suite-weights").json()
    assert weights["weights"][sid] == 1.0


def test_replay_and_causal_demo_flow(client):
    r = client.post("/api/nextgen/simulation/replay",
                    json={"scenario_id": "demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["data_label"] == "COUNTERFACTUAL"   # label survives to report
    assert body["open_loop"]["frame_recall"] is not None
    assert body["metrics"]["min_separation_m"] is not None

    r = client.post("/api/nextgen/causal/replay", json={"scenario_id": "demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "BEHAVIORALLY_CONSEQUENTIAL"
    assert len(body["causal_chain"]) == 3


def test_gauntlet_flow(client):
    r = client.post("/api/nextgen/gauntlet/run",
                    json={"effects": {"safety_critical": -0.08},
                          "budget_units": 40_000})
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    st = client.get(f"/api/nextgen/gauntlet/{run_id}/status").json()
    assert st["halted"] is True
    assert st["recommendation"] == "DO_NOT_LAUNCH"

    res = client.get(f"/api/nextgen/gauntlet/{run_id}/results").json()
    assert res["strata"][0]["decision"] == "REGRESSION"
    assert client.get("/api/nextgen/gauntlet/nope/status").status_code == 404


def test_compute_benchmark_and_report(client):
    assert client.get("/api/nextgen/compute/report").status_code == 404
    r = client.post("/api/nextgen/compute/benchmark",
                    json={"n_scenarios": 2, "frames_per_sequence": 8})
    assert r.status_code == 200
    assert r.json()["hit_rate"] == 0.5
    assert client.get("/api/nextgen/compute/report").status_code == 200


def test_safety_report_endpoints(client):
    demo = client.post("/api/nextgen/metrics/safety-report", json={}).json()
    assert demo["deltas"]["overall_recall"] > 0
    assert demo["deltas"]["safety_critical_recall"] < 0

    custom = client.post("/api/nextgen/metrics/safety-report", json={
        "objects": [{"x": 6.0, "y": 0.0, "class_name": "pedestrian",
                     "detected": True}],
        "ego_speed_mps": 12.0, "friction_mu": 0.5}).json()
    assert custom["region_params"]["friction_mu"] == 0.5
    assert custom["safety_informed"]["safety_critical_recall"] == 1.0


def test_distribution_and_docs(client):
    r = client.post("/api/nextgen/counterfactuals/generate", json={
        "recipe": [{"kind": "environment.day_to_night", "params": {}}],
        "n_scenarios": 1})
    sid = r.json()["scenarios"][0]["scenario_id"]
    r = client.post("/api/nextgen/distribution/analyze",
                    json={"scenario_id": sid})
    assert r.status_code == 200
    assert r.json()["data_labels"] == ["COUNTERFACTUAL"]

    r = client.get("/api/nextgen/architecture/docs")
    assert r.status_code == 200
    docs = r.json()["docs"]
    assert "Three-Way Architecture Comparison" in docs["comparison"]["content"]
    assert "Build / Reuse / Don't-Build" in docs["adr"]["content"]
    assert "Phases 0-6" in docs["rollout"]["content"]
    assert "Generative Architecture Comparison" in docs["worldmodel"]["content"]


def test_invalid_inputs_rejected(client):
    assert client.post("/api/nextgen/counterfactuals/generate",
                       json={"recipe": []}).status_code == 422
    assert client.post("/api/nextgen/counterfactuals/generate", json={
        "recipe": [{"kind": "nope.nope", "params": {}}]}).status_code == 422
    assert client.post("/api/nextgen/simulation/replay", json={
        "scenario_id": "demo", "engine": "nope"}).status_code == 422
    assert client.post("/api/nextgen/simulation/replay", json={
        "scenario_id": "missing-id"}).status_code == 404
