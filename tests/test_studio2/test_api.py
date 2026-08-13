"""API lifecycle over the studio2 router mounted on a bare FastAPI app."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.studio2.api import router


@pytest.fixture()
def client(studio2_env):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_status_and_registry_summary(client):
    r = client.get("/api/studio2/status")
    assert r.status_code == 200
    body = r.json()
    assert body["dependencies"]["sensorflow.safety"] == "importable"
    r = client.get("/api/studio2/registry/summary")
    assert r.status_code == 200
    assert "REGRESSION" in r.json()["dataset_roles"]


def test_registry_dataset_lifecycle_and_contamination_guard(client):
    r = client.post("/api/studio2/registry/datasets",
                    json={"name": "launch-eval", "role": "LAUNCH",
                          "actor": "tester"})
    assert r.status_code == 200
    ds = r.json()
    assert ds["protected_evaluation"] is True

    # guarded transition without override -> 409
    r = client.post(f"/api/studio2/registry/datasets/{ds['entity_id']}/role",
                    json={"new_role": "TRAINING", "actor": "tester"})
    assert r.status_code == 409
    assert "contamination boundary" in r.json()["detail"]

    # with override -> allowed and recorded
    r = client.post(f"/api/studio2/registry/datasets/{ds['entity_id']}/role",
                    json={"new_role": "TRAINING", "actor": "tester",
                          "override_reason": "signed off in review"})
    assert r.status_code == 200
    assert r.json()["role"] == "TRAINING"

    r = client.get(f"/api/studio2/registry/datasets/{ds['entity_id']}")
    assert r.json()["governance_overrides"][0]["reason"] == "signed off in review"

    # audit trail captured it
    events = client.get("/api/studio2/audit").json()["events"]
    assert any(e["action"] == "governance_override" for e in events)


def test_registry_unknown_kind_404(client):
    assert client.get("/api/studio2/registry/frobnicators").status_code == 404
    assert client.get("/api/studio2/registry/models/nope").status_code == 404


def test_release_evaluate_decide_approve_flow(client):
    payload = {
        "safety_metrics": {"decision": "RELEASE_READY", "blocking_gates": [],
                           "candidate_run_id": "eval-x",
                           "evidence_package_id": "sep-x"},
        "regression_results": {"run_id": "seq-x", "decision": "PASS",
                               "samples_used": 100, "affected_strata": []},
        "distribution_shift": {"run_id": "eval-x", "shifts": []},
        "agentic_outcome": {"outcome": "CONTINUE_INVESTIGATION",
                            "policy_version": "p", "failure_id": "f",
                            "severity": "S1"},
        "closed_loop": {"scenario_id": "s", "verdict": "METRIC_ONLY"},
        "hardware_matrix": {"status": "PASS", "matrix_id": "m",
                            "n_combinations": 3, "insufficient": []},
    }
    r = client.post("/api/studio2/release/evaluate", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "GO"
    assert d["deployment_authorized"] is False

    # decisions history
    hist = client.get("/api/studio2/release/decisions").json()["decisions"]
    assert any(x["entity_id"] == d["entity_id"] for x in hist)

    # approval without rationale rejected by schema-level check
    r = client.post(f"/api/studio2/release/decisions/{d['entity_id']}/approve",
                    json={"approver": "warda", "rationale": " "})
    assert r.status_code == 409

    r = client.post(f"/api/studio2/release/decisions/{d['entity_id']}/approve",
                    json={"approver": "warda", "rationale": "evidence reviewed"})
    assert r.status_code == 200
    assert r.json()["deployment_authorized"] is True


def test_release_evaluate_with_missing_inputs_is_review(client):
    r = client.post("/api/studio2/release/evaluate", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "REVIEW"
    assert len(d["degraded_inputs"]) == 3


def test_approve_unknown_decision_404(client):
    r = client.post("/api/studio2/release/decisions/rd-nope/approve",
                    json={"approver": "a", "rationale": "b"})
    assert r.status_code == 404


def test_funnel_endpoint(client):
    r = client.get("/api/studio2/funnel")
    assert r.status_code == 200
    body = r.json()
    assert {s["stage"] for s in body["stages"]} == {
        "raw", "selected", "simulated", "evaluated", "failed", "hitl",
        "regression"}
    for s in body["stages"]:
        assert "available" in s


def test_docs_endpoints(client):
    r = client.get("/api/studio2/docs")
    assert r.status_code == 200
    docs = r.json()["docs"]
    assert "studio2-review.md" in docs
    r = client.get("/api/studio2/docs/studio2-review.md")
    assert r.status_code == 200
    assert "Architecture Review" in r.json()["content"]
    assert client.get("/api/studio2/docs/evil.md").status_code == 404
    assert client.get("/api/studio2/docs/studio2-..%2Fsecrets.md").status_code == 404


def test_registry_ingest_endpoint(client):
    r = client.post("/api/studio2/registry/ingest")
    assert r.status_code == 200
    assert "registered" in r.json()
