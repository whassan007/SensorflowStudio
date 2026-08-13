"""API lifecycle over the /api/agentic router via TestClient."""

from __future__ import annotations

import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.agentic.api import router


@pytest.fixture(scope="module")
def client(agentic_env):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="module")
def ped_cone_id(client):
    res = client.post("/api/agentic/failures/detect", json={})
    assert res.status_code == 200
    events = res.json()["failures"]
    assert events
    flip = next(e for e in events
                if e["kind"] == "classification_flip"
                and "pedestrian" in e["title"]
                and "construction_cone" in e["title"])
    return flip["failure_id"]


def test_detect_and_queue(client, ped_cone_id):
    res = client.get("/api/agentic/failures")
    assert res.status_code == 200
    queue = res.json()["failures"]
    assert any(f["failure_id"] == ped_cone_id for f in queue)
    res = client.get(f"/api/agentic/failures/{ped_cone_id}")
    basis = res.json()["failure"]["detection_basis"]
    assert basis["denominator"] > 0 and basis["candidate_events"] > 0


def test_staged_analysis_and_evidence(client, ped_cone_id):
    # step stage by stage; stage order is enforced by the orchestrator
    for stage in ("EVIDENCE_AGGREGATION", "FAILURE_ANALYSIS",
                  "LAUNCH_DECISION"):
        res = client.post(
            f"/api/agentic/failures/{ped_cone_id}/analyze?stage={stage}",
            json={})
        assert res.status_code == 200, res.text
        state = res.json()["state"]
        rec = next(s for s in state["stages"] if s["stage"] == stage)
        assert rec["status"] == "complete"

    res = client.get(f"/api/agentic/failures/{ped_cone_id}/evidence")
    graph = res.json()["graph"]
    assert {n["node_type"] for n in graph["nodes"]} >= {"Object", "GroundTruth",
                                                        "Frequency"}
    res = client.get(f"/api/agentic/failures/{ped_cone_id}/snippets")
    assert res.json()["snippets"]

    res = client.post(f"/api/agentic/failures/{ped_cone_id}/cluster", json={})
    assert res.status_code == 200
    res = client.post(
        f"/api/agentic/failures/{ped_cone_id}/safety-assessment", json={})
    assert res.status_code == 200

    res = client.post(
        f"/api/agentic/failures/{ped_cone_id}/launch-assessment", json={})
    body = res.json()
    assert body["policy_evaluation"]["outcome"] in (
        "AUTOMATIC_STOP_SHIP", "LAUNCH_REVIEW_REQUIRED",
        "CONTINUE_INVESTIGATION", "NO_LAUNCH_IMPACT", "INDETERMINATE")
    assert body["policy_evaluation"]["policy_version"]


def test_human_review_and_flywheel(client, ped_cone_id):
    res = client.post(
        f"/api/agentic/failures/{ped_cone_id}/human-review",
        json={"reviewer": "api-test-reviewer",
              "decision": "confirm_failure",
              "rationale": "GT verified against synthetic vendor labels",
              "evidence_reviewed": ["evidence_graph", "snippets"]})
    assert res.status_code == 200
    res = client.get(f"/api/agentic/failures/{ped_cone_id}/human-review")
    assert any(d["reviewer"] == "api-test-reviewer"
               for d in res.json()["decisions"])

    # LEARNING_FLYWHEEL now permitted (failure validated by the human)
    res = client.post(
        f"/api/agentic/failures/{ped_cone_id}/analyze?stage=LEARNING_FLYWHEEL",
        json={})
    assert res.status_code == 200

    res = client.get("/api/agentic/evaluation-suites")
    suites = res.json()["suites"]
    assert suites
    suite = suites[0]
    res = client.get(f"/api/agentic/evaluation-suites/{suite['suite_id']}")
    assert res.status_code == 200

    # contamination guard over the API: promotion without override -> 4xx
    member_id = suite["members"][0]["member_id"]
    res = client.post(
        f"/api/agentic/evaluation-suites/{suite['suite_id']}/promote-member",
        json={"member_id": member_id})
    assert res.status_code in (400, 403, 409, 422)


def test_policy_endpoints(client):
    res = client.get("/api/agentic/policy")
    assert res.status_code == 200
    doc = res.json()["policy"]
    assert doc["policy_version"]
    mutated = copy.deepcopy(doc)
    mutated["expected_loss"]["delay_cost"] = 123.0
    res = client.post("/api/agentic/policy",
                      json={"policy": mutated, "actor": "api-test"})
    assert res.status_code == 200
    new_version = res.json()["policy_version"]
    assert new_version != doc["policy_version"]
    res = client.get(f"/api/agentic/policy?version={doc['policy_version']}")
    assert res.status_code == 200
    res = client.get("/api/agentic/policy/versions")
    assert len(res.json()["versions"]) >= 2
    # restore the original as active for other tests
    res = client.post("/api/agentic/policy",
                      json={"policy": doc, "actor": "api-test"})
    assert res.status_code == 200


def test_regression_scorecard_audit_and_worked_example(client, ped_cone_id):
    res = client.post("/api/agentic/regression/evaluate", json={})
    assert res.status_code == 200
    assert res.json()["suites"]

    res = client.get(f"/api/agentic/audit/{ped_cone_id}")
    body = res.json()
    assert body["records"]
    assert body["chain"]["valid"] is True

    res = client.get("/api/agentic/worked-example")
    assert res.status_code == 200
    we = res.json()["walkthrough"]
    assert we["layers"]["4_launch_decision"]["outcome"]["value"]
    card_id = we["scorecard_id"]
    res = client.get(f"/api/agentic/scorecards/{card_id}")
    assert res.status_code == 200
    card = res.json()["scorecard"]
    for field in ("frequency", "exposure", "severity", "confidence",
                  "novelty", "concentration", "downstream_impact",
                  "mitigations", "residual_risk"):
        f = card[field]
        assert f["tag"] in ("OBSERVED", "PREDICTED", "HYPOTHETICAL",
                            "REQUIRED_EVIDENCE")
        assert f["evidence_ref"]
