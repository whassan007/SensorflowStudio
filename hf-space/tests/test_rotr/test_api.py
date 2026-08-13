"""API lifecycle over an isolated router: bank -> run -> query ->
attribution -> consequence -> HITL -> flywheel -> gate."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.rotr.api import router

SPEC_QUERY = ("failed to yield to pedestrian at uncontrolled intersection "
              "during low visibility")


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def run(client):
    r = client.post("/api/rotr/runs",
                    json={"n_scenarios": 28, "seed": 7,
                          "model_version": "stack-v1"})
    assert r.status_code == 200, r.text
    return r.json()


class TestLifecycle:
    def test_health_and_rules(self, client):
        assert client.get("/api/rotr/health").json()["status"] == "ok"
        rules = client.get("/api/rotr/rules").json()
        assert len(rules["rules"]) == 6
        assert "ILLUSTRATIVE" in rules["config_note"]

    def test_bank_endpoints(self, client):
        r = client.post("/api/rotr/banks",
                        json={"n_scenarios": 14, "seed": 3,
                              "model_version": "stack-v1"})
        assert r.status_code == 200
        bank = r.json()
        assert client.get(f"/api/rotr/banks/{bank['bank_id']}").status_code == 200
        sid = bank["scenarios"][0]["scenario_id"]
        sc = client.get(f"/api/rotr/banks/{bank['bank_id']}/scenarios/{sid}").json()
        assert sc["ego"] and sc["actual_context"]
        assert client.post("/api/rotr/banks",
                           json={"model_version": "nope"}).status_code == 422

    def test_run_and_metrics(self, client, run):
        assert run["metrics"]["rotr_recall"] == 1.0
        assert run["metrics"]["false_accusation_rate"] == 0.0
        assert run["gate"]["outcome"] == "NO_GO"
        got = client.get(f"/api/rotr/runs/{run['run_id']}").json()
        assert got["run_id"] == run["run_id"]
        assert client.get("/api/rotr/runs").json()

    def test_violation_listing_and_taxonomy_filters(self, client, run):
        rid = run["run_id"]
        allv = client.get(f"/api/rotr/runs/{rid}/violations").json()
        assert allv["n_results"] > 0
        vru = client.get(f"/api/rotr/runs/{rid}/violations",
                         params={"vulnerability": "VRU"}).json()
        assert 0 < vru["n_results"] < allv["n_results"]
        assert all(x["taxonomy"]["vulnerability"] == "VRU"
                   for x in vru["results"])

    def test_spec_structured_query(self, client, run):
        r = client.post("/api/rotr/query",
                        json={"run_id": run["run_id"], "text": SPEC_QUERY})
        assert r.status_code == 200
        out = r.json()
        assert out["query"]["actor"] == "pedestrian"
        assert out["query"]["visibility"] == "low"
        assert out["n_results"] >= 1
        for item in out["results"]:
            assert item["taxonomy"]["legality"] == "YIELD"
            assert item["environment"]["visibility"] == "low"
            assert item["cluster_id"]
            assert item["provenance"]["model_version"] == "stack-v1"

    def test_attribution_matrix_and_consequence(self, client, run):
        rid = run["run_id"]
        matrix = client.get(f"/api/rotr/runs/{rid}/attribution").json()
        assert matrix["rows"]
        row = matrix["rows"][0]
        assert set(row["layers"]) == {
            "perception", "prediction", "planning", "localization", "map",
            "control", "policy_rule", "data_label"}
        vid = row["violation_id"]
        detail = client.get(f"/api/rotr/runs/{rid}/violations/{vid}").json()
        assert detail["attribution"]["primary_layer"]
        cf = client.get(
            f"/api/rotr/runs/{rid}/violations/{vid}/consequence").json()
        assert cf["planner_evaluation"]["observed_trajectory"]
        assert cf["planner_evaluation"]["corrected_trajectory"]
        assert "scenario_geometry" in cf

    def test_clusters_and_gate(self, client, run):
        rid = run["run_id"]
        clusters = client.get(f"/api/rotr/runs/{rid}/clusters").json()
        assert clusters and clusters[0]["count"] >= 1
        gate = client.get(f"/api/rotr/runs/{rid}/gate").json()
        assert gate["outcome"] in ("GO", "NO_GO")
        policy = client.get("/api/rotr/stopship/policy").json()
        assert policy["policy_version"]


class TestHITLAndFlywheel:
    def test_validate_then_guard_then_override(self, client, run):
        rid = run["run_id"]
        queue = client.get(f"/api/rotr/runs/{rid}/hitl").json()
        assert queue and queue[0]["status"] == "PENDING"
        review = queue[0]

        r = client.post("/api/rotr/hitl/action",
                        json={"run_id": rid, "review_id": review["review_id"],
                              "action": "VALIDATE", "actor": "reviewer",
                              "notes": "confirmed"})
        assert r.status_code == 200
        cand = r.json()["candidate"]
        assert cand["dataset_role"] == "REGRESSION"

        suite = client.get("/api/rotr/flywheel/suite").json()
        assert any(m["candidate_id"] == cand["candidate_id"]
                   for m in suite["members"])

        promote = client.post("/api/rotr/flywheel/promote",
                              json={"candidate_id": cand["candidate_id"],
                                    "actor": "sneaky"})
        assert promote.status_code == 409          # contamination guard

        override = client.post("/api/rotr/flywheel/override",
                               json={"candidate_id": cand["candidate_id"],
                                     "actor": "safety-lead",
                                     "reason": "governed exception"})
        assert override.status_code == 200
        promote2 = client.post("/api/rotr/flywheel/promote",
                               json={"candidate_id": cand["candidate_id"],
                                     "actor": "lead"})
        assert promote2.status_code == 200
        assert promote2.json()["guard_state"] == "OVERRIDDEN"


class TestDocs:
    def test_architecture_doc_served(self, client):
        r = client.get("/api/rotr/docs/architecture")
        assert r.status_code == 200
        assert "Right-of-the-Road" in r.json()["markdown"]
