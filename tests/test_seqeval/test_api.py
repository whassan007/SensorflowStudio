"""REST surface: run lifecycle, evidence, attribution, policy, trajectories."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.seqeval.api import router
from tests.test_seqeval.conftest import FAST_POLICY


@pytest.fixture(scope="module")
def client(seq_env):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="module")
def finished_run(seq_env, client):
    body = {
        "population_id": seq_env["meta"]["population_id"],
        "baseline": {"model_version": "api-base-v1"},
        "candidate": {"model_version": "api-cand-v1",
                      "effects": {"pedestrian|night": -0.03}},
        "policy": {**FAST_POLICY, "safety_floor": 1500},
        "sync": True,
    }
    r = client.post("/api/seqeval/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_unknown_population_404(client):
    r = client.post("/api/seqeval/runs", json={
        "population_id": "pop-nope",
        "baseline": {"model_version": "a"},
        "candidate": {"model_version": "b"}})
    assert r.status_code == 404


def test_run_state_shape(client, finished_run):
    rid = finished_run["run_id"]
    r = client.get(f"/api/seqeval/runs/{rid}")
    assert r.status_code == 200
    state = r.json()
    assert state["status"] == "done"
    assert state["decision"] in ("REGRESSION", "PASS", "INSUFFICIENT_EVIDENCE")
    assert state["gate"] in ("block", "allow", "expand_or_report")
    assert state["budget"]["samples_used"] > 0
    assert state["budget"]["planned_total"] > 0
    assert state["plan"]["plan_hash"]
    assert state["sanity"]["ok"] is True
    # per-node decision table for the dashboard
    keys = {n["node"] for n in state["nodes"]}
    assert "overall" in keys
    assert any(k.startswith("stratum:") for k in keys)
    node = next(n for n in state["nodes"] if n["node"] == "overall")
    assert "delta_ci" in node and "e_regression" in node and "decision" in node


def test_trajectories_include_boundaries(client, finished_run):
    rid = finished_run["run_id"]
    state = client.get(f"/api/seqeval/runs/{rid}").json()
    traj = state["trajectories"]
    assert "stratum:pedestrian|night" in traj
    t = traj["stratum:pedestrian|night"]
    assert len(t["points"]) >= 2
    p = t["points"][-1]
    for f in ("n", "log_e_regression", "log_e_pass", "delta_lower", "delta_upper"):
        assert f in p
    assert t["boundaries"]["log_e_regression"] > 0  # the stopping boundary to plot
    # evidence must be increasing in samples for a real regression
    assert t["points"][-1]["n"] >= t["points"][0]["n"]


def test_evidence_endpoint(client, finished_run):
    rid = finished_run["run_id"]
    r = client.get(f"/api/seqeval/runs/{rid}/evidence")
    assert r.status_code == 200
    body = r.json()
    assert body["records"], "expected ledger records"
    assert body["lineage"]["plan_hash"]
    assert set(body["required_fields"]).issubset(set(body["records"][0].keys()))


def test_attribution_endpoint(client, finished_run):
    rid = finished_run["run_id"]
    r = client.get(f"/api/seqeval/runs/{rid}/attribution")
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "REGRESSION"
    assert "stratum:pedestrian|night" in body["affected_strata"]
    row = body["regression_map"][0]
    for f in ("baseline_value", "candidate_value", "abs_delta", "rel_delta",
              "delta_ci", "n", "n_effective", "decision"):
        assert f in row


def test_runs_listing(client, finished_run):
    r = client.get("/api/seqeval/runs")
    assert r.status_code == 200
    assert any(run["run_id"] == finished_run["run_id"] for run in r.json()["runs"])


def test_policy_endpoint(client):
    r = client.get("/api/seqeval/policy")
    assert r.status_code == 200
    body = r.json()
    assert body["default_policy"]["delta_margin"] == 0.005
    assert body["default_policy"]["alpha"] == 0.05
    assert "REGRESSION" in body["decision_semantics"]
    assert "anytime-valid" in body["test_method"]


def test_404_on_unknown_run(client):
    assert client.get("/api/seqeval/runs/seq-nope").status_code == 404
    assert client.get("/api/seqeval/runs/seq-nope/evidence").status_code == 404
