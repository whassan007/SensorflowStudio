"""Live verification of every /api/safety/* endpoint through the real app
(in-process fastapi TestClient against app_backend.app — no running servers
are touched)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

LENIENT = {
    "scenario_quality": {"min_geometric_pass_rate": 0.05},
    "coverage": {"min_coverage_rate": 0.0, "min_production_weighted_coverage": 0.0,
                 "min_samples": 5, "max_ci_width": 0.99},
}


@pytest.fixture(scope="module")
def client():
    from app_backend import app
    return TestClient(app)


@pytest.fixture()
def api_env(mega_env, fresh_safety_root, tmp_path):
    """Isolated safety root + labeleval store with one dataset, plus mega runs."""
    from sensorflow.evaluation import synthetic
    from sensorflow.evaluation.records import reset_store

    store = reset_store(tmp_path / "eval")
    ds = synthetic.generate_dataset(store, num_sequences=2, frames_per_sequence=6,
                                    seed=23)
    synthetic.generate_labels(store, ds)
    return {"mega": mega_env, "store": store, "ds": ds}


def test_odd_endpoints(client, api_env):
    assert client.get("/api/safety/odd/taxonomy").status_code == 200

    run_id = api_env["mega"]["good"].run_id
    r = client.get("/api/safety/odd/coverage",
                   params={"run": run_id, "dims": "weather,lighting",
                           "min_samples": 10_000})
    assert r.status_code == 200
    cov = r.json()
    assert cov["run_id"] == run_id and cov["gaps"]

    gap = cov["gaps"][0]
    r = client.post("/api/safety/odd/fill-gap",
                    json={"run": run_id, "cell": gap["cell"], "num_sequences": 1,
                          "frames_per_sequence": 10})
    assert r.status_code == 200
    assert r.json()["objects_added"] > 0

    assert client.get("/api/safety/odd/coverage",
                      params={"run": "eval-nope"}).status_code == 404


def test_gates_and_evidence_endpoints(client, api_env):
    good = api_env["mega"]["good"].run_id
    bad = api_env["mega"]["bad"].run_id

    r = client.get("/api/safety/gates/policy")
    assert r.status_code == 200 and "coverage" in r.json()["policy"]

    r = client.post("/api/safety/gates/policy",
                    json={"overrides": {"safety": {"max_csi_increase_ratio": 0.05}}})
    assert r.status_code == 200
    assert r.json()["policy"]["safety"]["max_csi_increase_ratio"] == 0.05

    r = client.post("/api/safety/gates/evaluate",
                    json={"candidate_run": bad, "baseline_run": good,
                          "policy_overrides": LENIENT})
    assert r.status_code == 200
    res = r.json()
    assert res["decision"] == "BLOCKED"
    assert "regression" in res["blocking_gates"]

    r = client.get(f"/api/safety/gates/result/{bad}")
    assert r.status_code == 200 and r.json()["decision"] == "BLOCKED"

    r = client.get(f"/api/safety/evidence/{bad}")
    assert r.status_code == 200
    assert r.json()["package_id"] == f"sep-{bad}"

    r = client.get(f"/api/safety/evidence/{bad}", params={"format": "markdown"})
    assert r.status_code == 200
    assert "# Safety Evidence Package" in r.json()["markdown"]

    assert client.get("/api/safety/evidence/eval-none").status_code == 404


def test_ssam_endpoints(client, api_env):
    r = client.post("/api/safety/ssam/analyze",
                    json={"scenario": "crossing", "seed": 0, "reaction_delay_s": 0.9})
    assert r.status_code == 200
    body = r.json()
    assert body["aggregate"]["num_conflicts"] >= 1
    assert body["generated"]["simulated"] is True

    # explicit trajectories path
    states = [{"t": round(i * 0.1, 3), "x": 10.0 * i * 0.1, "y": 0.0,
               "speed": 10.0, "heading": 0.0} for i in range(21)]
    still = [{"t": round(i * 0.1, 3), "x": 30.0, "y": 0.0, "speed": 0.0,
              "heading": 0.0} for i in range(21)]
    r = client.post("/api/safety/ssam/analyze", json={"trajectories": [
        {"vehicle_id": "A", "states": states},
        {"vehicle_id": "B", "states": still}]})
    assert r.status_code == 200
    assert r.json()["aggregate"]["num_conflicts"] == 1

    run_id = api_env["mega"]["good"].run_id
    r = client.get("/api/safety/ssam/summary", params={"run": run_id})
    assert r.status_code == 200
    assert r.json()["aggregate_csi"] > 0


def test_calibration_endpoints(client, api_env):
    r = client.post("/api/safety/calibration/validate",
                    json={"mode": "miscalibrated", "rotation_offset_deg": 2.0})
    assert r.status_code == 200
    assert r.json()["status"] == "MISCALIBRATED"

    r = client.get("/api/safety/calibration/status")
    assert r.status_code == 200 and r.json()["status"] == "MISCALIBRATED"

    r = client.post("/api/safety/calibration/validate", json={"mode": "clean"})
    assert r.json()["status"] == "CALIBRATED"


def test_discrepancy_endpoints(client, api_env):
    ds_id = api_env["ds"].dataset_id
    r = client.post("/api/safety/discrepancy/mine",
                    json={"dataset_id": ds_id,
                          "profile": {"extra_miss_rate": 0.5}})
    assert r.status_code == 200
    report = r.json()
    assert report["totals"]["discrepancies"] > 0

    r = client.get("/api/safety/discrepancy/summary", params={"dataset_id": ds_id})
    assert r.status_code == 200
    assert r.json()["dataset_id"] == ds_id

    r = client.get("/api/safety/discrepancy/summary")
    assert r.status_code == 200 and len(r.json()["datasets"]) == 1


def test_scenario_endpoints(client, api_env):
    ds_id = api_env["ds"].dataset_id
    # seed content through discrepancy mining
    client.post("/api/safety/discrepancy/mine",
                json={"dataset_id": ds_id, "profile": {"extra_miss_rate": 0.5}})

    r = client.get("/api/safety/scenarios", params={"source": "discrepancy"})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["total"] > 0
    assert all(s["source"] == "discrepancy" for s in body["scenarios"])

    r = client.post("/api/safety/scenarios/populate")
    assert r.status_code == 200

    r = client.post("/api/safety/scenarios/export", json={"source": "discrepancy"})
    assert r.status_code == 200
    bundle = r.json()
    assert bundle["bundle_format"] == "sensorflow-scenario-bundle/v1"
    assert bundle["count"] == len(bundle["scenarios"]) > 0


def test_semantic_search_endpoint(client, api_env):
    r = client.post("/api/safety/semantic-search",
                    json={"concept": "hesitant vulnerable road user near lane edge",
                          "target": "containers", "k": 8, "use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert body["results"]
    assert body["stage2"]["provider"] == "offline_deterministic"
    assert body["results"][0]["explanations"]["stage2_reasoning"]

    r = client.post("/api/safety/semantic-search",
                    json={"concept": "night rain", "target": "containers",
                          "filters": {"lighting": "night"}, "use_llm": False})
    assert all(x["lighting"] == "night" for x in r.json()["results"])


def test_consensus_endpoint(client, api_env):
    store, ds = api_env["store"], api_env["ds"]
    from sensorflow.evaluation.graders import grade_annotation
    frames = {f.frame_id: f for f in store.where("frames", dataset_id=ds.dataset_id)}
    for ann in [a for a in store.where("annotations", dataset_id=ds.dataset_id)
                if a.bbox_3d][:6]:
        grade_annotation(store, ann, frames[ann.frame_id])

    r = client.get("/api/safety/consensus/summary",
                   params={"dataset_id": ds.dataset_id, "examples": 3})
    assert r.status_code == 200
    body = r.json()
    assert "kendall_tau_confidence_vs_consensus" in body["statistics"]
    assert body["examples"]
    ex = body["examples"][0]
    assert "consensus_score_vector" in ex and "mbr" in ex

    assert client.get("/api/safety/consensus/summary",
                      params={"dataset_id": "ds-none"}).status_code == 404
