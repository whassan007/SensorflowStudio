"""TestClient coverage for /api/hardening endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from sensorflow.hardening.api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_audit_json_served(client):
    body = client.get("/api/hardening/audit").json()
    assert body["findings"]
    assert body["summary"]["critical"] >= 1
    for f in body["findings"]:
        assert f["id"] and f["area"] and f["severity"] and f["refs"]


def test_audit_markdown_served(client):
    resp = client.get("/api/hardening/audit.md")
    assert resp.status_code == 200
    assert "Findings table" in resp.text


def test_readiness_scorecard(client):
    body = client.get("/api/hardening/readiness").json()
    assert body["overall_status"] == "NOT_PRODUCTION_READY"
    assert body["rule"]
    assert len(body["categories"]) >= 6


def test_thresholds_all_have_provenance(client):
    body = client.get("/api/hardening/thresholds").json()
    assert body["config_version"]
    for t in body["thresholds"]:
        assert t["provenance"] in ("FHWA_SSAM_DEFAULT", "ILLUSTRATIVE_THRESHOLD")
        assert t["source"]


def test_interfaces_all_labeled_local_or_mock(client):
    body = client.get("/api/hardening/interfaces").json()
    protocols = {i["protocol"] for i in body["implementations"]}
    assert {"VectorDB", "ObjectStorage", "DistributedCompute", "GPUInference",
            "FeatureCache", "MetadataStore", "ExperimentTracking",
            "Observability"} <= protocols
    for impl in body["implementations"]:
        assert impl["label"] in ("LOCAL", "MOCK")
        assert impl["production_options"]


def test_funnel(client):
    body = client.get("/api/hardening/funnel").json()
    if body["available"]:
        stages = {s["stage"]: s["count"] for s in body["stages"]}
        assert stages["annotations"] >= stages["human_reviews"]
    else:
        assert "note" in body


def test_demo_sampling_labeled_and_reweighted(client):
    body = client.get("/api/hardening/demo/sampling").json()
    assert body["simulated"] is True
    true_rate = body["true_population_rate"]
    assert abs(body["ht_reweighted_estimate"] - true_rate) < \
        abs(body["naive_sample_mean"] - true_rate)


def test_demo_sampling_deterministic(client):
    a = client.get("/api/hardening/demo/sampling", params={"seed": 5}).json()
    b = client.get("/api/hardening/demo/sampling", params={"seed": 5}).json()
    assert a == b


def test_demo_quality_routes_required_fixtures(client):
    body = client.get("/api/hardening/demo/quality").json()
    r = body["routing"]
    assert r["clean_vehicle"]["decision"] == "ACCEPT"
    assert r["vru_miss_in_boundary"]["decision"] in ("QUARANTINE", "HITL")
    assert r["phantom_obstacle"]["decision"] == "QUARANTINE"
    assert r["incomplete_telemetry"]["decision"] != "ACCEPT"
    assert body["grader_dependence_example"]["adjusted_confidence"] < \
        body["grader_dependence_example"]["naive_confidence"]


def test_demo_hitl(client):
    body = client.get("/api/hardening/demo/hitl").json()
    ranked_ids = [r["item_id"] for r in body["pareto_then_product"]]
    assert "max_risk_certain" in ranked_ids
    assert body["acceptance_metrics"]["critical_miss_rate"] is not None


def test_demo_power_monotone_tiers(client):
    body = client.get("/api/hardening/demo/power").json()
    ns = [t["n_stream_items"] for t in body["tiers"]]
    assert ns == sorted(ns)


def test_summary(client):
    body = client.get("/api/hardening/summary").json()
    assert body["summary"] and body["strengths"]
    assert body["readiness"]["overall_status"] == "NOT_PRODUCTION_READY"
