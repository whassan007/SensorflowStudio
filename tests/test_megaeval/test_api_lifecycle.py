"""Async run lifecycle, the query API (cache behavior + routing), 409 gating,
containers/forensic drill-down, review endpoints and performance SLOs —
all through the real HTTP API."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(mega_env):
    from app_backend import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def api_run(client, mega_env):
    """Create a fresh run via the API and wait for the async lifecycle to publish."""
    pop_id = mega_env["meta"]["population_id"]
    res = client.post("/api/megaeval/runs",
                      json={"population_id": pop_id, "model_version": "model-api",
                            "worker_delay_s": 0.05, "workers": 4})
    assert res.status_code == 200
    run = res.json()
    assert run["status"] in ("queued", "running")

    seen_statuses = {run["status"]}
    deadline = time.time() + 60
    while time.time() < deadline:
        r = client.get(f"/api/megaeval/runs/{run['run_id']}").json()
        seen_statuses.add(r["status"])
        if r["status"] in ("published", "failed"):
            break
        time.sleep(0.2)
    assert r["status"] == "published", r.get("error")
    return {"run": r, "seen_statuses": seen_statuses}


def test_async_lifecycle_and_progress(api_run):
    r = api_run["run"]
    assert api_run["seen_statuses"] & {"queued", "running", "reducing", "materializing"}
    assert r["percent"] == 100.0
    assert r["objects_processed"] == r["objects_total"] > 0
    assert r["throughput_objs_per_s"] > 0
    assert r["headline"]["n"] == r["objects_total"]
    lin = r["lineage"]
    for field in ("evaluation_id", "dataset_version", "model_version", "model_checkpoint",
                  "label_version", "evaluator_code_version", "metric_version",
                  "threshold_config", "sampling_config", "seed", "hardware", "timestamp"):
        assert field in lin


def test_results_blocked_until_published(client, mega_env):
    pop_id = mega_env["meta"]["population_id"]
    res = client.post("/api/megaeval/runs",
                      json={"population_id": pop_id, "model_version": "model-slow",
                            "worker_delay_s": 0.4, "workers": 2})
    run_id = res.json()["run_id"]
    blocked = client.get(f"/api/megaeval/runs/{run_id}/funnel")
    assert blocked.status_code == 409
    deadline = time.time() + 90
    while time.time() < deadline:
        if client.get(f"/api/megaeval/runs/{run_id}").json()["status"] == "published":
            break
        time.sleep(0.3)
    assert client.get(f"/api/megaeval/runs/{run_id}/funnel").status_code == 200


def test_query_api_cache_and_latency_slos(client, api_run):
    run_id = api_run["run"]["run_id"]
    body = {"evaluation_id": run_id,
            "filters": {"class": ["pedestrian"], "lighting": ["night"]},
            "metrics": ["n", "recall", "precision", "mean_iou"],
            "group_by": []}
    r1 = client.post("/api/evaluations/query", json=body).json()
    assert r1["meta"]["source"] == "cube"
    assert r1["meta"]["cache_hit"] is False
    assert r1["meta"]["exact"] is True
    assert r1["rows"][0]["n"] > 0
    assert r1["meta"]["latency_ms"] < 3000  # filtered cohort query SLO

    r2 = client.post("/api/evaluations/query", json=body).json()
    assert r2["meta"]["cache_hit"] is True
    assert r2["meta"]["source"] == "cache"
    assert r2["meta"]["latency_ms"] < 100   # cached query SLO
    assert r2["rows"] == r1["rows"]

    dash = client.post("/api/evaluations/query",
                       json={"evaluation_id": run_id, "group_by": ["class"],
                             "metrics": ["n", "precision", "recall", "f1"]}).json()
    assert dash["meta"]["latency_ms"] < 2000  # aggregate dashboard SLO
    assert len(dash["rows"]) == 6

    stats = client.get("/api/megaeval/cache").json()
    assert stats["hits"] >= 1


def test_query_sketch_metrics_marked_approximate(client, api_run):
    run_id = api_run["run"]["run_id"]
    res = client.post("/api/evaluations/query",
                      json={"evaluation_id": run_id,
                            "metrics": ["n", "recall", "confidence_p50", "iou_p90"]}).json()
    assert res["meta"]["exact"] is False
    assert set(res["meta"]["approximate_fields"]) == {"confidence_p50", "iou_p90"}
    assert 0 <= res["rows"][0]["confidence_p50"] <= 1


def test_containers_sort_presets_and_forensic_drilldown(client, api_run):
    run_id = api_run["run"]["run_id"]
    worst = client.get(f"/api/megaeval/runs/{run_id}/containers",
                       params={"sort": "worst_recall", "limit": 20}).json()
    assert worst["total"] > 0 and len(worst["rows"]) == 20
    recalls = [r["recall"] for r in worst["rows"] if r["recall"] is not None]
    assert recalls == sorted(recalls)

    risky = client.get(f"/api/megaeval/runs/{run_id}/containers",
                       params={"sort": "highest_risk", "limit": 5}).json()["rows"]
    risks = [r["risk_score"] for r in risky]
    assert risks == sorted(risks, reverse=True)
    assert risky[0]["status"] in ("ok", "warn", "critical")

    t0 = time.perf_counter()
    objs = client.get(
        f"/api/megaeval/runs/{run_id}/containers/{risky[0]['container_id']}/objects").json()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 1000  # object investigation SLO
    assert objs["objects"], "forensic drill-down must return annotation-level rows"
    outcomes = {o["outcome"] for o in objs["objects"]}
    assert outcomes <= {"TP", "FN", "FP", "LOCALIZATION"}
    gt_objs = [o for o in objs["objects"] if o["outcome"] != "FP"]
    assert len(gt_objs) == risky[0]["n_objects"]


def test_review_endpoints_full_cycle(client, api_run):
    run_id = api_run["run"]["run_id"]
    state = client.get(f"/api/megaeval/runs/{run_id}/review").json()
    assert state["planned"] is False

    plan = client.post(f"/api/megaeval/runs/{run_id}/review/plan",
                       json={"target_n": 600}).json()
    assert plan["planned"] and not plan["executed"]
    assert plan["funnel"]["statistically_selected"] > 0

    done = client.post(f"/api/megaeval/runs/{run_id}/review/execute").json()
    assert done["executed"]
    res = done["results"]
    for metric in ("precision", "recall"):
        est = res[metric]
        assert est["ci_low"] < est["estimate"] < est["ci_high"]

    funnel = client.get(f"/api/megaeval/runs/{run_id}/funnel").json()
    stages = {s["stage"]: s["count"] for s in funnel["stages"]}
    assert stages["Human Verified"] == done["funnel"]["reviewed"]
    assert funnel["estimated_precision"]["estimate"] is not None


def test_error_search_compare_shift_why_similarity_endpoints(client, api_run, mega_env):
    run_id = api_run["run"]["run_id"]
    es = client.post("/api/megaeval/errors/search",
                     json={"run_id": run_id, "error_types": ["FN"],
                           "filters": {"class": ["pedestrian"]},
                           "risk_min": 0.5}).json()
    assert es["matched_errors"] > 0 and es["worst_containers"]

    cmp = client.post("/api/megaeval/compare",
                      json={"candidate_run_id": mega_env["bad"].run_id,
                            "baseline_run_id": run_id}).json()
    assert cmp["recommendation"] == "DO_NOT_PROMOTE"

    sh = client.get(f"/api/megaeval/runs/{run_id}/shift").json()
    assert sh["shifts"]

    why = client.post("/api/megaeval/why",
                      json={"run_id": run_id, "filters": {"lighting": ["night"]},
                            "metric": "recall"}).json()
    assert why["failure_count"] > 0 and why["factors"]

    cid = es["worst_containers"][0]["container_id"]
    sim = client.post("/api/megaeval/similarity",
                      json={"run_id": run_id, "container_id": cid, "k": 5}).json()
    assert len(sim["results"]) == 5

    dist = client.get(f"/api/megaeval/runs/{run_id}/distributions").json()
    assert dist["exact"] is False
    assert "confidence" in dist and "50" in dist["confidence"]["percentiles"]
    assert dist["containers_exact"] > 0
