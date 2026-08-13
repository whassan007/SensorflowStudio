"""API lifecycle via TestClient: create -> diagnostics -> stage completion ->
scoreboard -> decision tree -> experiments -> report -> reveal."""

from __future__ import annotations


def _create(client, **kw):
    body = {"cause": "FEATURE_SKEW", "seed": 7}
    body.update(kw)
    res = client.post("/api/rca/investigations", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_meta_routes(rca_client):
    causes = rca_client.get("/api/rca/causes").json()["causes"]
    assert len(causes) == 8
    stages = rca_client.get("/api/rca/stages").json()["stages"]
    assert len(stages) == 13


def test_create_and_list(rca_client):
    inv = _create(rca_client)
    assert inv["scenario_cause"] == "FEATURE_SKEW"
    assert inv["claims"]["offline_delta_pp"] > 4.0
    assert inv["claims"]["shadow_delta_pp"] < -1.0
    listed = rca_client.get("/api/rca/investigations").json()["investigations"]
    assert any(r["id"] == inv["id"] for r in listed)


def test_training_mode_hides_cause_until_reveal(rca_client):
    inv = _create(rca_client, training_mode=True, cause=None, seed=123)
    assert "scenario_cause" not in inv
    got = rca_client.get(f"/api/rca/investigations/{inv['id']}").json()
    assert "scenario_cause" not in got
    reveal = rca_client.post(f"/api/rca/investigations/{inv['id']}/reveal").json()
    assert reveal["cause"] in {
        "TRUE_MODEL_REGRESSION", "DISTRIBUTION_SHIFT", "FEATURE_SKEW",
        "SERVING_MISMATCH", "LABEL_LATENCY", "SAMPLING_BIAS",
        "STATISTICAL_NOISE", "OFFLINE_CONTAMINATION"}
    assert reveal["explanation"]
    got = rca_client.get(f"/api/rca/investigations/{inv['id']}").json()
    assert got["scenario_cause"] == reveal["cause"]


def test_all_diagnostic_endpoints(rca_client):
    inv = _create(rca_client)
    for key in ("comparison_validity", "offline_audit", "population_validation",
                "distribution_shift", "conditional_performance",
                "paired_comparison", "statistical_significance",
                "feature_parity", "serving_parity", "shadow_traffic",
                "label_integrity"):
        res = rca_client.get(
            f"/api/rca/investigations/{inv['id']}/diagnostics/{key}")
        assert res.status_code == 200, f"{key}: {res.text}"
        body = res.json()
        assert body["stage"] == key
        assert body["data"]
        assert isinstance(body["findings"], list) and body["findings"]
    res = rca_client.get(
        f"/api/rca/investigations/{inv['id']}/diagnostics/nope")
    assert res.status_code == 404


def test_stage_ordering_enforced_via_api(rca_client):
    inv = _create(rca_client)
    res = rca_client.post(
        f"/api/rca/investigations/{inv['id']}/stages/3/complete", json={})
    assert res.status_code == 409
    res = rca_client.post(
        f"/api/rca/investigations/{inv['id']}/stages/0/complete", json={})
    assert res.status_code == 200


def test_unknown_ack_flow_via_api(rca_client):
    inv = _create(rca_client)  # FEATURE_SKEW -> CV_UNKNOWN:feature_pipeline_version
    iid = inv["id"]
    rca_client.get(f"/api/rca/investigations/{iid}/diagnostics/comparison_validity")
    res = rca_client.post(
        f"/api/rca/investigations/{iid}/stages/0/complete", json={})
    assert res.status_code == 409
    assert "UNKNOWN" in res.json()["detail"]
    res = rca_client.post(
        f"/api/rca/investigations/{iid}/stages/0/complete",
        json={"acknowledge_unknowns": True, "note": "pipeline team paged"})
    assert res.status_code == 200
    assert res.json()["stage"]["status"] == "complete_with_unknowns"


def test_record_finding_and_scoreboard(rca_client):
    inv = _create(rca_client)
    iid = inv["id"]
    res = rca_client.post(f"/api/rca/investigations/{iid}/findings", json={
        "stage": "feature_parity", "title": "Confirmed with pipeline team",
        "status": "MISMATCH", "severity": "CRITICAL",
        "detail": "fp-2.5 deployed online only",
        "code": "FP_FEATURE_SKEW:human_confirmed"})
    assert res.status_code == 200
    board = rca_client.get(f"/api/rca/investigations/{iid}/scoreboard").json()
    assert board["rows"][0]["hypothesis"] == "FEATURE_SKEW"
    # The human finding contributes to the score via the same rule table.
    top = board["rows"][0]
    assert any(e["finding_id"] and e["code"] == "FP_FEATURE_SKEW:human_confirmed"
               for e in top["evidence_for"])

    res = rca_client.post(f"/api/rca/investigations/{iid}/scoreboard/assess",
                          json={"hypothesis": "FEATURE_SKEW",
                                "confidence": "HIGH", "note": "confirmed"})
    assert res.status_code == 200
    row = next(r for r in res.json()["rows"] if r["hypothesis"] == "FEATURE_SKEW")
    assert row["human_confidence"] == "HIGH"
    assert row["auto_confidence"]  # both assessments are kept


def test_tree_experiments_report(rca_client):
    inv = _create(rca_client)
    iid = inv["id"]
    tree = rca_client.get(f"/api/rca/investigations/{iid}/decision-tree").json()
    assert tree["conclusion"] == "FEATURE_SKEW"
    assert tree["path"]
    exp = rca_client.get(f"/api/rca/investigations/{iid}/experiments").json()
    assert exp["minimum_additional_evidence"]
    rep = rca_client.get(f"/api/rca/investigations/{iid}/report").json()
    assert rep["executive_finding"]["conclusion"] == "FEATURE_SKEW"
    assert rep["markdown"].startswith("# RCA Report")
    assert len(rep["hypothesis_ranking"]) == 8
    assert len(rep["stage_summaries"]) == 11
    assert rep["remediation"]["containment"]


def test_persistence_roundtrip(rca_client):
    """Data survives the CSV round trip: fresh store caches cleared, then the
    scoreboard still ranks correctly."""
    from sensorflow.rca import store
    from sensorflow.rca import api as rca_api
    inv = _create(rca_client)
    store._cache.clear()
    store._data_cache.clear()
    rca_api._battery_cache.clear()
    board = rca_client.get(
        f"/api/rca/investigations/{inv['id']}/scoreboard").json()
    assert board["rows"][0]["hypothesis"] == "FEATURE_SKEW"


def test_recorded_only_scoreboard_evolves(rca_client):
    """The working-hypothesis banner must not leak unvisited stages: with no
    diagnostics run, all 8 hypotheses stay in play; evidence narrows it."""
    inv = _create(rca_client)
    iid = inv["id"]
    board = rca_client.get(
        f"/api/rca/investigations/{iid}/scoreboard?recorded_only=true").json()
    assert len(board["working_hypothesis_set"]) == 8
    for key in ("feature_parity", "offline_audit", "shadow_traffic",
                "label_integrity", "serving_parity"):
        rca_client.get(f"/api/rca/investigations/{iid}/diagnostics/{key}")
    board = rca_client.get(
        f"/api/rca/investigations/{iid}/scoreboard?recorded_only=true").json()
    ws = board["working_hypothesis_set"]
    assert "FEATURE_SKEW" in ws
    assert "OFFLINE_CONTAMINATION" not in ws  # ruled out by clean audit
    assert len(ws) < 8


def test_404s(rca_client):
    assert rca_client.get("/api/rca/investigations/nope").status_code == 404
    assert rca_client.get(
        "/api/rca/investigations/nope/scoreboard").status_code == 404
