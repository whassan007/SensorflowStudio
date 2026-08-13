"""Full API lifecycle via TestClient (in-process, no live server needed).

Uses a standalone FastAPI app with only the raremine router so the test does
not depend on other teams' in-progress routers in app_backend.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.raremine.api import router
from sensorflow.raremine.models import reset_store


@pytest.fixture()
def client(tmp_path):
    reset_store(tmp_path)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c
    reset_store()


def _mine(client) -> dict:
    r = client.post("/api/raremine/scenes/generate", json={"n": 60, "seed": 7})
    assert r.status_code == 200
    bank = r.json()["bank"]
    r = client.post("/api/raremine/mine", json={"diversity_budget": 10})
    assert r.status_code == 200
    return {"bank": bank, "run": r.json()}


def test_full_lifecycle(client):
    ctx = _mine(client)
    assert ctx["run"]["run"]["num_track_candidates"] > 0
    assert ctx["run"]["dedup_report"]["duplicates_archived"] >= 1

    # status summary
    status = client.get("/api/raremine/status").json()
    assert status["bank"]["bank_id"] == ctx["bank"]["bank_id"]
    assert status["priority_histogram"]

    # candidate list + filters
    r = client.get("/api/raremine/candidates").json()
    assert r["count"] > 0
    crit = client.get("/api/raremine/candidates", params={"priority": "CRITICAL"}).json()
    assert all(c["candidate"]["curation_priority"] == "CRITICAL" for c in crit["candidates"])

    # detail + scene view
    first = r["candidates"][0]
    tcid = first["track_candidate_id"]
    detail = client.get(f"/api/raremine/candidates/{tcid}").json()
    assert detail["frame_candidates"]
    assert detail["lineage"] is not None
    scene = client.get(f"/api/raremine/candidates/{tcid}/scene").json()
    assert any(o["is_candidate"] for o in scene["objects"])

    # track view + reports
    tracks = client.get("/api/raremine/tracks").json()
    assert tracks["count"] > 0
    assert client.get("/api/raremine/dedup/report").status_code == 200
    assert client.get("/api/raremine/diversity/report").status_code == 200
    assert client.get("/api/raremine/reports/quantval").status_code == 200
    assert client.get("/api/raremine/reports/curator").status_code == 200
    assert client.get("/api/raremine/reports/improvement").status_code == 200

    # approve into a rare-event dataset
    rev = client.post(f"/api/raremine/review/{tcid}",
                      json={"action": "approve", "note": "confirmed",
                            "destination": "RARE_EVENT_DATASET"}).json()
    assert rev["stage"] == "CURATED"
    assert rev["lineage"]["validation_status"] == "APPROVED"

    # reject another
    second = r["candidates"][1]["track_candidate_id"]
    rev = client.post(f"/api/raremine/review/{second}",
                      json={"action": "reject", "note": "statue"}).json()
    assert rev["stage"] == "ARCHIVED"

    # destinations reflect the decisions
    dest = client.get("/api/raremine/destinations").json()
    assert "RARE_EVENT_DATASET" in dest["counts"]

    # explain endpoint always answers (deterministic offline fallback)
    ex = client.post(f"/api/raremine/candidates/{tcid}/explain").json()
    assert ex["status"] == "ok"
    assert "Separate confidences" in ex["analysis"] or ex["provider"] != "offline_deterministic"


def test_leakage_guard_over_api(client):
    ctx = _mine(client)
    cands = client.get("/api/raremine/candidates").json()["candidates"]
    tcid = cands[0]["track_candidate_id"]

    # approve into a PROTECTED evaluation set
    rev = client.post(f"/api/raremine/review/{tcid}",
                      json={"action": "approve",
                            "destination": "SAFETY_CRITICAL_EVALUATION_SET"}).json()
    assert rev["lineage"]["protected_evaluation"] is True
    assert rev["lineage"]["training_eligible"] is False

    # silent training promotion must be blocked with 403
    r = client.post("/api/raremine/governance/promote-training",
                    json={"track_candidate_id": tcid})
    assert r.status_code == 403
    assert "protected evaluation set" in r.json()["detail"]
    lin = client.get(f"/api/raremine/lineage/{tcid}").json()
    assert lin["training_eligible"] is False

    # override without actor/reason fails
    r = client.post("/api/raremine/governance/override",
                    json={"track_candidate_id": tcid, "actor": " ", "reason": ""})
    assert r.status_code == 400

    # explicit override (who + why) then promotion succeeds and is recorded
    r = client.post("/api/raremine/governance/override",
                    json={"track_candidate_id": tcid, "actor": "safety-lead",
                          "reason": "example rotated out of the protected eval set"})
    assert r.status_code == 200
    r = client.post("/api/raremine/governance/promote-training",
                    json={"track_candidate_id": tcid, "curator": "safety-lead"})
    assert r.status_code == 200
    lin = r.json()["lineage"]
    assert lin["training_eligible"] is True
    assert lin["governance_overrides"][0]["actor"] == "safety-lead"


def test_unapproved_promotion_blocked(client):
    _mine(client)
    cands = client.get("/api/raremine/candidates").json()["candidates"]
    tcid = cands[0]["track_candidate_id"]
    r = client.post("/api/raremine/governance/promote-training",
                    json={"track_candidate_id": tcid})
    assert r.status_code == 403
    assert "unverified" in r.json()["detail"]


def test_validation_errors(client):
    _mine(client)
    cands = client.get("/api/raremine/candidates").json()["candidates"]
    tcid = cands[0]["track_candidate_id"]
    assert client.post(f"/api/raremine/review/{tcid}",
                       json={"action": "maybe"}).status_code == 400
    assert client.post(f"/api/raremine/review/{tcid}",
                       json={"action": "approve", "destination": "NOT_A_SET"}).status_code == 400
    assert client.get("/api/raremine/candidates/nope-123").status_code == 404
