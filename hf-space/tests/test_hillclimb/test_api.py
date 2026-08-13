"""End-to-end API lifecycle through the FastAPI router (in-process TestClient)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.hillclimb.api import router

from .conftest import STRONG_LEADERSHIP_ANSWER, STRONG_TECH_ANSWER, WEAK_ANSWER


@pytest.fixture()
def client(isolated_store):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_full_lifecycle(client):
    # profile
    p = client.get("/api/hillclimb/profile").json()
    assert p["user_id"] == "default"
    p["target_role"] = "EM, ML Infrastructure"
    assert client.put("/api/hillclimb/profile", json=p).json()["target_role"] == "EM, ML Infrastructure"

    # blueprint + graph
    bp = client.get("/api/hillclimb/blueprint").json()
    assert bp["source"] == "reconstructed-from-spec"
    assert len(bp["phases"]) == 4
    graph = client.get("/api/hillclimb/graph").json()
    assert graph["problems"] == []
    assert len(graph["nodes"]) >= 30 and graph["edges"]

    # journey starts NOT_STARTED
    assert client.get("/api/hillclimb/journey").json()["state"] == "NOT_STARTED"

    # diagnostic
    dx = client.post("/api/hillclimb/diagnostic/start", json={"seed": 11}).json()
    assert dx["current_question"]["scenario"]
    for _ in range(6):
        dx = client.post(f"/api/hillclimb/diagnostic/{dx['diagnostic_id']}/answer",
                         json={"answer": STRONG_TECH_ANSWER}).json()
    assert dx["status"] == "complete"
    assert client.get("/api/hillclimb/journey").json()["state"] == "LEARNING"

    # exercise generate + submit -> coached, evidence-backed evaluation
    ex = client.post("/api/hillclimb/exercise/generate",
                     json={"competency_id": "p1.regression_detection", "seed": 77}).json()
    sub = client.post("/api/hillclimb/exercise/submit",
                      json={"exercise_id": ex["exercise_id"], "answer": STRONG_TECH_ANSWER}).json()
    assert sub["evaluation"]["score"] >= 4
    assert sub["evaluation"]["evidence"]
    assert sub["coaching"]
    assert sub["linked_tool"]["api"] == "/api/rca"

    # STAR
    star = client.post("/api/hillclimb/star/diagnose", json={
        "text": ("When I joined the team of 12 our deploys failed under a hard deadline. I was "
                 "asked to fix it. I decided to build a gate and convinced the tech lead who "
                 "disagreed. As a result failures dropped from 14% to 3% and we sustained it "
                 "with weekly monitoring since then."),
        "save_evidence": True}).json()
    assert star["evidence_id"]
    assert len(star["components"]) == 4

    # design lab
    challenges = client.get("/api/hillclimb/design/challenges").json()
    assert len(challenges["challenges"]) == 4
    grade = client.post("/api/hillclimb/design/submit", json={
        "challenge_id": "parallel_inference",
        "components": [
            {"id": "s", "type": "source", "name": "traffic", "note": "20000 qps"},
            {"id": "i1", "type": "inference", "name": "gpu pool a"},
            {"id": "i2", "type": "inference", "name": "gpu pool b"},
            {"id": "st", "type": "storage", "name": "results"},
            {"id": "m", "type": "monitoring", "name": "slo monitor"},
        ],
        "edges": [{"source": "s", "target": "i1"}, {"source": "s", "target": "i2"},
                  {"source": "i1", "target": "st"}, {"source": "i2", "target": "st"},
                  {"source": "st", "target": "m"}],
        "rationales": {"batching": "dynamic batching keeps p99 latency under 100 ms with 30% "
                                   "headroom; the tradeoff is cost, and replicated zones give "
                                   "reliability; monitoring alerts on scalability limits and "
                                   "failure handling uses degraded single-model fallback"},
    }).json()
    assert grade["structural"]["missing_stages"] == []
    assert grade["overall_score"] >= 2

    # simulation: 3 turns
    sim = client.post("/api/hillclimb/simulation/start", json={"seed": 42, "max_turns": 3}).json()
    for hyp, iid in [("Monitoring will improve safety by 5", "add_monitoring"),
                     ("Pausing launches raises safety by 8", "pause_launches"),
                     ("1:1s will recover team morale by 5", "team_1on1s")]:
        sim = client.post(f"/api/hillclimb/simulation/{sim['sim_id']}/step",
                          json={"hypothesis": hyp, "intervention_id": iid}).json()
    assert sim["status"] == "complete"
    assert sim["debrief"]["competency_mappings"]
    catalog = client.get("/api/hillclimb/simulation/catalog").json()
    assert catalog["interventions"] and catalog["scenarios"]

    # interview: 2 questions
    iv = client.post("/api/hillclimb/interview/start", json={"mode": "management", "seed": 5}).json()
    iv = client.post(f"/api/hillclimb/interview/{iv['session_id']}/answer",
                     json={"answer": WEAK_ANSWER}).json()
    assert iv["turns"][-1]["question_type"] == "probe"
    iv = client.post(f"/api/hillclimb/interview/{iv['session_id']}/answer",
                     json={"answer": STRONG_LEADERSHIP_ANSWER}).json()
    iv = client.post(f"/api/hillclimb/interview/{iv['session_id']}/end").json()
    assert iv["status"] == "complete" and iv["evidence_id"]

    # evidence library reflects everything above
    ev = client.get("/api/hillclimb/evidence").json()
    types = {e["artifact_type"] for e in ev["evidence"]}
    assert {"star_story", "design_submission", "interview_transcript",
            "exercise_attempt", "diagnostic"} <= types

    # readiness matrix + bottleneck + next best action
    rd = client.get("/api/hillclimb/readiness").json()
    assert rd["matrix"] and rd["dimensions"]
    assert any(r["knowledge_score"] > 0 for r in rd["matrix"])
    nba = client.get("/api/hillclimb/next-action").json()
    assert nba["concept"] and nba["exercise"] and nba["assessment"]


def test_error_paths(client):
    assert client.get("/api/hillclimb/exercise/nope").status_code == 404
    assert client.post("/api/hillclimb/exercise/generate",
                       json={"competency_id": "bogus"}).status_code == 400
    assert client.post("/api/hillclimb/star/diagnose", json={"text": "  "}).status_code == 400
    assert client.get("/api/hillclimb/simulation/nope").status_code == 404
    assert client.post("/api/hillclimb/interview/start",
                       json={"mode": "bogus"}).status_code == 400
    assert client.post("/api/hillclimb/journey/advance",
                       json={"event": "assessment_result", "passed": True}).status_code == 400


def test_blueprint_edit_validation(client):
    bp = client.get("/api/hillclimb/blueprint").json()
    bp["competencies"][0]["prerequisites"] = ["does.not.exist"]
    res = client.put("/api/hillclimb/blueprint", json=bp)
    assert res.status_code == 400
    assert "unknown prerequisite" in str(res.json())
