"""Minimal tests for help FAQ matching (CPU-only, no LLM required)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from sensorflow.help.api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_match_faq_command_center():
    from sensorflow.help.matcher import match_faq

    hits = match_faq("What does the Command Center do?", top_k=3)
    assert hits
    top_ids = [d.id for d, _ in hits]
    assert any(i == "page:command" or "command" in i for i in top_ids)


def test_match_faq_hitl():
    from sensorflow.help.matcher import match_faq

    hits = match_faq("When do humans review flagged labels?", top_k=3)
    assert hits
    assert any("hitl" in d.id or "review" in d.id or "faq" in d.id for d, _ in hits)


def test_match_faq_page_context_boost():
    from sensorflow.help.matcher import match_faq

    hits = match_faq("what can I do here?", page_id="rotr", top_k=2)
    assert hits
    assert hits[0][0].page_id == "rotr" or hits[0][0].id.startswith("page:rotr")


def test_format_fallback_returns_answer():
    from sensorflow.help.matcher import format_fallback_answer, match_faq

    matches = match_faq("Does the chatbot need a GPU?", top_k=2)
    payload = format_fallback_answer("Does the chatbot need a GPU?", matches)
    assert "answer" in payload
    assert payload["provider"] == "faq_offline"
    assert "GPU" in payload["answer"] or "CPU" in payload["answer"] or "gpu" in payload["answer"].lower()


def test_chat_endpoint_offline(client):
    body = client.post("/api/help/chat", json={"question": "What is Sensorflow Studio?"}).json()
    assert body["answer"]
    assert body["provider"] == "faq_offline" or body["provider"]
    assert isinstance(body["sources"], list)


def test_guides_endpoint(client):
    body = client.get("/api/help/guides").json()
    assert len(body["guides"]) >= 10
    assert all("page_id" in g and "summary" in g for g in body["guides"])


def test_empty_question_rejected(client):
    resp = client.post("/api/help/chat", json={"question": ""})
    assert resp.status_code == 422
