"""Minimal tests for GET /api/about and GET /api/version."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from sensorflow.about.api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_about_payload(client):
    body = client.get("/api/about").json()
    assert body["name"] == "Sensorflow Studio"
    assert body["version"]
    assert body["description"]
    assert body["links"]["github"].startswith("https://github.com/")
    assert "huggingface.co" in body["links"]["hf_space"]
    assert isinstance(body["releases"], list) and len(body["releases"]) >= 2
    latest = body["releases"][0]
    assert latest["version"] == body["version"]
    assert latest["date"]
    assert latest["title"]
    assert latest["highlights"]
    dates = [r["date"] for r in body["releases"]]
    assert dates == sorted(dates, reverse=True)


def test_version_payload(client):
    about = client.get("/api/about").json()
    version = client.get("/api/version").json()
    assert version["version"] == about["version"]
    assert version["name"] == about["name"]
    assert version["releases"] == about["releases"]
