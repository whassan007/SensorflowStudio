"""Shared fixtures for the RCA workbench tests.

The full diagnostic battery per planted cause is computed once per session
(it is deterministic), and the API tests run against an isolated runs/rca
root in a tmp dir.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sensorflow.rca import diagnostics as dg, scoring, store
from sensorflow.rca.models import ROOT_CAUSES
from sensorflow.rca.scenario import generate_scenario

SEED = 7


@pytest.fixture(scope="session")
def batteries():
    """{cause: (bundle, battery, scoreboard, tree)} computed once."""
    out = {}
    for cause in ROOT_CAUSES:
        bundle = generate_scenario(cause, seed=SEED)
        battery = dg.run_all(bundle)
        board = scoring.build_scoreboard(bundle, battery=battery)
        tree = scoring.evaluate_decision_tree(bundle, battery=battery)
        out[cause] = (bundle, battery, board, tree)
    return out


@pytest.fixture()
def rca_client(tmp_path):
    store.set_rca_root(str(tmp_path / "rca"))
    from sensorflow.rca import api as rca_api
    rca_api._battery_cache.clear()
    app = FastAPI()
    app.include_router(rca_api.router)
    with TestClient(app) as client:
        yield client
    store.set_rca_root("runs/rca")
    rca_api._battery_cache.clear()
