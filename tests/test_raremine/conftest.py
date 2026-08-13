"""Shared fixtures: an isolated raremine store with a generated scene bank and
one completed mining run (deterministic seed)."""

from __future__ import annotations

import pytest

from sensorflow.raremine import pipeline, scenes
from sensorflow.raremine.models import reset_store


@pytest.fixture(scope="session")
def mined_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("raremine")
    store = reset_store(root)
    bank = scenes.generate_scene_bank(store, n_scenes=60, seed=7)
    run = pipeline.run_full_pipeline(store, bank.bank_id)
    yield {"store": store, "bank": bank, "run": run, "root": root}
    reset_store()  # restore defaults for later suites


@pytest.fixture()
def store(mined_env):
    return mined_env["store"]


@pytest.fixture()
def bank(mined_env):
    return mined_env["bank"]
