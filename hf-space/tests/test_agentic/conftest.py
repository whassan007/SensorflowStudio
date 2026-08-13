"""Shared fixtures: isolated agentic storage root + one worked-example run
reused by every test module (the pipeline is deterministic, so sharing is
safe and keeps the suite fast)."""

from __future__ import annotations

import pytest

from sensorflow.agentic import data as data_mod
from sensorflow.agentic import store as store_mod


@pytest.fixture(scope="session", autouse=True)
def agentic_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("agentic_env")
    store_mod.set_agentic_root(str(root))
    data_mod.reset_data_cache()
    yield {"root": root}
    store_mod.set_agentic_root("runs/agentic")
    data_mod.reset_data_cache()


@pytest.fixture(scope="session")
def walkthrough(agentic_env):
    from sensorflow.agentic import worked_example as we_mod
    return we_mod.run_worked_example()


@pytest.fixture(scope="session")
def ped_cone_failure(walkthrough):
    from sensorflow.agentic import pipeline as pipeline_mod
    return pipeline_mod.get_failure(walkthrough["failure_id"])


@pytest.fixture(scope="session")
def ped_cone_state(walkthrough):
    from sensorflow.agentic import pipeline as pipeline_mod
    return pipeline_mod.get_state(walkthrough["failure_id"])
