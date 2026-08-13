"""Shared fixtures for the safety layer test suite.

- mega_env: isolated megaeval root with a small population and two published
  runs (a good candidate and a deliberately regressed one) — mirrors
  tests/test_megaeval/conftest.py.
- safety_root: session-scoped isolated runs/safety replacement.
- fresh_safety_root: function-scoped pristine safety root (for tests that need
  an empty scenario DB / no cached CSI / no supplements).
- eval_env: fresh labeleval store with a small synthetic dataset + labels.
"""

from __future__ import annotations

import os

import pytest

from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.runs import get_mega_store, reset_mega_store
from sensorflow.safety import scenario_db
from sensorflow.safety import store as safety_store


@pytest.fixture(scope="session")
def mega_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("megaeval-safety")
    pop_mod.set_mega_root(str(root))
    reset_mega_store()
    meta = pop_mod.generate_population("safety-pop", num_objects=8_000, seed=17)
    store = get_mega_store()

    good = store.create_run(population_id=meta["population_id"],
                            model_version="model-v41", worker_delay_s=0.0)
    store.execute_sync(good)
    assert good.status == "published", good.error

    bad = store.create_run(population_id=meta["population_id"],
                           model_version="model-v42-regressed",
                           overrides={"night_penalty": 0.35, "vru_penalty": 0.12},
                           worker_delay_s=0.0)
    store.execute_sync(bad)
    assert bad.status == "published", bad.error

    yield {"root": root, "meta": meta, "store": store, "good": good, "bad": bad}

    pop_mod.set_mega_root("runs/megaeval")
    reset_mega_store()


@pytest.fixture(scope="session")
def safety_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("safety-root")
    safety_store.set_safety_root(str(root))
    scenario_db.reset_db()
    yield root
    safety_store.set_safety_root(os.path.join("runs", "safety"))
    scenario_db.reset_db()


@pytest.fixture()
def fresh_safety_root(safety_root, tmp_path):
    """Pristine safety root for one test; restores the session root after."""
    fresh = tmp_path / "safety"
    safety_store.set_safety_root(str(fresh))
    scenario_db.reset_db()
    yield fresh
    safety_store.set_safety_root(str(safety_root))
    scenario_db.reset_db()


@pytest.fixture()
def eval_env(tmp_path):
    from sensorflow.evaluation import synthetic
    from sensorflow.evaluation.records import reset_store

    store = reset_store(tmp_path)
    ds = synthetic.generate_dataset(store, num_sequences=2, frames_per_sequence=8,
                                    seed=11)
    synthetic.generate_labels(store, ds)
    return store, ds
