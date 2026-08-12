"""Shared fixtures: an isolated megaeval root with a small population and two
published runs (a good candidate and a deliberately regressed one)."""

from __future__ import annotations

import pytest

from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.runs import get_mega_store, reset_mega_store


@pytest.fixture(scope="session")
def mega_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("megaeval")
    pop_mod.set_mega_root(str(root))
    reset_mega_store()
    meta = pop_mod.generate_population("test-pop", num_objects=8_000, seed=17)
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

    # restore defaults for any later suites
    pop_mod.set_mega_root("runs/megaeval")
    reset_mega_store()
