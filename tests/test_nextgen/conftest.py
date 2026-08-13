"""Shared fixtures: isolated runs/nextgen root + reusable scenario bundles."""

from __future__ import annotations

import pytest

from sensorflow.nextgen import store
from sensorflow.nextgen.cache import reset_feature_cache
from sensorflow.nextgen.models import TransformationStep


@pytest.fixture(autouse=True)
def nextgen_env(tmp_path):
    store.set_nextgen_root(str(tmp_path / "nextgen"))
    reset_feature_cache()
    yield
    store.set_nextgen_root("runs/nextgen")
    reset_feature_cache()


@pytest.fixture()
def emergence_bundle():
    from sensorflow.nextgen import counterfactual as cf
    recipe = [TransformationStep(kind="actors.occluded_emergence",
                                 params={"t_emerge_s": 1.8})]
    scenario = cf.generate_counterfactuals(recipe, seed=3, n_scenarios=1,
                                           frames_per_sequence=50)[0]
    return cf.load_bundle(scenario.scenario_id)
