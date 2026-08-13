"""Shared fixtures: isolated runs/studio2 root + a fresh registry."""

from __future__ import annotations

import pytest

from sensorflow.studio2 import store as studio2_store
from sensorflow.studio2.registry import Registry


@pytest.fixture(autouse=True)
def studio2_env(tmp_path):
    studio2_store.set_studio2_root(str(tmp_path / "studio2"))
    yield
    studio2_store.set_studio2_root("runs/studio2")


@pytest.fixture()
def registry():
    return Registry()


# ------------------------------------------------------------------ planted
# release-gate inputs shaped exactly like the real engines' outputs


@pytest.fixture()
def passing_safety():
    return {"decision": "RELEASE_READY", "blocking_gates": [],
            "candidate_run_id": "eval-test", "baseline_run_id": "eval-base",
            "evidence_package_id": "sep-eval-test", "gates": []}


@pytest.fixture()
def blocked_safety():
    return {"decision": "BLOCKED", "blocking_gates": ["safety"],
            "candidate_run_id": "eval-test", "baseline_run_id": "eval-base",
            "evidence_package_id": "sep-eval-test", "gates": []}


@pytest.fixture()
def passing_seqeval():
    return {"run_id": "seq-test", "gate": "ALLOW", "decision": "PASS",
            "stopping_reason": "pass_confirmed", "samples_used": 4200,
            "affected_strata": [], "regression_map": []}


@pytest.fixture()
def regressed_seqeval():
    return {"run_id": "seq-test", "gate": "BLOCK", "decision": "REGRESSION",
            "stopping_reason": "regression_confirmed", "samples_used": 6100,
            "affected_strata": ["stratum:pedestrian|night"],
            "regression_map": [{"stratum": "pedestrian|night", "delta": -0.05}]}


@pytest.fixture()
def clean_shift():
    return {"run_id": "eval-test", "shifts": []}


@pytest.fixture()
def bad_shift():
    return {"run_id": "eval-test", "shifts": [
        {"cohort": "pedestrian/fog/night", "relative_change": 1.6,
         "recall_gap": -0.11, "eval_count": 900}]}
