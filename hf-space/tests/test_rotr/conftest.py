"""Isolated persistence roots per test (rotr always; studio2/agentic when
importable, since the flywheel/stop-ship mirror into them best-effort)."""

from __future__ import annotations

import pytest

from sensorflow.rotr import store as rotr_store


@pytest.fixture(autouse=True)
def rotr_env(tmp_path):
    rotr_store.set_rotr_root(str(tmp_path / "rotr"))
    restores = []
    try:
        from sensorflow.studio2 import store as s2_store
        old = s2_store.STUDIO2_ROOT
        s2_store.set_studio2_root(str(tmp_path / "studio2"))
        restores.append(lambda: s2_store.set_studio2_root(old))
    except Exception:
        pass
    try:
        from sensorflow.agentic import store as ag_store
        old_ag = ag_store.agentic_root()
        ag_store.set_agentic_root(str(tmp_path / "agentic"))
        restores.append(lambda: ag_store.set_agentic_root(old_ag))
    except Exception:
        pass
    yield
    rotr_store.set_rotr_root("runs/rotr")
    for r in restores:
        r()


@pytest.fixture(scope="session")
def bank_v1():
    """Deterministic in-memory bank (no store dependency)."""
    from sensorflow.rotr.scenes import generate_bank
    return generate_bank(n_scenarios=28, seed=7, model_version="stack-v1")


@pytest.fixture(scope="session")
def detections_v1(bank_v1):
    """Scenario -> violations for the seeded bank (pure, in memory)."""
    from sensorflow.rotr.rules import detect
    return {sc.scenario_id: detect(sc) for sc in bank_v1}
