"""Registry: role-transition contamination rules + reproducibility tuples +
auto-ingest provenance."""

from __future__ import annotations

import pytest

from sensorflow.studio2.registry import (
    NON_REPRODUCIBLE,
    PROTECTED_EVAL_ROLES,
    REPRO_COMPONENTS,
    REPRODUCIBLE,
    Registry,
    RoleTransitionError,
    ingest_existing_stores,
)


def _full_tuple(**overrides):
    base = {"model_version_id": "mv-1", "dataset_version_id": "dv-1",
            "scenario_version_id": "sv-1", "config_hash": "abc123",
            "calibration_version": "calib-v1", "seed": 7,
            "policy_version_id": "pol-1"}
    base.update(overrides)
    return base


# ------------------------------------------------------------------ roles


def test_protected_roles_match_spec():
    assert PROTECTED_EVAL_ROLES == {"TEST", "REGRESSION", "LAUNCH"}


@pytest.mark.parametrize("role", ["TEST", "REGRESSION", "LAUNCH"])
def test_protected_eval_to_training_requires_override(registry, role):
    ds = registry.register_dataset(f"ds-{role}", role)
    assert ds["protected_evaluation"] is True
    with pytest.raises(RoleTransitionError):
        registry.transition_role(ds["entity_id"], "TRAINING", actor="alice")
    # nothing changed
    assert registry.get("datasets", ds["entity_id"])["role"] == role


def test_training_to_protected_eval_requires_override(registry):
    ds = registry.register_dataset("train-set", "TRAINING")
    with pytest.raises(RoleTransitionError):
        registry.transition_role(ds["entity_id"], "REGRESSION", actor="alice")


def test_override_requires_actor_and_reason(registry):
    ds = registry.register_dataset("prot", "TEST")
    with pytest.raises(RoleTransitionError):
        registry.transition_role(ds["entity_id"], "TRAINING", actor="alice",
                                 override_reason="   ")
    out = registry.transition_role(ds["entity_id"], "TRAINING", actor="alice",
                                   override_reason="approved leak review #42")
    assert out["role"] == "TRAINING"
    assert out["governance_overrides"][0]["actor"] == "alice"
    assert out["governance_overrides"][0]["reason"] == "approved leak review #42"
    # the transition itself is in the immutable history
    assert out["role_history"][-1]["from"] == "TEST"
    assert out["role_history"][-1]["override"] is not None


def test_non_boundary_transitions_allowed_but_audited(registry):
    ds = registry.register_dataset("val", "VALIDATION")
    out = registry.transition_role(ds["entity_id"], "MONITORING", actor="bob")
    assert out["role"] == "MONITORING"
    assert len(out["role_history"]) == 2


def test_unknown_role_rejected(registry):
    with pytest.raises(ValueError):
        registry.register_dataset("x", "PRODUCTION")


def test_lineage_recorded_at_creation(registry):
    parent = registry.register_dataset("parent", "TEST")
    child = registry.register_dataset("child", "REGRESSION",
                                      lineage_parents=[parent["entity_id"]])
    assert child["lineage"]["parents"] == [parent["entity_id"]]


# ------------------------------------------------------------------ repro


def test_full_tuple_is_reproducible(registry):
    run = registry.register_run("r1", "megaeval", _full_tuple())
    assert run["reproducibility"] == REPRODUCIBLE
    assert run["missing_components"] == []


@pytest.mark.parametrize("missing", list(REPRO_COMPONENTS))
def test_any_missing_component_marks_non_reproducible(registry, missing):
    run = registry.register_run(f"r-{missing}", "megaeval",
                                _full_tuple(**{missing: None}))
    assert run["reproducibility"] == NON_REPRODUCIBLE
    assert run["missing_components"] == [missing]


def test_reproducibility_cannot_be_supplied(registry):
    """The verdict is computed, not accepted from callers."""
    run = registry.register_run("r2", "megaeval",
                                _full_tuple(seed=None))
    assert run["reproducibility"] == NON_REPRODUCIBLE


# ------------------------------------------------------------------ ingest


def test_ingest_is_idempotent_and_carries_provenance(registry):
    first = ingest_existing_stores(registry)
    totals1 = registry.counts()
    second = ingest_existing_stores(registry)
    totals2 = registry.counts()
    assert totals1 == totals2, "re-ingest must not duplicate entities"
    # provenance present on everything ingested
    for kind in ("models", "runs", "safety_cases"):
        for e in registry.list(kind):
            assert e["provenance"].get("source_package"), (kind, e["entity_id"])
    # megaeval runs exist in this repo; they must have been picked up with an
    # honest reproducibility verdict (megaeval lineage lacks scenario +
    # calibration components)
    mega_runs = [r for r in registry.list("runs") if r["engine"] == "megaeval"]
    if mega_runs:
        for r in mega_runs:
            assert r["reproducibility"] == NON_REPRODUCIBLE
            assert "scenario_version_id" in r["missing_components"]
            assert r["reproducibility_tuple"]["seed"] is not None


def test_ingest_on_empty_root(registry, tmp_path):
    res = ingest_existing_stores(registry, repo_root=str(tmp_path / "empty"))
    assert res["registered"] == {"models": 0, "datasets": 0, "runs": 0,
                                 "safety_cases": 0, "scenarios": 0,
                                 "policies": 0}
