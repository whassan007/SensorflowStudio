"""Phase 3 tests: tool registry permissions, audit completeness, allowlist."""

from __future__ import annotations

import json
import time

import pytest
from pydantic import BaseModel

from sensorflow.retro import store as retro_store
from sensorflow.retro.tools.registry import ToolPermissionError, ToolRegistry


class _EchoIn(BaseModel):
    msg: str


class _EchoOut(BaseModel):
    echoed: str


def test_read_only_is_the_default(registry):
    specs = {s.name: s for s in registry.specs()}
    assert specs["log_reader"].read_only is True
    assert specs["safety_standard_rag"].read_only is True
    assert specs["metric_calculator"].read_only is True
    assert specs["historical_failure_search"].read_only is True
    assert specs["distribution_analysis"].read_only is True
    # exactly one write tool exists
    write_tools = [s for s in specs.values() if not s.read_only]
    assert [t.name for t in write_tools] == ["create_evaluation_case"]


def test_write_tool_requires_explicit_authorization(registry, retro_root):
    args = {"title": "Rain-night pedestrian case",
            "description": "derived from retrospective EVAL-2026-0802-MP01",
            "scenario_tags": ["rain", "night", "pedestrian"],
            "source_evaluation_id": "EVAL-2026-0802-MP01"}
    with pytest.raises(ToolPermissionError):
        registry.call("create_evaluation_case", args)
    denied = [r for r in registry.audit_log if r.status == "denied"]
    assert denied and denied[0].tool == "create_evaluation_case"

    res = registry.call("create_evaluation_case", args, policy_authorization=True)
    assert res.ok
    case_path = res.result["path"]
    assert json.loads(open(case_path).read())["title"] == args["title"]
    authorized = [r for r in registry.audit_log
                  if r.tool == "create_evaluation_case" and r.status == "ok"]
    assert authorized and authorized[0].authorized_write is True


def test_audit_completeness(registry, retro_root):
    registry.call("safety_standard_rag", {"query": "phantom braking", "k": 2})
    with pytest.raises(Exception):
        registry.call("metric_calculator", {"operation": "nope", "params": {}})
    with pytest.raises(KeyError):
        registry.call("not_a_tool", {})

    # every call — ok, error, unknown — is audited with args + timestamp
    assert len(registry.audit_log) == 3
    for rec in registry.audit_log:
        assert rec.timestamp and rec.args is not None and rec.call_id
    ok = registry.audit_log[0]
    assert ok.status == "ok" and len(ok.result_hash) == 64
    assert registry.audit_log[1].status == "error"
    assert registry.audit_log[2].status == "error"

    # persisted to the isolated runs root as jsonl
    persisted = retro_store.read_audit("test-analysis")
    assert len(persisted) == 3


def test_result_hash_is_deterministic(registry, retro_root):
    r1 = registry.call("metric_calculator", {
        "operation": "stopping_distance", "params": {"velocity_mps": 10.0}})
    r2 = registry.call("metric_calculator", {
        "operation": "stopping_distance", "params": {"velocity_mps": 10.0}})
    hashes = [rec.result_hash for rec in registry.audit_log if rec.status == "ok"]
    assert hashes[0] == hashes[1]
    assert r1.result == r2.result


def test_path_allowlist_enforced(registry, tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text("{}")
    with pytest.raises(PermissionError, match="allowlist"):
        registry.call("log_reader", {"path": str(secret)})
    with pytest.raises(PermissionError):
        registry.call("log_reader", {"fixture_id": "../../../etc/passwd"})
    # uploads dir IS allowed
    up = retro_store.uploads_dir() / "ok.json"
    up.write_text(json.dumps({"evaluation_id": "X", "ground_truth": {"class": "car"}}))
    res = registry.call("log_reader", {"path": str(up)})
    assert res.ok and res.result["log"]["evaluation_id"] == "X"


def test_timeout_enforced(retro_root):
    reg = ToolRegistry(analysis_id="timeout-test")

    def slow(_inp: _EchoIn) -> _EchoOut:
        time.sleep(1.0)
        return _EchoOut(echoed="late")

    reg.register("slow_echo", slow, _EchoIn, _EchoOut, "slow", timeout_s=0.1)
    with pytest.raises(TimeoutError):
        reg.call("slow_echo", {"msg": "hi"})
    assert reg.audit_log[-1].status == "timeout"


def test_error_behavior_return_error(retro_root):
    reg = ToolRegistry(analysis_id="errbehavior-test")

    def boom(_inp: _EchoIn) -> _EchoOut:
        raise RuntimeError("kaput")

    reg.register("boom", boom, _EchoIn, _EchoOut, "explodes",
                 error_behavior="return_error")
    res = reg.call("boom", {"msg": "x"})
    assert not res.ok and "kaput" in res.error
    assert reg.audit_log[-1].status == "error"


def test_tool_schemas_declared(registry):
    for spec in registry.specs():
        assert spec.input_schema.get("properties") is not None
        assert spec.output_schema.get("properties") is not None
        assert spec.timeout_s > 0
        assert spec.description


# ---------------------------------------------- megaeval delegation (real data)

@pytest.fixture(scope="module")
def mega_run(tmp_path_factory):
    """A real, published megaeval run in an isolated store (public API only)."""
    from sensorflow.megaeval import population as pop_mod
    from sensorflow.megaeval.runs import get_mega_store, reset_mega_store

    root = tmp_path_factory.mktemp("retro-megaeval")
    pop_mod.set_mega_root(str(root))
    reset_mega_store()
    # 20k objects: large enough that skewed cohorts clear distribution_shift's
    # default min_eval_count=300 / rel_threshold=0.35 (8k leaves zero eligible)
    meta = pop_mod.generate_population("retro-delegation-pop",
                                       num_objects=20_000, seed=23)
    store = get_mega_store()
    run = store.create_run(population_id=meta["population_id"],
                           model_version="retro-delegation-model",
                           worker_delay_s=0.0)
    store.execute_sync(run)
    assert run.status == "published", run.error

    yield run

    pop_mod.set_mega_root("runs/megaeval")
    reset_mega_store()


def test_distribution_analysis_delegates_to_megaeval(registry, mega_run):
    """The SUCCESS path: import + call chain into megaeval actually works."""
    res = registry.call("distribution_analysis", {"run_id": mega_run.run_id})
    assert res.ok
    out = res.result
    # delegation succeeded — not the graceful-degradation branch
    assert out["source"] == "sensorflow.megaeval.analysis.distribution_shift"
    assert out["shift"]["run_id"] == mega_run.run_id
    assert "train mix vs eval mix" in out["shift"]["method"]
    # the synthetic training mix is deliberately skewed (night/rain cohorts),
    # so a real run surfaces genuine shifted cohorts
    shifts = out["shift"]["shifts"]
    assert len(shifts) > 0
    for s in shifts:
        assert {"cohort", "train_share", "eval_share",
                "relative_change"} <= set(s)
    assert any("cohort" in f for f in out["findings"])
    # the call is audited like any other tool call
    audited = [r for r in registry.audit_log
               if r.tool == "distribution_analysis" and r.status == "ok"]
    assert audited and len(audited[0].result_hash) == 64


def test_distribution_analysis_degrades_for_unknown_run(registry, mega_run):
    """Graceful degradation is reserved for genuinely missing data."""
    res = registry.call("distribution_analysis", {"run_id": "eval-nonexistent"})
    assert res.ok
    assert res.result["source"] == "megaeval-delegation-failed"
    assert res.result["shift"] is None
    assert "no distribution claim is made" in res.result["findings"][0]
