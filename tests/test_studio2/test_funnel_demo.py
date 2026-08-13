"""Funnel honesty + closed-loop demo determinism."""

from __future__ import annotations

from sensorflow.studio2.demo import run_demo
from sensorflow.studio2.funnel import build_funnel


def test_funnel_on_empty_root_reports_unavailable_not_fabricated(registry, tmp_path):
    empty = tmp_path / "empty-repo"
    empty.mkdir()
    f = build_funnel(repo_root=str(empty), registry=registry)
    for stage in f["stages"]:
        if stage["stage"] in ("raw", "selected", "evaluated", "regression"):
            assert stage["available"] is False, stage
            assert stage.get("status") == "UNAVAILABLE"
            assert stage.get("reason")
            assert "data" not in stage
    assert f["model_comparison"]["available"] is False
    assert f["drift"]["available"] is False


def test_funnel_flags_match_data_presence_with_real_run(registry, tmp_path):
    """Same honesty contract, against a root holding a REAL published megaeval
    run. Seeds the run in an isolated root rather than assuming the dev
    checkout's runs/ artifacts exist (they are gitignored, so a clean worktree
    has none)."""
    from sensorflow.megaeval import population as pop_mod
    from sensorflow.megaeval.runs import get_mega_store, reset_mega_store

    root = tmp_path / "repo"
    pop_mod.set_mega_root(str(root / "runs" / "megaeval"))
    reset_mega_store()
    try:
        meta = pop_mod.generate_population("funnel-honesty-pop",
                                           num_objects=2_000, seed=7)
        store = get_mega_store()
        run = store.create_run(population_id=meta["population_id"],
                               model_version="funnel-honesty-model",
                               worker_delay_s=0.0)
        store.execute_sync(run)
        assert run.status == "published", run.error

        f = build_funnel(repo_root=str(root), registry=registry)
        for stage in f["stages"]:
            if stage["available"]:
                assert isinstance(stage.get("data"), dict) and stage["data"], stage
            else:
                assert stage.get("reason"), stage
        # the seeded published run must surface as real evaluated data
        evaluated = next(s for s in f["stages"] if s["stage"] == "evaluated")
        assert evaluated["available"] is True
        assert evaluated["data"]["objects_evaluated"] > 0
    finally:
        pop_mod.set_mega_root("runs/megaeval")
        reset_mega_store()


def test_demo_is_deterministic_and_registers_regression_dataset(registry):
    d1 = run_demo(seed=123, registry=registry, persist=False)
    d2 = run_demo(seed=123, registry=registry, persist=False)

    assert d1["decision"]["status"] == d2["decision"]["status"]
    assert (d1["decision"]["blocking_conditions"]
            == d2["decision"]["blocking_conditions"])
    assert (d1["decision"]["evidence_completeness"]
            == d2["decision"]["evidence_completeness"])
    assert ([s["available"] for s in d1["steps"]]
            == [s["available"] for s in d2["steps"]])

    # the planted pedestrian|night regression must surface as NO_GO or, when
    # a subsystem is unavailable in this environment, at least REVIEW
    assert d1["decision"]["status"] in ("NO_GO", "REVIEW")
    # the failure feeds the flywheel as a protected REGRESSION dataset
    assert d1["regression_dataset"] is not None
    assert d1["regression_dataset"]["role"] == "REGRESSION"
    assert d1["regression_dataset"]["protected_evaluation"] is True
    # evidence tuple fully recorded
    t = d1["decision"]["evidence_tuple"]
    assert "release_policy_version" in t
    for k in ("safety_gates", "sequential_regression", "distribution_shift"):
        assert k in t


def test_demo_steps_report_availability_honestly(registry):
    d = run_demo(seed=99, registry=registry, persist=False)
    names = [s["step"] for s in d["steps"]]
    assert names[0] == "scenario_and_perception"
    assert "release_decision" in names
    for s in d["steps"]:
        if not s["available"]:
            assert s.get("reason"), s
