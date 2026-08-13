"""Gauntlet scheduler: priority order, early stopping, budget, caching."""

from __future__ import annotations

from sensorflow.nextgen import scheduler as sched
from sensorflow.nextgen.cache import get_feature_cache


def test_priority_order_respected_without_regressions():
    g = sched.run_gauntlet(effects={}, seed=11, persist=False)
    order = g["processed_order"]
    expected = [s for s in sched.PRIORITY_ORDER]
    assert order == expected
    assert g["scale"]["total_units_defined"] >= 100_000
    assert g["status"] == "completed"


def test_early_stop_halts_on_planted_catastrophic_regression():
    g = sched.run_gauntlet(effects={"safety_critical": -0.08}, seed=11,
                           persist=False)
    assert g["halted"] is True
    assert g["events"][0]["event"] == "CATASTROPHIC_HALT"
    assert g["recommendation"]["recommendation"] == "DO_NOT_LAUNCH"
    # only the safety-critical stratum was evaluated before the halt
    assert [s["stratum"] for s in g["strata"]] == ["safety_critical"]
    # early stopping saved most of the stratum's units
    s0 = g["strata"][0]
    assert s0["units_evaluated"] < s0["units_available"]
    assert s0["decision"] == "REGRESSION"


def test_regression_promotes_related_strata():
    g = sched.run_gauntlet(effects={"new_odd": -0.03}, seed=11, persist=False)
    ev = next(e for e in g["events"]
              if e["event"] == "REGRESSION_PRIORITIZE_RELATED")
    assert ev["stratum"] == "new_odd"
    order = g["processed_order"]
    # distribution_shift (related to new_odd) processed immediately after it
    assert order.index("distribution_shift") == order.index("new_odd") + 1


def test_budget_respected():
    budget = 20_000
    g = sched.run_gauntlet(effects={}, seed=11, persist=False,
                           config={"budget_units": budget})
    assert g["scale"]["units_evaluated"] <= budget
    assert g["scale"]["budget_remaining"] >= 0


def test_rerun_hits_outcome_cache():
    cache = get_feature_cache()
    sched.run_gauntlet(effects={}, seed=13, persist=False)
    hits0 = cache.stats.hits
    g2 = sched.run_gauntlet(effects={}, seed=13, persist=False)
    assert cache.stats.hits > hits0
    assert g2["cache"]["hit_rate"] == 1.0


def test_anytime_valid_decisions_only_and_lineage_present():
    g = sched.run_gauntlet(effects={}, seed=11, persist=False)
    assert "anytime-valid" in g["statistical_validity"]
    for s in g["strata"]:
        assert s["decision"] in ("REGRESSION", "PASS", "INSUFFICIENT_EVIDENCE")
        assert "e_regression" in s  # seqeval e-process snapshot
    assert g["lineage_valid"] is True
    assert g["lineage"]["seeds"] == {"gauntlet": 11}
    # provenance labels of every stratum flow into the recommendation
    assert set(g["recommendation"]["data_labels"]) == {
        "COUNTERFACTUAL", "REPLAYED", "SIMULATED"}
