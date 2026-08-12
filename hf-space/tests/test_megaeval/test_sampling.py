"""Review sampling: funnel structure, CI correctness (must cover the true
population metric), and cube review-counter updates."""

from __future__ import annotations

import pytest

from sensorflow.megaeval import cube as cube_mod
from sensorflow.megaeval import sampling


@pytest.fixture(scope="module")
def reviewed(mega_env):
    store, run = mega_env["store"], mega_env["good"]
    plan = sampling.build_review_plan(store, run, target_n=1200)
    result = sampling.execute_reviews(store, run)
    return {"plan": plan, "result": result, "run": run, "store": store}


def test_sampling_funnel_shape(reviewed):
    f = reviewed["result"]["funnel"]
    assert f["population_objects"] > f["containers"] > 0
    assert f["suspicious_containers"] <= f["containers"]
    # per-stratum rounding can overshoot by at most one unit per stratum (10 strata)
    assert f["statistically_selected"] <= reviewed["result"]["target_n"] + 10
    assert f["reviewed"] == f["statistically_selected"]


def test_stratified_ci_covers_true_metrics(reviewed):
    run = reviewed["run"]
    res = reviewed["result"]["results"]
    slack = 0.02  # human-fidelity noise (98.5%) can bias estimates slightly
    for metric in ("precision", "recall"):
        true = run.headline[metric]
        est = res[metric]
        assert est["ci_low"] < est["estimate"] < est["ci_high"]
        assert est["ci_low"] - slack <= true <= est["ci_high"] + slack, (
            f"{metric}: true {true} outside CI [{est['ci_low']}, {est['ci_high']}]")
        assert abs(est["estimate"] - true) < 0.06
        assert (est["ci_high"] - est["ci_low"]) < 0.25
        assert est["n_reviewed"] > 0
        assert "stratified" in est["method"]


def test_strata_are_risk_weighted(reviewed):
    """Missed/safety strata must be oversampled relative to their share."""
    rec = reviewed["result"]["results"]["recall"]
    by_label = {s["stratum"]: s for s in rec["strata"]}
    detected = by_label.get("detected")
    missed = by_label.get("missed")
    assert detected and missed
    assert (missed["n"] / max(missed["N"], 1)) > (detected["n"] / max(detected["N"], 1))


def test_cube_review_counters_updated(reviewed):
    store, run = reviewed["store"], reviewed["run"]
    cube = store.artifacts(run.run_id)["cube"]
    rows, _ = cube_mod.aggregate(cube, None, None, ["n", "reviewed", "verified"], 1)
    total = rows[0]
    assert total["reviewed"] > 0
    assert 0 < total["verified"] <= total["reviewed"]
    # containers table too
    containers = store.artifacts(run.run_id)["containers"]
    assert int(containers["reviewed"].sum()) == total["reviewed"]
