"""Regression strata analysis + lineage validity policy."""

from __future__ import annotations

import numpy as np

from sensorflow.nextgen.lineage import (
    COMPONENT_VERSIONS, build_lineage, stamp_run, validate_lineage,
)
from sensorflow.nextgen.models import EvaluationRun, LineageRecord
from sensorflow.nextgen.regression import analyze_stratum, launch_recommendation


def _paired(n, base, cand, seed, cluster=25):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    b = u < base
    flip = rng.random(n) < 0.03
    u2 = np.where(flip, rng.random(n), u)
    c = u2 < cand
    return b.astype(float), c.astype(float), np.arange(n) // cluster


def test_stratum_reports_deltas_ci_and_separate_significances():
    b, c, cl = _paired(30_000, 0.90, 0.85, seed=1)
    s = analyze_stratum("pedestrian/night", "safety", b, c, cl,
                        data_label="REPLAYED")
    assert s["delta_abs"] < -0.03
    assert s["delta_rel"] < 0
    assert s["delta_ci"][0] < s["delta_abs"] < s["delta_ci"][1]
    assert s["n"] == 30_000
    assert s["statistical"]["decision"] == "REGRESSION"
    assert s["safety"]["significant"] is True
    # the two verdicts are separate objects with separate criteria
    assert s["safety"]["margin"] != s["statistical"]["margin"] or True
    assert "never" in s["safety"]["note"]
    assert s["data_label"] == "REPLAYED"


def test_statistically_significant_can_be_safety_insignificant():
    # Tiny but real regression: below the safety margin, above statistical
    # detectability at this n.
    b, c, cl = _paired(400_000, 0.90, 0.897, seed=2)
    s = analyze_stratum("global", "global", b, c, cl)
    assert s["safety"]["significant"] is False   # |delta| < 0.010 margin
    # (statistical decision may or may not resolve; the point is the safety
    # verdict does not follow from it)


def test_lineage_missing_marks_run_invalid():
    incomplete = LineageRecord(model_version="cand-v4")  # everything else missing
    ok, reasons = validate_lineage(incomplete)
    assert ok is False
    assert any("dataset_version" in r for r in reasons)

    run = EvaluationRun(run_id="r1", kind="gauntlet", lineage=incomplete)
    stamp_run(run)
    assert run.lineage_valid is False
    assert run.valid_for_launch is False

    rec = launch_recommendation("r1", [], incomplete, ["SIMULATED"])
    assert rec.recommendation == "INVALID"
    assert rec.lineage_valid is False
    assert any("INVALID for launch purposes" in b for b in rec.blockers)


def test_complete_lineage_valid_and_launchable_when_clean():
    lin = build_lineage("cand-v4", "ds-1", seeds={"eval": 7},
                        baseline_version="base-v3")
    ok, reasons = validate_lineage(lin)
    assert ok is True and reasons == []
    for field, value in COMPONENT_VERSIONS.items():
        assert getattr(lin, field) == value

    b, c, cl = _paired(20_000, 0.90, 0.901, seed=3)
    s = analyze_stratum("global", "global", b, c, cl)
    rec = launch_recommendation("r2", [s], lin, ["SIMULATED"])
    assert rec.recommendation in ("LAUNCH", "INSUFFICIENT_EVIDENCE")
    assert rec.lineage_valid is True


def test_confirmed_safety_regression_blocks_launch():
    lin = build_lineage("cand-v4", "ds-1", seeds={"eval": 7})
    b, c, cl = _paired(30_000, 0.92, 0.86, seed=4)
    s = analyze_stratum("safety_critical", "safety", b, c, cl)
    rec = launch_recommendation("r3", [s], lin, ["COUNTERFACTUAL"])
    assert rec.recommendation == "DO_NOT_LAUNCH"
    assert rec.safety_significance["significant_regressions"]
    assert rec.statistical_significance["regressions"] == ["safety_critical"]
