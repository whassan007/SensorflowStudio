"""For EACH planted root cause: run the full diagnostic battery and assert the
scoring board ranks the true cause on top with the right signature findings,
and that the decision tree walks to the right conclusion."""

from __future__ import annotations

import numpy as np
import pytest

from sensorflow.rca.models import ROOT_CAUSES
from sensorflow.rca.scenario import generate_scenario

SEED = 7


def _codes(battery):
    return {f.code for _, (_, fs) in battery.items() for f in fs}


def _rank(board, cause):
    return next(r["rank"] for r in board["rows"] if r["hypothesis"] == cause)


# ------------------------------------------------------------- global shape


@pytest.mark.parametrize("cause", ROOT_CAUSES)
def test_claims_shape(batteries, cause):
    bundle, _, _, _ = batteries[cause]
    m = bundle.meta["measured"]
    assert 0.040 <= m["offline_delta"] <= 0.060, "offline claim should be ~+5pp"
    assert -0.035 <= m["shadow_delta"] <= -0.005, "shadow claim should be ~-2pp"


@pytest.mark.parametrize("cause", ROOT_CAUSES)
def test_determinism(cause):
    a = generate_scenario(cause, seed=SEED)
    b = generate_scenario(cause, seed=SEED)
    assert a.offline["b_correct"].tolist() == b.offline["b_correct"].tolist()
    assert a.shadow["a_correct"].tolist() == b.shadow["a_correct"].tolist()
    assert a.meta["measured"] == b.meta["measured"]


# ------------------------------------------------------- ranking + signatures


@pytest.mark.parametrize("cause", ROOT_CAUSES)
def test_true_cause_ranks_top(batteries, cause):
    _, _, board, _ = batteries[cause]
    assert _rank(board, cause) <= 2, (
        f"{cause} ranked {_rank(board, cause)}: "
        f"{[(r['hypothesis'], r['score']) for r in board['rows'][:3]]}")
    top = board["rows"][0]
    if top["hypothesis"] == cause:
        assert top["auto_confidence"] in ("HIGH", "MEDIUM")


@pytest.mark.parametrize("cause", ROOT_CAUSES)
def test_decision_tree_conclusion(batteries, cause):
    _, _, _, tree = batteries[cause]
    assert tree["conclusion"] == cause
    if cause == "STATISTICAL_NOISE":
        assert tree["conclusion_kind"] == "insufficient_evidence"
    assert tree["path"][-1] == tree["nodes"][
        [n["id"] for n in tree["nodes"]].index(tree["path"][-1])]["id"]


def test_feature_skew_signature(batteries):
    bundle, battery, _, _ = batteries["FEATURE_SKEW"]
    codes = _codes(battery)
    assert any(c.startswith("FP_FEATURE_SKEW:obj_distance_m") for c in codes)
    # The skewed feature must TOP the parity ranking.
    rows = battery["feature_parity"][0]["rows"]
    assert rows[0]["feature"] == "obj_distance_m"
    assert rows[0]["skew_flag"]
    # Median ratio reflects the meters->feet unit error.
    assert 2.5 <= rows[0]["median_ratio"] <= 4.2
    # Regression concentrated where the feature matters (highway).
    assert any(c.startswith("CP_SEGMENT_CONCENTRATED:scene=highway")
               for c in codes)


def test_contamination_signature(batteries):
    _, battery, _, _ = batteries["OFFLINE_CONTAMINATION"]
    codes = _codes(battery)
    assert "OA_LEAKAGE_DUPLICATES" in codes, "leakage scan must fire"
    assert "OA_REPRO_FAIL" in codes, "reproducibility must fail"
    assert "OA_SPLIT_ROW_LEVEL" in codes
    leak = battery["offline_audit"][0]["leakage"]
    assert leak["n_duplicates"] > 0.15 * leak["offline_n"]
    # Leaked units carry the inflation; clean units do not.
    assert leak["dup_delta_pp"] > leak["clean_delta_pp"] + 5.0


def test_noise_signature(batteries):
    _, battery, _, _ = batteries["STATISTICAL_NOISE"]
    codes = _codes(battery)
    assert "SS_INSUFFICIENT_EVIDENCE" in codes
    assert "SS_LOW_ESS" in codes
    assert "POP_VOLUME_LOW" in codes
    sig = battery["statistical_significance"][0]
    assert sig["outcome"] == "insufficient_evidence"
    # CI must span the practical margin on both sides.
    sp = sig["shadow_paired"]
    assert sp["ci_low"] < -0.01 < 0.0 < sp["ci_high"] + 0.01
    assert sp["effective_n"] < 800


def test_true_regression_signature(batteries):
    _, battery, _, _ = batteries["TRUE_MODEL_REGRESSION"]
    codes = _codes(battery)
    assert "SS_SIGNIFICANT_REGRESSION" in codes
    assert "CP_UNIFORM_REGRESSION" in codes
    # All artifact channels clean.
    for clean in ("OA_LEAKAGE_CLEAN", "FP_CLEAN", "SP_CLEAN",
                  "ST_SAMPLE_FAIR", "LI_CLEAN"):
        assert clean in codes, f"{clean} expected for a genuine regression"


def test_distribution_shift_signature(batteries):
    _, battery, _, _ = batteries["DISTRIBUTION_SHIFT"]
    codes = _codes(battery)
    assert "DS_SHIFT_HIGH:time_of_day" in codes
    assert "CP_SIMPSONS_DETECTED" in codes
    # Simpson's: aggregate signs flip while within-segment deltas agree.
    agg = battery["conditional_performance"][0]["aggregate"]
    assert agg["offline_delta_pp"] > 0 > agg["shadow_delta_pp"]
    assert agg["sign_consistency"] >= 0.55
    # Night segments regress in BOTH environments (the per-segment truth).
    rows = battery["conditional_performance"][0]["rows"]
    night = [r for r in rows if r["time_of_day"] == "night"
             and r["interpretation"] != "low_volume"]
    assert night and all(r["offline_delta_pp"] < 0 and r["shadow_delta_pp"] < 0
                         for r in night)


def test_serving_mismatch_signature(batteries):
    _, battery, _, _ = batteries["SERVING_MISMATCH"]
    codes = _codes(battery)
    assert "SP_CONFIG_DIFF:confidence_threshold" in codes
    assert "SP_CONFIG_DIFF:quantization" in codes
    assert "PC_CONF_BAND_CONCENTRATION" in codes
    bands = battery["paired_comparison"][0]["by_band"]
    focus = next(b for b in bands if b["segment"].startswith("[0.35"))
    assert focus["lift"] >= 1.5


def test_label_latency_signature(batteries):
    _, battery, _, _ = batteries["LABEL_LATENCY"]
    codes = _codes(battery)
    assert "LI_MATURE_DIVERGES" in codes
    assert "LI_PROVISIONAL_HIGH" in codes
    li = battery["label_integrity"][0]
    assert li["provisional_fraction"] > 0.3
    # Verdict flips on mature labels: B is actually ahead.
    assert li["mature_delta_pp"] > 0 > li["provisional_delta_pp"]


def test_sampling_bias_signature(batteries):
    _, battery, _, _ = batteries["SAMPLING_BIAS"]
    codes = _codes(battery)
    assert "ST_SELECTION_BIAS" in codes
    sel = battery["shadow_traffic"][0]["selection"]
    # Sampled stream regresses; the rest of the eligible stream does not.
    assert sel["sampled_delta_pp"] < 0 < sel["unsampled_delta_pp"]
    assert sel["difficulty_psi"] >= 0.05


# ------------------------------------------------------ cross-cause exclusion


@pytest.mark.parametrize("cause", ROOT_CAUSES)
def test_no_foreign_critical_signatures(batteries, cause):
    """Discriminating CRITICAL signatures must not fire for other causes."""
    _, battery, _, _ = batteries[cause]
    codes = _codes(battery)
    if cause != "OFFLINE_CONTAMINATION":
        assert "OA_LEAKAGE_DUPLICATES" not in codes
        assert "OA_REPRO_FAIL" not in codes
    if cause != "FEATURE_SKEW":
        assert not any(c.startswith("FP_FEATURE_SKEW") for c in codes)
    if cause != "SERVING_MISMATCH":
        assert not any(c.startswith("SP_CONFIG_DIFF") for c in codes)
    if cause != "LABEL_LATENCY":
        assert "LI_MATURE_DIVERGES" not in codes
    if cause != "SAMPLING_BIAS":
        assert "ST_SELECTION_BIAS" not in codes
    if cause != "DISTRIBUTION_SHIFT":
        assert "CP_SIMPSONS_DETECTED" not in codes


# ------------------------------------------------------------ experiments


@pytest.mark.parametrize("cause", ROOT_CAUSES)
def test_experiments_target_top_hypothesis(batteries, cause):
    from sensorflow.rca import scoring
    bundle, battery, board, _ = batteries[cause]
    rec = scoring.recommend_experiments(bundle, board, battery)
    assert rec["minimum_additional_evidence"]
    assert len(rec["experiments"]) == 5
    ranked = rec["experiments"]
    assert ranked == sorted(ranked, key=lambda d: d["priority"], reverse=True)
    # The top experiment must discriminate the leading hypothesis.
    top_hyp = board["rows"][0]["hypothesis"]
    assert any(top_hyp in d["discriminates"] for d in ranked[:2])
