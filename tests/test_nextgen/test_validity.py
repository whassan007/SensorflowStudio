"""Validity gate: rejection of implausible scenarios + weight-cap policy."""

from __future__ import annotations

from sensorflow.nextgen import counterfactual as cf
from sensorflow.nextgen import validity as v
from sensorflow.nextgen.models import TransformationStep, ValidityReport


def _validate(scenario):
    b = cf.load_bundle(scenario.scenario_id)
    return v.validate_scenario(
        scenario.scenario_id, b["sequence"], b["actors"],
        scenario.environment, scenario.provenance.seed,
        source_overlap_pairs=b["source_overlap_pairs"],
        source_features=b["source_features"])


def test_gate_rejects_planted_physically_implausible_scenario():
    s = cf.generate_counterfactuals(
        [TransformationStep(kind="actors.teleport", params={"jump_m": 30.0})],
        seed=7, n_scenarios=1, frames_per_sequence=40)[0]
    rep = _validate(s)
    assert rep.accepted is False
    assert rep.evaluation_weight == 0.0
    assert any("acceleration" in r or "mismatch" in r for r in rep.reasons)
    phys = next(c for c in rep.checks if c["check"] == "physical_plausibility")
    assert phys["passed"] is False


def test_gate_accepts_plausible_transformation_with_full_scores():
    s = cf.generate_counterfactuals(
        [TransformationStep(kind="actors.occluded_emergence")],
        seed=7, n_scenarios=1, frames_per_sequence=40)[0]
    rep = _validate(s)
    assert rep.accepted is True
    assert rep.evaluation_weight == 1.0
    assert 0.8 <= rep.simulation_fidelity_score <= 1.0
    assert 0.5 <= rep.counterfactual_validity <= 1.0
    assert {c["check"] for c in rep.checks} == {
        "physical_plausibility", "temporal_consistency", "sensor_consistency",
        "identity_trajectory_consistency", "distribution_similarity"}


def test_per_scenario_weight_policy():
    assert v.weight_policy(False, 0.95) == (0.0, False)   # rejected -> 0
    assert v.weight_policy(True, 0.95) == (1.0, False)    # high fidelity
    w, capped = v.weight_policy(True, 0.65)               # low fidelity
    assert capped is True and w == v.LOW_FIDELITY_WEIGHT_CAP


def _report(sid, accepted, fidelity):
    w, capped = v.weight_policy(accepted, fidelity)
    return ValidityReport(scenario_id=sid, accepted=accepted,
                          simulation_fidelity_score=fidelity,
                          counterfactual_validity=fidelity,
                          realism_confidence=fidelity,
                          evaluation_weight=w, weight_capped=capped)


def test_low_fidelity_scenarios_cannot_dominate_suite():
    # 10 low-fidelity accepted + 2 high-fidelity: uncapped, low fidelity
    # would carry 3.0 of 5.0 total weight (60%). The suite policy must cap
    # the low-fidelity share at LOW_FIDELITY_SUITE_SHARE.
    reports = ([_report(f"low-{i}", True, 0.6) for i in range(10)]
               + [_report(f"high-{i}", True, 0.95) for i in range(2)]
               + [_report("rejected", False, 0.9)])
    out = v.apply_suite_weight_policy(reports)
    assert out["scaled_down"] is True
    assert out["low_fidelity_share"] <= v.LOW_FIDELITY_SUITE_SHARE + 1e-9
    assert out["weights"]["rejected"] == 0.0
    assert out["weights"]["high-0"] == 1.0
    assert out["weights"]["low-0"] < v.LOW_FIDELITY_WEIGHT_CAP


def test_suite_policy_leaves_high_fidelity_majorities_alone():
    reports = ([_report(f"high-{i}", True, 0.9) for i in range(8)]
               + [_report("low-0", True, 0.7)])
    out = v.apply_suite_weight_policy(reports)
    assert out["scaled_down"] is False
    assert out["weights"]["low-0"] == v.LOW_FIDELITY_WEIGHT_CAP
