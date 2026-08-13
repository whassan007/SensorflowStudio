"""Phase 4 tests: asymmetric severity model, adjudication, launch gate."""

from __future__ import annotations

from sensorflow.retro.policy import (POLICY_CONFIG, SEVERITY_ORDER,
                                     SeverityContext, adjudicate_severity,
                                     compute_severity, launch_gate)
from sensorflow.retro.scorecard import LaunchRecommendation, Severity


def _rank(sev: Severity) -> int:
    return SEVERITY_ORDER.index(sev)


# --------------------------------------------------------- asymmetric costs

def test_benign_distant_fn_ranks_below_phantom_braking_fp():
    """The spec's pinned ordering: NOT every FN outranks every FP."""
    benign_fn = compute_severity(SeverityContext(
        failure_type="FALSE_NEGATIVE", object_class="vehicle",
        distance_m=87.0, stopping_distance_m=8.7,
        closing_velocity_mps=-0.2, ego_speed_mps=8.3))
    phantom_fp = compute_severity(SeverityContext(
        failure_type="FALSE_POSITIVE", object_class="plastic_bag",
        intervention_decel_mps2=6.2, following_gap_s=0.9, ego_speed_mps=24.6))
    assert benign_fn.severity == Severity.BENIGN
    assert phantom_fp.severity == Severity.CRITICAL
    assert _rank(benign_fn.severity) < _rank(phantom_fp.severity)
    assert benign_fn.score < phantom_fp.score


def test_fn_vru_inside_stopping_distance_is_critical():
    r = compute_severity(SeverityContext(
        failure_type="FALSE_NEGATIVE", object_class="pedestrian",
        distance_m=26.0, stopping_distance_m=26.2, closing_velocity_mps=13.9,
        ttc_s=1.69, total_reaction_time_s=0.53, near_miss=True))
    assert r.severity == Severity.CRITICAL
    assert any("stopping" in t for t in r.rule_trace)


def test_fn_collision_is_fatal():
    r = compute_severity(SeverityContext(
        failure_type="FALSE_NEGATIVE", object_class="pedestrian",
        distance_m=10.0, stopping_distance_m=25.0, closing_velocity_mps=12.0,
        collision_occurred=True))
    assert r.severity == Severity.FATAL


def test_fp_gentle_slowdown_without_traffic_is_benign():
    r = compute_severity(SeverityContext(
        failure_type="FALSE_POSITIVE", object_class="traffic_sign",
        intervention_decel_mps2=1.2, following_gap_s=4.2, ego_speed_mps=12.0))
    assert r.severity == Severity.BENIGN


def test_fp_hard_brake_rule_requires_tailgater():
    tailgated = compute_severity(SeverityContext(
        failure_type="FALSE_POSITIVE", intervention_decel_mps2=4.5,
        following_gap_s=1.0, ego_speed_mps=20.0))
    clear_road = compute_severity(SeverityContext(
        failure_type="FALSE_POSITIVE", intervention_decel_mps2=4.5,
        following_gap_s=4.0, ego_speed_mps=20.0))
    assert tailgated.severity == Severity.CRITICAL
    assert _rank(clear_road.severity) < _rank(tailgated.severity)


def test_severity_is_deterministic():
    ctx = SeverityContext(failure_type="FALSE_NEGATIVE", object_class="cyclist",
                          distance_m=18.0, stopping_distance_m=20.0,
                          closing_velocity_mps=9.0, ttc_s=1.8,
                          total_reaction_time_s=0.6)
    results = {compute_severity(ctx).model_dump_json() for _ in range(5)}
    assert len(results) == 1


def test_rule_trace_is_explainable():
    r = compute_severity(SeverityContext(
        failure_type="FALSE_POSITIVE", intervention_decel_mps2=6.0,
        following_gap_s=0.8, ego_speed_mps=25.0))
    assert r.rule_trace[0].startswith("policy retro-policy/")
    assert len(r.rule_trace) >= 4


# ------------------------------------------------------------- adjudication

def test_ai_severity_divergence_recorded_and_policy_wins():
    policy = compute_severity(SeverityContext(
        failure_type="FALSE_POSITIVE", intervention_decel_mps2=1.0,
        following_gap_s=5.0, ego_speed_mps=10.0))
    adj = adjudicate_severity(policy, Severity.CRITICAL)
    assert adj.final_severity == policy.severity  # policy always authoritative
    assert adj.divergence
    assert "overstated" in adj.divergence_note

    agree = adjudicate_severity(policy, policy.severity)
    assert not agree.divergence
    no_proposal = adjudicate_severity(policy, None)
    assert not no_proposal.divergence


# --------------------------------------------------------------- launch gate

def test_gate_insufficient_evidence_never_pass():
    d = launch_gate(Severity.BENIGN, has_evidence_gaps=True,
                    scr_impact=0.001, scr_significant=False)
    assert d.recommendation == LaunchRecommendation.INSUFFICIENT_EVIDENCE
    assert any("never be converted to PASS" in r for r in d.rationale)
    # even a FATAL with gaps stays INSUFFICIENT_EVIDENCE (gaps block first,
    # and certainly never improve the outcome to PASS)
    d2 = launch_gate(Severity.FATAL, has_evidence_gaps=True,
                     scr_impact=None, scr_significant=None)
    assert d2.recommendation != LaunchRecommendation.PASS


def test_gate_critical_fails():
    d = launch_gate(Severity.CRITICAL, has_evidence_gaps=False,
                    scr_impact=0.0, scr_significant=False)
    assert d.recommendation == LaunchRecommendation.FAIL


def test_gate_scr_regression_with_significance_fails():
    tol = POLICY_CONFIG["scr_regression_tolerance"]
    d = launch_gate(Severity.BENIGN, has_evidence_gaps=False,
                    scr_impact=-(tol * 2), scr_significant=True)
    assert d.recommendation == LaunchRecommendation.FAIL
    # same delta without established significance -> conditional, not fail
    d2 = launch_gate(Severity.BENIGN, has_evidence_gaps=False,
                     scr_impact=-(tol * 2), scr_significant=None)
    assert d2.recommendation == LaunchRecommendation.CONDITIONAL_PASS


def test_gate_disruptive_conditional_and_benign_pass():
    assert launch_gate(Severity.DISRUPTIVE, False, 0.0, False).recommendation \
        == LaunchRecommendation.CONDITIONAL_PASS
    assert launch_gate(Severity.BENIGN, False, 0.001, False).recommendation \
        == LaunchRecommendation.PASS


def test_gate_determinism():
    outs = {launch_gate(Severity.DISRUPTIVE, False, -0.001,
                        False).model_dump_json() for _ in range(5)}
    assert len(outs) == 1
