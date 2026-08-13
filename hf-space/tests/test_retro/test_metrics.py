"""Deterministic metric tests: stopping-distance parameterization, TTC
validity flags, SCR criticality, behavioral comparison."""

from __future__ import annotations

import pytest

from sensorflow.retro import metrics


# ---------------------------------------------------------- stopping distance

def test_stopping_distance_baseline_case():
    r = metrics.stopping_distance(velocity_mps=20.0, reaction_time_s=1.0,
                                  friction=0.7, grade=0.0)
    # reaction 20m + braking 400/(2*6.8646) = 29.13m
    assert r.reaction_distance_m == pytest.approx(20.0)
    assert r.braking_distance_m == pytest.approx(29.13, abs=0.05)
    assert r.stopping_distance_m == pytest.approx(49.13, abs=0.1)


def test_stopping_distance_latency_charged_against_budget():
    base = metrics.stopping_distance(velocity_mps=15.0, reaction_time_s=0.25)
    lat = metrics.stopping_distance(velocity_mps=15.0, reaction_time_s=0.25,
                                    system_latency_s=0.3, planner_latency_s=0.1)
    assert lat.total_reaction_time_s == pytest.approx(0.65)
    assert lat.stopping_distance_m - base.stopping_distance_m == pytest.approx(
        15.0 * 0.4, abs=0.02)


def test_stopping_distance_grade_and_friction():
    flat = metrics.stopping_distance(velocity_mps=20.0, friction=0.5, grade=0.0)
    downhill = metrics.stopping_distance(velocity_mps=20.0, friction=0.5, grade=-0.08)
    uphill = metrics.stopping_distance(velocity_mps=20.0, friction=0.5, grade=0.08)
    assert downhill.stopping_distance_m > flat.stopping_distance_m > uphill.stopping_distance_m
    wet = metrics.stopping_distance(velocity_mps=20.0, friction=0.3)
    dry = metrics.stopping_distance(velocity_mps=20.0, friction=0.9)
    assert wet.stopping_distance_m > dry.stopping_distance_m


def test_stopping_distance_explicit_decel_and_impossible_grade():
    explicit = metrics.stopping_distance(velocity_mps=20.0, decel_mps2=5.0)
    assert explicit.effective_decel_mps2 == pytest.approx(5.0)
    impossible = metrics.stopping_distance(velocity_mps=20.0, friction=0.05,
                                           grade=-0.2)
    assert impossible.stopping_distance_m is None
    assert "undefined" in impossible.unknown_reason


def test_stopping_distance_missing_velocity_is_unknown():
    r = metrics.stopping_distance(velocity_mps=None)
    assert r.stopping_distance_m is None
    assert "UNKNOWN" in r.unknown_reason


# ------------------------------------------------------------------------ TTC

def test_ttc_closing_matches_closed_form():
    r = metrics.time_to_collision(distance_m=30.0, closing_velocity_mps=10.0,
                                  ego_length_m=4.5, obj_length_m=0.5)
    # gap = 30 - (4.5+0.5)/2 = 27.5 -> 2.75 s
    assert r.ttc_s == pytest.approx(2.75, abs=0.05)
    assert r.closed_form_ttc_s == pytest.approx(2.75, abs=0.01)
    assert any("constant-velocity" in f for f in r.validity_flags)
    assert any("ssam_ext" in f for f in r.validity_flags)


def test_ttc_not_closing_flagged_undefined():
    r = metrics.time_to_collision(distance_m=30.0, closing_velocity_mps=-2.0)
    assert r.ttc_s is None
    assert any("not closing" in f for f in r.validity_flags)
    assert r.unknown_reason is None  # defined situation, not missing data


def test_ttc_missing_telemetry_unknown():
    r = metrics.time_to_collision(distance_m=None, closing_velocity_mps=5.0)
    assert r.ttc_s is None
    assert "UNKNOWN" in r.unknown_reason


def test_ttc_beyond_lookahead_flagged():
    r = metrics.time_to_collision(distance_m=200.0, closing_velocity_mps=1.0,
                                  look_ahead_s=5.0)
    assert r.ttc_s is None
    assert any("look-ahead" in f for f in r.validity_flags)


# ------------------------------------------------------------------------ SCR

def test_scr_impact_computation():
    r = metrics.scr_impact({"critical_object_count": 1000,
                            "baseline_missed_critical": 2,
                            "candidate_missed_critical": 5})
    assert r.scr_baseline == pytest.approx(0.998)
    assert r.scr_candidate == pytest.approx(0.995)
    assert r.scr_impact == pytest.approx(-0.003)
    assert r.denominator == 1000
    assert "stopping distance" in r.criticality_policy


def test_scr_impact_unknown_without_denominator():
    r = metrics.scr_impact({"baseline_missed_critical": 2})
    assert r.scr_impact is None
    assert "UNKNOWN" in r.unknown_reason
    assert metrics.scr_impact(None).scr_impact is None


# ---------------------------------------------------------- behavioral impact

def test_behavioral_impact_phantom_intervention():
    r = metrics.behavioral_impact(
        {"action": "emergency_brake", "decel_mps2": 6.2, "response_time_s": 0.4},
        {"action": "maintain_speed", "decel_mps2": 0.0, "response_time_s": 0.4})
    assert r.action_changed and r.decel_delta_mps2 == pytest.approx(6.2)
    assert "unwarranted intervention" in r.consequence


def test_behavioral_impact_delayed_response():
    r = metrics.behavioral_impact(
        {"action": "emergency_brake", "decel_mps2": 8.0, "response_time_s": 2.4},
        {"action": "controlled_brake", "decel_mps2": 3.5, "response_time_s": 0.5})
    assert r.response_delay_s == pytest.approx(1.9)
    assert "DELAYED" in r.consequence


def test_behavioral_impact_unknown_without_counterfactual():
    r = metrics.behavioral_impact(
        {"action": "emergency_brake", "decel_mps2": 8.0}, None)
    assert r.consequence == "UNKNOWN"
    assert "counterfactual" in r.unknown_reason
