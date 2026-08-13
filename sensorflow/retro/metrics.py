"""Deterministic safety metric calculators (the safety boundary).

Everything here is pure, parameterized computation — no LLM involvement.
TTC reuses the platform's SSAM rectangle conflict-point projection
(sensorflow/safety/ssam_ext.py) and cross-checks with the closed-form
distance/closing-speed ratio. Every result carries explicit
assumption/validity flags; inputs that are absent produce UNKNOWN results,
never guesses.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.safety.ssam_ext import projected_ttc

G = 9.80665  # m/s^2


class StoppingDistanceResult(BaseModel):
    stopping_distance_m: Optional[float]
    reaction_distance_m: Optional[float]
    braking_distance_m: Optional[float]
    effective_decel_mps2: Optional[float]
    total_reaction_time_s: Optional[float]
    params: Dict[str, Any]
    assumptions: List[str]
    unknown_reason: Optional[str] = None


def stopping_distance(velocity_mps: Optional[float],
                      reaction_time_s: float = 0.25,
                      system_latency_s: float = 0.0,
                      planner_latency_s: float = 0.0,
                      friction: float = 0.7,
                      grade: float = 0.0,
                      decel_mps2: Optional[float] = None) -> StoppingDistanceResult:
    """Stopping distance = v * t_react_total + v^2 / (2 * a_eff).

    a_eff comes from an explicit decel limit if provided, otherwise from
    friction and road grade: a = g * (friction * cos(theta) + sin(theta)),
    with theta = atan(grade); a downhill (negative) grade reduces available
    deceleration. t_react_total charges perception system latency and
    planner latency against the reaction budget.
    """
    params = {"velocity_mps": velocity_mps, "reaction_time_s": reaction_time_s,
              "system_latency_s": system_latency_s,
              "planner_latency_s": planner_latency_s, "friction": friction,
              "grade": grade, "decel_mps2": decel_mps2}
    if velocity_mps is None:
        return StoppingDistanceResult(
            stopping_distance_m=None, reaction_distance_m=None,
            braking_distance_m=None, effective_decel_mps2=None,
            total_reaction_time_s=None, params=params, assumptions=[],
            unknown_reason="ego velocity missing from telemetry -> UNKNOWN")

    assumptions = ["point-mass braking model on uniform surface",
                   "constant deceleration during braking phase",
                   "reaction budget includes perception system latency "
                   "and planner latency"]
    theta = math.atan(grade)
    if decel_mps2 is not None:
        a_eff = float(decel_mps2) + G * math.sin(theta)
        assumptions.append("deceleration limit supplied explicitly; grade "
                           "component added (downhill reduces)")
    else:
        a_eff = G * (friction * math.cos(theta) + math.sin(theta))
        assumptions.append(f"deceleration derived from friction={friction} "
                           f"and grade={grade}")
    if a_eff <= 0.05:
        return StoppingDistanceResult(
            stopping_distance_m=None, reaction_distance_m=None,
            braking_distance_m=None, effective_decel_mps2=round(a_eff, 4),
            total_reaction_time_s=None, params=params, assumptions=assumptions,
            unknown_reason="effective deceleration <= 0 (grade exceeds "
                           "friction) -> stopping distance undefined")

    t_react = reaction_time_s + system_latency_s + planner_latency_s
    reaction_d = velocity_mps * t_react
    braking_d = velocity_mps ** 2 / (2 * a_eff)
    return StoppingDistanceResult(
        stopping_distance_m=round(reaction_d + braking_d, 2),
        reaction_distance_m=round(reaction_d, 2),
        braking_distance_m=round(braking_d, 2),
        effective_decel_mps2=round(a_eff, 3),
        total_reaction_time_s=round(t_react, 3),
        params=params, assumptions=assumptions)


class TTCResult(BaseModel):
    ttc_s: Optional[float]
    method: str
    validity_flags: List[str]
    closed_form_ttc_s: Optional[float] = None
    unknown_reason: Optional[str] = None


def time_to_collision(distance_m: Optional[float],
                      closing_velocity_mps: Optional[float],
                      ego_length_m: float = 4.5, ego_width_m: float = 1.9,
                      obj_length_m: float = 0.6, obj_width_m: float = 0.6,
                      look_ahead_s: float = 12.0) -> TTCResult:
    """TTC via the SSAM rectangle conflict-point projection (ssam_ext),
    cross-checked against the closed-form gap/closing ratio.

    closing_velocity_mps > 0 means the range is decreasing.
    """
    flags = ["constant-velocity extrapolation (no acceleration model)",
             "straight-line relative motion assumed",
             "rectangle footprint projection (ssam_ext.projected_ttc)"]
    if distance_m is None or closing_velocity_mps is None:
        return TTCResult(ttc_s=None, method="ssam_ext.projected_ttc",
                         validity_flags=flags,
                         unknown_reason="distance or relative velocity missing "
                                        "from telemetry -> UNKNOWN")
    if closing_velocity_mps <= 0:
        flags.append("object not closing: TTC undefined (no predicted collision)")
        return TTCResult(ttc_s=None, method="ssam_ext.projected_ttc",
                         validity_flags=flags)

    ego_state = {"x": 0.0, "y": 0.0, "speed": float(closing_velocity_mps),
                 "heading": 0.0}
    obj_state = {"x": float(distance_m), "y": 0.0, "speed": 0.0, "heading": 0.0}
    ttc = projected_ttc(ego_state, (ego_length_m, ego_width_m),
                        obj_state, (obj_length_m, obj_width_m),
                        look_ahead=look_ahead_s, dt=0.02)
    gap = max(distance_m - (ego_length_m + obj_length_m) / 2, 0.0)
    closed_form = gap / closing_velocity_mps
    if ttc is None:
        flags.append(f"no projected collision within {look_ahead_s}s look-ahead")
    return TTCResult(ttc_s=None if ttc is None else round(ttc, 3),
                     method="ssam_ext.projected_ttc",
                     validity_flags=flags,
                     closed_form_ttc_s=round(closed_form, 3))


class SCRImpactResult(BaseModel):
    scr_baseline: Optional[float]
    scr_candidate: Optional[float]
    scr_impact: Optional[float]           # candidate - baseline
    criticality_policy: str
    denominator: Optional[int]
    unknown_reason: Optional[str] = None


def scr_impact(evaluation_context: Optional[Dict[str, Any]]) -> SCRImpactResult:
    """Safety-critical recall delta against the policy-defined criticality
    population (objects inside 1.5x stopping distance, closing, VRU or
    collision-relevant mass — see SFS-EVAL-005 EVAL-SCR-01 [SYNTHETIC])."""
    policy = ("critical = inside 1.5x stopping distance AND closing AND "
              "(VRU class OR collision-relevant mass)")
    ctx = evaluation_context or {}
    n = ctx.get("critical_object_count")
    mb = ctx.get("baseline_missed_critical")
    mc = ctx.get("candidate_missed_critical")
    if n is None or mb is None or mc is None or n <= 0:
        return SCRImpactResult(scr_baseline=None, scr_candidate=None,
                               scr_impact=None, criticality_policy=policy,
                               denominator=None,
                               unknown_reason="criticality-context counts absent "
                                              "-> SCR impact UNKNOWN (inadmissible "
                                              "as launch evidence)")
    scr_b = 1.0 - mb / n
    scr_c = 1.0 - mc / n
    return SCRImpactResult(scr_baseline=round(scr_b, 6),
                           scr_candidate=round(scr_c, 6),
                           scr_impact=round(scr_c - scr_b, 6),
                           criticality_policy=policy, denominator=int(n))


class BehavioralImpactResult(BaseModel):
    observed_action: Optional[str]
    observed_decel_mps2: Optional[float]
    corrected_action: Optional[str]
    corrected_decel_mps2: Optional[float]
    action_changed: Optional[bool]
    decel_delta_mps2: Optional[float]
    response_delay_s: Optional[float]
    consequence: str
    unknown_reason: Optional[str] = None


def behavioral_impact(observed: Optional[Dict[str, Any]],
                      counterfactual: Optional[Dict[str, Any]]) -> BehavioralImpactResult:
    """Observed planner response vs the corrected-perception counterfactual.

    The counterfactual must be supplied by the evaluation artifact (replay or
    simulation output); this function only compares — it never invents one.
    """
    if not observed:
        return BehavioralImpactResult(
            observed_action=None, observed_decel_mps2=None,
            corrected_action=None, corrected_decel_mps2=None,
            action_changed=None, decel_delta_mps2=None, response_delay_s=None,
            consequence="UNKNOWN",
            unknown_reason="planner response missing from artifact")
    if not counterfactual:
        return BehavioralImpactResult(
            observed_action=observed.get("action"),
            observed_decel_mps2=observed.get("decel_mps2"),
            corrected_action=None, corrected_decel_mps2=None,
            action_changed=None, decel_delta_mps2=None, response_delay_s=None,
            consequence="UNKNOWN",
            unknown_reason="corrected-perception counterfactual not provided "
                           "in artifact -> behavioral delta UNKNOWN")
    obs_a, cf_a = observed.get("action"), counterfactual.get("action")
    obs_d = float(observed.get("decel_mps2") or 0.0)
    cf_d = float(counterfactual.get("decel_mps2") or 0.0)
    obs_t = observed.get("response_time_s")
    cf_t = counterfactual.get("response_time_s")
    delay = (round(float(obs_t) - float(cf_t), 3)
             if obs_t is not None and cf_t is not None else None)
    changed = obs_a != cf_a
    delta = round(obs_d - cf_d, 3)
    if delay is not None and delay > 0.5:
        consequence = (f"perception error DELAYED the planner response by "
                       f"{delay:.2f} s ('{cf_a}' at t={cf_t}s became '{obs_a}' "
                       f"at t={obs_t}s) -> reaction-time budget consumed, "
                       "residual collision risk")
    elif changed and delta > 3.0:
        consequence = (f"perception error converted '{cf_a}' into '{obs_a}' with "
                       f"{delta:+.1f} m/s^2 excess deceleration (unwarranted "
                       "intervention)")
    elif changed and delta < -3.0:
        consequence = (f"perception error suppressed '{cf_a}' "
                       f"({delta:+.1f} m/s^2 missing deceleration -> residual "
                       "collision risk)")
    elif changed:
        consequence = f"planner action changed ('{cf_a}' -> '{obs_a}'), moderate magnitude"
    else:
        consequence = "no behavioral change attributable to the perception error"
    return BehavioralImpactResult(
        observed_action=obs_a, observed_decel_mps2=obs_d,
        corrected_action=cf_a, corrected_decel_mps2=cf_d,
        action_changed=changed, decel_delta_mps2=delta, response_delay_s=delay,
        consequence=consequence)
