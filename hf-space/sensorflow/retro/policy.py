"""Deterministic severity framework and launch-recommendation gate.

This module OWNS the safety boundary. The LLM may propose a severity with
evidence; this engine validates or overrides it, records divergence, and is
the ONLY producer of launch determinations.

Severity model (asymmetric FN/FP costs, context-dependent — deliberately NOT
"every FN outranks every FP"):
  FALSE NEGATIVE cost ~ VRU weight x safety-relevant-distance factor x
      relative-motion factor x remaining-reaction-time factor
  FALSE POSITIVE cost ~ intervention magnitude x traffic disruption exposure
A benign distant non-closing FN scores below a hard-braking phantom FP with
a tailgater; the test suite pins this ordering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from sensorflow.retro.scorecard import LaunchRecommendation, Severity

POLICY_VERSION = "retro-policy/1.0.0"

# Versioned policy configuration: all thresholds live here, not in code paths.
POLICY_CONFIG: Dict[str, Any] = {
    "version": POLICY_VERSION,
    "vru_weights": {
        "pedestrian": 1.0, "cyclist": 0.95, "motorcycle": 0.9,
        "wheelchair": 1.0, "vehicle": 0.6, "truck": 0.65, "bus": 0.65,
        "unknown": 0.5, "debris": 0.15, "plastic_bag": 0.05, "static": 0.2,
    },
    "fn_severity_cutoffs": {"critical": 0.45, "disruptive": 0.18},
    "fp_severity_cutoffs": {"critical": 0.55, "disruptive": 0.28},
    "hard_brake_g": 0.35,               # from SFS-SAFE-001 REQ-FP-02 [SYNTHETIC]
    "tailgater_gap_s": 1.5,
    "scr_regression_tolerance": 0.002,  # 0.2 pp, SFS-LAUNCH-004 GATE-02 [SYNTHETIC]
    "collision_risk_ttc_s": 1.5,
}


class SeverityContext(BaseModel):
    """Deterministic inputs to the severity computation (facts + derived)."""
    failure_type: str                       # FALSE_NEGATIVE | FALSE_POSITIVE | ...
    object_class: Optional[str] = None      # ground-truth class
    distance_m: Optional[float] = None
    stopping_distance_m: Optional[float] = None
    closing_velocity_mps: Optional[float] = None
    ttc_s: Optional[float] = None
    total_reaction_time_s: Optional[float] = None
    collision_occurred: bool = False
    near_miss: bool = False
    intervention_decel_mps2: Optional[float] = None   # observed planner decel
    following_gap_s: Optional[float] = None            # rear traffic time gap
    ego_speed_mps: Optional[float] = None


class SeverityResult(BaseModel):
    severity: Severity
    score: float
    rule_trace: List[str] = Field(default_factory=list)
    computable: bool = True


def _fn_severity(ctx: SeverityContext, cfg: Dict[str, Any],
                 trace: List[str]) -> Tuple[Severity, float]:
    w = cfg["vru_weights"].get((ctx.object_class or "unknown").lower(), 0.5)
    trace.append(f"FN branch: class '{ctx.object_class}' weight={w}")

    if ctx.distance_m is not None and ctx.stopping_distance_m:
        ratio = ctx.distance_m / ctx.stopping_distance_m
        if ratio <= 1.0:
            dist_f = 1.0
        elif ratio <= 1.5:
            dist_f = 0.6
        elif ratio <= 3.0:
            dist_f = 0.25
        else:
            dist_f = 0.05
        trace.append(f"distance {ctx.distance_m:.1f} m = {ratio:.2f}x stopping "
                     f"distance -> factor {dist_f}")
    else:
        dist_f = 0.5
        trace.append("distance vs stopping distance not computable -> neutral 0.5")

    closing = ctx.closing_velocity_mps or 0.0
    if closing > 0:
        motion_f = min(1.0, 0.3 + closing / 15.0)
        trace.append(f"closing at {closing:.1f} m/s -> motion factor {motion_f:.2f}")
    else:
        motion_f = 0.1
        trace.append("not closing -> motion factor 0.1")

    if ctx.ttc_s is not None and ctx.total_reaction_time_s is not None:
        remaining = ctx.ttc_s - ctx.total_reaction_time_s
        if remaining <= 0:
            rrt_f = 1.0
        elif remaining < 1.0:
            rrt_f = 0.8
        elif remaining < 2.5:
            rrt_f = 0.5
        else:
            rrt_f = 0.2
        trace.append(f"remaining reaction time {remaining:.2f} s -> factor {rrt_f}")
    elif closing > 0:
        rrt_f = 0.6
        trace.append("TTC/reaction budget unknown while closing -> conservative 0.6")
    else:
        rrt_f = 0.1
        trace.append("no predicted collision path -> reaction factor 0.1")

    score = w * dist_f * (0.5 * motion_f + 0.5 * rrt_f)
    cuts = cfg["fn_severity_cutoffs"]
    if ctx.collision_occurred:
        trace.append("collision occurred -> FATAL override")
        return Severity.FATAL, max(score, 1.0)
    if score >= cuts["critical"] or (ctx.near_miss and score >= cuts["disruptive"]):
        return Severity.CRITICAL, score
    if score >= cuts["disruptive"]:
        return Severity.DISRUPTIVE, score
    if ctx.near_miss:
        trace.append("near-miss recorded despite low computed score -> "
                     "DISRUPTIVE floor")
        return Severity.DISRUPTIVE, score
    return Severity.BENIGN, score


def _fp_severity(ctx: SeverityContext, cfg: Dict[str, Any],
                 trace: List[str]) -> Tuple[Severity, float]:
    decel = ctx.intervention_decel_mps2 or 0.0
    intervention_f = min(1.0, decel / 9.0)
    trace.append(f"FP branch: intervention decel {decel:.1f} m/s^2 "
                 f"({decel / 9.80665:.2f} g) -> factor {intervention_f:.2f}")

    if ctx.following_gap_s is not None:
        if ctx.following_gap_s < 1.0:
            disruption_f = 1.0
        elif ctx.following_gap_s < cfg["tailgater_gap_s"]:
            disruption_f = 0.8
        elif ctx.following_gap_s < 3.0:
            disruption_f = 0.5
        else:
            disruption_f = 0.3
        trace.append(f"following gap {ctx.following_gap_s:.1f} s -> "
                     f"disruption factor {disruption_f}")
    else:
        disruption_f = 0.4
        trace.append("following traffic unknown -> disruption factor 0.4 (assumed)")

    speed_f = min(1.0, (ctx.ego_speed_mps or 0.0) / 30.0)
    trace.append(f"ego speed factor {speed_f:.2f}")

    score = intervention_f * (0.55 + 0.45 * disruption_f) * (0.6 + 0.4 * speed_f)
    cuts = cfg["fp_severity_cutoffs"]
    hard_brake = decel >= cfg["hard_brake_g"] * 9.80665
    tailgated = (ctx.following_gap_s is not None
                 and ctx.following_gap_s < cfg["tailgater_gap_s"])
    if ctx.collision_occurred:
        trace.append("collision occurred (induced) -> FATAL override")
        return Severity.FATAL, max(score, 1.0)
    if hard_brake and tailgated:
        trace.append(f"hard brake (>= {cfg['hard_brake_g']} g) with rear gap "
                     f"< {cfg['tailgater_gap_s']} s -> CRITICAL rule "
                     "(REQ-FP-02 [SYNTHETIC])")
        return Severity.CRITICAL, max(score, cuts["critical"])
    if score >= cuts["critical"]:
        return Severity.CRITICAL, score
    if score >= cuts["disruptive"]:
        return Severity.DISRUPTIVE, score
    return Severity.BENIGN, score


def compute_severity(ctx: SeverityContext,
                     config: Optional[Dict[str, Any]] = None) -> SeverityResult:
    """Deterministic severity from contextual evidence. Pure function."""
    cfg = config or POLICY_CONFIG
    trace: List[str] = [f"policy {cfg['version']}"]
    ft = ctx.failure_type.upper()
    if ft in ("FALSE_NEGATIVE", "MISSED_DETECTION", "LATE_DETECTION"):
        sev, score = _fn_severity(ctx, cfg, trace)
    elif ft in ("FALSE_POSITIVE", "PHANTOM_OBJECT", "MISCLASSIFICATION_FP"):
        sev, score = _fp_severity(ctx, cfg, trace)
    elif ft == "MISCLASSIFICATION":
        # Consequence-dominant: score both directions, take the worse.
        t1: List[str] = []
        t2: List[str] = []
        s1 = _fn_severity(ctx, cfg, t1)
        s2 = _fp_severity(ctx, cfg, t2)
        order = [Severity.BENIGN, Severity.DISRUPTIVE, Severity.CRITICAL, Severity.FATAL]
        if order.index(s1[0]) >= order.index(s2[0]):
            sev, score = s1
            trace += ["misclassification scored on FN consequence branch"] + t1
        else:
            sev, score = s2
            trace += ["misclassification scored on FP consequence branch"] + t2
    else:
        trace.append(f"failure type '{ctx.failure_type}' has no severity model "
                     "-> not computable, conservative DISRUPTIVE floor")
        return SeverityResult(severity=Severity.DISRUPTIVE, score=0.0,
                              rule_trace=trace, computable=False)
    return SeverityResult(severity=sev, score=round(score, 4), rule_trace=trace)


SEVERITY_ORDER = [Severity.BENIGN, Severity.DISRUPTIVE, Severity.CRITICAL,
                  Severity.FATAL]


class SeverityAdjudication(BaseModel):
    policy_severity: Severity
    ai_proposed_severity: Optional[Severity]
    final_severity: Severity            # always the policy severity
    divergence: bool
    divergence_note: str = ""


def adjudicate_severity(policy: SeverityResult,
                        ai_proposed: Optional[Severity]) -> SeverityAdjudication:
    """Policy validates/overrides the LLM proposal; divergence is recorded
    and (by the caller) flagged for human review."""
    divergence = ai_proposed is not None and ai_proposed != policy.severity
    note = ""
    if divergence:
        direction = ("understated" if SEVERITY_ORDER.index(ai_proposed)
                     < SEVERITY_ORDER.index(policy.severity) else "overstated")
        note = (f"AI proposed {ai_proposed.value} but deterministic policy "
                f"computed {policy.severity.value} (AI {direction} severity); "
                "policy severity is authoritative and the divergence requires "
                "human review")
    return SeverityAdjudication(policy_severity=policy.severity,
                                ai_proposed_severity=ai_proposed,
                                final_severity=policy.severity,
                                divergence=divergence, divergence_note=note)


class LaunchDecision(BaseModel):
    recommendation: LaunchRecommendation
    rationale: List[str]
    policy_version: str = POLICY_VERSION


def launch_gate(severity: Severity,
                has_evidence_gaps: bool,
                scr_impact: Optional[float],
                scr_significant: Optional[bool],
                config: Optional[Dict[str, Any]] = None) -> LaunchDecision:
    """Deterministic launch recommendation. INSUFFICIENT_EVIDENCE never
    becomes PASS (SFS-LAUNCH-004 GATE-03 [SYNTHETIC], enforced here)."""
    cfg = config or POLICY_CONFIG
    rationale: List[str] = [f"policy {cfg['version']}"]

    if has_evidence_gaps:
        rationale.append("required evidence fields are UNKNOWN -> "
                         "INSUFFICIENT_EVIDENCE (can never be converted to PASS; "
                         "resolve evidence gaps and re-run)")
        return LaunchDecision(recommendation=LaunchRecommendation.INSUFFICIENT_EVIDENCE,
                              rationale=rationale)

    if severity in (Severity.CRITICAL, Severity.FATAL):
        rationale.append(f"unresolved {severity.value} severity -> FAIL (GATE-01)")
        return LaunchDecision(recommendation=LaunchRecommendation.FAIL,
                              rationale=rationale)

    tol = cfg["scr_regression_tolerance"]
    if scr_impact is not None and scr_impact < -tol and scr_significant:
        rationale.append(f"safety-critical recall regressed {scr_impact:+.4f} "
                         f"beyond tolerance {tol} with statistical significance "
                         "-> FAIL (GATE-02)")
        return LaunchDecision(recommendation=LaunchRecommendation.FAIL,
                              rationale=rationale)

    if severity == Severity.DISRUPTIVE:
        rationale.append("DISRUPTIVE severity -> CONDITIONAL_PASS pending "
                         "mitigation tracking")
        return LaunchDecision(recommendation=LaunchRecommendation.CONDITIONAL_PASS,
                              rationale=rationale)

    if scr_impact is not None and scr_impact < -tol and scr_significant is None:
        rationale.append(f"SCR delta {scr_impact:+.4f} beyond tolerance but "
                         "significance not established -> CONDITIONAL_PASS "
                         "pending seqeval confirmation")
        return LaunchDecision(recommendation=LaunchRecommendation.CONDITIONAL_PASS,
                              rationale=rationale)

    rationale.append("BENIGN severity, no significant safety-critical recall "
                     "regression -> PASS")
    return LaunchDecision(recommendation=LaunchRecommendation.PASS,
                          rationale=rationale)
