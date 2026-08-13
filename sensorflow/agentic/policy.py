"""Deterministic, hash-versioned stop-ship policy engine.

This module is the launch authority. It is pure deterministic code: no LLM
output enters it, and its inputs are typed measurements/verdicts produced by
deterministic services (plus clearly-tagged agent hypotheses, which the
engine treats only as *escalation* signals, never as evidence of safety).

Semantics
---------
* Policies are JSON documents persisted under runs/agentic/policies/ and
  addressed by the SHA-256 of their canonical serialization. Editing a policy
  produces a new version; evaluations record the exact version they used.
* Outcomes are four-way (AUTOMATIC_STOP_SHIP / LAUNCH_REVIEW_REQUIRED /
  CONTINUE_INVESTIGATION / NO_LAUNCH_IMPACT) plus the INDETERMINATE
  fail-safe. INDETERMINATE fires whenever required evidence is missing
  (ground truth, lineage, policy version, telemetry denominator), sensor
  fusion verification failed, or agents conflict — and is NEVER a pass; it
  forces human escalation.
* AUTOMATIC_STOP_SHIP fires only on pre-authorized conditions written in the
  policy document; nothing else can trigger it.
* The option-selection matrix is evaluated top-down; the first row whose
  condition holds selects the recommended option. Expected loss is computed
  for every option for transparency, but SAFETY CONSTRAINTS ARE HARD
  CONSTRAINTS: infeasible (unsafe) options are excluded from selection no
  matter how cheap they are — business optimization happens only inside the
  safe feasible region.

All numeric thresholds in DEFAULT_POLICY are EXAMPLE PLACEHOLDERS (flagged
`placeholder_values: true`); a real deployment must set organization-approved
values through the versioned policy API.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.agentic import store as store_mod
from sensorflow.agentic.models import now_iso

SEVERITY_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]
CONFIDENCE_ORDER = ["INSUFFICIENT_EVIDENCE", "POSSIBLE", "LIKELY", "CONFIRMED"]

# Concentration dimension -> ODD taxonomy dimension used for the
# "reliably detectable" check in Option C (see sensorflow.safety.odd.taxonomy).
ODD_DIMENSION_MAP = {
    "construction_zone": "scenario",   # proxy: construction handled as a scenario/dynamic element
    "lighting": "lighting",
    "geo_bucket": "geography",         # declared but NOT instrumented in the taxonomy
}

DEFAULT_POLICY: Dict[str, Any] = {
    "policy_name": "example-stop-ship-policy",
    "policy_semver": "1.0.0",
    "placeholder_values": True,
    "note": ("ALL numeric thresholds below are EXAMPLE PLACEHOLDERS shipped "
             "for demonstration; they are not invented safety claims and must "
             "be replaced with organization-approved values."),
    "severity_taxonomy": {
        "S0": "No functional impact (cosmetic / logging only).",
        "S1": "Functional degradation with no safety relevance.",
        "S2": "Perception error on a non-safety-critical object, or a "
              "safety-critical class error that is fully contained downstream.",
        "S3": "Perception error on a safety-critical class (VRU) with "
              "uncertain or degraded downstream behavior.",
        "S4": "Observed unsafe downstream behavior (hard braking / evasive "
              "action / failure to yield) attributable to the error.",
        "S5": "Observed collision or imminent-collision behavior "
              "attributable to the error.",
    },
    # Ordered, exact criteria: the first matching rule assigns the severity.
    "severity_rules": [
        {"severity": "S5", "when": {"collision_observed": True}},
        {"severity": "S4", "when": {"safety_critical_class": True,
                                    "behavioral_evidence": "observed_unsafe"}},
        {"severity": "S3", "when": {"safety_critical_class": True,
                                    "downstream_contained": False}},
        {"severity": "S2", "when": {"safety_critical_class": True,
                                    "downstream_contained": True}},
        {"severity": "S1", "when": {"functional_impact": True}},
        {"severity": "S0", "when": {}},
    ],
    # Pre-authorized automatic stop-ship conditions. ONLY these can fire
    # AUTOMATIC_STOP_SHIP.
    "automatic_stop_ship": [
        {"condition_id": "ASS-1",
         "description": "Confirmed S4+ regression at statistically significant "
                        "frequency",
         "min_severity": "S4", "min_confidence": "CONFIRMED",
         "requires_significant": True},
        {"condition_id": "ASS-2",
         "description": "Confirmed S3+ novel regression with frequency lower "
                        "bound above the example ceiling",
         "min_severity": "S3", "min_confidence": "CONFIRMED",
         "requires_significant": True, "requires_novel": True,
         "min_rate_lower_bound": 1.0e-4},
    ],
    # Four-way outcome rules over Severity x Exposure x Frequency x
    # Confidence x Novelty x DownstreamConsequence (evaluated after the
    # stop-ship conditions; first match wins).
    "outcome_rules": [
        {"rule_id": "OR-1", "outcome": "LAUNCH_REVIEW_REQUIRED",
         "description": "significant S3+ regression",
         "min_severity": "S3", "requires_significant": True,
         "min_confidence": "LIKELY"},
        {"rule_id": "OR-2", "outcome": "LAUNCH_REVIEW_REQUIRED",
         "description": "novel safety-critical pattern with uncertain "
                        "downstream consequence",
         "min_severity": "S3", "requires_novel": True,
         "downstream_in": ["uncertain", "observed_degraded",
                           "observed_critical"]},
        {"rule_id": "OR-3", "outcome": "CONTINUE_INVESTIGATION",
         "description": "possible regression, evidence not yet sufficient",
         "min_severity": "S2", "max_confidence": "POSSIBLE"},
        {"rule_id": "OR-4", "outcome": "CONTINUE_INVESTIGATION",
         "description": "significant but low-severity change",
         "min_severity": "S1", "requires_significant": True},
        {"rule_id": "OR-5", "outcome": "NO_LAUNCH_IMPACT",
         "description": "not significant, not novel, no safety relevance",
         "max_severity": "S1"},
    ],
    # Option selection matrix — all seven rows, evaluated top-down.
    "option_matrix": [
        {"row": 1, "condition": "GATE_VIOLATED", "option": "STOP_SHIP",
         "description": "a release gate or pre-authorized stop-ship condition fired"},
        {"row": 2, "condition": "HIGH_CONFIDENCE_UNCONTAINED_CRITICAL",
         "option": "OPTION_A_DELAY",
         "description": "confirmed/likely S3+ failure with no validated "
                        "containment: delay launch until fixed"},
        {"row": 3, "condition": "VALIDATED_MITIGATION",
         "option": "OPTION_B_MITIGATION",
         "description": "a validated mitigation exists: ship with mitigation"},
        {"row": 4, "condition": "RELIABLY_DETECTABLE_ODD_CONCENTRATION",
         "option": "OPTION_C_REDUCED_ODD",
         "description": "failure concentrates in an ODD region that is "
                        "reliably detectable at runtime: ship with reduced ODD"},
        {"row": 5, "condition": "INSUFFICIENT_EVIDENCE",
         "option": "EXPAND_EVALUATION",
         "description": "evidence insufficient for any launch claim"},
        {"row": 6, "condition": "AGENT_OR_MODALITY_CONFLICT",
         "option": "HUMAN_SAFETY_REVIEW",
         "description": "agents or sensor modalities conflict"},
        {"row": 7, "condition": "NO_SAFETY_IMPACT", "option": "PROCEED",
         "description": "no material safety impact established"},
    ],
    # Option C parameters (EXAMPLE placeholders; never invented at runtime).
    "option_c": {
        "MAX_ACCEPTABLE_FAILURE_PROBABILITY": 1.0e-5,
        "MIN_REQUIRED_SAFETY_RECALL": 0.995,
        "MAX_ACCEPTABLE_RESIDUAL_RISK": 5.0e-5,
        "MIN_REQUIRED_EVIDENCE_CONFIDENCE": "LIKELY",
        "MAX_ALLOWED_ODD_EXCLUSION": 0.20,
        "placeholder_values": True,
    },
    # Expected-loss model (EXAMPLE placeholder unit costs).
    "expected_loss": {
        "unit": "abstract cost units (example placeholders)",
        "delay_cost": 100.0,
        "mitigation_cost": 40.0,
        "odd_reduction_cost_per_exposure_share": 300.0,
        "expand_evaluation_cost": 25.0,
        "incident_cost": 2_000_000.0,
        "exposure_events_per_release": 1_000_000,
        "mitigation_effectiveness": 0.9,
    },
    # Hard safety constraint applied to option feasibility.
    "hard_constraints": {
        "MAX_ACCEPTABLE_RESIDUAL_RISK": 5.0e-5,
        "applies_from_severity": "S3",
        "note": "options whose residual failure rate exceeds this are "
                "infeasible regardless of expected loss",
    },
}


# ------------------------------------------------------------------ versioning


def policy_hash(doc: Dict) -> str:
    blob = json.dumps({k: v for k, v in doc.items()
                       if k not in ("policy_version", "created_at")},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_policy(doc: Dict, actor: str = "system") -> Dict:
    doc = dict(doc)
    version = policy_hash(doc)
    doc["policy_version"] = version
    doc["created_at"] = doc.get("created_at") or now_iso()
    store_mod.write_json(doc, "policies", f"{version}.json")
    store_mod.write_json({"active": version}, "policies", "_active.json")
    store_mod.audit("policy_saved", None, actor,
                    detail=f"policy {doc.get('policy_name')} version {version}")
    return doc


def get_policy(version: Optional[str] = None) -> Dict:
    if version:
        doc = store_mod.read_json("policies", f"{version}.json")
        if doc is None:
            raise KeyError(f"Unknown policy version {version}")
        return doc
    active = store_mod.read_json("policies", "_active.json")
    if active and active.get("active"):
        doc = store_mod.read_json("policies", f"{active['active']}.json")
        if doc is not None:
            return doc
    return save_policy(DEFAULT_POLICY)


def list_policies() -> List[Dict]:
    out = []
    for name in store_mod.list_dir("policies"):
        if name.startswith("_") or not name.endswith(".json"):
            continue
        doc = store_mod.read_json("policies", name)
        if doc:
            out.append({"policy_version": doc.get("policy_version"),
                        "policy_name": doc.get("policy_name"),
                        "policy_semver": doc.get("policy_semver"),
                        "placeholder_values": doc.get("placeholder_values"),
                        "created_at": doc.get("created_at")})
    return out


# ------------------------------------------------------------------ input schema


class PolicyInput(BaseModel):
    """Typed, deterministic inputs for a policy evaluation."""

    failure_id: str
    # severity inputs
    safety_critical_class: bool = False
    collision_observed: bool = False
    behavioral_evidence: str = "none"    # observed_unsafe|observed_contained|uncertain|none
    downstream_contained: bool = False
    functional_impact: bool = True
    # frequency / statistics (deterministic, from the statistical agent)
    rate: Optional[float] = None
    rate_ci: Optional[List[float]] = None
    denominator: Optional[int] = None
    significant: bool = False
    small_sample: bool = False
    # exposure / novelty / confidence
    exposure_share: Optional[float] = None
    novelty: str = "unknown"             # novel|known_regression|known_stable|unknown
    evidence_confidence: str = "INSUFFICIENT_EVIDENCE"
    downstream_consequence: str = "uncertain"
    # integrity signals
    gt_available: bool = True
    lineage_complete: bool = True
    telemetry_available: bool = True
    fusion_verdict: str = "multi_modal_supported"
    agent_conflict: bool = False
    conflict_details: List[str] = Field(default_factory=list)
    # gates + mitigation + ODD concentration
    gate_violated: bool = False
    violated_gates: List[str] = Field(default_factory=list)
    mitigation_validated: bool = False
    mitigation_description: str = ""
    concentration_dimension: Optional[str] = None
    concentrated: bool = False
    odd_detector_recall: Optional[float] = None
    odd_exclusion_share: Optional[float] = None
    residual_rate_outside_odd: Optional[float] = None


# ------------------------------------------------------------------ helpers


def _sev_ge(a: str, b: str) -> bool:
    return SEVERITY_ORDER.index(a) >= SEVERITY_ORDER.index(b)


def _conf_ge(a: str, b: str) -> bool:
    return CONFIDENCE_ORDER.index(a) >= CONFIDENCE_ORDER.index(b)


def assign_severity(policy: Dict, inp: PolicyInput) -> Dict:
    """Deterministic severity from the policy's ordered exact criteria."""
    facts = {
        "collision_observed": inp.collision_observed,
        "safety_critical_class": inp.safety_critical_class,
        "behavioral_evidence": inp.behavioral_evidence,
        "downstream_contained": inp.downstream_contained,
        "functional_impact": inp.functional_impact,
    }
    for rule in policy["severity_rules"]:
        if all(facts.get(k) == v for k, v in rule["when"].items()):
            return {"severity": rule["severity"], "matched_rule": rule,
                    "facts": facts,
                    "taxonomy_description": policy["severity_taxonomy"][rule["severity"]]}
    return {"severity": "S0", "matched_rule": None, "facts": facts,
            "taxonomy_description": policy["severity_taxonomy"]["S0"]}


def _odd_instrumented(dimension: Optional[str]) -> Dict:
    """Reliably-detectable check against the safety ODD taxonomy's
    instrumented dimensions (sensorflow.safety.odd)."""
    if dimension is None:
        return {"dimension": None, "odd_dimension": None, "instrumented": False,
                "reason": "no concentration dimension supplied"}
    odd_dim = ODD_DIMENSION_MAP.get(dimension, dimension)
    try:
        from sensorflow.safety.odd import taxonomy
        dims = taxonomy()["dimensions"]
        entry = dims.get(odd_dim)
        instrumented = bool(entry and entry.get("instrumented"))
        reason = (f"ODD taxonomy dimension '{odd_dim}' instrumented="
                  f"{instrumented}")
    except Exception as e:  # taxonomy unavailable -> fail safe (not detectable)
        instrumented = False
        reason = f"ODD taxonomy unavailable ({e}); failing safe"
    return {"dimension": dimension, "odd_dimension": odd_dim,
            "instrumented": instrumented, "reason": reason,
            "mapping_note": ("construction_zone is proxied to the 'scenario' "
                             "taxonomy dimension" if dimension == "construction_zone" else None)}


def evaluate_option_c(policy: Dict, inp: PolicyInput) -> Dict:
    """Option C (reduced ODD) feasibility from config parameters only."""
    cfg = policy["option_c"]
    checks = []

    odd = _odd_instrumented(inp.concentration_dimension)
    checks.append({"check": "odd_dimension_instrumented",
                   "actual": odd["instrumented"], "required": True,
                   "passed": odd["instrumented"], "detail": odd["reason"]})
    checks.append({"check": "concentrated", "actual": inp.concentrated,
                   "required": True, "passed": inp.concentrated})

    recall = inp.odd_detector_recall
    checks.append({
        "check": "MIN_REQUIRED_SAFETY_RECALL",
        "actual": recall, "required": cfg["MIN_REQUIRED_SAFETY_RECALL"],
        "passed": recall is not None and recall >= cfg["MIN_REQUIRED_SAFETY_RECALL"]})

    resid = inp.residual_rate_outside_odd
    checks.append({
        "check": "MAX_ACCEPTABLE_RESIDUAL_RISK",
        "actual": resid, "required": cfg["MAX_ACCEPTABLE_RESIDUAL_RISK"],
        "passed": resid is not None and resid <= cfg["MAX_ACCEPTABLE_RESIDUAL_RISK"]})

    checks.append({
        "check": "MIN_REQUIRED_EVIDENCE_CONFIDENCE",
        "actual": inp.evidence_confidence,
        "required": cfg["MIN_REQUIRED_EVIDENCE_CONFIDENCE"],
        "passed": _conf_ge(inp.evidence_confidence,
                           cfg["MIN_REQUIRED_EVIDENCE_CONFIDENCE"])})

    excl = inp.odd_exclusion_share
    checks.append({
        "check": "MAX_ALLOWED_ODD_EXCLUSION",
        "actual": excl, "required": cfg["MAX_ALLOWED_ODD_EXCLUSION"],
        "passed": excl is not None and excl <= cfg["MAX_ALLOWED_ODD_EXCLUSION"]})

    rate_in = inp.rate
    checks.append({
        "check": "MAX_ACCEPTABLE_FAILURE_PROBABILITY (post-exclusion)",
        "actual": resid, "required": cfg["MAX_ACCEPTABLE_FAILURE_PROBABILITY"],
        "passed": resid is not None and resid <= cfg["MAX_ACCEPTABLE_FAILURE_PROBABILITY"],
        "detail": f"in-ODD rate before exclusion: {rate_in}"})

    return {"feasible": all(c["passed"] for c in checks), "checks": checks,
            "config": cfg, "odd_taxonomy_check": odd}


# ------------------------------------------------------------------ expected loss


def expected_loss_table(policy: Dict, inp: PolicyInput, severity: str) -> List[Dict]:
    cfg = policy["expected_loss"]
    hard = policy["hard_constraints"]
    rate = inp.rate or 0.0
    resid_c = inp.residual_rate_outside_odd if inp.residual_rate_outside_odd is not None else rate
    excl = inp.odd_exclusion_share or 0.0
    eff = cfg["mitigation_effectiveness"] if inp.mitigation_validated else 0.0
    exposure = cfg["exposure_events_per_release"]
    safety_scoped = _sev_ge(severity, hard["applies_from_severity"])
    max_resid = hard["MAX_ACCEPTABLE_RESIDUAL_RISK"]

    rows = []
    for option, business, residual_rate in [
        ("STOP_SHIP", cfg["delay_cost"], 0.0),
        ("OPTION_A_DELAY", cfg["delay_cost"], 0.0),
        ("OPTION_B_MITIGATION", cfg["mitigation_cost"], rate * (1.0 - eff)),
        ("OPTION_C_REDUCED_ODD",
         cfg["odd_reduction_cost_per_exposure_share"] * excl, resid_c),
        ("EXPAND_EVALUATION", cfg["expand_evaluation_cost"], rate),
        ("HUMAN_SAFETY_REVIEW", cfg["expand_evaluation_cost"], rate),
        ("PROCEED", 0.0, rate),
    ]:
        risk_cost = residual_rate * exposure * cfg["incident_cost"] / max(exposure, 1)
        # normalized: expected incidents per release * incident cost
        expected_incidents = residual_rate * exposure
        total = business + expected_incidents * cfg["incident_cost"] / 1000.0
        feasible = (not safety_scoped) or (residual_rate <= max_resid)
        rows.append({
            "option": option,
            "business_cost": round(business, 2),
            "residual_failure_rate": residual_rate,
            "expected_incidents_per_release": round(expected_incidents, 4),
            "expected_loss": round(total, 2),
            "risk_cost_component": round(risk_cost, 6),
            "feasible": feasible,
            "infeasible_reason": (None if feasible else
                                  f"residual rate {residual_rate:.2e} exceeds hard "
                                  f"constraint {max_resid:.2e} at severity {severity}"),
        })
    return rows


# ------------------------------------------------------------------ evaluation


def _integrity_failures(inp: PolicyInput) -> List[str]:
    problems = []
    if not inp.gt_available:
        problems.append("ground truth unavailable or disputed")
    if not inp.lineage_complete:
        problems.append("lineage (model/dataset fingerprints) incomplete")
    if not inp.telemetry_available or not inp.denominator:
        problems.append("telemetry denominator unavailable — no valid rate claim")
    if inp.fusion_verdict == "verification_failed":
        problems.append("sensor fusion verification failed")
    if inp.agent_conflict:
        problems.append("agents produced conflicting verdicts: "
                        + "; ".join(inp.conflict_details or ["unspecified"]))
    return problems


def _matrix_condition_holds(condition: str, inp: PolicyInput, severity: str,
                            option_c: Dict, stop_ship_fired: bool) -> bool:
    if condition == "GATE_VIOLATED":
        return inp.gate_violated or stop_ship_fired
    if condition == "HIGH_CONFIDENCE_UNCONTAINED_CRITICAL":
        return (_sev_ge(severity, "S3")
                and _conf_ge(inp.evidence_confidence, "LIKELY")
                and not inp.downstream_contained
                and not inp.mitigation_validated)
    if condition == "VALIDATED_MITIGATION":
        return inp.mitigation_validated
    if condition == "RELIABLY_DETECTABLE_ODD_CONCENTRATION":
        return option_c["feasible"]
    if condition == "INSUFFICIENT_EVIDENCE":
        return (inp.evidence_confidence == "INSUFFICIENT_EVIDENCE"
                or inp.small_sample)
    if condition == "AGENT_OR_MODALITY_CONFLICT":
        return inp.agent_conflict or inp.fusion_verdict == "modality_conflict"
    if condition == "NO_SAFETY_IMPACT":
        return True
    return False


def _outcome_rule_matches(rule: Dict, inp: PolicyInput, severity: str) -> bool:
    if "min_severity" in rule and not _sev_ge(severity, rule["min_severity"]):
        return False
    if "max_severity" in rule and _sev_ge(severity, rule["max_severity"]) \
            and severity != rule["max_severity"]:
        return False
    if rule.get("requires_significant") and not inp.significant:
        return False
    if rule.get("requires_novel") and inp.novelty != "novel":
        return False
    if "min_confidence" in rule and not _conf_ge(inp.evidence_confidence,
                                                 rule["min_confidence"]):
        return False
    if "max_confidence" in rule and _conf_ge(inp.evidence_confidence,
                                             rule["max_confidence"]) \
            and inp.evidence_confidence != rule["max_confidence"]:
        return False
    if "downstream_in" in rule and inp.downstream_consequence not in rule["downstream_in"]:
        return False
    return True


def _stop_ship_condition_matches(cond: Dict, inp: PolicyInput, severity: str) -> bool:
    if "min_severity" in cond and not _sev_ge(severity, cond["min_severity"]):
        return False
    if "min_confidence" in cond and not _conf_ge(inp.evidence_confidence,
                                                 cond["min_confidence"]):
        return False
    if cond.get("requires_significant") and not inp.significant:
        return False
    if cond.get("requires_novel") and inp.novelty != "novel":
        return False
    if "min_rate_lower_bound" in cond:
        lb = (inp.rate_ci or [0.0, 0.0])[0]
        if lb < cond["min_rate_lower_bound"]:
            return False
    return True


def evaluate(inp: PolicyInput, policy: Optional[Dict] = None,
             actor: str = "policy_engine") -> Dict:
    """Deterministic policy evaluation. Returns outcome + fired rules +
    option selection + expected-loss table, and audits itself."""
    doc = policy or get_policy()
    version = doc.get("policy_version") or policy_hash(doc)

    result: Dict[str, Any] = {
        "failure_id": inp.failure_id,
        "policy_version": version,
        "policy_name": doc.get("policy_name"),
        "placeholder_values": doc.get("placeholder_values", False),
        "evaluated_at": now_iso(),
        "input": inp.model_dump(),
        "authority": "deterministic policy engine + human approval; agents are advisory",
    }

    # 1. INDETERMINATE fail-safe (never a pass).
    problems = _integrity_failures(inp)
    severity_info = assign_severity(doc, inp)
    severity = severity_info["severity"]
    result["severity"] = severity
    result["severity_assignment"] = severity_info

    if problems:
        result.update({
            "outcome": "INDETERMINATE",
            "indeterminate_reasons": problems,
            "recommended_option": "HUMAN_SAFETY_REVIEW",
            "matrix_row_fired": {"row": 6,
                                 "condition": "AGENT_OR_MODALITY_CONFLICT"
                                 if inp.agent_conflict else "INTEGRITY_FAILURE",
                                 "option": "HUMAN_SAFETY_REVIEW"},
            "expected_loss_table": expected_loss_table(doc, inp, severity),
            "option_c_evaluation": evaluate_option_c(doc, inp),
            "note": "INDETERMINATE is fail-safe: it can never authorize a "
                    "launch and always escalates to human safety review.",
        })
        store_mod.audit("policy_evaluated", inp.failure_id, actor,
                        detail=f"INDETERMINATE ({'; '.join(problems)})",
                        payload={"policy_version": version})
        return result

    # 2. Pre-authorized automatic stop-ship conditions.
    fired_stop = None
    for cond in doc["automatic_stop_ship"]:
        if _stop_ship_condition_matches(cond, inp, severity):
            fired_stop = cond
            break

    # 3. Four-way outcome.
    if fired_stop:
        outcome = "AUTOMATIC_STOP_SHIP"
        fired_rule = fired_stop
    else:
        outcome, fired_rule = "CONTINUE_INVESTIGATION", None
        for rule in doc["outcome_rules"]:
            if _outcome_rule_matches(rule, inp, severity):
                outcome, fired_rule = rule["outcome"], rule
                break

    # 4. Option matrix + hard-constrained expected loss.
    option_c = evaluate_option_c(doc, inp)
    loss_table = expected_loss_table(doc, inp, severity)
    feasible = {r["option"] for r in loss_table if r["feasible"]}

    matrix_row = None
    for row in doc["option_matrix"]:
        if not _matrix_condition_holds(row["condition"], inp, severity,
                                       option_c, fired_stop is not None):
            continue
        if row["option"] in ("STOP_SHIP", "OPTION_A_DELAY", "EXPAND_EVALUATION",
                             "HUMAN_SAFETY_REVIEW") or row["option"] in feasible:
            matrix_row = row
            break
        # matched but infeasible under the hard safety constraint -> next row
    if matrix_row is None:
        matrix_row = {"row": 6, "condition": "AGENT_OR_MODALITY_CONFLICT",
                      "option": "HUMAN_SAFETY_REVIEW",
                      "description": "no feasible matrix row — fail safe"}

    result.update({
        "outcome": outcome,
        "outcome_rule_fired": fired_rule,
        "automatic_stop_ship_condition": fired_stop,
        "recommended_option": matrix_row["option"],
        "matrix_row_fired": matrix_row,
        "option_c_evaluation": option_c,
        "expected_loss_table": loss_table,
        "hard_constraint": doc["hard_constraints"],
        "note": ("expected loss is reported for every option, but options "
                 "violating the hard safety constraint were excluded from "
                 "selection regardless of cost"),
    })
    store_mod.audit("policy_evaluated", inp.failure_id, actor,
                    detail=f"{outcome} -> {matrix_row['option']} "
                           f"(policy {version})",
                    payload={"policy_version": version, "severity": severity})
    return result
