"""ROTR metric hierarchy + baseline-vs-candidate regression.

Metrics (architecture doc §6):
* ROTR recall / precision proxy (false-accusation rate on planted
  NON-violations) — the scenario bank supplies a true denominator.
* Safety-critical ROTR recall — EXPOSURE-DERIVED weights: the weight of a
  stratum is its measured fraction of SAFETY_CRITICAL consequence outcomes
  on the evaluated bank, normalized by the overall fraction (documented
  calibration, monotone in measured harm — NOT arbitrary class
  multipliers). ILLUSTRATIVE on synthetic data by construction.
* Behavioral Consequence Rate (BCR), Critical Failure Rate (CFR, with a
  Wilson interval — REUSE rca.stats.wilson_ci).
* Regression: claims are delegated to seqeval's sequential engine
  (REUSE seqeval.evaluate_regression over a megaeval population, rare-event
  cluster handling included) and forced into the SIX-outcome distinction:
  observed / statistically significant / practically significant /
  safety-critical / insufficient evidence / distribution-driven (the last
  via cohort-mix divergence, megaeval-shift + rca comparison-validity
  concepts).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# REUSE (landed, read-only).
from sensorflow.rca.stats import js_divergence_categorical, wilson_ci

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr.models import Provenance, RegressionResult

CALIBRATION_VERSION = "rotr-weights-1.0.0"

CONSEQUENCE_RANK = {"NO_MATERIAL_CONSEQUENCE": 0, "DEGRADED_COMFORT": 1,
                    "PLANNER_INTERVENTION": 2, "SAFETY_CRITICAL": 3}

# ILLUSTRATIVE practical-significance thresholds.
PRACTICAL_ROTR_RECALL_PP = 0.05
PRACTICAL_SC_RECALL_PP = 0.02
DISTRIBUTION_JS_THRESHOLD = 0.05


# ------------------------------------------------------------ run metrics


def _stratum(scenario_summary: Dict) -> str:
    return f"{scenario_summary['kind']}|{scenario_summary['vulnerability']}"


def compute_metrics(scenarios: List[Dict], violations: List[Dict],
                    consequences: Dict[str, Dict]) -> Dict:
    """scenarios: summaries with planted truth; violations: detected records;
    consequences: violation_id -> {consequence_class}."""
    by_scenario: Dict[str, List[Dict]] = {}
    for v in violations:
        by_scenario.setdefault(v["scenario_id"], []).append(v)

    committed = [s for s in scenarios if s["committed"]]
    negatives = [s for s in scenarios if not s["committed"]]

    def _detected(s: Dict) -> bool:
        return any(v["rule_id"] == s["expected_rule_id"]
                   for v in by_scenario.get(s["scenario_id"], []))

    n_detected = sum(1 for s in committed if _detected(s))
    rotr_recall = n_detected / len(committed) if committed else None

    false_acc = sum(1 for s in negatives
                    if by_scenario.get(s["scenario_id"]))
    false_accusation_rate = false_acc / len(negatives) if negatives else None

    # --- exposure-derived weight calibration (documented, monotone) ------
    strata: Dict[str, Dict] = {}
    for s in committed:
        st = strata.setdefault(_stratum(s), {"n": 0, "detected": 0, "sc": 0})
        st["n"] += 1
        if _detected(s):
            st["detected"] += 1
        cons = [consequences.get(v["violation_id"], {}).get("consequence_class")
                for v in by_scenario.get(s["scenario_id"], [])]
        if "SAFETY_CRITICAL" in cons or s.get("planted_sc_hint"):
            st["sc"] += 1
    total_n = sum(st["n"] for st in strata.values()) or 1
    total_sc = sum(st["sc"] for st in strata.values())
    overall_harm = total_sc / total_n if total_n else 0.0
    calibration = {}
    for name, st in sorted(strata.items()):
        harm = st["sc"] / st["n"] if st["n"] else 0.0
        # weight = measured harm exposure relative to overall harm; floor at
        # a small epsilon so zero-harm strata still count, never dominate.
        weight = (harm / overall_harm) if overall_harm > 0 else 1.0
        weight = max(weight, 0.1)
        calibration[name] = {"n": st["n"], "harm_fraction": round(harm, 4),
                             "weight": round(weight, 4),
                             "recall": round(st["detected"] / st["n"], 4)
                             if st["n"] else None}
    sc_strata = {k: v for k, v in calibration.items() if v["harm_fraction"] > 0
                 or k.endswith("|VRU")}
    num = sum(v["weight"] * v["recall"] * v["n"] for v in sc_strata.values()
              if v["recall"] is not None)
    den = sum(v["weight"] * v["n"] for v in sc_strata.values())
    sc_rotr_recall = (num / den) if den else None

    # --- consequence-level rates ----------------------------------------
    detected_vids = [v["violation_id"] for v in violations]
    consequential = sum(
        1 for vid in detected_vids
        if CONSEQUENCE_RANK.get(
            consequences.get(vid, {}).get("consequence_class", ""), 0) >= 1)
    bcr = consequential / len(detected_vids) if detected_vids else None

    sc_count = sum(1 for vid in detected_vids
                   if consequences.get(vid, {}).get("consequence_class")
                   == "SAFETY_CRITICAL")
    n_scen = len(scenarios)
    cfr = sc_count / n_scen if n_scen else None
    cfr_ci = wilson_ci(sc_count, n_scen) if n_scen else (0.0, 1.0)

    # --- cohort breakdowns ----------------------------------------------
    cohorts: Dict[str, Dict] = {}
    for s in committed:
        for key in (f"rule:{s['expected_rule_id']}",
                    f"visibility:{s['visibility']}",
                    f"lighting:{s['lighting']}",
                    f"vulnerability:{s['vulnerability']}"):
            c = cohorts.setdefault(key, {"n": 0, "detected": 0})
            c["n"] += 1
            c["detected"] += 1 if _detected(s) else 0
    for c in cohorts.values():
        c["recall"] = round(c["detected"] / c["n"], 4) if c["n"] else None

    # --- stack-behavior rates (what regresses between model versions) ----
    # ROTR recall measures the EVALUATOR; violation rates measure the STACK:
    # committed violations per planted opportunity, per vulnerability cohort.
    opportunities = [s for s in scenarios if s["is_violation_opportunity"]]
    behavior_rates: Dict[str, Dict] = {}
    for coh in ("VRU", "NON_VRU", "ALL"):
        opp = [s for s in opportunities
               if coh == "ALL" or s["vulnerability"] == coh]
        com = [s for s in opp if s["committed"]]
        behavior_rates[coh] = {
            "opportunities": len(opp), "committed": len(com),
            "violation_rate": round(len(com) / len(opp), 4) if opp else None}

    return {
        "n_scenarios": n_scen,
        "n_committed_violations": len(committed),
        "n_planted_non_violations": len(negatives),
        "n_detected": n_detected,
        "n_violation_records": len(violations),
        "rotr_recall": _r(rotr_recall),
        "false_accusation_rate": _r(false_accusation_rate),
        "sc_rotr_recall": _r(sc_rotr_recall),
        "bcr": _r(bcr),
        "cfr": _r(cfr),
        "cfr_wilson_95": [round(cfr_ci[0], 4), round(cfr_ci[1], 4)],
        "weight_calibration": {
            "version": CALIBRATION_VERSION,
            "method": "stratum weight = measured SAFETY_CRITICAL harm "
                      "fraction / overall harm fraction (exposure-derived, "
                      "monotone in measured harm; floored at 0.1). "
                      "ILLUSTRATIVE: calibrated on the synthetic bank itself.",
            "strata": calibration,
        },
        "cohorts": cohorts,
        "behavior_rates": behavior_rates,
        "surrogate_caveat": "TTC/PET are surrogate conflict measures; they "
                            "prioritize triage and never gate alone.",
    }


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 4)


# ------------------------------------------------------------ regression


def _seqeval_effects(baseline: Dict, candidate: Dict) -> Dict[str, float]:
    """Translate stack-behavior deltas into seqeval stratum effects.

    The regressing quantity between MODEL versions is the committed
    violation RATE per cohort (ROTR recall measures the evaluator and is
    invariant across stack profiles). A rate INCREASE is a performance
    DECREASE, so the effect is the negated rate delta in probability
    points. seqeval strata are megaeval "class|lighting" strata: the VRU
    cohort maps onto the pedestrian strata, NON_VRU onto vehicle strata.
    """
    effects: Dict[str, float] = {}
    b = baseline.get("behavior_rates", {})
    c = candidate.get("behavior_rates", {})
    mapping = {"VRU": "pedestrian|night", "NON_VRU": "vehicle|day",
               "ALL": "__global__"}
    for coh, stratum in mapping.items():
        rb = (b.get(coh) or {}).get("violation_rate")
        rc = (c.get(coh) or {}).get("violation_rate")
        if rb is None or rc is None:
            continue
        delta = -(rc - rb)
        if abs(delta) > 1e-9:
            effects[stratum] = round(delta, 4)
    return effects


def evaluate_candidate(regression_id: str, baseline_run: Dict,
                       candidate_run: Dict, seed: int = 11) -> RegressionResult:
    bm, cm = baseline_run["metrics"], candidate_run["metrics"]
    deltas = {}
    for k in ("rotr_recall", "sc_rotr_recall", "bcr", "cfr",
              "false_accusation_rate"):
        if bm.get(k) is not None and cm.get(k) is not None:
            deltas[k] = round(cm[k] - bm[k], 4)
    for coh in ("VRU", "NON_VRU", "ALL"):
        rb = (bm.get("behavior_rates", {}).get(coh) or {}).get("violation_rate")
        rc = (cm.get("behavior_rates", {}).get(coh) or {}).get("violation_rate")
        if rb is not None and rc is not None:
            deltas[f"violation_rate_{coh}"] = round(rc - rb, 4)

    # --- delegation: statistical claims go through seqeval ---------------
    seq_payload = None
    seq_decision = None
    try:
        from sensorflow.megaeval import population as pop_mod
        from sensorflow.seqeval.controller import evaluate_regression

        pop_name = "rotr-regression-pop"
        pop_id = next((p["population_id"] for p in pop_mod.list_populations()
                       if p.get("name") == pop_name), None)
        if pop_id is None:
            pop_id = pop_mod.generate_population(
                pop_name, num_objects=60_000, seed=17)["population_id"]
        effects = _seqeval_effects(bm, cm)
        seq_payload = evaluate_regression(
            pop_id,
            {"model_version": baseline_run.get("model_version", "baseline")},
            {"model_version": candidate_run.get("model_version", "candidate"),
             "effects": effects},
            policy={"target_n": 6000, "safety_floor": 800,
                    "min_per_stratum": 120,
                    "escalation": {"enabled": True,
                                   "max_extra_per_stratum": 600,
                                   "batch_objects": 300}},
            seed=seed)
        seq_payload["translated_effects"] = effects
        seq_decision = seq_payload.get("decision")
    except Exception as e:                             # honest degradation
        seq_payload = {"error": f"seqeval delegation unavailable: {e}"}

    # --- distribution-driven check (megaeval shift + rca concepts) -------
    # Compare the OPPORTUNITY mix (bank composition), not the committed mix:
    # committed-mix changes are model behavior, which is the signal, not a
    # confound. Same-seed banks have identical opportunity mixes (JS = 0).
    def _mix(run: Dict) -> List[str]:
        return [f"{s['kind']}|{s['visibility']}|{s['lighting']}"
                for s in run["scenario_summaries"]]

    js = js_divergence_categorical(_mix(baseline_run), _mix(candidate_run)) \
        if baseline_run.get("scenario_summaries") and \
        candidate_run.get("scenario_summaries") else 0.0
    distribution_driven = js > DISTRIBUTION_JS_THRESHOLD
    dist_note = (f"cohort-mix JS divergence {js:.4f} "
                 + ("EXCEEDS" if distribution_driven else "below")
                 + f" {DISTRIBUTION_JS_THRESHOLD} — "
                 + ("observed deltas may reflect a different committed-"
                    "violation mix, not model behavior; verify before acting"
                    if distribution_driven else
                    "committed-violation mix comparable between runs"))

    # --- six-outcome distinction -----------------------------------------
    six = {
        "observed_difference": any(abs(d) > 0 for d in deltas.values()),
        "statistically_significant": seq_decision == "REGRESSION",
        "practically_significant": (
            abs(deltas.get("violation_rate_ALL", 0.0))
            >= PRACTICAL_ROTR_RECALL_PP
            or abs(deltas.get("violation_rate_VRU", 0.0))
            >= PRACTICAL_SC_RECALL_PP
            or abs(deltas.get("rotr_recall", 0.0)) >= PRACTICAL_ROTR_RECALL_PP),
        "safety_critical": (deltas.get("violation_rate_VRU", 0.0)
                            >= PRACTICAL_SC_RECALL_PP
                            or deltas.get("sc_rotr_recall", 0.0)
                            <= -PRACTICAL_SC_RECALL_PP
                            or deltas.get("cfr", 0.0) > 0),
        "insufficient_evidence": seq_decision == "INSUFFICIENT"
        or seq_decision is None,
        "distribution_driven": distribution_driven,
    }
    if six["safety_critical"] and six["statistically_significant"]:
        primary = "SAFETY_CRITICAL_REGRESSION"
    elif six["statistically_significant"]:
        primary = "STATISTICALLY_SIGNIFICANT_REGRESSION"
    elif six["insufficient_evidence"]:
        primary = "INSUFFICIENT_EVIDENCE"
    elif six["distribution_driven"]:
        primary = "DISTRIBUTION_DRIVEN"
    elif six["practically_significant"]:
        primary = "PRACTICALLY_SIGNIFICANT_DIFFERENCE"
    elif six["observed_difference"]:
        primary = "OBSERVED_DIFFERENCE_ONLY"
    else:
        primary = "NO_DIFFERENCE"

    return RegressionResult(
        regression_id=regression_id,
        baseline_run_id=baseline_run["run_id"],
        candidate_run_id=candidate_run["run_id"],
        baseline_model=baseline_run.get("model_version", ""),
        candidate_model=candidate_run.get("model_version", ""),
        metric_deltas=deltas, six_outcomes=six, primary_outcome=primary,
        seqeval=seq_payload, distribution_note=dist_note,
        provenance=Provenance(
            software_version=f"{SOFTWARE_VERSION}/metrics",
            calibration_version=CALIBRATION_VERSION, source="SYNTHETIC"))
