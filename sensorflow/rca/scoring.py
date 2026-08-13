"""Root-cause scoring board, decision-tree evaluation, and experiment ranking.

The scoreboard is rule-based and auditable: each of the 8 hypotheses has an
explicit list of finding codes that count for/against it, with weights. Every
score is traceable to the stage findings that produced it (evidence links),
and the user can overlay their own confidence assessment without erasing the
auto one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sensorflow.rca import diagnostics as dg
from sensorflow.rca.models import (Finding, HYPOTHESIS_LABELS, Investigation,
                                   ROOT_CAUSES)
from sensorflow.rca.scenario import ScenarioBundle

# (code_prefix, weight). A finding matches a rule when its code starts with
# the prefix, so FP_FEATURE_SKEW matches FP_FEATURE_SKEW:obj_distance_m.
_RULES: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "OFFLINE_CONTAMINATION": {
        "for": [("OA_LEAKAGE_DUPLICATES", 3.0), ("OA_REPRO_FAIL", 2.5),
                ("OA_SPLIT_ROW_LEVEL", 1.5), ("OA_PINS_MISSING", 1.0),
                ("OA_TEMPORAL_LEAK", 2.0)],
        "against": [("OA_LEAKAGE_CLEAN", 2.5), ("OA_REPRO_OK", 1.5),
                    ("OA_SPLIT_ENTITY_LEVEL", 0.5)],
    },
    "DISTRIBUTION_SHIFT": {
        "for": [("DS_SHIFT_HIGH", 2.5), ("CP_SIMPSONS_DETECTED", 3.0)],
        "against": [("DS_SHIFT_LOW", 2.5), ("DS_VOLUME_TOO_SMALL", 0.5),
                    ("CP_NO_PATTERN", 0.5)],
    },
    "FEATURE_SKEW": {
        "for": [("FP_FEATURE_SKEW", 3.5), ("DS_FEATURE_ONLY_SHIFT", 1.5),
                ("SP_UNKNOWN:feature_pipeline_version", 1.0),
                ("CV_UNKNOWN:feature_pipeline_version", 0.5),
                ("CP_SEGMENT_CONCENTRATED", 0.75)],
        "against": [("FP_CLEAN", 3.0)],
    },
    "SERVING_MISMATCH": {
        "for": [("SP_CONFIG_DIFF", 2.0), ("PC_CONF_BAND_CONCENTRATION", 2.5),
                ("CV_MISMATCH:confidence_threshold", 0.5),
                ("CV_MISMATCH:quantization", 0.5)],
        "against": [("SP_CLEAN", 3.0)],
    },
    "LABEL_LATENCY": {
        "for": [("LI_MATURE_DIVERGES", 3.0), ("LI_PROVISIONAL_HIGH", 1.5),
                ("LI_DIFFICULTY_CORRELATED", 1.5), ("LI_POLICY_DIFF", 1.0),
                ("CV_MISMATCH:label_policy_version", 0.5)],
        "against": [("LI_CLEAN", 3.0)],
    },
    "SAMPLING_BIAS": {
        "for": [("ST_SELECTION_BIAS", 3.5), ("ST_DROPS_HIGH", 1.0)],
        "against": [("ST_SAMPLE_FAIR", 3.0)],
    },
    "STATISTICAL_NOISE": {
        "for": [("SS_INSUFFICIENT_EVIDENCE", 2.0), ("SS_LOW_ESS", 2.0),
                ("POP_VOLUME_LOW", 1.5), ("POP_SMALL_OFFLINE", 1.0),
                ("DS_VOLUME_TOO_SMALL", 0.5)],
        "against": [("SS_SIGNIFICANT_REGRESSION", 3.0), ("SS_NO_SIG_DIFF", 1.0),
                    ("POP_VOLUME_OK", 1.5)],
    },
    "TRUE_MODEL_REGRESSION": {
        "for": [("SS_SIGNIFICANT_REGRESSION", 2.5), ("CP_UNIFORM_REGRESSION", 1.5),
                ("CV_WINDOW_STALE", 0.75),
                # Clean bill of health on every artifact channel is itself
                # evidence that the regression is real.
                ("OA_LEAKAGE_CLEAN", 0.6), ("FP_CLEAN", 0.6), ("SP_CLEAN", 0.6),
                ("ST_SAMPLE_FAIR", 0.6), ("LI_CLEAN", 0.6), ("DS_SHIFT_LOW", 0.6)],
        "against": [("OA_LEAKAGE_DUPLICATES", 2.0), ("OA_REPRO_FAIL", 1.5),
                    ("FP_FEATURE_SKEW", 2.0), ("SP_CONFIG_DIFF", 1.5),
                    ("ST_SELECTION_BIAS", 2.0), ("LI_MATURE_DIVERGES", 2.0),
                    ("CP_SIMPSONS_DETECTED", 2.0), ("DS_SHIFT_HIGH", 1.5),
                    ("SS_INSUFFICIENT_EVIDENCE", 1.5),
                    ("PC_CONF_BAND_CONCENTRATION", 1.0)],
    },
}

_NEXT_TEST: Dict[str, str] = {
    "TRUE_MODEL_REGRESSION":
        "Split the shadow window into halves and check the paired delta is "
        "stable; re-run the offline eval on freshly-labeled recent data "
        "(same model, different data).",
    "DISTRIBUTION_SHIFT":
        "Reweight the scored shadow units to the offline segment mix and "
        "recompute the paired delta -- if it turns positive, the mix is the "
        "whole story.",
    "FEATURE_SKEW":
        "Replay ~500 shadow frames through the offline feature pipeline and "
        "diff feature values unit-by-unit (same data, different pipeline); "
        "start with the top-ranked parity feature.",
    "SERVING_MISMATCH":
        "Score the offline eval set through the shadow serving stack (same "
        "data, different pipeline) with configs pinned one at a time "
        "(threshold first, then quantization).",
    "LABEL_LATENCY":
        "Freeze the cohort and re-grade after labels mature past the audit "
        "threshold; compare against the provisional-label verdict.",
    "SAMPLING_BIAS":
        "Score a uniform random sample of the eligible stream (bypass the "
        "triage sampler) and compare the paired delta.",
    "STATISTICAL_NOISE":
        "Extend shadow collection until the cluster-aware CI half-width is "
        "below the practical margin; pre-register the stopping rule "
        "(anytime-valid sequential test).",
    "OFFLINE_CONTAMINATION":
        "Dedup the offline eval set against B's training data at entity "
        "level, pin versions, and re-run the offline comparison.",
}


def _confidence(score: float, matched: int) -> str:
    """UNKNOWN = no evidence touches this hypothesis yet. Otherwise the score
    maps to confidence THAT THIS IS THE CAUSE (negative = evidence against)."""
    if matched == 0:
        return "UNKNOWN"
    if score >= 5.0:
        return "HIGH"
    if score >= 2.5:
        return "MEDIUM"
    return "LOW"


def build_scoreboard(bundle: ScenarioBundle,
                     inv: Optional[Investigation] = None,
                     battery: Optional[Dict] = None,
                     recorded_only: bool = False) -> Dict:
    """Score all 8 hypotheses from the full diagnostic battery plus any
    human-recorded findings; overlay human assessments if present.

    With recorded_only=True, only findings already recorded on the
    investigation are scored (auto findings accrue as the engineer works
    through the stages) -- used for the evolving working-hypothesis banner so
    the board never leaks conclusions from stages not yet visited."""
    if recorded_only and inv is not None:
        findings: List[Finding] = list(inv.findings)
    else:
        battery = battery or dg.run_all(bundle)
        findings = [f for _, (_, fs) in battery.items() for f in fs]
        if inv is not None:
            findings = findings + [f for f in inv.findings if f.source == "human"]

    rows = []
    for hyp in ROOT_CAUSES:
        rules = _RULES[hyp]
        ev_for, ev_against = [], []
        score = 0.0
        for f in findings:
            for prefix, w in rules["for"]:
                if f.code.startswith(prefix):
                    score += w
                    ev_for.append({"code": f.code, "stage": f.stage,
                                   "title": f.title, "weight": w,
                                   "severity": f.severity, "status": f.status,
                                   "finding_id": f.id})
                    break
            else:
                for prefix, w in rules["against"]:
                    if f.code.startswith(prefix):
                        score -= w
                        ev_against.append({"code": f.code, "stage": f.stage,
                                           "title": f.title, "weight": w,
                                           "severity": f.severity,
                                           "status": f.status,
                                           "finding_id": f.id})
                        break
        matched = len(ev_for) + len(ev_against)
        auto_conf = _confidence(score, matched)
        human = (inv.human_assessments.get(hyp) if inv is not None else None) or {}
        rows.append({
            "hypothesis": hyp,
            "label": HYPOTHESIS_LABELS[hyp],
            "score": round(score, 2),
            "evidence_for": ev_for,
            "evidence_against": ev_against,
            "auto_confidence": auto_conf,
            "human_confidence": human.get("confidence"),
            "human_note": human.get("note", ""),
            "next_discriminating_test": _NEXT_TEST[hyp],
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    top = rows[0]
    runner = rows[1] if len(rows) > 1 else None
    # Skeptical semantics: a hypothesis leaves the working set only when the
    # net recorded evidence is AGAINST it -- weak or absent evidence keeps it
    # in play (never a premature single conclusion).
    working_set = [r["hypothesis"] for r in rows
                   if r["score"] >= -0.5 or r["auto_confidence"] == "UNKNOWN"]
    return {
        "rows": rows,
        "top_hypothesis": top["hypothesis"],
        "top_confidence": top["auto_confidence"],
        "score_gap": round(top["score"] - (runner["score"] if runner else 0.0), 2),
        "working_hypothesis_set": working_set or [top["hypothesis"]],
        "explainer": ("Rule-based, auditable scoring: every point traces to a "
                      "stage finding. Human overrides are recorded next to -- "
                      "never instead of -- the automatic assessment."),
    }


# ------------------------------------------------------------- decision tree


def _has(findings: List[Finding], prefix: str) -> Optional[Finding]:
    for f in findings:
        if f.code.startswith(prefix):
            return f
    return None


def evaluate_decision_tree(bundle: ScenarioBundle,
                           battery: Optional[Dict] = None) -> Dict:
    """Walk the measurement-validity -> distribution -> parity -> significance
    causal chain with this investigation's actual findings at each node."""
    battery = battery or dg.run_all(bundle)
    findings = [f for _, (_, fs) in battery.items() for f in fs]

    def node(nid, question, yes_when_absent, bad_prefixes, unknown_prefixes,
             conclusion_if_no, next_id):
        bad = [p for p in bad_prefixes if _has(findings, p)]
        unk = [p for p in unknown_prefixes if _has(findings, p)]
        if bad:
            answer = "no"
        elif unk:
            answer = "unknown"
        else:
            answer = "yes" if yes_when_absent else "unknown"
        basis = [f.code for f in findings
                 if any(f.code.startswith(p) for p in
                        list(bad_prefixes) + list(unknown_prefixes))]
        return {"id": nid, "question": question, "answer": answer,
                "basis": basis, "conclusion_if_no": conclusion_if_no,
                "next": next_id}

    nodes = [
        node("offline_trustworthy",
             "Is the offline +5% measurement itself trustworthy "
             "(reproducible, leak-free)?", True,
             ["OA_LEAKAGE_DUPLICATES", "OA_REPRO_FAIL", "OA_TEMPORAL_LEAK"],
             ["OA_PINS_MISSING"],
             "OFFLINE_CONTAMINATION", "labels_trustworthy"),
        node("labels_trustworthy",
             "Are shadow ground-truth labels mature and unbiased?", True,
             ["LI_MATURE_DIVERGES"],
             ["LI_PROVISIONAL_HIGH"],
             "LABEL_LATENCY", "sample_representative"),
        node("sample_representative",
             "Is the scored shadow sample representative of eligible "
             "traffic?", True,
             ["ST_SELECTION_BIAS"], [],
             "SAMPLING_BIAS", "same_population"),
        node("same_population",
             "Do offline and shadow score comparable populations?", True,
             ["DS_SHIFT_HIGH", "CP_SIMPSONS_DETECTED"],
             ["DS_VOLUME_TOO_SMALL"],
             "DISTRIBUTION_SHIFT", "feature_parity"),
        node("feature_parity",
             "Do the models see the same feature values online as offline?",
             True,
             ["FP_FEATURE_SKEW"],
             ["SP_UNKNOWN:feature_pipeline_version"],
             "FEATURE_SKEW", "serving_parity"),
        node("serving_parity",
             "Is the serving configuration at parity with the offline "
             "harness?", True,
             ["SP_CONFIG_DIFF"],
             ["SP_UNKNOWN"],
             "SERVING_MISMATCH", "significant"),
    ]

    sig = _has(findings, "SS_SIGNIFICANT_REGRESSION")
    insuff = _has(findings, "SS_INSUFFICIENT_EVIDENCE")
    nodes.append({
        "id": "significant",
        "question": ("With all artifact channels checked, is the shadow "
                     "regression significant beyond the practical margin?"),
        "answer": "yes" if sig else ("unknown" if insuff else "no"),
        "basis": [f.code for f in findings if f.code.startswith("SS_")],
        "conclusion_if_no": None,
        "next": None,
    })

    # Walk the live path.
    path: List[str] = []
    conclusion = None
    conclusion_kind = "root_cause"
    for n in nodes:
        path.append(n["id"])
        if n["id"] == "significant":
            if n["answer"] == "yes":
                conclusion = "TRUE_MODEL_REGRESSION"
            elif n["answer"] == "unknown":
                conclusion = "STATISTICAL_NOISE"
                conclusion_kind = "insufficient_evidence"
            else:
                conclusion = "NO_PRACTICAL_REGRESSION"
                conclusion_kind = "no_regression"
            break
        if n["answer"] == "no":
            conclusion = n["conclusion_if_no"]
            break
        # 'unknown' proceeds but is recorded on the node; the stage state
        # machine is what forces the acknowledgment.

    return {"nodes": nodes, "path": path, "conclusion": conclusion,
            "conclusion_kind": conclusion_kind,
            "explainer": ("The causal chain rendered with this "
                          "investigation's actual answers. A 'no' anywhere "
                          "upstream invalidates naive readings of every "
                          "number downstream.")}


# ---------------------------------------------------------------- experiments


def recommend_experiments(bundle: ScenarioBundle,
                          scoreboard: Optional[Dict] = None,
                          battery: Optional[Dict] = None) -> Dict:
    battery = battery or dg.run_all(bundle)
    scoreboard = scoreboard or build_scoreboard(bundle, battery=battery)
    rows = scoreboard["rows"]
    by_hyp = {r["hypothesis"]: r for r in rows}
    top, runner = rows[0], rows[1]

    sig_data = battery["statistical_significance"][0]
    eff_n = sig_data["shadow_paired"]["effective_n"]
    margin = sig_data["practical_margin_pp"]
    se = sig_data["shadow_paired"]["se"]
    # n needed for the CI half-width to fit inside the practical margin.
    needed_n = int(eff_n * (1.96 * se / max(1e-9, margin / 100.0)) ** 2) + 1

    fp_rows = battery["feature_parity"][0]["rows"]
    top_feature = fp_rows[0]["feature"] if fp_rows else "top-ranked feature"

    designs = [
        {"id": "same_data_diff_pipeline",
         "design": "Same data, different pipeline",
         "description": (f"Replay ~500 scored shadow frames through the "
                         f"offline feature + serving pipeline and diff both "
                         f"feature values (start with {top_feature}) and "
                         "model outputs unit-by-unit."),
         "discriminates": ["FEATURE_SKEW", "SERVING_MISMATCH"],
         "cost": "low", "expected_days": 1},
        {"id": "same_model_diff_data",
         "design": "Same model, different data",
         "description": ("Re-run the offline evaluation on a fresh, "
                         "entity-deduped, recent-window eval set (and "
                         "reweighted to the shadow segment mix)."),
         "discriminates": ["OFFLINE_CONTAMINATION", "DISTRIBUTION_SHIFT",
                           "TRUE_MODEL_REGRESSION"],
         "cost": "medium", "expected_days": 2},
        {"id": "uniform_shadow_sample",
         "design": "Same traffic, different sampler",
         "description": ("Score a uniform random sample of eligible shadow "
                         "traffic, bypassing eligibility filters and the "
                         "triage sampler."),
         "discriminates": ["SAMPLING_BIAS"],
         "cost": "low", "expected_days": 1},
        {"id": "mature_label_regrade",
         "design": "Same units, different labels",
         "description": ("Freeze the scored cohort and re-grade once all "
                         "labels pass the maturity threshold; compare "
                         "verdicts."),
         "discriminates": ["LABEL_LATENCY"],
         "cost": "medium", "expected_days": 4},
        {"id": "extend_collection",
         "design": "More data, pre-registered stop",
         "description": (f"Extend shadow collection to ~{needed_n:,} "
                         f"effective units so the CI half-width fits inside "
                         f"the {margin:.1f}pp practical margin; use an "
                         "anytime-valid sequential test."),
         "discriminates": ["STATISTICAL_NOISE", "TRUE_MODEL_REGRESSION"],
         "cost": "medium", "expected_days": 5},
    ]

    # Information gain proxy: how much unresolved probability mass the
    # experiment can move (sum of positive scores it discriminates among,
    # with the top-2 gap counted double), divided by cost.
    cost_w = {"low": 1.0, "medium": 2.0, "high": 4.0}
    for d in designs:
        mass = sum(max(0.0, by_hyp[h]["score"]) for h in d["discriminates"])
        bonus = 3.0 if (top["hypothesis"] in d["discriminates"]
                        or runner["hypothesis"] in d["discriminates"]) else 0.0
        d["information_gain"] = round(mass + bonus, 2)
        d["priority"] = round((mass + bonus) / cost_w[d["cost"]], 2)
    designs.sort(key=lambda d: d["priority"], reverse=True)
    for i, d in enumerate(designs):
        d["rank"] = i + 1

    minimum = (
        f"Current leader: {top['hypothesis']} "
        f"({top['auto_confidence']}, score {top['score']}) vs runner-up "
        f"{runner['hypothesis']} (score {runner['score']}). The minimum "
        f"additional evidence to conclude is: {designs[0]['description']} "
        f"If that comes back clean, fall back to: {designs[1]['description']}")

    return {"experiments": designs, "minimum_additional_evidence": minimum,
            "power": {"effective_n": eff_n, "needed_effective_n": needed_n,
                      "practical_margin_pp": margin},
            "explainer": ("Counterfactual designs instantiated with this "
                          "investigation's specifics, ranked by information "
                          "gain per unit cost.")}
