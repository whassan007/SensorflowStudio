"""Final investigation report: markdown + JSON assembly."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from sensorflow.rca import diagnostics as dg, scoring
from sensorflow.rca.models import (HYPOTHESIS_LABELS, Investigation, STAGES)
from sensorflow.rca.scenario import ScenarioBundle

_REMEDIATION: Dict[str, Dict[str, List[str]]] = {
    "TRUE_MODEL_REGRESSION": {
        "containment": ["Halt the rollout / keep B out of the serving path",
                        "Pin A as the serving model; keep B in shadow only"],
        "short_term": ["Slice the regression by segment and error type to "
                       "find what B lost",
                       "Fine-tune B on recent-window data covering the "
                       "regressing segments"],
        "long_term": ["Add a recent-window eval set to the offline gate",
                      "Automate freshness checks on eval data before any "
                      "launch decision"],
    },
    "DISTRIBUTION_SHIFT": {
        "containment": ["Do not treat either aggregate as a launch signal; "
                        "gate on segment-level deltas"],
        "short_term": ["Reweight offline eval to the production mix and "
                       "re-decide", "Target B's night/urban weakness with "
                       "focused training data"],
        "long_term": ["Continuously monitor production mix vs eval mix (PSI "
                      "alerting)", "Stratified launch gates per segment"],
    },
    "FEATURE_SKEW": {
        "containment": ["Freeze the rollout; the online model is consuming "
                        "corrupted features"],
        "short_term": ["Fix the unit/normalization difference in the online "
                       "feature pipeline; add a unit test on the skewed "
                       "feature", "Re-run shadow after the fix"],
        "long_term": ["Continuous training-serving feature parity monitoring "
                      "with per-feature SMD alerts",
                      "Version-pin and log feature pipeline versions in both "
                      "environments"],
    },
    "SERVING_MISMATCH": {
        "containment": ["Align shadow serving config (threshold, precision) "
                        "with the evaluated artifact before reading any "
                        "shadow number"],
        "short_term": ["Re-run shadow with the offline config; if the gap "
                       "closes, ship config alignment",
                       "Calibrate B's confidence to the serving threshold"],
        "long_term": ["Config parity checks in CI between eval harness and "
                      "serving stack", "One artifact, one config: promote the "
                      "exact evaluated bundle"],
    },
    "LABEL_LATENCY": {
        "containment": ["Suspend shadow-based verdicts until labels mature"],
        "short_term": ["Re-grade the frozen cohort post-maturity",
                       "Exclude provisional labels from headline metrics"],
        "long_term": ["Separate provisional and mature metrics in dashboards",
                      "Track label-age distribution as a first-class metric"],
    },
    "SAMPLING_BIAS": {
        "containment": ["Stop reading the triage-sampled stream as an "
                        "unbiased estimate"],
        "short_term": ["Add a parallel uniform sample for measurement",
                       "Reweight existing sample by inclusion probability "
                       "(if logged)"],
        "long_term": ["Log inclusion probabilities for every sampled unit",
                      "Keep measurement sampling separate from triage "
                      "sampling"],
    },
    "STATISTICAL_NOISE": {
        "containment": ["Make no launch decision on current evidence; both "
                        "+5% and -2% are within noise"],
        "short_term": ["Extend collection to the pre-computed effective-n "
                       "target", "Use paired, cluster-aware tests only"],
        "long_term": ["Pre-register power calculations before shadow "
                      "experiments", "Adopt anytime-valid sequential tests "
                      "(seqeval) to allow early stopping without p-hacking"],
    },
    "OFFLINE_CONTAMINATION": {
        "containment": ["Retract the +5% claim; do not use the current "
                        "offline set for any decision"],
        "short_term": ["Rebuild the eval set with entity-level dedup against "
                       "B's training data; re-run both models",
                       "Pin versions and record environment locks"],
        "long_term": ["Automated leakage scans in the eval-set build "
                      "pipeline", "Entity-level splitting enforced by "
                      "tooling, not convention"],
    },
    "NO_PRACTICAL_REGRESSION": {
        "containment": ["No containment needed"],
        "short_term": ["Proceed with a guarded rollout with segment-level "
                       "monitoring"],
        "long_term": ["Tighten practical-margin definitions per surface"],
    },
}

_STAGE_HEADLINE = {
    "comparison_validity": "Are the two numbers commensurable?",
    "offline_audit": "Is the +5% real and reproducible?",
    "population_validation": "Enough volume, comparable populations?",
    "distribution_shift": "Did the world move between the two evals?",
    "conditional_performance": "Where does the delta live?",
    "paired_comparison": "Which units actually flipped?",
    "statistical_significance": "Is -2% distinguishable from noise?",
    "feature_parity": "Same features online and offline?",
    "serving_parity": "Same config online and offline?",
    "shadow_traffic": "Is the scored sample fair?",
    "label_integrity": "Is shadow ground truth actually true?",
}


def build_report(bundle: ScenarioBundle, inv: Investigation,
                 battery: Optional[Dict] = None) -> Dict:
    battery = battery or dg.run_all(bundle)
    scoreboard = scoring.build_scoreboard(bundle, inv, battery=battery)
    tree = scoring.evaluate_decision_tree(bundle, battery=battery)
    experiments = scoring.recommend_experiments(bundle, scoreboard, battery)

    top = scoreboard["rows"][0]
    conclusion = tree["conclusion"]
    remediation = _REMEDIATION.get(conclusion or top["hypothesis"],
                                   _REMEDIATION[top["hypothesis"]])

    stage_summaries = []
    for key, (data, findings) in battery.items():
        mism = [f for f in findings if f.status == "MISMATCH"]
        unk = [f for f in findings if f.status == "UNKNOWN"]
        stage_summaries.append({
            "stage": key,
            "headline": _STAGE_HEADLINE.get(key, key),
            "mismatches": [f.title for f in mism],
            "unknowns": [f.title for f in unk],
            "n_pass": sum(1 for f in findings if f.status == "PASS"),
            "verdict": ("mismatch" if mism else
                        "unknown" if unk else "clean"),
        })

    acked = [e for e in inv.events if e["kind"] == "unknowns_acknowledged"]

    payload = {
        "investigation_id": inv.id,
        "name": inv.name,
        "generated_at": time.time(),
        "models": {"baseline": inv.baseline_model,
                   "candidate": inv.candidate_model},
        "claims": inv.claims,
        "executive_finding": {
            "conclusion": conclusion,
            "conclusion_kind": tree["conclusion_kind"],
            "top_hypothesis": top["hypothesis"],
            "label": HYPOTHESIS_LABELS.get(top["hypothesis"], top["hypothesis"]),
            "confidence": top["auto_confidence"],
            "score": top["score"],
            "score_gap_to_runner_up": scoreboard["score_gap"],
        },
        "stage_summaries": stage_summaries,
        "hypothesis_ranking": [
            {"rank": r["rank"], "hypothesis": r["hypothesis"],
             "score": r["score"], "confidence": r["auto_confidence"],
             "human_confidence": r["human_confidence"],
             "evidence_for": len(r["evidence_for"]),
             "evidence_against": len(r["evidence_against"])}
            for r in scoreboard["rows"]],
        "decision_path": tree["path"],
        "minimum_additional_evidence": experiments["minimum_additional_evidence"],
        "recommended_experiments": experiments["experiments"],
        "remediation": remediation,
        "acknowledged_unknowns": acked,
        "human_findings": [f.to_dict() for f in inv.findings
                           if f.source == "human"],
    }
    payload["markdown"] = _to_markdown(payload, inv)
    return payload


def _to_markdown(p: Dict, inv: Investigation) -> str:
    ef = p["executive_finding"]
    lines = [
        f"# RCA Report — {p['name']}",
        "",
        f"**Baseline:** `{p['models']['baseline']}` &nbsp; "
        f"**Candidate:** `{p['models']['candidate']}`",
        "",
        f"**Claims under investigation:** offline "
        f"{p['claims'].get('offline_delta_pp', '+5')}pp vs shadow "
        f"{p['claims'].get('shadow_delta_pp', '-2')}pp "
        f"({p['claims'].get('metric', 'accuracy')})",
        "",
        "## Executive finding",
        "",
        f"**{ef['conclusion'] or ef['top_hypothesis']}** — "
        f"{ef['label']} (confidence: **{ef['confidence']}**, score "
        f"{ef['score']}, gap to runner-up {ef['score_gap_to_runner_up']}).",
        "",
        "## Hypothesis ranking",
        "",
        "| # | Hypothesis | Score | Confidence | For | Against |",
        "|---|-----------|-------|------------|-----|---------|",
    ]
    for r in p["hypothesis_ranking"]:
        hc = f" (human: {r['human_confidence']})" if r["human_confidence"] else ""
        lines.append(f"| {r['rank']} | {r['hypothesis']} | {r['score']} | "
                     f"{r['confidence']}{hc} | {r['evidence_for']} | "
                     f"{r['evidence_against']} |")

    lines += ["", "## Stage summaries", ""]
    for s in p["stage_summaries"]:
        badge = {"mismatch": "MISMATCH", "unknown": "UNKNOWN",
                 "clean": "clean"}[s["verdict"]]
        lines.append(f"- **{s['stage']}** ({s['headline']}) — {badge}")
        for m in s["mismatches"]:
            lines.append(f"    - ⚠ {m}")
        for u in s["unknowns"]:
            lines.append(f"    - ? {u}")

    if p["acknowledged_unknowns"]:
        lines += ["", "## Acknowledged unknowns", ""]
        for e in p["acknowledged_unknowns"]:
            lines.append(f"- {e['message']} — note: "
                         f"{e['data'].get('note', '')}")

    lines += ["", "## Minimum additional evidence", "",
              p["minimum_additional_evidence"], "",
              "## Recommended experiments", ""]
    for d in p["recommended_experiments"]:
        lines.append(f"{d['rank']}. **{d['design']}** (cost: {d['cost']}, "
                     f"gain {d['information_gain']}) — {d['description']}")

    lines += ["", "## Remediation", "", "**Containment (now):**"]
    lines += [f"- {x}" for x in p["remediation"]["containment"]]
    lines += ["", "**Short term:**"]
    lines += [f"- {x}" for x in p["remediation"]["short_term"]]
    lines += ["", "**Long term:**"]
    lines += [f"- {x}" for x in p["remediation"]["long_term"]]
    return "\n".join(lines)
