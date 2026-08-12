"""MITL Copilot: advisory analysis of evaluation failures.

Routes to local Ollama/Gemma when reachable; otherwise produces a deterministic
offline analysis derived from the actual evaluation evidence. The Copilot is
advisory only: it never changes metrics, approves/rejects labels, or overrides
gates (spec §35).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import httpx

from sensorflow.evaluation import reporting
from sensorflow.evaluation.records import EvalStore

OLLAMA_ENDPOINTS = [
    {"url": "http://localhost:11434/api/chat", "model": "gemma3:latest"},
    {"url": "http://dgx-spark.tail16d8d9.ts.net:11434/api/chat", "model": "gemma4:26b"},
]

REASON_HINTS = {
    "LOW_IOU": ("Geometric mismatch vs reference", "box regression head under-fitting for this class/range"),
    "POSITION_ERROR": ("Center offset beyond tolerance", "LiDAR-camera extrinsic drift or late fusion latency"),
    "ORIENTATION_ERROR": ("Heading estimate off", "symmetric object ambiguity / yaw flipping in the detector"),
    "INSUFFICIENT_POINT_SUPPORT": ("Too few LiDAR returns inside box", "occlusion, range sparsity, or hallucinated detection"),
    "ANOMALY": ("Statistical outlier vs population", "feature combination rarely seen in training distribution"),
    "GRADER_DISAGREEMENT": ("Independent graders disagree", "genuinely ambiguous object class or degraded viewpoint"),
    "TRACK_FRAGMENTATION": ("Track split into fragments", "missed detections breaking association continuity"),
    "ID_SWITCH": ("Identity switched mid-track", "crossing objects with similar appearance/kinematics"),
    "MODEL_REGRESSION": ("Model version regressed", "training data shift or a bad checkpoint promotion"),
    "LOW_CONFIDENCE": ("Detector confidence very low", "borderline detection near the score threshold"),
    "SENSOR_DISAGREEMENT": ("Camera and LiDAR disagree", "single-sensor failure mode (glare, rain scatter, dropout)"),
}


def _gather_context(store: EvalStore, req: Dict) -> Dict:
    ctx: Dict = {"context_type": req.get("context_type", "general")}
    aid = req.get("annotation_id")
    if aid:
        rec = reporting.evaluation_record(store, aid)
        if rec:
            ctx["evaluation_record"] = rec
    eid = req.get("event_id")
    if eid:
        ev = store.get("rare_events", eid)
        if ev:
            ctx["rare_event"] = ev.model_dump()
    if req.get("context_type") == "regression" or req.get("model_version"):
        regs = sorted(store.all("regressions"), key=lambda r: r.date)
        if req.get("model_version"):
            regs = [r for r in regs if r.model_version == req["model_version"]] or regs
        if regs:
            ctx["regression"] = regs[-1].model_dump()
    return ctx


def _offline_analysis(ctx: Dict) -> Dict:
    """Deterministic rule-based analysis assembled from real evidence."""
    ctype = ctx.get("context_type", "general")
    evidence: List[str] = []
    factors: List[str] = []
    investigation: List[str] = []
    classification = ctype.replace("_", " ").title()
    likely_cause = "Insufficient evidence gathered for a specific cause."
    confidence = 0.5

    rec = ctx.get("evaluation_record")
    if rec:
        geo = rec["geometry"]
        an = rec["anomaly"]
        gr = rec["grading"]
        dec = rec.get("decision") or {}
        reasons = dec.get("failure_reasons", [])
        if reasons:
            classification = reasons[0].replace("_", " ").title()
        if geo.get("iou_3d") is not None:
            evidence.append(f"3D IoU vs reference = {geo['iou_3d']:.3f}")
        if geo.get("position_error") is not None:
            evidence.append(f"Position error = {geo['position_error']:.2f} m")
        if geo.get("orientation_error_deg") is not None:
            evidence.append(f"Orientation error = {geo['orientation_error_deg']:.1f}°")
        evidence.append(f"Anomaly ensemble score = {an['score']:.3f} "
                        f"(threshold {an['decision_threshold']:.2f}, strategy {an['ensemble_strategy']})")
        if an.get("detector_scores"):
            top = sorted(an["normalized_scores"].items(), key=lambda kv: -kv[1])[:3]
            evidence.append("Highest detector votes: " + ", ".join(f"{k}={v:.2f}" for k, v in top))
        if gr.get("consensus") is not None:
            evidence.append(f"Grader consensus = {gr['consensus']:.2%} across {gr['grader_count']} graders")
        if rec["tracking"]["id_switch"]:
            evidence.append("Tracking evidence shows an identity switch on this object's track")
        if rec["tracking"]["fragmentation"]:
            evidence.append("Tracking evidence shows track fragmentation")
        primary = dec.get("primary_failure_reason")
        if primary and primary in REASON_HINTS:
            headline, cause = REASON_HINTS[primary]
            likely_cause = f"{headline}: most consistent with {cause}."
            confidence = 0.72
        for r in reasons[1:4]:
            if r in REASON_HINTS:
                factors.append(f"{r}: {REASON_HINTS[r][1]}")
        investigation.extend([
            "Open the frame in HITL review and compare camera vs LiDAR evidence",
            "Check neighboring frames of the same track for consistent behavior",
            "Verify the reference ground-truth type and coverage before trusting the comparison",
        ])

    ev = ctx.get("rare_event")
    if ev:
        classification = f"Rare Event: {ev['scenario_type'].replace('_', ' ').title()}"
        evidence.append(f"Scenario {ev['scenario_type']} severity={ev['severity']} "
                        f"rarity={ev['rarity_score']:.2f} anomaly={ev['anomaly_score']:.2f}")
        for sensor, detail in ev.get("sensor_evidence", {}).items():
            evidence.append(f"{sensor}: {detail}")
        likely_cause = "Genuine rare scenario rather than a labeling defect; value lies in adding it to training."
        investigation.append("Verify the event, then include its frames in the next training dataset")
        confidence = 0.68

    reg = ctx.get("regression")
    if reg and ctype == "regression":
        classification = "Model Performance Regression"
        for d in reg.get("deltas", [])[:6]:
            if d["regressed"]:
                evidence.append(f"{d['metric']}: {d['baseline']:.3f} → {d['current']:.3f} "
                                f"(Δ {d['delta']:+.3f}, tolerance {d['tolerance']})")
        if reg.get("affected_classes"):
            factors.append("Affected classes: " + ", ".join(reg["affected_classes"]))
        likely_cause = ("Training-data distribution shift or label-quality drop between model versions; "
                        "the regressed metrics cluster in " + ", ".join(reg.get("kinds", ["performance"])) + ".")
        investigation.extend([
            "Diff the training dataset lineage between the two model versions",
            "Re-run the benchmark on the affected classes only",
            "Inspect flagged labels from the newer model for systematic geometry errors",
        ])
        confidence = 0.7

    if not evidence:
        evidence.append("No structured evidence was attached to this request.")
        investigation.append("Select a specific annotation, rare event, or regression entry and retry")

    hypothesis = (f"HYPOTHESIS (not a conclusion): {likely_cause} "
                  "This is generated from deterministic evidence rules and requires human confirmation.")

    lines = [f"# {classification}", "", "## Observed evidence"]
    lines += [f"- {e}" for e in evidence]
    if factors:
        lines += ["", "## Contributing factors"] + [f"- {f}" for f in factors]
    lines += ["", "## Likely cause", likely_cause, "", "## Hypothesis", hypothesis,
              "", "## Recommended investigation"] + [f"- {i}" for i in investigation]
    lines += ["", f"_Advisory only — confidence {confidence:.0%}. The Copilot never changes metrics or overrides gates._"]

    return {
        "analysis": "\n".join(lines),
        "structured": {
            "failure_classification": classification,
            "observed_evidence": evidence,
            "likely_cause": likely_cause,
            "contributing_factors": factors,
            "hypothesis": hypothesis,
            "recommended_investigation": investigation,
            "confidence": confidence,
        },
    }


def explain(store: EvalStore, req: Dict) -> Dict:
    """Try Ollama first; fall back to the deterministic offline analysis."""
    ctx = _gather_context(store, req)
    offline = _offline_analysis(ctx)

    prompt = (
        "You are an advisory MITL Copilot for an autonomous-driving label evaluation platform. "
        "You never change metrics or approve/reject labels; you explain evidence.\n\n"
        f"CONTEXT TYPE: {ctx.get('context_type')}\n"
        f"EVIDENCE (JSON):\n{json.dumps({k: v for k, v in ctx.items() if k != 'context_type'}, indent=2)[:6000]}\n\n"
        "Produce: failure classification, observed evidence, likely cause, contributing factors, "
        "a clearly-labeled hypothesis, recommended investigation steps, and a confidence estimate. "
        "Be concise and concrete."
    )
    for ep in OLLAMA_ENDPOINTS:
        try:
            res = httpx.post(ep["url"], json={
                "model": ep["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }, timeout=20.0)
            if res.status_code == 200:
                text = res.json().get("message", {}).get("content", "")
                if text:
                    return {
                        "status": "ok",
                        "provider": ep["url"],
                        "analysis": text,
                        "structured": offline["structured"],  # deterministic evidence summary retained
                    }
        except Exception:
            continue

    return {
        "status": "ok",
        "provider": "offline_deterministic",
        "analysis": offline["analysis"],
        "structured": offline["structured"],
    }
