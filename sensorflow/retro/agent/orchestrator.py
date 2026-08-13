"""Retrospective orchestrator: the typed four-stage agent pipeline.

    Log Agent          reads the artifact through the audited registry,
                       derives the failure type, extracts TIER1 facts,
                       and explicitly lists missing fields (UNKNOWN).
    Evidence Analyzer  drives the deterministic MetricCalculatorTool
                       (stopping distance, TTC, SCR impact, behavioral
                       impact) -> TIER2 derived facts.
    RAG Agent          builds failure-conditioned queries, retrieves
                       safety-case requirements (TIER3), historical
                       matches, and distribution findings.
    Safety Synthesizer LLM (Ollama) or deterministic scripted analysis
                       (mock) -> TIER4 hypotheses + severity PROPOSAL.

The deterministic policy engine then validates/overrides the proposal and
produces the launch determination; the assembled RetrospectiveScorecard is
persisted with its full audit trail.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from sensorflow.retro import AGENT_VERSION, store
from sensorflow.retro.inference.client import InferenceBackend, get_backend
from sensorflow.retro.agent.prompts import SYSTEM_PROMPT, synthesis_prompt
from sensorflow.retro.agent.scripted import (scripted_hypotheses,
                                             scripted_severity_proposal)
from sensorflow.retro.policy import (POLICY_VERSION, SeverityContext,
                                     adjudicate_severity, compute_severity,
                                     launch_gate)
from sensorflow.retro.scorecard import (EvidenceItem, EvidenceTier,
                                        RetrievedStandard,
                                        RetrospectiveScorecard,
                                        RootCauseHypothesis, Severity,
                                        StatSignificance, UNKNOWN,
                                        UncertaintyReport)
from sensorflow.retro.tools.builtin import build_registry
from sensorflow.retro.tools.registry import ToolRegistry

NON_HAZARD_CLASSES = {"plastic_bag", "debris", "cardboard", "balloon",
                      "traffic_sign", "vegetation", "steam", "shadow"}
DEFAULT_REACTION_TIME_S = 0.25


# --------------------------------------------------------- stage contracts

class LogExtract(BaseModel):
    log: Dict[str, Any]
    source_path: str
    present_fields: List[str]
    missing_fields: List[str]
    failure_type: str
    failure_type_rationale: str
    observed_evidence: List[EvidenceItem]


class EvidenceAnalysis(BaseModel):
    stopping: Dict[str, Any]
    ttc: Dict[str, Any]
    scr: Dict[str, Any]
    behavioral: Dict[str, Any]
    derived_evidence: List[EvidenceItem]
    unknown_metrics: List[str]


class RAGFindings(BaseModel):
    standards: List[RetrievedStandard]
    retrieved_evidence: List[EvidenceItem]
    historical_matches: List[Dict[str, Any]] = Field(default_factory=list)
    distribution_findings: List[str] = Field(default_factory=list)
    distribution_shift: Optional[Dict[str, Any]] = None
    queries: List[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    hypotheses: List[RootCauseHypothesis]
    ai_proposed_severity: Optional[Severity]
    behavioral_consequence: str
    provider: str


# ----------------------------------------------------------- 1. Log Agent

def _derive_failure_type(log: Dict[str, Any]) -> Tuple[str, str]:
    gt = log.get("ground_truth") or {}
    pred = log.get("prediction")
    planner = log.get("planner_response") or {}
    if gt and pred is None:
        return "FALSE_NEGATIVE", ("ground truth object present with no "
                                  "corresponding prediction/track")
    if pred and (pred.get("detection_delay_s") or 0.0) > 0.5:
        return "FALSE_NEGATIVE", (f"detection arrived "
                                  f"{pred.get('detection_delay_s')}s late — the "
                                  "object was effectively missed during the "
                                  "safety-relevant window (late detection)")
    gt_class = (gt.get("class") or "").lower()
    pred_class = (pred.get("class") or "").lower() if pred else ""
    if (gt_class in NON_HAZARD_CLASSES and pred_class
            and pred_class not in NON_HAZARD_CLASSES):
        if (planner.get("decel_mps2") or 0.0) > 0.5:
            return "FALSE_POSITIVE", (f"non-hazard '{gt_class}' predicted as "
                                      f"'{pred_class}' and the planner "
                                      "intervened: phantom intervention")
        return "FALSE_POSITIVE", (f"non-hazard '{gt_class}' predicted as "
                                  f"hazard '{pred_class}' (no intervention)")
    if pred and gt_class and pred_class and gt_class != pred_class:
        return "MISCLASSIFICATION", (f"ground truth '{gt_class}' predicted as "
                                     f"'{pred_class}'")
    return "UNKNOWN_FAILURE", "no deterministic rule matched this artifact"


def _obs(key: str, value: Any, statement: str, provenance: str) -> EvidenceItem:
    return EvidenceItem(tier=EvidenceTier.OBSERVED, key=key,
                        value=UNKNOWN if value is None else str(value),
                        statement=statement, provenance=provenance)


def run_log_agent(registry: ToolRegistry, fixture_id: Optional[str] = None,
                  path: Optional[str] = None) -> LogExtract:
    res = registry.call("log_reader",
                        {"fixture_id": fixture_id, "path": path})
    out = res.result or {}
    log = out["log"]
    failure_type, rationale = _derive_failure_type(log)

    gt = log.get("ground_truth") or {}
    pred = log.get("prediction") or {}
    ego = log.get("ego") or {}
    scenario = log.get("scenario") or {}
    planner = log.get("planner_response") or {}
    sensors = log.get("sensor_metadata") or {}
    traffic = (log.get("traffic_context") or {}).get("following_vehicle") or {}
    dis = log.get("disengagement") or {}
    col = log.get("collision") or {}
    models = log.get("models") or {}
    tracks = log.get("tracks") or []

    src = "eval log field"
    ev: List[EvidenceItem] = [
        _obs("ego_speed_mps", ego.get("speed_mps"),
             "ego speed at event time", f"{src} ego.speed_mps"),
        _obs("gt_class", gt.get("class"),
             "human-verified ground-truth object class", f"{src} ground_truth.class"),
        _obs("gt_distance", gt.get("distance_m"),
             "ground-truth object range from ego", f"{src} ground_truth.distance_m"),
        _obs("gt_relative_velocity", gt.get("relative_velocity_mps"),
             "ground-truth closing velocity (positive = closing)",
             f"{src} ground_truth.relative_velocity_mps"),
        _obs("pred_class", pred.get("class") if pred else None,
             "predicted object class", f"{src} prediction.class"),
        _obs("pred_confidence", pred.get("confidence") if pred else None,
             "detector confidence", f"{src} prediction.confidence"),
        _obs("pred_detection_delay", pred.get("detection_delay_s") if pred else None,
             "detection delay versus ground-truth appearance",
             f"{src} prediction.detection_delay_s"),
        _obs("pred_relative_velocity", pred.get("relative_velocity_mps") if pred else None,
             "predicted closing velocity", f"{src} prediction.relative_velocity_mps"),
        _obs("planner_action", planner.get("action"),
             "observed planner action", f"{src} planner_response.action"),
        _obs("planner_decel", planner.get("decel_mps2"),
             "observed planner deceleration (m/s^2)", f"{src} planner_response.decel_mps2"),
        _obs("planner_response_time", planner.get("response_time_s"),
             "planner response time after object appearance",
             f"{src} planner_response.response_time_s"),
        _obs("weather", scenario.get("weather"),
             "scenario weather", f"{src} scenario.weather"),
        _obs("odd_status", scenario.get("odd_status"),
             "inside/outside operational design domain", f"{src} scenario.odd_status"),
        _obs("following_gap", traffic.get("time_gap_s"),
             "rear traffic time gap", f"{src} traffic_context.following_vehicle.time_gap_s"),
        _obs("disengagement_occurred", dis.get("occurred"),
             "safety-driver disengagement", f"{src} disengagement.occurred"),
        _obs("collision_near_miss", col.get("near_miss"),
             "near-miss recorded", f"{src} collision.near_miss"),
        _obs("models", f"{models.get('baseline')} -> {models.get('candidate')}",
             "baseline and candidate model versions", f"{src} models"),
    ]
    if tracks:
        ev.append(_obs("track_class_flicker", tracks[0].get("class_flicker"),
                       "track-level class flicker observed", f"{src} tracks[0].class_flicker"))
        ev.append(_obs("track_velocity_variance", tracks[0].get("velocity_variance"),
                       "track velocity variance level", f"{src} tracks[0].velocity_variance"))
    if sensors:
        for name in ("camera", "lidar"):
            status = (sensors.get(name) or {}).get("status")
            if status is not None:
                ev.append(_obs(f"sensor_{name}_status", status,
                               f"{name} health/degradation state",
                               f"{src} sensor_metadata.{name}.status"))
    for f in out["missing_fields"]:
        ev.append(EvidenceItem(
            tier=EvidenceTier.OBSERVED, key=f"missing:{f}", value=UNKNOWN,
            statement=f"required field '{f}' is absent from the artifact — "
                      "treated as UNKNOWN, never guessed",
            provenance="field absent from artifact"))

    return LogExtract(log=log, source_path=out["source_path"],
                      present_fields=out["present_fields"],
                      missing_fields=out["missing_fields"],
                      failure_type=failure_type,
                      failure_type_rationale=rationale,
                      observed_evidence=ev)


# --------------------------------------------------- 2. Evidence Analyzer

def _drv(key: str, value: Any, statement: str, provenance: str) -> EvidenceItem:
    return EvidenceItem(tier=EvidenceTier.DERIVED, key=key,
                        value=UNKNOWN if value is None else str(value),
                        statement=statement, provenance=provenance)


def run_evidence_analyzer(registry: ToolRegistry,
                          extract: LogExtract) -> EvidenceAnalysis:
    log = extract.log
    ego = log.get("ego") or {}
    gt = log.get("ground_truth") or {}
    timing = log.get("timing") or {}

    sd_params = {
        "velocity_mps": ego.get("speed_mps"),
        "reaction_time_s": DEFAULT_REACTION_TIME_S,
        "system_latency_s": (timing.get("detection_latency_ms") or 0) / 1000.0,
        "planner_latency_s": (timing.get("planner_latency_ms") or 0) / 1000.0,
        "friction": ego.get("road_friction_estimate", 0.7),
        "grade": ego.get("grade", 0.0),
    }
    stopping = registry.call("metric_calculator", {
        "operation": "stopping_distance", "params": sd_params}).result["result"]

    obj_dims = gt.get("dimensions_m") or [0.6, 0.6]
    ttc = registry.call("metric_calculator", {
        "operation": "ttc",
        "params": {"distance_m": gt.get("distance_m"),
                   "closing_velocity_mps": gt.get("relative_velocity_mps"),
                   "obj_length_m": obj_dims[0], "obj_width_m": obj_dims[1]},
    }).result["result"]

    scr = registry.call("metric_calculator", {
        "operation": "scr_impact",
        "params": {"evaluation_context": log.get("evaluation_context")},
    }).result["result"]

    behavioral = registry.call("metric_calculator", {
        "operation": "behavioral_impact",
        "params": {"observed": log.get("planner_response"),
                   "counterfactual": log.get("counterfactual_planner_response")},
    }).result["result"]

    prov = "MetricCalculatorTool (deterministic)"
    ev: List[EvidenceItem] = []
    unknown: List[str] = []

    if stopping.get("stopping_distance_m") is not None:
        ev.append(_drv("stopping_distance", stopping["stopping_distance_m"],
                       f"stopping distance {stopping['stopping_distance_m']} m "
                       f"(reaction {stopping['reaction_distance_m']} m + braking "
                       f"{stopping['braking_distance_m']} m at "
                       f"{stopping['effective_decel_mps2']} m/s^2 effective decel; "
                       f"total reaction budget {stopping['total_reaction_time_s']} s "
                       "incl. perception+planner latency)",
                       f"{prov} stopping_distance"))
        if gt.get("distance_m") is not None:
            ratio = gt["distance_m"] / stopping["stopping_distance_m"]
            ev.append(_drv("distance_vs_stopping", round(ratio, 3),
                           f"object range {gt['distance_m']} m is {ratio:.2f}x "
                           "the stopping distance -> "
                           + ("INSIDE the stopping envelope"
                              if ratio <= 1.0 else
                              ("inside 1.5x criticality envelope" if ratio <= 1.5
                               else "outside the criticality envelope")),
                           f"{prov} stopping_distance + observed range"))
    else:
        unknown.append("stopping_distance")
        ev.append(_drv("stopping_distance", None,
                       f"stopping distance UNKNOWN: {stopping.get('unknown_reason')}",
                       f"{prov} stopping_distance"))

    if ttc.get("ttc_s") is not None:
        ev.append(_drv("ttc", ttc["ttc_s"],
                       f"projected TTC {ttc['ttc_s']} s (SSAM rectangle "
                       f"projection; closed-form cross-check "
                       f"{ttc.get('closed_form_ttc_s')} s); validity: "
                       + "; ".join(ttc.get("validity_flags", [])),
                       f"{prov} ttc"))
    else:
        if ttc.get("unknown_reason"):
            unknown.append("ttc")
            ev.append(_drv("ttc", None,
                           f"TTC UNKNOWN: {ttc['unknown_reason']}", f"{prov} ttc"))
        else:
            ev.append(_drv("ttc", "no_predicted_collision",
                           "no projected collision path (object not closing or "
                           "beyond look-ahead); validity: "
                           + "; ".join(ttc.get("validity_flags", [])),
                           f"{prov} ttc"))

    if scr.get("scr_impact") is not None:
        ev.append(_drv("scr_impact", scr["scr_impact"],
                       f"safety-critical recall {scr['scr_baseline']:.4f} -> "
                       f"{scr['scr_candidate']:.4f} (delta {scr['scr_impact']:+.4f} "
                       f"over {scr['denominator']} critical objects; criticality: "
                       f"{scr['criticality_policy']})",
                       f"{prov} scr_impact"))
    else:
        unknown.append("safety_critical_recall_impact")
        ev.append(_drv("scr_impact", None,
                       f"SCR impact UNKNOWN: {scr.get('unknown_reason')}",
                       f"{prov} scr_impact"))

    if behavioral.get("consequence") not in (None, "UNKNOWN"):
        ev.append(_drv("behavioral_impact", behavioral.get("decel_delta_mps2"),
                       f"behavioral impact: {behavioral['consequence']}",
                       f"{prov} behavioral_impact (observed vs corrected-"
                       "perception counterfactual)"))
    else:
        unknown.append("behavioral_impact")
        ev.append(_drv("behavioral_impact", None,
                       f"behavioral impact UNKNOWN: {behavioral.get('unknown_reason')}",
                       f"{prov} behavioral_impact"))

    return EvidenceAnalysis(stopping=stopping, ttc=ttc, scr=scr,
                            behavioral=behavioral, derived_evidence=ev,
                            unknown_metrics=unknown)


# ---------------------------------------------------------- 3. RAG Agent

def _build_queries(extract: LogExtract) -> List[str]:
    log = extract.log
    gt_class = ((log.get("ground_truth") or {}).get("class") or "object").lower()
    weather = ((log.get("scenario") or {}).get("weather") or "").lower()
    queries: List[str] = []
    if extract.failure_type == "FALSE_NEGATIVE":
        queries.append(f"{gt_class} missed detection inside stopping distance "
                       "recall requirement")
        if weather in ("rain", "snow", "fog"):
            queries.append(f"{weather} degradation detection recall confidence "
                           "threshold fusion")
    elif extract.failure_type == "FALSE_POSITIVE":
        queries.append(f"phantom braking false positive {gt_class} road debris "
                       "misclassification limits")
        queries.append("asymmetric false negative false positive cost context")
    else:
        queries.append(f"{gt_class} misclassification requirements "
                       "class stability")
    queries.append("launch criteria severity gate insufficient evidence")
    return queries


def run_rag_agent(registry: ToolRegistry, extract: LogExtract,
                  extra_queries: Optional[List[str]] = None,
                  k: int = 3) -> RAGFindings:
    queries = _build_queries(extract) + list(extra_queries or [])
    seen: Dict[str, RetrievedStandard] = {}
    for q in queries:
        res = registry.call("safety_standard_rag", {"query": q, "k": k})
        for hit in (res.result or {}).get("hits", []):
            std = RetrievedStandard(**hit)
            if std.chunk_id not in seen or std.relevance_score > seen[std.chunk_id].relevance_score:
                seen[std.chunk_id] = std
    standards = sorted(seen.values(), key=lambda s: -s.relevance_score)[:8]

    retrieved_ev = [EvidenceItem(
        tier=EvidenceTier.RETRIEVED, key=f"standard:{s.doc_id}:{s.chunk_id}",
        value=f"relevance {s.relevance_score:.2f}",
        statement=f"[{s.label}] {s.document} v{s.version} §{s.section}: "
                  f"{s.retrieved_text[:180]}…",
        provenance=f"SafetyStandardRAGTool hit {s.chunk_id} "
                   f"(source {s.source})") for s in standards]

    hist = registry.call("historical_failure_search", {
        "query_text": f"{extract.failure_type} "
                      f"{json.dumps((extract.log.get('scenario') or {}))[:300]}",
        "k": 3, "exclude_evaluation_id": extract.log.get("evaluation_id"),
    }).result or {}

    dist = registry.call("distribution_analysis", {
        "distribution_shift": (extract.log.get("evaluation_context") or {})
        .get("distribution_shift"),
    }).result or {}
    for m in hist.get("matches", []):
        retrieved_ev.append(EvidenceItem(
            tier=EvidenceTier.RETRIEVED, key=f"historical:{m['evaluation_id']}",
            value=f"similarity {m['similarity']:.2f}",
            statement=f"prior analyzed failure {m['evaluation_id']} "
                      f"({m.get('failure_type')}, {m.get('severity')}) is similar",
            provenance="HistoricalFailureSearchTool"))
    for f in dist.get("findings", []):
        retrieved_ev.append(EvidenceItem(
            tier=EvidenceTier.RETRIEVED, key="distribution_finding",
            value=None, statement=f,
            provenance=f"DistributionAnalysisTool ({dist.get('source')})"))

    return RAGFindings(standards=standards, retrieved_evidence=retrieved_ev,
                       historical_matches=hist.get("matches", []),
                       distribution_findings=dist.get("findings", []),
                       distribution_shift=dist.get("shift"),
                       queries=queries)


# -------------------------------------------------- 4. Safety Synthesizer

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_synthesizer(backend: InferenceBackend, registry: ToolRegistry,
                    extract: LogExtract, analysis: EvidenceAnalysis,
                    rag: RAGFindings) -> Tuple[Synthesis, RAGFindings]:
    """Returns the synthesis plus a possibly-augmented RAGFindings (the LLM
    may request up to two follow-up retrieval queries, ReAct-style)."""
    scripted = Synthesis(
        hypotheses=scripted_hypotheses(extract.failure_type, extract.log,
                                       extract.missing_fields),
        ai_proposed_severity=scripted_severity_proposal(extract.log,
                                                        extract.failure_type),
        behavioral_consequence=analysis.behavioral.get("consequence", UNKNOWN),
        provider="scripted-deterministic")

    if backend.name == "mock":
        return scripted, rag

    prompt = synthesis_prompt(
        extract.failure_type,
        [e.model_dump() for e in extract.observed_evidence],
        [e.model_dump() for e in analysis.derived_evidence],
        [s.model_dump() for s in rag.standards],
        extract.missing_fields)
    try:
        res = backend.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1400)
        parsed = _extract_json(res.text)
    except Exception:
        parsed = None
    if not parsed or not parsed.get("hypotheses"):
        # Honest fallback: model unreachable/unparseable -> scripted analysis,
        # provider label says so.
        scripted.provider = f"{backend.name}-fallback-scripted"
        return scripted, rag

    hyps: List[RootCauseHypothesis] = []
    for h in parsed["hypotheses"][:4]:
        try:
            hyps.append(RootCauseHypothesis(
                hypothesis=str(h.get("hypothesis", ""))[:600],
                confidence=max(0.0, min(1.0, float(h.get("confidence", 0.5)))),
                supporting_evidence_keys=[str(k) for k in
                                          h.get("supporting_evidence_keys", [])][:8],
                missing_evidence=[str(k) for k in
                                  h.get("missing_evidence", [])][:8]))
        except (TypeError, ValueError):
            continue
    sev = None
    try:
        sev = Severity(str(parsed.get("proposed_severity", "")).upper())
    except ValueError:
        pass

    extra_q = [str(q) for q in parsed.get("additional_queries", [])][:2]
    if extra_q:
        rag = run_rag_agent(registry, extract, extra_queries=extra_q)

    model_label = getattr(backend, "model", None) or backend.name
    return Synthesis(
        hypotheses=hyps or scripted.hypotheses,
        ai_proposed_severity=sev or scripted.ai_proposed_severity,
        behavioral_consequence=str(
            parsed.get("behavioral_consequence")
            or analysis.behavioral.get("consequence", UNKNOWN))[:500],
        provider=f"{backend.name}:{model_label}"), rag


# ------------------------------------------------------------- assembly

def _assemble_scorecard(extract: LogExtract, analysis: EvidenceAnalysis,
                        rag: RAGFindings, synth: Synthesis,
                        backend_name: str) -> RetrospectiveScorecard:
    log = extract.log
    ego = log.get("ego") or {}
    gt = log.get("ground_truth") or {}
    pred = log.get("prediction")
    ec = log.get("evaluation_context") or {}
    dis = log.get("disengagement") or {}
    traffic = ((log.get("traffic_context") or {}).get("following_vehicle") or {})

    sev_ctx = SeverityContext(
        failure_type=extract.failure_type,
        object_class=gt.get("class"),
        distance_m=gt.get("distance_m"),
        stopping_distance_m=analysis.stopping.get("stopping_distance_m"),
        closing_velocity_mps=gt.get("relative_velocity_mps"),
        ttc_s=analysis.ttc.get("ttc_s"),
        total_reaction_time_s=analysis.stopping.get("total_reaction_time_s"),
        collision_occurred=bool((log.get("collision") or {}).get("occurred")),
        near_miss=bool((log.get("collision") or {}).get("near_miss")),
        intervention_decel_mps2=(log.get("planner_response") or {}).get("decel_mps2"),
        following_gap_s=traffic.get("time_gap_s"),
        ego_speed_mps=ego.get("speed_mps"))
    sev_result = compute_severity(sev_ctx)
    adjudication = adjudicate_severity(sev_result, synth.ai_proposed_severity)

    uncertainty = UncertaintyReport(
        missing_fields=extract.missing_fields,
        unknown_metrics=analysis.unknown_metrics,
        notes=([f"severity adjudication: {adjudication.divergence_note}"]
               if adjudication.divergence else []))

    sig_raw = ec.get("significance")
    significance = (StatSignificance(**sig_raw) if sig_raw else
                    StatSignificance(method="seqeval.PairedSequentialTest",
                                     significant=None,
                                     detail="no paired-run significance artifact "
                                            "attached — not evaluable here"))

    decision = launch_gate(
        severity=adjudication.final_severity,
        has_evidence_gaps=uncertainty.has_gaps,
        scr_impact=analysis.scr.get("scr_impact"),
        scr_significant=significance.significant)

    policy_ev = [EvidenceItem(
        tier=EvidenceTier.DERIVED, key="policy_severity",
        value=adjudication.final_severity.value,
        statement="deterministic severity: " + " | ".join(sev_result.rule_trace),
        provenance=f"policy engine {POLICY_VERSION} (compute_severity)")]
    if adjudication.divergence:
        policy_ev.append(EvidenceItem(
            tier=EvidenceTier.DERIVED, key="severity_divergence",
            value=f"AI={adjudication.ai_proposed_severity.value} "
                  f"policy={adjudication.final_severity.value}",
            statement=adjudication.divergence_note,
            provenance=f"policy engine {POLICY_VERSION} (adjudicate_severity)"))

    hypothesis_ev = [EvidenceItem(
        tier=EvidenceTier.AI_HYPOTHESIS, key=f"hypothesis_{i + 1}",
        value=f"confidence {h.confidence:.0%}",
        statement=h.hypothesis,
        provenance=f"Safety Synthesizer ({synth.provider}) — inference, not fact")
        for i, h in enumerate(synth.hypotheses)]

    review_reasons: List[str] = []
    if adjudication.final_severity in (Severity.CRITICAL, Severity.FATAL):
        review_reasons.append(f"severity {adjudication.final_severity.value}")
    if adjudication.divergence:
        review_reasons.append("AI severity proposal diverged from policy")
    if uncertainty.has_gaps:
        review_reasons.append("required evidence UNKNOWN")
    if len(rag.standards) < 2:
        review_reasons.append("fewer than two independent retrieved sources")

    return RetrospectiveScorecard(
        evaluation_id=log.get("evaluation_id", "UNKNOWN-EVAL"),
        policy_version=POLICY_VERSION,
        agent_version=AGENT_VERSION,
        backend_used=synth.provider if backend_name != "mock" else "mock",
        failure_type=extract.failure_type,
        severity=adjudication.final_severity,
        ai_proposed_severity=adjudication.ai_proposed_severity,
        severity_divergence=adjudication.divergence,
        safety_critical_recall_impact=analysis.scr.get("scr_impact"),
        scr_impact_detail=analysis.scr.get("unknown_reason")
                          or analysis.scr.get("criticality_policy", ""),
        behavioral_consequence=synth.behavioral_consequence,
        launch_recommendation=decision.recommendation,
        launch_rationale=decision.rationale,
        baseline_model=(log.get("models") or {}).get("baseline"),
        candidate_model=(log.get("models") or {}).get("candidate"),
        scenario=log.get("scenario") or {},
        object_class=gt.get("class"),
        ground_truth=gt or None,
        prediction=pred,
        confidence=(pred or {}).get("confidence") if pred else None,
        ego_speed_mps=ego.get("speed_mps"),
        distance_to_object_m=gt.get("distance_m"),
        relative_velocity_mps=gt.get("relative_velocity_mps"),
        stopping_distance_m=analysis.stopping.get("stopping_distance_m"),
        ttc_s=analysis.ttc.get("ttc_s"),
        ttc_validity=analysis.ttc.get("validity_flags", []),
        planner_response=log.get("planner_response"),
        disengagement_probability=dis.get("probability_model"),
        metric_delta=ec.get("metric_delta"),
        statistical_significance=significance,
        distribution_shift=rag.distribution_shift,
        root_cause_hypotheses=synth.hypotheses,
        evidence=(extract.observed_evidence + analysis.derived_evidence
                  + rag.retrieved_evidence + policy_ev + hypothesis_ev),
        retrieved_standards=rag.standards,
        uncertainty=uncertainty,
        human_review_required=bool(review_reasons),
        human_review_reasons=review_reasons)


# ------------------------------------------------------------ entrypoints

def analyze(fixture_id: Optional[str] = None, path: Optional[str] = None,
            backend: str = "mock", persist: bool = True,
            registry: Optional[ToolRegistry] = None) -> RetrospectiveScorecard:
    """Full pipeline on a fixture id or an allowlisted artifact path."""
    be = get_backend(backend)
    if be.name != "mock":
        status = be.health()
        if not status.available:
            raise RuntimeError(f"backend '{backend}' unavailable: {status.detail}")

    analysis_key = fixture_id or (path or "adhoc").rsplit("/", 1)[-1].replace(".json", "")
    reg = registry or build_registry(analysis_id=analysis_key,
                                     persist_audit=persist)

    extract = run_log_agent(reg, fixture_id=fixture_id, path=path)
    analysis = run_evidence_analyzer(reg, extract)
    rag = run_rag_agent(reg, extract)
    synth, rag = run_synthesizer(be, reg, extract, analysis, rag)
    scorecard = _assemble_scorecard(extract, analysis, rag, synth, be.name)

    if persist:
        store.save_analysis(scorecard.evaluation_id, {
            "scorecard": json.loads(scorecard.model_dump_json()),
            "markdown": scorecard.render_markdown(),
            "audit_analysis_id": reg.analysis_id,
            "stage_summary": {
                "failure_type_rationale": extract.failure_type_rationale,
                "rag_queries": rag.queries,
                "synthesis_provider": synth.provider,
            },
        })
    return scorecard


def analyze_fixture(fixture_id: str, backend: str = "mock",
                    persist: bool = True) -> RetrospectiveScorecard:
    return analyze(fixture_id=fixture_id, backend=backend, persist=persist)
