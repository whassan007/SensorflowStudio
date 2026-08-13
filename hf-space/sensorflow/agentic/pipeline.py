"""Five-layer pipeline orchestrator.

FAILURE DETECTION -> EVIDENCE AGGREGATION -> FAILURE ANALYSIS ->
LAUNCH DECISION -> LEARNING FLYWHEEL

Explicit staged state (pydantic PipelineState), orchestrated by this
deterministic module — never one autonomous agent controlling the chain.
Deterministic components own metrics, thresholds, statistical tests, lineage,
policy, gates and the audit log; agents contribute typed advisory results.

Every stage transition, agent output and policy evaluation is appended to the
immutable audit trail (store.audit).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.agentic import concentration as conc_mod
from sensorflow.agentic import data as data_mod
from sensorflow.agentic import evidence as evidence_mod
from sensorflow.agentic import flywheel as flywheel_mod
from sensorflow.agentic import policy as policy_mod
from sensorflow.agentic import review as review_mod
from sensorflow.agentic import scorecard as scorecard_mod
from sensorflow.agentic import snippets as snippets_mod
from sensorflow.agentic import store as store_mod
from sensorflow.agentic.agents import (EvalFlywheelAgent, FailureDetectionAgent,
                                       LaunchDecisionAgent, SafetyImpactAgent,
                                       ScenarioMiningAgent,
                                       SensorFusionVerificationAgent,
                                       StatisticalRegressionAgent,
                                       VLMSceneAnalysisAgent)
from sensorflow.agentic.models import (STAGES, EvidenceGraph, FailureEvent,
                                       PipelineState, StatisticalAssessment,
                                       now_iso)

SAFETY_CRITICAL = {"pedestrian", "cyclist", "motorcycle"}
MAX_SNIPPETS = 6


# ------------------------------------------------------------------ persistence


def save_failure(failure: FailureEvent) -> None:
    store_mod.write_json(failure, "failures", f"{failure.failure_id}.json")


def get_failure(failure_id: str) -> Optional[FailureEvent]:
    doc = store_mod.read_json("failures", f"{failure_id}.json")
    return FailureEvent.model_validate(doc) if doc else None


def list_failures() -> List[Dict]:
    out = []
    for name in store_mod.list_dir("failures"):
        doc = store_mod.read_json("failures", name)
        if doc:
            doc["instances"] = doc.get("instances", [])[:3]  # queue view: trim
            out.append(doc)
    return sorted(out, key=lambda d: d.get("created_at", ""))


def save_state(state: PipelineState) -> None:
    state.updated_at = now_iso()
    store_mod.write_json(state, "pipeline", f"{state.failure_id}.json")


def get_state(failure_id: str) -> Optional[PipelineState]:
    doc = store_mod.read_json("pipeline", f"{failure_id}.json")
    return PipelineState.model_validate(doc) if doc else None


def get_evidence_graph(failure_id: str) -> Optional[EvidenceGraph]:
    doc = store_mod.read_json("evidence", f"{failure_id}.json")
    return EvidenceGraph.model_validate(doc) if doc else None


def get_snippets(failure_id: str) -> List[Dict]:
    return [store_mod.read_json("snippets", failure_id, n)
            for n in store_mod.list_dir("snippets", failure_id)]


def _transition(state: PipelineState, stage: str, status: str,
                detail: str = "") -> None:
    rec = state.stage_record(stage)
    if status == "running":
        rec.started_at = now_iso()
    else:
        rec.finished_at = now_iso()
    rec.status = status
    rec.detail = detail
    save_state(state)
    store_mod.audit("stage_transition", state.failure_id, "pipeline",
                    detail=f"{stage} -> {status}" + (f": {detail}" if detail else ""))


def _record_agent(state: PipelineState, result) -> None:
    state.agent_results[result.agent] = result
    store_mod.audit("agent_output", state.failure_id, result.agent,
                    detail=f"status={result.status} confidence={result.confidence}",
                    payload={"agent_version": result.agent_version,
                             "escalation": result.escalation.model_dump(),
                             "epistemic_status": result.epistemic_status})


# ------------------------------------------------------------------ layer 1


def detect_failures(seed: int = data_mod.DEFAULT_SEED,
                    use_llm: bool = False) -> List[FailureEvent]:
    agent = FailureDetectionAgent(use_llm=use_llm)
    events, summary = agent.detect(seed)
    scan_result = agent.run("_detection_scan", seed=seed)
    for failure in events:
        save_failure(failure)
        state = PipelineState(failure_id=failure.failure_id)
        _transition(state, "FAILURE_DETECTION", "running")
        scan_copy = scan_result.model_copy(update={"failure_id": failure.failure_id})
        _record_agent(state, scan_copy)
        _transition(state, "FAILURE_DETECTION", "complete",
                    detail=f"{failure.title} "
                           f"(candidate {failure.detection_basis.candidate_events} "
                           f"vs baseline {failure.detection_basis.baseline_events} "
                           f"over n={failure.detection_basis.denominator})")
    store_mod.audit("detection_scan", None, "failure_detection",
                    detail=f"{len(events)} failure patterns emitted",
                    payload={"scan": {k: v for k, v in summary.items()
                                      if k != "checks"}})
    return events


# ------------------------------------------------------------------ layer 2


def run_evidence_aggregation(failure: FailureEvent, state: PipelineState,
                             seed: int) -> None:
    _transition(state, "EVIDENCE_AGGREGATION", "running")
    campaign = data_mod.get_campaign(seed)
    graph = evidence_mod.build_graph(failure, campaign)
    store_mod.write_json(graph, "evidence", f"{failure.failure_id}.json")
    n_snips = 0
    for inst in failure.instances[:MAX_SNIPPETS]:
        snip = snippets_mod.build_snippet(failure, inst, campaign)
        store_mod.write_json(snip, "snippets", failure.failure_id,
                             f"{snip.snippet_id}.json")
        n_snips += 1
    _transition(state, "EVIDENCE_AGGREGATION", "complete",
                detail=f"evidence graph ({len(graph.nodes)} nodes) + "
                       f"{n_snips} snippets")


# ------------------------------------------------------------------ layer 3


def _is_rate_population_failure(failure: FailureEvent) -> bool:
    return (failure.kind == "classification_flip"
            and failure.gt_class == "pedestrian"
            and failure.predicted_class == "construction_cone")


def run_failure_analysis(failure: FailureEvent, state: PipelineState,
                         seed: int, use_llm: bool = False) -> None:
    _transition(state, "FAILURE_ANALYSIS", "running")
    campaign = data_mod.get_campaign(seed)

    vlm = VLMSceneAnalysisAgent(use_llm=use_llm).run(
        failure.failure_id, failure=failure)
    _record_agent(state, vlm)

    fusion = SensorFusionVerificationAgent(use_llm=use_llm).run(
        failure.failure_id, failure=failure, seed=seed)
    _record_agent(state, fusion)

    mining = ScenarioMiningAgent(use_llm=use_llm).run(
        failure.failure_id, failure=failure)
    _record_agent(state, mining)

    stat_agent = StatisticalRegressionAgent(use_llm=use_llm)
    if _is_rate_population_failure(failure):
        stat = stat_agent.run(failure.failure_id, seed=seed)
        state.statistical = StatisticalAssessment.model_validate(stat.output)
        state.concentration = conc_mod.analyze_concentration(failure.failure_id, seed)
    else:
        stat = stat_agent.run(failure.failure_id, seed=seed)
        stat.output["scope_note"] = (
            "rate-population telemetry instruments the pedestrian->"
            "construction_cone channel; for this failure the campaign counts "
            "in detection_basis are the denominator of record")
        state.statistical = None
        state.concentration = None
    _record_agent(state, stat)

    safety = SafetyImpactAgent(use_llm=use_llm).run(
        failure.failure_id, failure=failure)
    _record_agent(state, safety)

    # ---- deterministic evidence-graph updates -------------------------
    graph = get_evidence_graph(failure.failure_id)
    if graph is not None:
        if fusion.status != "failed":
            evidence_mod.set_node(
                graph, "Sensors",
                "OBSERVED" if fusion.output.get("overall_verdict")
                != "verification_failed" else "UNAVAILABLE",
                f"fusion verdict: {fusion.output.get('overall_verdict')}",
                {"verdict": fusion.output.get("overall_verdict"),
                 "verdict_counts": fusion.output.get("verdict_counts")},
                "sensor fusion verification agent (deterministic re-simulation)")
        if mining.status == "ok":
            evidence_mod.set_node(
                graph, "HistoricalSimilarity", "DERIVED",
                f"novelty: {mining.output.get('novelty')} "
                f"({mining.output.get('exact_precedent_count')} precedents)",
                {"novelty": mining.output.get("novelty"),
                 "top_similar": mining.output.get("similar_historical", [])[:3],
                 "clusters": mining.output.get("clusters", [])},
                "scenario mining agent (deterministic retrieval)")
        if state.statistical is not None:
            s = state.statistical
            evidence_mod.set_node(
                graph, "Frequency", "OBSERVED",
                f"candidate {s.candidate.rate:.6f} "
                f"(CI {s.candidate.wilson_ci}) vs baseline "
                f"{s.baseline.rate:.6f}; significant={s.significant}",
                {"statistical": s.model_dump()},
                "statistical regression agent (deterministic; stats via "
                "seqeval + exact binomial)")
        if safety.status != "failed":
            observed = safety.output.get("behavioral_evidence") == "observed"
            evidence_mod.set_node(
                graph, "SafetyConsequence",
                "OBSERVED" if observed else "UNAVAILABLE",
                safety.output.get("assessment", ""),
                {"behavioral_evidence": safety.output.get("behavioral_evidence"),
                 "classification": safety.output.get("behavioral_classification"),
                 "worst_case": safety.output.get("worst_case"),
                 "chain": safety.output.get("chain")},
                "safety impact agent (replay evidence via safety.ssam_ext)")
            if observed:
                evidence_mod.set_node(
                    graph, "Planner", "OBSERVED",
                    f"replayed {safety.output.get('replayed_instances')} "
                    "planner-response trace(s)",
                    {"replays": safety.output.get("replays", [])},
                    "safety impact agent (safety.ssam_ext replay)")
        # VLM hypotheses attach to Environment as clearly-labeled hypotheses
        env = graph.node("Environment")
        if env is not None and vlm.status == "ok":
            env.fields["scene_hypotheses"] = vlm.output.get("hypotheses", [])
            env.caveats.append("scene_hypotheses are HYPOTHESIS-status agent "
                               "conjectures, not observations")
        store_mod.write_json(graph, "evidence", f"{failure.failure_id}.json")

    failure.status = "investigating"
    save_failure(failure)
    _transition(state, "FAILURE_ANALYSIS", "complete",
                detail=f"5 analysis agents ran; fusion="
                       f"{fusion.output.get('overall_verdict')}")


# ------------------------------------------------------------------ layer 4


def evidence_confidence(state: PipelineState) -> str:
    s = state.statistical
    if s is None:
        return "INSUFFICIENT_EVIDENCE"
    if s.significant and not s.small_sample_flags:
        return "CONFIRMED"
    if s.significant:
        return "LIKELY"
    if s.candidate.events > 0:
        return "POSSIBLE"
    return "INSUFFICIENT_EVIDENCE"


def build_policy_input(failure: FailureEvent, state: PipelineState,
                       overrides: Optional[Dict] = None) -> policy_mod.PolicyInput:
    """Deterministic mapping from pipeline artifacts to the policy input."""
    safety = state.agent_results.get("safety_impact")
    fusion = state.agent_results.get("sensor_fusion_verification")
    mining = state.agent_results.get("scenario_mining")

    behavioral = "none"
    contained = False
    downstream = "uncertain"
    if safety is not None and safety.status != "failed":
        ev = safety.output.get("behavioral_evidence")
        if ev == "observed":
            cls = safety.output.get("behavioral_classification")
            behavioral = cls or "observed_contained"
            contained = cls == "observed_contained"
            downstream = ("observed_critical" if cls == "observed_unsafe"
                          else "observed_degraded")
        else:
            behavioral, downstream = "none", "uncertain"

    stat = state.statistical
    conc = state.concentration
    concentrated_dim = (conc.concentrated_dimensions[0]
                        if conc and conc.concentrated_dimensions else None)
    resid_outside = None
    excl_share = None
    if conc is not None and concentrated_dim == "construction_zone":
        row_out = next((s for s in conc.strata
                        if s.stratum == "non_construction"), None)
        row_in = next((s for s in conc.strata
                       if s.stratum == "construction"), None)
        if row_out is not None:
            resid_outside = row_out.stratum_rate
        if row_in is not None:
            excl_share = row_in.exposure_share

    agent_failures = [r.agent for r in state.agent_results.values()
                      if r.status == "failed"]

    kwargs = dict(
        failure_id=failure.failure_id,
        safety_critical_class=(failure.gt_class in SAFETY_CRITICAL
                               if failure.gt_class else False),
        collision_observed=False,
        behavioral_evidence=behavioral,
        downstream_contained=contained,
        functional_impact=True,
        rate=stat.candidate.rate if stat else None,
        rate_ci=stat.candidate.wilson_ci if stat else None,
        denominator=stat.candidate.denominator if stat
        else failure.detection_basis.denominator,
        significant=stat.significant if stat else False,
        small_sample=bool(stat.small_sample_flags) if stat else True,
        exposure_share=excl_share,
        novelty=(mining.output.get("novelty", "unknown")
                 if mining is not None and mining.status == "ok" else "unknown"),
        evidence_confidence=evidence_confidence(state),
        downstream_consequence=downstream,
        gt_available=True,
        lineage_complete=bool(failure.dataset_fingerprint
                              and failure.candidate_model
                              and failure.baseline_model),
        telemetry_available=failure.detection_basis.denominator > 0,
        fusion_verdict=(fusion.output.get("overall_verdict",
                                          "verification_failed")
                        if fusion is not None and fusion.status != "failed"
                        else "verification_failed"),
        agent_conflict=bool(agent_failures),
        conflict_details=[f"agent {a} failed" for a in agent_failures],
        gate_violated=False,
        mitigation_validated=False,
        concentration_dimension=concentrated_dim,
        concentrated=bool(conc and conc.determination == "concentrated"),
        odd_detector_recall=None,   # REQUIRED-EVIDENCE: no zone detector measured
        odd_exclusion_share=excl_share,
        residual_rate_outside_odd=resid_outside,
    )
    kwargs.update(overrides or {})
    return policy_mod.PolicyInput(**kwargs)


def run_launch_decision(failure: FailureEvent, state: PipelineState,
                        seed: int, use_llm: bool = False,
                        policy_input_overrides: Optional[Dict] = None) -> Dict:
    _transition(state, "LAUNCH_DECISION", "running")
    pol_input = build_policy_input(failure, state, policy_input_overrides)
    evaluation = policy_mod.evaluate(pol_input)
    state.policy_evaluation = evaluation

    narrative = LaunchDecisionAgent(use_llm=use_llm).run(
        failure.failure_id, failure=failure, policy_evaluation=evaluation)
    _record_agent(state, narrative)

    card = scorecard_mod.build_scorecard(
        failure, state.statistical, state.concentration, evaluation,
        state.agent_results)
    state.scorecard_id = card.scorecard_id

    triggers = review_mod.mandatory_review_triggers(
        evaluation, pol_input.fusion_verdict, pol_input.gt_available,
        pol_input.small_sample, pol_input.behavioral_evidence)
    evaluation["mandatory_review_triggers"] = triggers
    evaluation["human_review_required"] = any(t["fired"] for t in triggers)

    failure.severity = evaluation.get("severity")
    failure.policy_outcome = evaluation.get("outcome")
    failure.status = "decided"
    save_failure(failure)
    _transition(state, "LAUNCH_DECISION", "complete",
                detail=f"{evaluation['outcome']} -> "
                       f"{evaluation['recommended_option']} "
                       f"(policy {evaluation['policy_version']})")
    return evaluation


# ------------------------------------------------------------------ layer 5


def run_learning_flywheel(failure: FailureEvent, state: PipelineState,
                          use_llm: bool = False) -> Dict:
    _transition(state, "LEARNING_FLYWHEEL", "running")
    agent = EvalFlywheelAgent(use_llm=use_llm)
    result = agent.run(failure.failure_id, failure=failure,
                       concentration=state.concentration)
    _record_agent(state, result)
    if result.output.get("proposal") is None:
        _transition(state, "LEARNING_FLYWHEEL", "blocked",
                    detail="failure not human-validated; flywheel refused")
        return {"created": False, "reason": result.output.get("reason")}
    suite = flywheel_mod.create_or_update_suite(failure, result.output["proposal"])
    state.suite_ids.append(suite.suite_id)
    _transition(state, "LEARNING_FLYWHEEL", "complete",
                detail=f"suite {suite.name} v{suite.version} "
                       f"({len(suite.members)} members)")
    return {"created": True, "suite_id": suite.suite_id,
            "suite_name": suite.name, "version": suite.version,
            "members": len(suite.members)}


# ------------------------------------------------------------------ dispatcher


STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}


def run_stage(failure_id: str, stage: Optional[str] = None,
              seed: int = data_mod.DEFAULT_SEED, use_llm: bool = False,
              policy_input_overrides: Optional[Dict] = None) -> PipelineState:
    """Run one stage (the next pending one when stage is None)."""
    failure = get_failure(failure_id)
    state = get_state(failure_id)
    if failure is None or state is None:
        raise KeyError(f"Unknown failure {failure_id}")

    if stage is None:
        pending = [s.stage for s in state.stages if s.status == "pending"]
        if not pending:
            return state
        stage = pending[0]
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {stage}")
    for prior in state.stages:
        if STAGE_ORDER[prior.stage] < STAGE_ORDER[stage] \
                and prior.status not in ("complete", "blocked"):
            raise ValueError(f"stage {stage} requires {prior.stage} to be "
                             f"complete (currently {prior.status})")

    if stage == "EVIDENCE_AGGREGATION":
        run_evidence_aggregation(failure, state, seed)
    elif stage == "FAILURE_ANALYSIS":
        run_failure_analysis(failure, state, seed, use_llm)
    elif stage == "LAUNCH_DECISION":
        run_launch_decision(failure, state, seed, use_llm,
                            policy_input_overrides)
    elif stage == "LEARNING_FLYWHEEL":
        run_learning_flywheel(failure, state, use_llm)
    elif stage == "FAILURE_DETECTION":
        pass  # detection happens at detect_failures(); already complete
    return get_state(failure_id) or state
