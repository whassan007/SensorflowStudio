"""FastAPI router for the agentic launch-readiness subsystem (/api/agentic/*).

Endpoint map:
  POST /api/agentic/failures/detect                 layer-1 scan
  GET  /api/agentic/failures                        failure queue
  GET  /api/agentic/failures/{id}                   one failure
  GET  /api/agentic/failures/{id}/state             pipeline state
  POST /api/agentic/failures/{id}/analyze[?stage=]  advance the pipeline
  GET  /api/agentic/failures/{id}/evidence          evidence graph
  GET  /api/agentic/failures/{id}/snippets          failure snippets
  POST /api/agentic/failures/{id}/cluster           scenario mining on demand
  POST /api/agentic/failures/{id}/safety-assessment safety impact on demand
  POST /api/agentic/failures/{id}/launch-assessment launch decision stage
  POST /api/agentic/failures/{id}/human-review      record a human decision
  GET  /api/agentic/policy[?version=]               active/specific policy
  POST /api/agentic/policy                          save a new version
  GET  /api/agentic/policy/versions                 version list
  POST /api/agentic/evaluation-suites               flywheel stage for a failure
  GET  /api/agentic/evaluation-suites               list suites
  GET  /api/agentic/evaluation-suites/{id}          one suite
  POST /api/agentic/evaluation-suites/{id}/promote-member  guarded promotion
  POST /api/agentic/regression/evaluate             suite regression hook
  GET  /api/agentic/scorecards/{id}                 scorecard
  GET  /api/agentic/audit/{failure_id}              immutable audit trail
  GET  /api/agentic/worked-example                  deterministic walkthrough
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sensorflow.agentic import data as data_mod
from sensorflow.agentic import flywheel as flywheel_mod
from sensorflow.agentic import pipeline as pipeline_mod
from sensorflow.agentic import policy as policy_mod
from sensorflow.agentic import review as review_mod
from sensorflow.agentic import scorecard as scorecard_mod
from sensorflow.agentic import store as store_mod
from sensorflow.agentic import worked_example as we_mod
from sensorflow.agentic.agents import SafetyImpactAgent, ScenarioMiningAgent
from sensorflow.agentic.models import STAGES

router = APIRouter(prefix="/api/agentic", tags=["agentic"])


def _failure_or_404(failure_id: str):
    failure = pipeline_mod.get_failure(failure_id)
    if failure is None:
        raise HTTPException(404, f"Unknown failure {failure_id}")
    return failure


# ------------------------------------------------------------------ detection


class DetectRequest(BaseModel):
    seed: int = data_mod.DEFAULT_SEED
    use_llm: bool = False


@router.post("/failures/detect")
def detect(req: Optional[DetectRequest] = None):
    req = req or DetectRequest()
    events = pipeline_mod.detect_failures(seed=req.seed, use_llm=req.use_llm)
    return {"status": "ok", "detected": len(events),
            "failures": [{"failure_id": e.failure_id, "kind": e.kind,
                          "title": e.title,
                          "candidate_events": e.detection_basis.candidate_events,
                          "denominator": e.detection_basis.denominator}
                         for e in events]}


@router.get("/failures")
def failure_queue():
    return {"status": "ok", "failures": pipeline_mod.list_failures(),
            "stages": STAGES}


@router.get("/failures/{failure_id}")
def failure_detail(failure_id: str):
    return {"status": "ok", "failure": _failure_or_404(failure_id).model_dump()}


@router.get("/failures/{failure_id}/state")
def failure_state(failure_id: str):
    state = pipeline_mod.get_state(failure_id)
    if state is None:
        raise HTTPException(404, f"Unknown failure {failure_id}")
    return {"status": "ok", "state": state.model_dump()}


# ------------------------------------------------------------------ pipeline


class AnalyzeRequest(BaseModel):
    seed: int = data_mod.DEFAULT_SEED
    use_llm: bool = False
    policy_input_overrides: Dict = Field(default_factory=dict)


@router.post("/failures/{failure_id}/analyze")
def analyze(failure_id: str, req: Optional[AnalyzeRequest] = None,
            stage: Optional[str] = Query(default=None)):
    _failure_or_404(failure_id)
    req = req or AnalyzeRequest()
    try:
        state = pipeline_mod.run_stage(
            failure_id, stage, seed=req.seed, use_llm=req.use_llm,
            policy_input_overrides=req.policy_input_overrides or None)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": "ok", "state": state.model_dump()}


@router.get("/failures/{failure_id}/evidence")
def evidence(failure_id: str):
    graph = pipeline_mod.get_evidence_graph(failure_id)
    if graph is None:
        raise HTTPException(404, "Evidence graph not built yet; run the "
                                 "EVIDENCE_AGGREGATION stage first")
    return {"status": "ok", "graph": graph.model_dump()}


@router.get("/failures/{failure_id}/snippets")
def snippets(failure_id: str):
    _failure_or_404(failure_id)
    return {"status": "ok", "snippets": pipeline_mod.get_snippets(failure_id)}


@router.post("/failures/{failure_id}/cluster")
def cluster(failure_id: str, req: Optional[AnalyzeRequest] = None):
    failure = _failure_or_404(failure_id)
    req = req or AnalyzeRequest()
    result = ScenarioMiningAgent(use_llm=req.use_llm).run(
        failure_id, failure=failure)
    return {"status": "ok", "result": result.model_dump()}


@router.post("/failures/{failure_id}/safety-assessment")
def safety_assessment(failure_id: str, req: Optional[AnalyzeRequest] = None):
    failure = _failure_or_404(failure_id)
    req = req or AnalyzeRequest()
    result = SafetyImpactAgent(use_llm=req.use_llm).run(
        failure_id, failure=failure)
    return {"status": "ok", "result": result.model_dump()}


@router.post("/failures/{failure_id}/launch-assessment")
def launch_assessment(failure_id: str, req: Optional[AnalyzeRequest] = None):
    _failure_or_404(failure_id)
    req = req or AnalyzeRequest()
    try:
        state = pipeline_mod.run_stage(
            failure_id, "LAUNCH_DECISION", seed=req.seed, use_llm=req.use_llm,
            policy_input_overrides=req.policy_input_overrides or None)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": "ok", "policy_evaluation": state.policy_evaluation,
            "scorecard_id": state.scorecard_id}


# ------------------------------------------------------------------ human review


class HumanReviewRequest(BaseModel):
    reviewer: str
    decision: str
    rationale: str
    approved_option: Optional[str] = None
    evidence_reviewed: List[str] = Field(default_factory=list)
    override_reason: Optional[str] = None


@router.post("/failures/{failure_id}/human-review")
def human_review(failure_id: str, req: HumanReviewRequest):
    failure = _failure_or_404(failure_id)
    state = pipeline_mod.get_state(failure_id)
    policy_version = (state.policy_evaluation or {}).get("policy_version", "") \
        if state else ""
    try:
        rec = review_mod.record_decision(
            failure, reviewer=req.reviewer, decision=req.decision,
            rationale=req.rationale, evidence_reviewed=req.evidence_reviewed,
            policy_version=policy_version,
            approved_option=req.approved_option,
            override_reason=req.override_reason)
    except Exception as e:
        raise HTTPException(400, str(e))
    if req.decision == "confirm_failure":
        failure.validated = True
    elif req.decision == "reject_failure":
        failure.validated = False
        failure.status = "closed"
    pipeline_mod.save_failure(failure)
    return {"status": "ok", "review": rec.model_dump(),
            "decisions": review_mod.decisions_for(failure_id)}


@router.get("/failures/{failure_id}/human-review")
def human_review_list(failure_id: str):
    _failure_or_404(failure_id)
    return {"status": "ok", "decisions": review_mod.decisions_for(failure_id)}


# ------------------------------------------------------------------ policy


@router.get("/policy")
def get_policy(version: Optional[str] = None):
    try:
        return {"status": "ok", "policy": policy_mod.get_policy(version)}
    except KeyError as e:
        raise HTTPException(404, str(e))


class PolicyRequest(BaseModel):
    policy: Dict
    actor: str = "api"


@router.post("/policy")
def save_policy(req: PolicyRequest):
    doc = policy_mod.save_policy(req.policy, actor=req.actor)
    return {"status": "ok", "policy_version": doc["policy_version"],
            "policy": doc}


@router.get("/policy/versions")
def policy_versions():
    return {"status": "ok", "versions": policy_mod.list_policies()}


# ------------------------------------------------------------------ flywheel


class SuiteRequest(BaseModel):
    failure_id: str
    use_llm: bool = False


@router.post("/evaluation-suites")
def create_suite(req: SuiteRequest):
    _failure_or_404(req.failure_id)
    try:
        state = pipeline_mod.run_stage(req.failure_id, "LEARNING_FLYWHEEL",
                                       use_llm=req.use_llm)
    except ValueError as e:
        raise HTTPException(409, str(e))
    rec = state.stage_record("LEARNING_FLYWHEEL")
    return {"status": "ok", "stage_status": rec.status, "detail": rec.detail,
            "suite_ids": state.suite_ids}


@router.get("/evaluation-suites")
def list_suites():
    return {"status": "ok", "suites": flywheel_mod.list_suites()}


@router.get("/evaluation-suites/{suite_id}")
def suite_detail(suite_id: str):
    suite = flywheel_mod.get_suite(suite_id)
    if suite is None:
        raise HTTPException(404, f"Unknown suite {suite_id}")
    return {"status": "ok", "suite": suite.model_dump()}


class PromoteRequest(BaseModel):
    member_id: str
    actor: Optional[str] = None
    override_reason: Optional[str] = None


@router.post("/evaluation-suites/{suite_id}/promote-member")
def promote_member(suite_id: str, req: PromoteRequest):
    try:
        result = flywheel_mod.promote_member_to_training(
            suite_id, req.member_id, actor=req.actor,
            override_reason=req.override_reason)
    except flywheel_mod.LeakageError as e:
        raise HTTPException(403, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"status": "ok", **result}


class RegressionRequest(BaseModel):
    seed: int = data_mod.DEFAULT_SEED


@router.post("/regression/evaluate")
def regression_evaluate(req: Optional[RegressionRequest] = None):
    req = req or RegressionRequest()
    return {"status": "ok", **flywheel_mod.regression_evaluate(seed=req.seed)}


# ------------------------------------------------------------------ scorecards / audit


@router.get("/scorecards/{scorecard_id}")
def scorecard(scorecard_id: str):
    card = scorecard_mod.load_scorecard(scorecard_id)
    if card is None:
        raise HTTPException(404, f"Unknown scorecard {scorecard_id}")
    return {"status": "ok", "scorecard": card}


@router.get("/audit/{failure_id}")
def audit(failure_id: str):
    trail = store_mod.audit_trail(failure_id)
    return {"status": "ok", "records": trail,
            "chain": store_mod.verify_audit_chain(failure_id)}


# ------------------------------------------------------------------ worked example


@router.get("/worked-example")
def worked_example(refresh: bool = False, use_llm: bool = False):
    if not refresh:
        cached = store_mod.read_json("worked_example", "latest.json")
        if cached:
            return {"status": "ok", "walkthrough": cached, "cached": True}
    walkthrough = we_mod.run_worked_example(use_llm=use_llm)
    return {"status": "ok", "walkthrough": walkthrough, "cached": False}
