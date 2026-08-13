"""FastAPI router for Hill Climbing EM (mounted at /api/hillclimb)."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.hillclimb import adaptive, coach, content, design_lab, evaluate, interviewer
from sensorflow.hillclimb import readiness as readiness_mod
from sensorflow.hillclimb import simulation as sim_mod
from sensorflow.hillclimb import star as star_mod
from sensorflow.hillclimb.blueprint import (Blueprint, competency_index, load_blueprint,
                                            save_blueprint, validate_graph)
from sensorflow.hillclimb.models import Attempt, Evidence, UserProfile, get_store, now_iso

router = APIRouter(prefix="/api/hillclimb", tags=["hillclimb"])


# --------------------------------------------------------------- request bodies


class GenerateExerciseRequest(BaseModel):
    competency_id: str
    difficulty: int = Field(default=2, ge=1, le=3)
    seed: Optional[int] = None


class SubmitExerciseRequest(BaseModel):
    exercise_id: str
    answer: str
    user_id: str = "default"
    as_assessment: bool = False


class StarRequest(BaseModel):
    text: str
    user_id: str = "default"
    save_evidence: bool = True


class DiagnosticStartRequest(BaseModel):
    user_id: str = "default"
    seed: int = 11


class DiagnosticAnswerRequest(BaseModel):
    answer: str


class SimulationStartRequest(BaseModel):
    user_id: str = "default"
    scenario_id: str = "inherited_perception_team"
    seed: int = 42
    max_turns: int = 8


class SimulationStepRequest(BaseModel):
    hypothesis: str
    intervention_id: str
    revert_previous: bool = False


class InterviewStartRequest(BaseModel):
    mode: str = "hybrid"
    user_id: str = "default"
    seed: int = 7


class InterviewAnswerRequest(BaseModel):
    answer: str


class JourneyAdvanceRequest(BaseModel):
    user_id: str = "default"
    event: str
    competency_id: Optional[str] = None
    passed: Optional[bool] = None


# --------------------------------------------------------------------- profile


@router.get("/profile")
def get_profile(user_id: str = "default"):
    store = get_store()
    raw = store.get("profiles", user_id)
    if raw is None:
        profile = UserProfile(user_id=user_id)
        store.put("profiles", user_id, profile)
        return profile
    return raw


@router.put("/profile")
def put_profile(profile: UserProfile):
    store = get_store()
    profile.updated_at = now_iso()
    store.put("profiles", profile.user_id, profile)
    return profile


# -------------------------------------------------------------------- blueprint


@router.get("/blueprint")
def get_blueprint():
    return load_blueprint(get_store())


@router.put("/blueprint")
def put_blueprint(bp: Blueprint):
    problems = validate_graph(bp)
    if problems:
        raise HTTPException(status_code=400, detail={"problems": problems})
    save_blueprint(bp, get_store())
    return bp


@router.get("/graph")
def get_graph():
    bp = load_blueprint(get_store())
    return {
        "nodes": [c.model_dump() for c in bp.competencies],
        "edges": [{"source": p, "target": c.id}
                  for c in bp.competencies for p in c.prerequisites],
        "problems": validate_graph(bp),
    }


# ------------------------------------------------------------------- diagnostic


@router.post("/diagnostic/start")
def diagnostic_start(req: DiagnosticStartRequest):
    dx = adaptive.start_diagnostic(req.user_id, seed=req.seed, store=get_store())
    return _diagnostic_view(dx)


@router.post("/diagnostic/{diagnostic_id}/answer")
def diagnostic_answer(diagnostic_id: str, req: DiagnosticAnswerRequest):
    try:
        dx = adaptive.answer_diagnostic(diagnostic_id, req.answer, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _diagnostic_view(dx)


@router.get("/diagnostic/{diagnostic_id}")
def diagnostic_get(diagnostic_id: str):
    dx = adaptive.get_diagnostic(diagnostic_id, get_store())
    if dx is None:
        raise HTTPException(status_code=404, detail="diagnostic not found")
    return _diagnostic_view(dx)


def _diagnostic_view(dx) -> Dict:
    view = dx.model_dump()
    if dx.current_exercise_id:
        ex = content.get_exercise(dx.current_exercise_id, get_store())
        view["current_question"] = {"exercise_id": ex.exercise_id,
                                    "competency_id": ex.competency_id,
                                    "scenario": ex.scenario}
    return view


# -------------------------------------------------------------------- exercises


@router.post("/exercise/generate")
def exercise_generate(req: GenerateExerciseRequest):
    try:
        ex = content.generate_exercise(req.competency_id, difficulty=req.difficulty,
                                       seed=req.seed, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ex


@router.get("/exercise/{exercise_id}")
def exercise_get(exercise_id: str):
    ex = content.get_exercise(exercise_id, get_store())
    if ex is None:
        raise HTTPException(status_code=404, detail="exercise not found")
    return ex


@router.post("/exercise/submit")
def exercise_submit(req: SubmitExerciseRequest):
    store = get_store()
    ex = content.get_exercise(req.exercise_id, store)
    if ex is None:
        raise HTTPException(status_code=404, detail="exercise not found")
    result = evaluate.evaluate_response(ex, req.answer)
    feedback = coach.build_feedback(ex, result)

    attempt = Attempt(user_id=req.user_id, exercise_id=ex.exercise_id,
                      competency_id=ex.competency_id, kind="exercise",
                      responses={"answer": req.answer}, evaluation=result)
    readiness_mod.record_attempt(attempt, store)

    ev = Evidence(user_id=req.user_id, competency_ids=[ex.competency_id],
                  artifact_type="exercise_attempt",
                  source=f"Exercise {ex.template_id} ({ex.competency_id})",
                  summary=f"Scored {result.score}/5 on a {ex.family} exercise.",
                  quotes=result.evidence[:3], score=float(result.score),
                  confidence=result.confidence,
                  payload={"attempt_id": attempt.attempt_id, "exercise_id": ex.exercise_id})
    store.put("evidence", ev.evidence_id, ev)

    journey = None
    if req.as_assessment:
        try:
            journey = adaptive.advance_journey(req.user_id, "assessment_result",
                                               competency_id=ex.competency_id,
                                               passed=result.score >= 4, store=store)
        except ValueError:
            journey = None

    matrix = readiness_mod.compute_matrix(req.user_id, store)
    return {
        "evaluation": result.model_dump(),
        "coaching": feedback,
        "attempt_id": attempt.attempt_id,
        "evidence_id": ev.evidence_id,
        "linked_tool": ex.linked_tool,
        "readiness_for_competency": matrix[ex.competency_id].model_dump()
        if ex.competency_id in matrix else None,
        "journey": journey.model_dump() if journey else None,
    }


# ------------------------------------------------------------------------ STAR


@router.post("/star/diagnose")
def star_diagnose(req: StarRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="empty story")
    return star_mod.diagnose_story(req.text, user_id=req.user_id,
                                   save_evidence=req.save_evidence, store=get_store())


# ------------------------------------------------------------------ design lab


@router.get("/design/challenges")
def design_challenges():
    return {"challenges": [c.model_dump() for c in design_lab.CHALLENGES],
            "component_types": design_lab.COMPONENT_TYPES}


@router.post("/design/submit")
def design_submit(sub: design_lab.DesignSubmission):
    try:
        return design_lab.grade_submission(sub, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------ simulation


@router.get("/simulation/catalog")
def simulation_catalog():
    return {"scenarios": [s.model_dump() for s in sim_mod.SCENARIOS],
            "interventions": [i.model_dump() for i in sim_mod.INTERVENTIONS],
            "metrics": sim_mod.METRICS, "inverted_metrics": sorted(sim_mod.INVERTED),
            "safety_floor": sim_mod.SAFETY_FLOOR, "morale_floor": sim_mod.MORALE_FLOOR}


@router.post("/simulation/start")
def simulation_start(req: SimulationStartRequest):
    return sim_mod.create_simulation(req.user_id, req.scenario_id, req.seed,
                                     req.max_turns, store=get_store())


@router.post("/simulation/{sim_id}/step")
def simulation_step(sim_id: str, req: SimulationStepRequest):
    try:
        return sim_mod.step_simulation(sim_id, req.hypothesis, req.intervention_id,
                                       revert_previous=req.revert_previous, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/simulation/{sim_id}")
def simulation_get(sim_id: str):
    sim = sim_mod.get_simulation(sim_id, get_store())
    if sim is None:
        raise HTTPException(status_code=404, detail="simulation not found")
    return sim


# -------------------------------------------------------------------- interview


@router.post("/interview/start")
def interview_start(req: InterviewStartRequest):
    try:
        return interviewer.start_interview(req.mode, req.user_id, req.seed, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/interview/{session_id}/answer")
def interview_answer(session_id: str, req: InterviewAnswerRequest):
    store = get_store()
    try:
        session = interviewer.submit_answer(session_id, req.answer, store=store)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # answered turns also feed the readiness matrix as knowledge attempts
    answered = session.turns[-2]
    attempt = Attempt(user_id=session.user_id, exercise_id=answered.exercise_id,
                      competency_id=answered.competency_id, kind="interview",
                      responses={"answer": req.answer}, evaluation=answered.evaluation)
    readiness_mod.record_attempt(attempt, store)
    return session


@router.post("/interview/{session_id}/end")
def interview_end(session_id: str):
    try:
        return interviewer.end_interview(session_id, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/interview/{session_id}")
def interview_get(session_id: str):
    session = interviewer.get_interview(session_id, get_store())
    if session is None:
        raise HTTPException(status_code=404, detail="interview not found")
    return session


# ------------------------------------------------------- evidence / readiness


@router.get("/evidence")
def evidence_list(user_id: str = "default", competency_id: Optional[str] = None):
    store = get_store()
    items = store.where("evidence", user_id=user_id)
    if competency_id:
        items = [e for e in items if competency_id in e.get("competency_ids", [])]
    items.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return {"evidence": items, "count": len(items)}


@router.get("/readiness")
def readiness_view(user_id: str = "default"):
    store = get_store()
    matrix = readiness_mod.compute_matrix(user_id, store)
    bp = load_blueprint(store)
    idx = competency_index(bp)
    rows: List[Dict] = []
    for cid, r in matrix.items():
        comp = idx[cid]
        rows.append({**r.model_dump(), "name": comp.name, "phase": comp.phase,
                     "dimension": comp.dimension.value, "prerequisites": comp.prerequisites})
    return {
        "matrix": rows,
        "dimensions": readiness_mod.dimension_summary(user_id, store, matrix),
        "bottleneck": readiness_mod.bottleneck_analysis(user_id, store, matrix),
    }


@router.get("/next-action")
def next_action(user_id: str = "default"):
    return adaptive.next_best_action(user_id, get_store())


# ---------------------------------------------------------------------- journey


@router.get("/journey")
def journey_get(user_id: str = "default"):
    return adaptive.get_journey(user_id, get_store())


@router.post("/journey/advance")
def journey_advance(req: JourneyAdvanceRequest):
    try:
        return adaptive.advance_journey(req.user_id, req.event, req.competency_id,
                                        req.passed, store=get_store())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
