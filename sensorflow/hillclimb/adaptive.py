"""Adaptive engine: journey state machine, diagnostic assessment, next-best-action.

Journey: NOT_STARTED → DIAGNOSTIC → LEARNING → PRACTICE → ASSESSMENT →
(PASS → next competency | FAIL → REMEDIATION → REASSESS). Failing an advanced
item routes BACKWARD: the engine diagnoses the weakest prerequisite of the
failed competency and assigns remediation there.

Next-best-action always returns exactly {1 concept, 1 exercise, 1 assessment}
aimed at the current bottleneck.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.hillclimb import content, evaluate, readiness
from sensorflow.hillclimb.blueprint import competency_index, load_blueprint
from sensorflow.hillclimb.models import (Attempt, Evidence, Journey, JourneyState,
                                         Store, get_store, new_id, now_iso)

# ------------------------------------------------------------------ journey


def get_journey(user_id: str = "default", store: Optional[Store] = None) -> Journey:
    store = store or get_store()
    raw = store.get("journeys", user_id)
    if raw:
        return Journey(**raw)
    j = Journey(user_id=user_id)
    store.put("journeys", user_id, j)
    return j


def _save_journey(j: Journey, store: Store) -> None:
    j.updated_at = now_iso()
    store.put("journeys", j.user_id, j)


def diagnose_missing_prerequisite(competency_id: str, user_id: str = "default",
                                  store: Optional[Store] = None) -> str:
    """Failing an advanced item -> find the weakest prerequisite to remediate.

    Returns the prerequisite (transitively closest) with the lowest combined
    readiness; if the competency has no prerequisites, remediation stays on
    the competency itself.
    """
    store = store or get_store()
    bp = load_blueprint(store)
    idx = competency_index(bp)
    if competency_id not in idx or not idx[competency_id].prerequisites:
        return competency_id
    matrix = readiness.compute_matrix(user_id, store)
    prereqs = [p for p in idx[competency_id].prerequisites if p in matrix]
    weak = sorted(prereqs, key=lambda p: (readiness.combined_score(matrix[p]), p))
    for p in weak:
        if matrix[p].readiness_state not in readiness.COMPETENT_STATES:
            return p
    return competency_id  # all prereqs fine -> remediate the competency itself


TRANSITIONS = {
    (JourneyState.NOT_STARTED, "start_diagnostic"): JourneyState.DIAGNOSTIC,
    (JourneyState.DIAGNOSTIC, "diagnostic_complete"): JourneyState.LEARNING,
    (JourneyState.LEARNING, "begin_practice"): JourneyState.PRACTICE,
    (JourneyState.PRACTICE, "request_assessment"): JourneyState.ASSESSMENT,
    (JourneyState.REMEDIATION, "remediation_complete"): JourneyState.REASSESS,
}


def advance_journey(user_id: str, event: str, competency_id: Optional[str] = None,
                    passed: Optional[bool] = None, store: Optional[Store] = None) -> Journey:
    store = store or get_store()
    j = get_journey(user_id, store)
    prior = j.state

    if event == "assessment_result":
        if j.state not in (JourneyState.ASSESSMENT, JourneyState.REASSESS):
            raise ValueError(f"assessment_result invalid in state {j.state}")
        target = competency_id or j.current_competency or ""
        if passed:
            j.state = JourneyState.LEARNING
            j.remediation_target = None
            j.current_competency = _next_competency(target, user_id, store)
        else:
            j.state = JourneyState.REMEDIATION
            j.remediation_target = diagnose_missing_prerequisite(target, user_id, store)
    else:
        key = (j.state, event)
        if key not in TRANSITIONS:
            raise ValueError(f"event '{event}' invalid in state {j.state}")
        j.state = TRANSITIONS[key]
        if competency_id:
            j.current_competency = competency_id

    j.history.append({"from": prior, "event": event, "to": j.state,
                      "competency": competency_id, "passed": passed,
                      "remediation_target": j.remediation_target, "at": now_iso()})
    _save_journey(j, store)
    return j


def _next_competency(after: str, user_id: str, store: Store) -> Optional[str]:
    """Next non-competent competency in blueprint order after the given one."""
    bp = load_blueprint(store)
    matrix = readiness.compute_matrix(user_id, store)
    ordered = [c.id for c in bp.competencies]
    start = ordered.index(after) + 1 if after in ordered else 0
    for cid in ordered[start:] + ordered[:start]:
        if cid != after and matrix[cid].readiness_state not in readiness.COMPETENT_STATES:
            return cid
    return None


# ------------------------------------------------------------------ diagnostic


class DiagnosticSession(BaseModel):
    diagnostic_id: str = Field(default_factory=lambda: new_id("dx"))
    user_id: str = "default"
    seed: int = 11
    total_questions: int = 6
    answered: int = 0
    status: str = "active"  # active | complete
    current_exercise_id: Optional[str] = None
    current_competency: Optional[str] = None
    asked: List[str] = Field(default_factory=list)
    results: List[Dict] = Field(default_factory=list)


DIAGNOSTIC_STARTERS = ["p1.precision_recall", "p2.distributed_fundamentals",
                       "p3.roadmap_ambiguity", "p1.regression_detection",
                       "p2.parallel_inference", "p3.prioritization_risk"]


def start_diagnostic(user_id: str = "default", seed: int = 11,
                     store: Optional[Store] = None) -> DiagnosticSession:
    store = store or get_store()
    dx = DiagnosticSession(user_id=user_id, seed=seed)
    first = DIAGNOSTIC_STARTERS[0]
    ex = content.generate_exercise(first, difficulty=2, seed=seed, store=store, use_llm=False)
    dx.current_exercise_id = ex.exercise_id
    dx.current_competency = first
    dx.asked.append(first)
    store.put("diagnostics", dx.diagnostic_id, dx)
    try:
        advance_journey(user_id, "start_diagnostic", competency_id=first, store=store)
    except ValueError:
        pass  # journey already past NOT_STARTED; diagnostic can be retaken
    return dx


def _adaptive_next(dx: DiagnosticSession, last_score: int, store: Store) -> Optional[str]:
    """Adaptive selection: weak answer drills into prerequisites, strong answer
    jumps ahead; otherwise walk the starter set."""
    bp = load_blueprint(store)
    idx = competency_index(bp)
    if last_score <= 2 and dx.current_competency:
        for p in idx[dx.current_competency].prerequisites:
            if p not in dx.asked:
                return p
    if last_score >= 4 and dx.current_competency:
        dependents = [c.id for c in bp.competencies
                      if dx.current_competency in c.prerequisites and c.id not in dx.asked]
        if dependents:
            rng = random.Random(dx.seed + dx.answered)
            return rng.choice(sorted(dependents))
    for cid in DIAGNOSTIC_STARTERS:
        if cid not in dx.asked:
            return cid
    remaining = [c.id for c in bp.competencies if c.id not in dx.asked]
    return remaining[0] if remaining else None


def answer_diagnostic(diagnostic_id: str, answer: str, use_llm: bool = True,
                      store: Optional[Store] = None) -> DiagnosticSession:
    store = store or get_store()
    raw = store.get("diagnostics", diagnostic_id)
    if raw is None:
        raise ValueError(f"unknown diagnostic '{diagnostic_id}'")
    dx = DiagnosticSession(**raw)
    if dx.status != "active":
        raise ValueError("diagnostic already complete")

    ex = content.get_exercise(dx.current_exercise_id, store)
    result = evaluate.evaluate_response(ex, answer, use_llm=use_llm)
    dx.answered += 1
    dx.results.append({"competency_id": dx.current_competency, "score": result.score,
                       "evaluation": result.model_dump()})

    attempt = Attempt(user_id=dx.user_id, exercise_id=ex.exercise_id,
                      competency_id=dx.current_competency, kind="diagnostic",
                      responses={"answer": answer}, evaluation=result)
    readiness.record_attempt(attempt, store)

    if dx.answered >= dx.total_questions:
        dx.status = "complete"
        dx.current_exercise_id = None
        dx.current_competency = None
        ev = Evidence(user_id=dx.user_id,
                      competency_ids=[r["competency_id"] for r in dx.results],
                      artifact_type="diagnostic", source="Adaptive diagnostic",
                      summary=f"Diagnostic across {len(dx.results)} competencies.",
                      quotes=[], score=_avg_score(dx), confidence=0.5,
                      payload={"diagnostic_id": dx.diagnostic_id})
        store.put("evidence", ev.evidence_id, ev)
        readiness.compute_matrix(dx.user_id, store)
        try:
            advance_journey(dx.user_id, "diagnostic_complete", store=store)
        except ValueError:
            pass
    else:
        nxt = _adaptive_next(dx, result.score, store)
        ex2 = content.generate_exercise(nxt, difficulty=2, seed=dx.seed + dx.answered * 37,
                                        store=store, use_llm=False)
        dx.current_exercise_id = ex2.exercise_id
        dx.current_competency = nxt
        dx.asked.append(nxt)

    store.put("diagnostics", dx.diagnostic_id, dx)
    return dx


def _avg_score(dx: DiagnosticSession) -> float:
    scores = [r["score"] for r in dx.results]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def get_diagnostic(diagnostic_id: str, store: Optional[Store] = None) -> Optional[DiagnosticSession]:
    raw = (store or get_store()).get("diagnostics", diagnostic_id)
    return DiagnosticSession(**raw) if raw else None


# ------------------------------------------------------------- next best action


def next_best_action(user_id: str = "default", store: Optional[Store] = None) -> Dict:
    """Exactly one concept + one exercise + one assessment, aimed at the bottleneck."""
    store = store or get_store()
    bp = load_blueprint(store)
    idx = competency_index(bp)
    matrix = readiness.compute_matrix(user_id, store)
    bn = readiness.bottleneck_analysis(user_id, store, matrix)

    if bn is None:  # everything competent — polish the weakest STRONG item
        ranked = sorted(matrix.values(), key=readiness.combined_score)
        target = ranked[0].competency_id
        explanation = "All competencies are at COMPETENT or better; polishing the weakest one."
    else:
        target = bn["competency_id"]
        explanation = bn["explanation"]

    comp = idx[target]
    phase = next(p for p in bp.phases if p.phase == comp.phase)
    exercise = content.generate_exercise(target, difficulty=2, store=store, use_llm=False)

    return {
        "bottleneck": bn,
        "explanation": explanation,
        "concept": {
            "competency_id": target,
            "name": comp.name,
            "study": comp.topics,
            "description": comp.description,
            "phase_objective": phase.objective,
        },
        "exercise": exercise.model_dump(),
        "assessment": {
            "competency_id": target,
            "kind": "scored_scenario",
            "description": (f"Timed assessment: answer a fresh {comp.name} scenario without notes; "
                            f"target score 4+/5 to mark this competency COMPETENT."),
        },
    }
