"""AI interviewer: stateful adaptive sessions (technical / management / hybrid).

There is NO fixed question list. Questions come from the competency-keyed
exercise generator; after every answer the evaluation contract drives the
sequencing policy:
  - weakness detected (score <= 2)    -> targeted follow-up probe from the
                                         exercise's follow_up_questions
  - strong answer (score >= 4)        -> escalate difficulty, or switch
                                         dimension once difficulty is maxed
  - shallow-but-polished (score 3
    with polish markers but thin
    concept coverage)                 -> depth probe on a missed criterion
  - otherwise                         -> advance to a new competency

The transcript is stored as an Evidence artifact.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.hillclimb import content, evaluate
from sensorflow.hillclimb.blueprint import competency_index, load_blueprint
from sensorflow.hillclimb.models import (EvaluationResult, Evidence, Store, get_store,
                                         new_id, now_iso)

MODES = {
    "technical": [1, 2],
    "management": [3],
    "hybrid": [1, 2, 3, 4],
}

POLISH_MARKERS = ["first", "second", "third", "in summary", "to summarize", "overall",
                  "furthermore", "moreover", "in conclusion", "let me walk", "great question"]


class InterviewTurn(BaseModel):
    index: int
    question: str
    question_type: str  # opening | probe | depth_probe | escalate | advance
    competency_id: str
    exercise_id: str
    difficulty: int
    answer: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    timestamp: str = Field(default_factory=now_iso)


class InterviewSession(BaseModel):
    session_id: str = Field(default_factory=lambda: new_id("iv"))
    user_id: str = "default"
    mode: str = "hybrid"
    seed: int = 7
    status: str = "active"  # active | complete
    turns: List[InterviewTurn] = Field(default_factory=list)
    asked_competencies: List[str] = Field(default_factory=list)
    probe_count_on_current: int = 0
    evidence_id: Optional[str] = None


def _pool(mode: str, store: Store) -> List[str]:
    bp = load_blueprint(store)
    phases = MODES.get(mode, MODES["hybrid"])
    return [c.id for c in bp.competencies if c.phase in phases]


def _pick_competency(session: InterviewSession, store: Store,
                     avoid_dimension_of: Optional[str] = None) -> str:
    bp = load_blueprint(store)
    idx = competency_index(bp)
    pool = [c for c in _pool(session.mode, store) if c not in session.asked_competencies]
    if not pool:
        pool = _pool(session.mode, store)
    if avoid_dimension_of and avoid_dimension_of in idx:
        dim = idx[avoid_dimension_of].dimension
        switched = [c for c in pool if idx[c].dimension != dim]
        if switched:
            pool = switched
    rng = random.Random(session.seed + len(session.turns) * 131)
    return rng.choice(sorted(pool))


def _new_exercise_turn(session: InterviewSession, competency_id: str, difficulty: int,
                       question_type: str, store: Store) -> InterviewTurn:
    ex = content.generate_exercise(competency_id, difficulty=difficulty,
                                   seed=session.seed + len(session.turns) * 977,
                                   store=store, use_llm=False)
    session.asked_competencies.append(competency_id)
    session.probe_count_on_current = 0
    return InterviewTurn(index=len(session.turns), question=ex.scenario,
                         question_type=question_type, competency_id=competency_id,
                         exercise_id=ex.exercise_id, difficulty=difficulty)


def start_interview(mode: str = "hybrid", user_id: str = "default", seed: int = 7,
                    store: Optional[Store] = None) -> InterviewSession:
    store = store or get_store()
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}' (technical|management|hybrid)")
    session = InterviewSession(user_id=user_id, mode=mode, seed=seed)
    first_comp = _pick_competency(session, store)
    session.turns.append(_new_exercise_turn(session, first_comp, difficulty=2,
                                            question_type="opening", store=store))
    store.put("interviews", session.session_id, session)
    return session


def _is_polished_but_shallow(result: EvaluationResult, answer: str) -> bool:
    low = answer.lower()
    polish = sum(1 for m in POLISH_MARKERS if m in low)
    concept_strengths = [s for s in result.strengths
                         if not s.lower().startswith(("quantifies", "states explicit tradeoff"))]
    return polish >= 2 and len(concept_strengths) <= 2


def submit_answer(session_id: str, answer: str, use_llm: bool = True,
                  store: Optional[Store] = None) -> InterviewSession:
    store = store or get_store()
    raw = store.get("interviews", session_id)
    if raw is None:
        raise ValueError(f"unknown interview session '{session_id}'")
    session = InterviewSession(**raw)
    if session.status != "active":
        raise ValueError("interview already complete")

    current = session.turns[-1]
    ex = content.get_exercise(current.exercise_id, store)
    result = evaluate.evaluate_response(ex, answer, use_llm=use_llm)
    current.answer = answer
    current.evaluation = result

    # ------------------------------------------------------ sequencing policy
    next_turn: Optional[InterviewTurn] = None
    if result.score <= 2 and session.probe_count_on_current < 2 and ex.follow_up_questions:
        probe_idx = min(session.probe_count_on_current, len(ex.follow_up_questions) - 1)
        session.probe_count_on_current += 1
        next_turn = InterviewTurn(index=len(session.turns),
                                  question=ex.follow_up_questions[probe_idx],
                                  question_type="probe", competency_id=current.competency_id,
                                  exercise_id=current.exercise_id, difficulty=current.difficulty)
    elif result.score >= 4:
        if current.difficulty < 3:
            next_turn = _new_exercise_turn(session, current.competency_id,
                                           difficulty=current.difficulty + 1,
                                           question_type="escalate", store=store)
        else:
            comp = _pick_competency(session, store, avoid_dimension_of=current.competency_id)
            next_turn = _new_exercise_turn(session, comp, difficulty=3,
                                           question_type="escalate", store=store)
    elif result.score == 3 and _is_polished_but_shallow(result, answer):
        missed = result.weaknesses[0] if result.weaknesses else "the mechanism behind your answer"
        next_turn = InterviewTurn(
            index=len(session.turns),
            question=(f"That was well structured, but let's go deeper. Specifically: "
                      f"{missed.rstrip('.').lower()}. Explain the underlying mechanism with a "
                      f"concrete example and real numbers."),
            question_type="depth_probe", competency_id=current.competency_id,
            exercise_id=current.exercise_id, difficulty=current.difficulty)
    else:
        comp = _pick_competency(session, store)
        next_turn = _new_exercise_turn(session, comp, difficulty=2,
                                       question_type="advance", store=store)

    session.turns.append(next_turn)
    store.put("interviews", session.session_id, session)
    return session


def end_interview(session_id: str, store: Optional[Store] = None) -> InterviewSession:
    store = store or get_store()
    raw = store.get("interviews", session_id)
    if raw is None:
        raise ValueError(f"unknown interview session '{session_id}'")
    session = InterviewSession(**raw)
    if session.status == "complete":
        return session
    session.status = "complete"

    answered = [t for t in session.turns if t.evaluation is not None]
    per_comp: Dict[str, List[int]] = {}
    for t in answered:
        per_comp.setdefault(t.competency_id, []).append(t.evaluation.score)
    avg = (sum(t.evaluation.score for t in answered) / len(answered)) if answered else 0.0

    ev = Evidence(
        user_id=session.user_id,
        competency_ids=list(per_comp.keys()),
        artifact_type="interview_transcript",
        source=f"AI interview ({session.mode})",
        summary=(f"{len(answered)} answers across {len(per_comp)} competencies, "
                 f"avg score {avg:.1f}/5."),
        quotes=[t.evaluation.evidence[0] for t in answered if t.evaluation.evidence][:4],
        score=round(avg, 2),
        confidence=min(0.85, 0.3 + 0.1 * len(answered)),
        payload={"session_id": session.session_id,
                 "per_competency": {k: sum(v) / len(v) for k, v in per_comp.items()},
                 "transcript": [t.model_dump() for t in session.turns]},
    )
    store.put("evidence", ev.evidence_id, ev)
    session.evidence_id = ev.evidence_id
    store.put("interviews", session.session_id, session)
    return session


def get_interview(session_id: str, store: Optional[Store] = None) -> Optional[InterviewSession]:
    raw = (store or get_store()).get("interviews", session_id)
    return InterviewSession(**raw) if raw else None
