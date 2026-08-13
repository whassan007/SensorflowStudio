"""Interviewer adaptivity: sequencing must depend on evaluations, no fixed list."""

from sensorflow.hillclimb.content import get_exercise
from sensorflow.hillclimb.interviewer import (end_interview, start_interview,
                                              submit_answer)

from .conftest import STRONG_LEADERSHIP_ANSWER, WEAK_ANSWER


def test_weak_answer_triggers_targeted_probe(isolated_store):
    s = start_interview("management", seed=5, store=isolated_store)
    first = s.turns[0]
    s = submit_answer(s.session_id, WEAK_ANSWER, use_llm=False, store=isolated_store)
    nxt = s.turns[-1]
    assert s.turns[0].evaluation.score <= 2
    assert nxt.question_type == "probe"
    # the probe comes from the exercise's own follow_up_questions
    ex = get_exercise(first.exercise_id, isolated_store)
    assert nxt.question in ex.follow_up_questions
    assert nxt.competency_id == first.competency_id


def test_strong_answer_escalates(isolated_store):
    s = start_interview("management", seed=5, store=isolated_store)
    s = submit_answer(s.session_id, STRONG_LEADERSHIP_ANSWER, use_llm=False, store=isolated_store)
    assert s.turns[0].evaluation.score >= 4
    nxt = s.turns[-1]
    assert nxt.question_type == "escalate"
    assert nxt.difficulty == 3  # difficulty went up


def test_sequencing_depends_on_answer_quality(isolated_store):
    weak = start_interview("management", seed=5, store=isolated_store)
    strong = start_interview("management", seed=5, store=isolated_store)
    # identical opening question (same seed) ...
    assert weak.turns[0].question == strong.turns[0].question
    weak = submit_answer(weak.session_id, WEAK_ANSWER, use_llm=False, store=isolated_store)
    strong = submit_answer(strong.session_id, STRONG_LEADERSHIP_ANSWER, use_llm=False,
                           store=isolated_store)
    # ... but different evaluations produce different next questions
    assert weak.turns[-1].question != strong.turns[-1].question
    assert weak.turns[-1].question_type != strong.turns[-1].question_type


def test_probe_limit_then_moves_on(isolated_store):
    s = start_interview("management", seed=5, store=isolated_store)
    s = submit_answer(s.session_id, WEAK_ANSWER, use_llm=False, store=isolated_store)
    s = submit_answer(s.session_id, WEAK_ANSWER, use_llm=False, store=isolated_store)
    assert s.turns[-1].question_type == "probe"
    s = submit_answer(s.session_id, WEAK_ANSWER, use_llm=False, store=isolated_store)
    assert s.turns[-1].question_type == "advance"  # max 2 probes, then new competency


def test_transcript_saved_as_evidence(isolated_store):
    s = start_interview("technical", seed=8, store=isolated_store)
    s = submit_answer(s.session_id, "Because dynamic batching amortizes overhead it can fail "
                                    "under overload; I would keep 30% headroom as the tradeoff.",
                      use_llm=False, store=isolated_store)
    s = end_interview(s.session_id, store=isolated_store)
    assert s.status == "complete" and s.evidence_id
    raw = isolated_store.get("evidence", s.evidence_id)
    assert raw["artifact_type"] == "interview_transcript"
    assert raw["payload"]["transcript"]
    assert raw["competency_ids"]


def test_modes_restrict_competency_pool(isolated_store):
    tech = start_interview("technical", seed=2, store=isolated_store)
    mgmt = start_interview("management", seed=2, store=isolated_store)
    assert tech.turns[0].competency_id.split(".")[0] in ("p1", "p2")
    assert mgmt.turns[0].competency_id.startswith("p3")
