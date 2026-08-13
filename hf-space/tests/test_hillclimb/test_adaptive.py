"""Diagnostic flow adaptivity and next-best-action shape."""

from sensorflow.hillclimb import adaptive
from sensorflow.hillclimb.blueprint import competency_index, load_blueprint

from .conftest import STRONG_TECH_ANSWER, WEAK_ANSWER


def test_diagnostic_seeds_matrix_and_completes(isolated_store):
    dx = adaptive.start_diagnostic(store=isolated_store)
    assert dx.status == "active" and dx.current_exercise_id
    answers = [WEAK_ANSWER, STRONG_TECH_ANSWER, WEAK_ANSWER,
               STRONG_TECH_ANSWER, WEAK_ANSWER, WEAK_ANSWER]
    for a in answers:
        dx = adaptive.answer_diagnostic(dx.diagnostic_id, a, use_llm=False, store=isolated_store)
    assert dx.status == "complete"
    assert dx.answered == 6
    # attempts recorded -> matrix seeded
    attempts = isolated_store.all("attempts")
    assert len(attempts) == 6
    assert all(a["kind"] == "diagnostic" for a in attempts)
    # journey advanced past diagnostic
    j = adaptive.get_journey(store=isolated_store)
    assert j.state.value == "LEARNING"


def test_diagnostic_adapts_weak_answer_drills_into_prerequisites(isolated_store):
    idx = competency_index(load_blueprint(isolated_store))
    dx = adaptive.start_diagnostic(store=isolated_store)
    first = dx.current_competency
    dx = adaptive.answer_diagnostic(dx.diagnostic_id, WEAK_ANSWER, use_llm=False,
                                    store=isolated_store)
    second = dx.current_competency
    prereqs = idx[first].prerequisites
    if prereqs:  # weak answer -> drill into a prerequisite
        assert second in prereqs
    else:  # no prereqs -> continue the starter set
        assert second != first


def test_next_best_action_returns_exactly_one_of_each(isolated_store):
    nba = adaptive.next_best_action(store=isolated_store)
    assert set(nba) >= {"concept", "exercise", "assessment", "explanation", "bottleneck"}
    assert isinstance(nba["concept"], dict) and nba["concept"]["competency_id"]
    assert isinstance(nba["exercise"], dict) and nba["exercise"]["scenario"]
    assert isinstance(nba["assessment"], dict) and nba["assessment"]["description"]
    # all three target the same bottleneck competency
    target = nba["concept"]["competency_id"]
    assert nba["exercise"]["competency_id"] == target
    assert nba["assessment"]["competency_id"] == target
