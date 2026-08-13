"""Journey state machine incl. FAIL -> REMEDIATION routing to the diagnosed prerequisite."""

import pytest

from sensorflow.hillclimb import adaptive, readiness
from sensorflow.hillclimb.models import Attempt, EvaluationResult, JourneyState


def _seed(store, cid, score):
    for kind in ("diagnostic", "exercise"):
        readiness.record_attempt(
            Attempt(user_id="default", competency_id=cid, kind=kind, exercise_id="x",
                    responses={},
                    evaluation=EvaluationResult(competency=cid, score=score, confidence=0.8,
                                                evidence=["q"])), store)


def test_happy_path_transitions(isolated_store):
    j = adaptive.get_journey(store=isolated_store)
    assert j.state == JourneyState.NOT_STARTED
    j = adaptive.advance_journey("default", "start_diagnostic", store=isolated_store)
    assert j.state == JourneyState.DIAGNOSTIC
    j = adaptive.advance_journey("default", "diagnostic_complete", store=isolated_store)
    assert j.state == JourneyState.LEARNING
    j = adaptive.advance_journey("default", "begin_practice",
                                 competency_id="p1.precision_recall", store=isolated_store)
    assert j.state == JourneyState.PRACTICE
    j = adaptive.advance_journey("default", "request_assessment", store=isolated_store)
    assert j.state == JourneyState.ASSESSMENT
    j = adaptive.advance_journey("default", "assessment_result", passed=True,
                                 competency_id="p1.precision_recall", store=isolated_store)
    assert j.state == JourneyState.LEARNING  # PASS -> next
    assert j.current_competency != "p1.precision_recall"
    assert j.history and j.history[-1]["passed"] is True


def test_invalid_transitions_rejected(isolated_store):
    with pytest.raises(ValueError):
        adaptive.advance_journey("default", "request_assessment", store=isolated_store)
    with pytest.raises(ValueError):
        adaptive.advance_journey("default", "assessment_result", passed=True, store=isolated_store)


def test_fail_routes_remediation_to_diagnosed_prerequisite(isolated_store):
    # weak prerequisite: distributed fundamentals; the advanced item fails
    _seed(isolated_store, "p2.distributed_fundamentals", 2)
    adaptive.advance_journey("default", "start_diagnostic", store=isolated_store)
    adaptive.advance_journey("default", "diagnostic_complete", store=isolated_store)
    adaptive.advance_journey("default", "begin_practice",
                             competency_id="p2.parallel_inference", store=isolated_store)
    adaptive.advance_journey("default", "request_assessment", store=isolated_store)
    j = adaptive.advance_journey("default", "assessment_result", passed=False,
                                 competency_id="p2.parallel_inference", store=isolated_store)
    assert j.state == JourneyState.REMEDIATION
    # backward routing: the diagnosed MISSING PREREQUISITE, not the failed item
    assert j.remediation_target == "p2.distributed_fundamentals"

    j = adaptive.advance_journey("default", "remediation_complete", store=isolated_store)
    assert j.state == JourneyState.REASSESS
    j = adaptive.advance_journey("default", "assessment_result", passed=True,
                                 competency_id="p2.parallel_inference", store=isolated_store)
    assert j.state == JourneyState.LEARNING
    assert j.remediation_target is None


def test_fail_without_prerequisites_remediates_in_place(isolated_store):
    adaptive.advance_journey("default", "start_diagnostic", store=isolated_store)
    adaptive.advance_journey("default", "diagnostic_complete", store=isolated_store)
    adaptive.advance_journey("default", "begin_practice",
                             competency_id="p1.precision_recall", store=isolated_store)
    adaptive.advance_journey("default", "request_assessment", store=isolated_store)
    j = adaptive.advance_journey("default", "assessment_result", passed=False,
                                 competency_id="p1.precision_recall", store=isolated_store)
    assert j.state == JourneyState.REMEDIATION
    assert j.remediation_target == "p1.precision_recall"


def test_reassess_fail_loops_back_to_remediation(isolated_store):
    _seed(isolated_store, "p2.distributed_fundamentals", 2)
    adaptive.advance_journey("default", "start_diagnostic", store=isolated_store)
    adaptive.advance_journey("default", "diagnostic_complete", store=isolated_store)
    adaptive.advance_journey("default", "begin_practice",
                             competency_id="p2.parallel_inference", store=isolated_store)
    adaptive.advance_journey("default", "request_assessment", store=isolated_store)
    adaptive.advance_journey("default", "assessment_result", passed=False,
                             competency_id="p2.parallel_inference", store=isolated_store)
    adaptive.advance_journey("default", "remediation_complete", store=isolated_store)
    j = adaptive.advance_journey("default", "assessment_result", passed=False,
                                 competency_id="p2.parallel_inference", store=isolated_store)
    assert j.state == JourneyState.REMEDIATION
