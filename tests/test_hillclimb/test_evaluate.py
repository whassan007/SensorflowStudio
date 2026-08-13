"""Evaluation contract, evidence-required rule, anti-gaming, coaching specificity."""

import pytest
from pydantic import ValidationError

from sensorflow.hillclimb import coach
from sensorflow.hillclimb.content import generate_exercise
from sensorflow.hillclimb.evaluate import rule_based_evaluate
from sensorflow.hillclimb.models import EvaluationResult

from .conftest import (CONCISE_PRECISE_ANSWER, STRONG_TECH_ANSWER, VERBOSE_WAFFLE,
                       WEAK_ANSWER)


def _exercise(seed=1):
    return generate_exercise("p1.regression_detection", seed=seed, use_llm=False)


def test_contract_schema_enforced():
    with pytest.raises(ValidationError):
        EvaluationResult(competency="x", score=0, confidence=0.5)
    with pytest.raises(ValidationError):
        EvaluationResult(competency="x", score=6, confidence=0.5)
    with pytest.raises(ValidationError):
        EvaluationResult(competency="x", score=3, confidence=1.5)


def test_strong_answer_scores_high_with_quoted_evidence():
    result = rule_based_evaluate(_exercise(), STRONG_TECH_ANSWER)
    assert result.score >= 4
    assert result.evidence, "no score without evidence"
    # evidence quotes the user's own statements
    for quote in result.evidence:
        assert quote.rstrip("…")[:40].lower() in STRONG_TECH_ANSWER.lower()
    assert result.follow_up_question in _exercise().follow_up_questions


def test_no_score_without_evidence():
    result = rule_based_evaluate(_exercise(), WEAK_ANSWER)
    assert result.score == 1
    assert result.evidence == []
    empty = rule_based_evaluate(_exercise(), "")
    assert empty.score == 1
    # the rule as an invariant: any score above 1 must carry evidence
    for answer in [STRONG_TECH_ANSWER, CONCISE_PRECISE_ANSWER, VERBOSE_WAFFLE, WEAK_ANSWER]:
        r = rule_based_evaluate(_exercise(), answer)
        if r.score > 1:
            assert r.evidence


def test_anti_gaming_verbose_waffle_scores_below_concise_precise():
    ex = _exercise()
    waffle = rule_based_evaluate(ex, VERBOSE_WAFFLE)
    concise = rule_based_evaluate(ex, CONCISE_PRECISE_ANSWER)
    assert len(VERBOSE_WAFFLE) > 4 * len(CONCISE_PRECISE_ANSWER)
    assert waffle.score < concise.score, (
        f"verbosity must not raise scores: waffle={waffle.score} concise={concise.score}")


def test_misconception_detection():
    ex = _exercise()
    r = rule_based_evaluate(ex, "This is definitely a true regression, no need to test more. "
                                "Offline eval is ground truth anyway.")
    assert r.misconceptions
    assert any("significance" in m.lower() or "premature" in m.lower() for m in r.misconceptions)


def test_missing_evidence_lists_specific_gaps():
    ex = _exercise()
    r = rule_based_evaluate(ex, CONCISE_PRECISE_ANSWER)
    joined = " ".join(r.missing_evidence).lower()
    assert "quantified" in joined or "tradeoff" in joined


def test_coaching_is_specific_not_generic():
    ex = _exercise()
    r = rule_based_evaluate(ex, CONCISE_PRECISE_ANSWER)
    feedback = coach.build_feedback(ex, r)
    # references what the answer DID establish...
    assert "establishes" in feedback
    # ...and points at concrete rubric gaps, not platitudes
    assert "Missing:" in feedback or "quantified" in feedback.lower()
    # recommended action mentions the competency / retry semantics
    assert r.recommended_action
    assert "p1.regression_detection" in r.recommended_action or "exercise" in r.recommended_action
