"""Exercise generation: required shape, seeded structural variation, canonical family."""

import pytest

from sensorflow.hillclimb.content import generate_exercise, get_exercise


def test_exercise_has_required_shape():
    ex = generate_exercise("p1.cusum", difficulty=2, seed=5, use_llm=False)
    assert ex.competency_id == "p1.cusum"
    assert ex.difficulty == 2
    assert ex.prerequisites == ["p1.statistical_evaluation"]
    assert len(ex.scenario) > 50
    assert ex.expected_reasoning
    assert ex.evaluation_rubric
    assert ex.common_failure_modes
    assert ex.follow_up_questions


def test_structurally_different_regeneration_for_same_competency():
    a = generate_exercise("p1.regression_detection", seed=101, use_llm=False)
    b = generate_exercise("p1.regression_detection", seed=202, use_llm=False)
    assert a.exercise_id != b.exercise_id
    assert a.scenario != b.scenario          # different slot values / variant
    assert a.template_id != b.template_id    # seeded template stamp differs
    # same competency, same family — the rubric skeleton stays comparable
    assert a.competency_id == b.competency_id
    assert a.family == b.family == "offline_shadow"


def test_same_seed_is_deterministic():
    a = generate_exercise("p2.parallel_inference", seed=7, use_llm=False)
    b = generate_exercise("p2.parallel_inference", seed=7, use_llm=False)
    assert a.scenario == b.scenario


def test_canonical_offline_shadow_family_rubric_distinguishes_seven_causes():
    ex = generate_exercise("p1.regression_detection", seed=1, use_llm=False)
    criteria = " | ".join(i.criterion.lower() for i in ex.evaluation_rubric)
    for cause in ["metric definition", "distribution shift", "sampling bias",
                  "feature", "serving", "noise", "true regression"]:
        assert cause in criteria, f"rubric missing cause: {cause}"
    # cross-linked to the live RCA workbench (read-only pointer)
    assert ex.linked_tool is not None
    assert ex.linked_tool["page"] == "rca"
    assert ex.linked_tool["api"] == "/api/rca"


def test_every_competency_can_generate(isolated_store):
    from sensorflow.hillclimb.blueprint import load_blueprint
    bp = load_blueprint(isolated_store)
    for comp in bp.competencies:
        ex = generate_exercise(comp.id, seed=3, use_llm=False)
        assert ex.evaluation_rubric and ex.scenario


def test_exercise_persisted_and_retrievable():
    ex = generate_exercise("p3.hiring", seed=9, use_llm=False)
    again = get_exercise(ex.exercise_id)
    assert again is not None and again.scenario == ex.scenario


def test_unknown_competency_rejected():
    with pytest.raises(ValueError):
        generate_exercise("p9.nope", use_llm=False)
