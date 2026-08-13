"""Coaching feedback generated from rubric gaps — specific, never generic.

Every feedback line points at a concrete gap between the answer and the
exercise rubric ("your answer establishes the technical problem but not
personal ownership; identify the decision you made..."), assembled from what
the evaluation actually found.
"""

from __future__ import annotations

from typing import List

from sensorflow.hillclimb.content import Exercise
from sensorflow.hillclimb.models import EvaluationResult

_GAP_TEMPLATES = {
    "quantified": (
        "Nothing in your answer is quantified. Interviewers discount unmeasured claims — add the "
        "numbers you'd actually use: a threshold, a capacity figure, or a before/after comparison."
    ),
    "tradeoff": (
        "You never state what your approach sacrifices. Name the explicit tradeoff (what gets worse, "
        "by how much, and why that's acceptable) — one honest tradeoff is worth three benefits."
    ),
}


def _keyword_gap_line(criterion: str, keywords: List[str]) -> str:
    hint = ", ".join(f"'{k}'" for k in keywords[:3]) if keywords else criterion.lower()
    return (f"Missing: {criterion.lower()}. Work it in explicitly — an evaluator is scanning for "
            f"substance in the territory of {hint}.")


def build_feedback(exercise: Exercise, result: EvaluationResult) -> str:
    lines: List[str] = []

    if result.strengths:
        established = "; ".join(s.split(" — ")[0].lower() for s in result.strengths[:3])
        lines.append(f"What your answer establishes: {established}.")
    else:
        lines.append(
            "Your answer does not yet engage any rubric criterion for this exercise — the evaluator "
            "found no quotable evidence. Re-read the scenario and answer its specific question."
        )

    covered_criteria = {s.split(" — ")[0] for s in result.strengths}
    for item in exercise.evaluation_rubric:
        if item.criterion in covered_criteria:
            continue
        if item.check in _GAP_TEMPLATES:
            lines.append(_GAP_TEMPLATES[item.check])
        else:
            lines.append(_keyword_gap_line(item.criterion, item.keywords))

    for m in result.misconceptions:
        lines.append(f"Watch out — likely misconception: {m}")

    if result.score >= 4:
        lines.append("This is a strong answer; the remaining gaps above are what separates 'strong' from 'expert'.")
    return "\n".join(lines[:9])


def recommended_action(exercise: Exercise, result: EvaluationResult) -> str:
    comp = exercise.competency_id
    if result.score <= 2:
        return (f"Review the underlying concepts for {comp}, then retry a fresh variant of this "
                f"exercise (a structurally different scenario will be generated — memorizing this one won't help).")
    if result.score == 3:
        return (f"You have the shape of the answer. Practice one more {comp} exercise focusing on the "
                f"missing criteria listed above, then take the assessment.")
    return (f"Strong. Escalate: take a difficulty-{min(3, exercise.difficulty + 1)} exercise for {comp} "
            f"or move to the next competency in your path.")


def pick_follow_up(exercise: Exercise, result: EvaluationResult) -> str:
    if not exercise.follow_up_questions:
        return ""
    # Weak answers get the earlier (more fundamental) probes; strong answers
    # get the later (escalation) probes.
    idx = 0 if result.score <= 2 else min(len(exercise.follow_up_questions) - 1, result.score - 2)
    return exercise.follow_up_questions[idx]
