"""Evaluation engine: rubric-driven, evidence-based scoring of free-text answers.

Rules enforced here:
- NO SCORE WITHOUT EVIDENCE: every score above 1 must reference specific user
  statements (quotes extracted from the answer). No evidence => score 1.
- ANTI-GAMING: score derives ONLY from concept coverage, tradeoff mentions and
  quantified results — never from answer length. A long waffle that covers no
  rubric concepts scores below a short precise answer.
- LLM output (when available) must validate against the EvaluationResult
  contract AND its evidence quotes must actually appear in the user's answer;
  otherwise the deterministic rule-based result is used. An LLM may never
  raise the score more than one point above the rule-based floor.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from sensorflow.hillclimb import coach, llm
from sensorflow.hillclimb.content import Exercise, RubricItem
from sensorflow.hillclimb.models import EvaluationResult

QUANT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent|ms\b|s\b|x\b|qps\b|rps\b|fps\b|gb\b|tb\b|pb\b|"
    r"days?\b|hours?\b|weeks?\b|engineers?\b|gpus?\b|nodes?\b|replicas?\b|frames?\b)|"
    r"\bfrom\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?|\bp9[059]\b|\d+(?:\.\d+)?\s*sigma",
    re.IGNORECASE,
)

TRADEOFF_MARKERS = [
    "trade-off", "tradeoff", "trade off", "at the cost of", "sacrific", "downside",
    "at the expense", " versus ", " vs ", "on the other hand", "the cost is",
    "we give up", "gives up", "in exchange", "instead of", "weighed against",
]

MISCONCEPTION_PATTERNS: List[Tuple[str, str]] = [
    (r"accuracy\s+is\s+the\s+best|just\s+use\s+accuracy",
     "Accuracy is a poor headline metric for imbalanced detection problems; prefer precision/recall/AP."),
    (r"average\s+latency\s+is\s+(fine|enough|what matters)",
     "Serving systems are judged by tail latency (p99), not the average."),
    (r"more\s+data\s+always\s+(helps|fixes|improves)",
     "More data does not fix distribution mismatch, label noise, or serving skew."),
    (r"offline\s+(metrics?|eval(uation)?)\s+(is|are)\s+ground\s+truth",
     "Offline evaluation is a proxy under its own sampling; it is not ground truth for live behavior."),
    (r"definitely\s+a\s+(true\s+)?regression|certainly\s+regressed",
     "Claiming certainty before significance testing / measurement validation is premature."),
]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clip(sentence: str, limit: int = 220) -> str:
    return sentence if len(sentence) <= limit else sentence[:limit].rsplit(" ", 1)[0] + "…"


def _find_sentence(sentences: List[str], predicate) -> Optional[str]:
    for s in sentences:
        if predicate(s):
            return s
    return None


def _criterion_evidence(item: RubricItem, answer_lower: str, sentences: List[str]) -> Optional[str]:
    """Return an evidence quote (user's own sentence) if the criterion is met."""
    if item.check == "quantified":
        hit = _find_sentence(sentences, lambda s: bool(QUANT_RE.search(s)))
        return hit
    if item.check == "tradeoff":
        return _find_sentence(sentences, lambda s: any(m in s.lower() for m in TRADEOFF_MARKERS))
    for kw in item.keywords:
        if kw.lower() in answer_lower:
            hit = _find_sentence(sentences, lambda s: kw.lower() in s.lower())
            if hit:
                return hit
    return None


def rule_based_evaluate(exercise: Exercise, answer: str) -> EvaluationResult:
    sentences = split_sentences(answer)
    answer_lower = answer.lower()

    covered: List[Tuple[RubricItem, str]] = []
    missed: List[RubricItem] = []
    evidence: List[str] = []
    for item in exercise.evaluation_rubric:
        quote = _criterion_evidence(item, answer_lower, sentences)
        if quote:
            covered.append((item, quote))
            clipped = _clip(quote)
            if clipped not in evidence:
                evidence.append(clipped)
        else:
            missed.append(item)

    total_weight = sum(i.weight for i in exercise.evaluation_rubric) or 1.0
    covered_weight = sum(i.weight for i, _ in covered)
    coverage = covered_weight / total_weight

    # Coverage → score. Deliberately independent of answer length.
    if coverage >= 0.85:
        score = 5
    elif coverage >= 0.65:
        score = 4
    elif coverage >= 0.45:
        score = 3
    elif coverage >= 0.20:
        score = 2
    else:
        score = 1

    # No score without evidence.
    if not evidence:
        score = 1

    misconceptions = [msg for pat, msg in MISCONCEPTION_PATTERNS if re.search(pat, answer_lower)]

    strengths = [f"{i.criterion} — \"{_clip(q, 140)}\"" for i, q in covered[:5]]
    weaknesses = [i.criterion for i in missed if i.check == "keywords"][:5]
    missing_evidence = []
    for i in missed:
        if i.check == "quantified":
            missing_evidence.append("No quantified claims: no numbers, thresholds, or before/after comparisons anywhere in the answer.")
        elif i.check == "tradeoff":
            missing_evidence.append("No explicit tradeoff stated: nothing about what your approach sacrifices.")
        else:
            missing_evidence.append(f"Not addressed: {i.criterion.lower()}.")

    confidence = round(min(0.9, 0.35 + 0.07 * len(covered)), 2)

    result = EvaluationResult(
        competency=exercise.competency_id,
        score=score,
        confidence=confidence,
        evidence=evidence,
        strengths=strengths,
        weaknesses=weaknesses,
        missing_evidence=missing_evidence[:6],
        misconceptions=misconceptions,
        evaluator="rule_based",
    )
    result.recommended_action = coach.recommended_action(exercise, result)
    result.follow_up_question = coach.pick_follow_up(exercise, result)
    return result


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .\"'…")


def _llm_result_valid(candidate: EvaluationResult, answer: str) -> bool:
    """LLM evidence must quote the user's actual statements."""
    if not candidate.evidence:
        return False
    normalized_answer = _normalize(answer)
    for quote in candidate.evidence:
        if _normalize(quote) not in normalized_answer:
            return False
    return True


def evaluate_response(exercise: Exercise, answer: str, use_llm: bool = True) -> EvaluationResult:
    rule_result = rule_based_evaluate(exercise, answer)
    if not (use_llm and llm.llm_enabled()):
        return rule_result

    rubric_lines = "\n".join(f"- {i.criterion}" for i in exercise.evaluation_rubric)
    prompt = (
        "You are grading an engineering-manager interview answer against a rubric.\n"
        f"SCENARIO:\n{exercise.scenario}\n\nRUBRIC:\n{rubric_lines}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON object with keys: competency (string), score (int 1-5), "
        "confidence (0-1), evidence (array of EXACT quotes copied verbatim from the answer), "
        "strengths (array), weaknesses (array), missing_evidence (array), misconceptions (array), "
        "recommended_action (string), follow_up_question (string). "
        "Every point awarded must be justified by a quote in evidence. Do not reward verbosity."
    )
    candidate = llm.generate_json(prompt, EvaluationResult, timeout=25.0)
    if candidate is None or not _llm_result_valid(candidate, answer):
        return rule_result

    candidate.competency = exercise.competency_id
    candidate.evaluator = "llm"
    # Anti-gaming guard: the LLM may refine, but never inflate more than one
    # point above the deterministic evidence-based floor.
    if candidate.score > rule_result.score + 1:
        candidate.score = rule_result.score + 1
    if not candidate.recommended_action:
        candidate.recommended_action = coach.recommended_action(exercise, candidate)
    if not candidate.follow_up_question:
        candidate.follow_up_question = coach.pick_follow_up(exercise, candidate)
    return candidate
