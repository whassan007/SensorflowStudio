"""STAR Story Box: diagnose (not just rewrite) unstructured experience stories.

Segments raw text into Situation/Task/Action/Result, runs per-component checks
(context clarity, personal ownership, constraints, tradeoffs, influence/
disagreement handling, measurable outcome, follow-through), flags unquantified
claims vs measurable evidence, and maps the story onto the Phase-3 leadership
competencies it actually evidences. Saved stories become Evidence artifacts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.hillclimb.evaluate import QUANT_RE, TRADEOFF_MARKERS, split_sentences
from sensorflow.hillclimb.models import Evidence, Store, get_store

CLAIM_RE = re.compile(
    r"\b(improved?|increased?|reduced?|decreased?|dropped?|fell|declined?|doubled?|halved?|"
    r"optimi[sz]ed?|accelerated?|boosted?|cut|grew|saved?|streamlined?|enhanced?|"
    r"significantly|dramatically|much (better|faster)|"
    r"more (efficient|reliable|scalable))\b", re.IGNORECASE)

ACTION_VERBS = ["led", "built", "decided", "wrote", "organized", "met", "proposed", "implemented",
                "negotiated", "created", "ran", "set up", "drove", "presented", "designed",
                "restructured", "escalated", "coached", "hired", "paused", "refactored", "convinced",
                "prioritized", "delegated", "launched", "defined", "pushed"]

S_CUES = ["at the time", "we had", "the team", "our system", "when i joined", "was facing",
          "inherited", "the situation", "context", "back in", "my team of", "our org"]
T_CUES = ["my goal", "i was asked", "needed to", "responsible for", "my task", "the goal was",
          "we needed", "i had to", "my job was", "the ask was", "was tasked"]
R_CUES = ["as a result", "result", "outcome", "ultimately", "in the end", "shipped", "launched",
          "improved", "reduced", "increased", "ended up", "afterwards", "since then"]


class ComponentDiagnosis(BaseModel):
    component: str  # S | T | A | R
    label: str
    sentences: List[str] = Field(default_factory=list)
    present: bool = False
    issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


class ClaimFlag(BaseModel):
    sentence: str
    kind: str  # unquantified_claim | measurable_evidence
    detail: str


class CheckResult(BaseModel):
    check: str
    label: str
    passed: bool
    detail: str
    evidence: Optional[str] = None


class StarDiagnosis(BaseModel):
    components: List[ComponentDiagnosis]
    checks: List[CheckResult]
    claim_flags: List[ClaimFlag]
    competencies: List[Dict] = Field(default_factory=list)
    overall_score: int = 1
    coaching: List[str] = Field(default_factory=list)
    evidence_id: Optional[str] = None


def _classify_sentences(sentences: List[str]) -> Dict[str, List[str]]:
    """Assign each sentence to S/T/A/R via cue scoring with a positional prior."""
    buckets: Dict[str, List[str]] = {"S": [], "T": [], "A": [], "R": []}
    n = max(1, len(sentences))
    for i, s in enumerate(sentences):
        low = s.lower()
        pos = i / n
        scores = {
            "S": sum(2 for c in S_CUES if c in low) + (1.5 if pos < 0.25 else 0),
            "T": sum(2 for c in T_CUES if c in low) + (0.5 if pos < 0.4 else 0),
            "A": sum(2 for v in ACTION_VERBS if re.search(rf"\bi\s+(?:\w+\s+)?{v}\b|\bi {v}\b", low)) +
                 (1 if re.search(r"\bi\b", low) and 0.2 <= pos <= 0.85 else 0),
            "R": sum(2 for c in R_CUES if c in low) + (2 if QUANT_RE.search(s) and pos > 0.4 else 0) +
                 (1 if pos > 0.75 else 0),
        }
        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            best = "S" if pos < 0.3 else ("A" if pos < 0.7 else "R")
        buckets[best].append(s)
    return buckets


def _claim_analysis(sentences: List[str]) -> List[ClaimFlag]:
    flags: List[ClaimFlag] = []
    for i, s in enumerate(sentences):
        if not CLAIM_RE.search(s):
            continue
        neighborhood = " ".join(sentences[max(0, i - 1):i + 2])
        if QUANT_RE.search(neighborhood):
            flags.append(ClaimFlag(sentence=s, kind="measurable_evidence",
                                   detail="Claim is backed by a number/before-after nearby — good."))
        else:
            claim_word = CLAIM_RE.search(s).group(0)
            flags.append(ClaimFlag(
                sentence=s, kind="unquantified_claim",
                detail=(f"'{claim_word}' is an unquantified claim. Strengthen it: what was the metric "
                        f"before and after, and what named mechanism caused the change?")))
    return flags


COMPETENCY_CUES = {
    "p3.conflict_resolution": ["disagree", "conflict", "pushback", "mediat", "tension", "clash"],
    "p3.hiring": ["hire", "hiring", "interview loop", "recruit", "offer", "onboard", "headcount"],
    "p3.mentorship": ["mentor", "coach", "junior", "grew into", "promotion", "growth plan"],
    "p3.performance_management": ["underperform", "pip", "performance review", "expectations",
                                  "difficult feedback", "low performer"],
    "p3.roadmap_ambiguity": ["roadmap", "ambigu", "unclear scope", "milestone", "no clear plan", "from scratch"],
    "p3.cross_functional": ["product manager", " pm ", "cross-functional", "partner team", "design team",
                            "legal", "ops team", "research team"],
    "p3.org_influence": ["exec", "leadership review", "vp", "director", "rfc", "org-wide", "influence"],
    "p3.prioritization_risk": ["priorit", "risk", "cut scope", "deprioritiz", "deadline", "trade"],
    "p3.safety_culture": ["safety", "incident", "postmortem", "sev", "regression escaped"],
    "p3.closed_loop_execution": ["measured", "metric", "iterated", "course-correct", "weekly review",
                                 "tracked", "dashboards"],
    "p3.business_impact": ["revenue", "cost sav", "customer", "business", "churn", "$", "contract"],
    "p3.technical_strategy": ["strategy", "platform", "migration", "build vs buy", "technical bet", "rewrite"],
}


def _map_competencies(text_lower: str) -> List[Dict]:
    hits: List[Dict] = []
    for cid, cues in COMPETENCY_CUES.items():
        matched = [c for c in cues if c in text_lower]
        if matched:
            hits.append({"competency_id": cid,
                         "matched_cues": matched[:3],
                         "reason": f"Story references {', '.join(matched[:3])}"})
    return hits


def diagnose_story(text: str, user_id: str = "default", save_evidence: bool = False,
                   store: Optional[Store] = None) -> StarDiagnosis:
    store = store or get_store()
    sentences = split_sentences(text)
    text_lower = text.lower()
    buckets = _classify_sentences(sentences)

    labels = {"S": "Situation", "T": "Task", "A": "Action", "R": "Result"}
    components: List[ComponentDiagnosis] = []
    for key in ["S", "T", "A", "R"]:
        sents = buckets[key]
        comp = ComponentDiagnosis(component=key, label=labels[key], sentences=sents, present=bool(sents))
        if not sents:
            comp.issues.append(f"No {labels[key]} content detected — the story never establishes it.")
        components.append(comp)
    by_key = {c.component: c for c in components}

    # ---------------------------------------------------------------- checks
    checks: List[CheckResult] = []

    def add_check(check: str, label: str, passed: bool, detail: str, evidence: Optional[str] = None):
        checks.append(CheckResult(check=check, label=label, passed=passed, detail=detail, evidence=evidence))

    s_text = " ".join(buckets["S"]).lower()
    context_tokens = bool(re.search(r"team|engineer|org|system|product|customer|quarter|year", s_text))
    context_scale = bool(QUANT_RE.search(" ".join(buckets["S"]))) or bool(re.search(r"\d", s_text))
    add_check("context_clarity", "Context clarity",
              bool(buckets["S"]) and context_tokens,
              "Situation should establish team, system, and scale so a stranger can follow."
              + ("" if context_scale else " Add scale (team size, data volume, timeline) as numbers."),
              buckets["S"][0] if buckets["S"] else None)

    a_text = " ".join(buckets["A"]).lower()
    i_count = len(re.findall(r"\bi\b", a_text))
    we_count = len(re.findall(r"\bwe\b", a_text))
    ownership = i_count >= 2 and (we_count == 0 or i_count >= we_count)
    add_check("personal_ownership", "Personal ownership",
              ownership,
              (f"Actions use 'I' {i_count}x vs 'we' {we_count}x. "
               + ("Clear personal ownership." if ownership else
                  "Mostly 'we' — identify the decision YOU made, the alternatives YOU rejected.")),
              buckets["A"][0] if buckets["A"] else None)

    constraint_hit = re.search(r"deadline|constraint|limited|only had|budget|headcount|pressure|"
                               r"within \d|couldn't|shortage|freeze", text_lower)
    add_check("constraints", "Constraints named", bool(constraint_hit),
              "Stories are credible when the constraints (time, people, budget) are explicit."
              if not constraint_hit else "Constraints are explicit — good.",
              _sentence_containing(sentences, constraint_hit.group(0)) if constraint_hit else None)

    tradeoff_hit = next((m for m in TRADEOFF_MARKERS if m in text_lower), None)
    add_check("tradeoffs", "Tradeoffs / alternatives", bool(tradeoff_hit),
              "State what you chose NOT to do and why." if not tradeoff_hit
              else "Alternatives/tradeoffs are visible.",
              _sentence_containing(sentences, tradeoff_hit) if tradeoff_hit else None)

    influence_hit = re.search(r"disagree|pushback|convinc|persuad|align|escalat|stakeholder|"
                              r"object|resist|bought in|buy-in", text_lower)
    add_check("influence_disagreement", "Influence & disagreement handling", bool(influence_hit),
              "EM stories should show navigating disagreement or influencing without authority."
              if not influence_hit else "Disagreement/influence handling present.",
              _sentence_containing(sentences, influence_hit.group(0)) if influence_hit else None)

    r_joined = " ".join(buckets["R"])
    measurable = bool(QUANT_RE.search(r_joined))
    add_check("measurable_outcome", "Measurable outcome", measurable,
              "The Result has no numbers. Add before/after values or a quantified business outcome."
              if not measurable else "Result is quantified.",
              next((s for s in buckets["R"] if QUANT_RE.search(s)), None))

    follow_hit = re.search(r"monitor|sustain|long.term|retro|followed up|since then|kept|maintained|"
                           r"checked back|still|a year later|months later", text_lower)
    add_check("follow_through", "Follow-through", bool(follow_hit),
              "Show the change stuck: monitoring, retros, or how it looked months later."
              if not follow_hit else "Follow-through demonstrated.",
              _sentence_containing(sentences, follow_hit.group(0)) if follow_hit else None)

    # ------------------------------------------------------- claims & mapping
    claim_flags = _claim_analysis(sentences)
    unquantified = [f for f in claim_flags if f.kind == "unquantified_claim"]
    competencies = _map_competencies(text_lower)

    passed = sum(1 for c in checks if c.passed)
    overall = max(1, min(5, 1 + round(passed * 4 / len(checks))))
    if unquantified and overall > 3:
        overall = 3  # unquantified claims cap the story quality

    coaching: List[str] = []
    for c in checks:
        if not c.passed:
            coaching.append(f"{c.label}: {c.detail}")
    for f in unquantified[:3]:
        coaching.append(f"Strengthen evidence — \"{f.sentence[:90]}\": {f.detail}")
    if not competencies:
        coaching.append("This story doesn't clearly evidence any Phase-3 competency; "
                        "anchor it to a leadership behavior (conflict, hiring, strategy, execution).")

    diagnosis = StarDiagnosis(components=components, checks=checks, claim_flags=claim_flags,
                              competencies=competencies, overall_score=overall, coaching=coaching)

    if save_evidence:
        quotes = [f.sentence for f in claim_flags if f.kind == "measurable_evidence"][:4]
        if not quotes and buckets["A"]:
            quotes = buckets["A"][:2]
        ev = Evidence(
            user_id=user_id,
            competency_ids=[c["competency_id"] for c in competencies],
            artifact_type="star_story",
            source="STAR Story Box",
            summary=(buckets["S"][0] if buckets["S"] else sentences[0] if sentences else "")[:160],
            quotes=quotes,
            score=float(overall),
            confidence=round(0.4 + 0.08 * passed, 2),
            payload={"checks": [c.model_dump() for c in checks],
                     "unquantified_claims": len(unquantified),
                     "story_text": text},
        )
        store.put("evidence", ev.evidence_id, ev)
        diagnosis.evidence_id = ev.evidence_id

    return diagnosis


def _sentence_containing(sentences: List[str], token: str) -> Optional[str]:
    tl = token.lower()
    for s in sentences:
        if tl in s.lower():
            return s
    return None
