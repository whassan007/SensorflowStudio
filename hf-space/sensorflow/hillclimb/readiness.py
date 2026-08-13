"""Readiness matrix + bottleneck analysis.

Per competency we track knowledge_score, application_score and evidence_score
separately (never collapsed), derive a readiness state, and link the evidence
behind every number (explainable scores: click a score -> see its evidence).

Bottleneck = the highest-LEVERAGE weakness: the prerequisite blocking the most
downstream competencies — explicitly NOT simply the lowest score.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.hillclimb.blueprint import (competency_index, display_dimensions_for,
                                            downstream_map, load_blueprint,
                                            DISPLAY_DIMENSIONS)
from sensorflow.hillclimb.models import (Attempt, CompetencyReadiness, ReadinessState,
                                         Store, get_store)

KNOWLEDGE_KINDS = {"diagnostic", "interview"}
APPLICATION_KINDS = {"exercise", "design", "simulation"}

COMPETENT_STATES = {ReadinessState.COMPETENT, ReadinessState.STRONG, ReadinessState.INTERVIEW_READY}


def record_attempt(attempt: Attempt, store: Optional[Store] = None) -> None:
    (store or get_store()).put("attempts", attempt.attempt_id, attempt)


def _avg(scores: List[float]) -> float:
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _derive_state(k: float, a: float, e: float, evidence_count: int) -> ReadinessState:
    if k == 0 and a == 0 and evidence_count == 0:
        return ReadinessState.NOT_STARTED
    best = max(k, a)
    if k > 0 and a > 0 and abs(k - a) >= 2.0:
        return ReadinessState.NEEDS_REVIEW
    if best < 2.0:
        return ReadinessState.LEARNING
    if k >= 4.5 and a >= 4.0 and e >= 3.5 and evidence_count >= 2:
        return ReadinessState.INTERVIEW_READY
    if k >= 4.0 and a >= 4.0:
        return ReadinessState.STRONG
    if k >= 3.0 and a >= 3.0:
        return ReadinessState.COMPETENT
    return ReadinessState.PRACTICING


def compute_matrix(user_id: str = "default", store: Optional[Store] = None) -> Dict[str, CompetencyReadiness]:
    store = store or get_store()
    bp = load_blueprint(store)
    attempts = [Attempt(**a) for a in store.where("attempts", user_id=user_id)]
    evidence = store.where("evidence", user_id=user_id)

    matrix: Dict[str, CompetencyReadiness] = {}
    for comp in bp.competencies:
        cid = comp.id
        k_scores = [a.evaluation.score for a in attempts
                    if a.competency_id == cid and a.kind in KNOWLEDGE_KINDS and a.evaluation]
        a_scores = [a.evaluation.score for a in attempts
                    if a.competency_id == cid and a.kind in APPLICATION_KINDS and a.evaluation]
        linked = [e for e in evidence if cid in e.get("competency_ids", [])]
        e_scores = [e.get("score", 0.0) for e in linked if e.get("score", 0.0) > 0]

        k, a, e = _avg(k_scores), _avg(a_scores), _avg(e_scores)
        # interview/exercise evidence also informs application when direct
        # application attempts are absent
        if a == 0 and e > 0:
            a = round(e * 0.8, 2)
        state = _derive_state(k, a, e, len(linked))
        matrix[cid] = CompetencyReadiness(
            competency_id=cid, knowledge_score=k, application_score=a,
            evidence_score=e, readiness_state=state,
            evidence_ids=[ev["evidence_id"] for ev in linked][:20],
        )

    store.put("readiness", user_id, {cid: r.model_dump() for cid, r in matrix.items()})
    return matrix


def combined_score(r: CompetencyReadiness) -> float:
    parts = [s for s in (r.knowledge_score, r.application_score) if s > 0]
    return round(sum(parts) / len(parts), 2) if parts else 0.0


def bottleneck_analysis(user_id: str = "default", store: Optional[Store] = None,
                        matrix: Optional[Dict[str, CompetencyReadiness]] = None) -> Optional[Dict]:
    """Highest-leverage weakness: prerequisite blocking the most downstream work."""
    store = store or get_store()
    bp = load_blueprint(store)
    idx = competency_index(bp)
    matrix = matrix or compute_matrix(user_id, store)
    downstream = downstream_map(bp)

    candidates = [cid for cid, r in matrix.items() if r.readiness_state not in COMPETENT_STATES]
    if not candidates:
        return None

    def blocked_count(cid: str) -> int:
        return sum(1 for d in downstream.get(cid, set())
                   if matrix[d].readiness_state not in COMPETENT_STATES)

    ranked = sorted(candidates,
                    key=lambda cid: (-blocked_count(cid), combined_score(matrix[cid]), cid))
    best = ranked[0]
    blocked = sorted(d for d in downstream.get(best, set())
                     if matrix[d].readiness_state not in COMPETENT_STATES)
    comp = idx[best]
    blocked_names = [idx[b].name for b in blocked[:4]]
    if blocked_names:
        explanation = (f"{comp.name} is limiting {len(blocked)} downstream competencies "
                       f"(e.g. {', '.join(blocked_names[:3])}). Strengthening it unblocks the most "
                       f"of your remaining path — that makes it higher-leverage than your lowest raw score.")
    else:
        explanation = (f"{comp.name} is your weakest unblocked competency; nothing depends on it, "
                       f"but it is required for the phase-{comp.phase} completion criteria.")
    return {
        "competency_id": best,
        "name": comp.name,
        "phase": comp.phase,
        "dimension": comp.dimension.value,
        "readiness": matrix[best].model_dump(),
        "blocked_competencies": blocked,
        "blocked_count": len(blocked),
        "explanation": explanation,
    }


def dimension_summary(user_id: str = "default", store: Optional[Store] = None,
                      matrix: Optional[Dict[str, CompetencyReadiness]] = None) -> List[Dict]:
    """Aggregate readiness into the dashboard's display dimensions."""
    store = store or get_store()
    bp = load_blueprint(store)
    matrix = matrix or compute_matrix(user_id, store)
    buckets: Dict[str, List[CompetencyReadiness]] = {d: [] for d in DISPLAY_DIMENSIONS}
    for comp in bp.competencies:
        for d in display_dimensions_for(comp):
            buckets[d].append(matrix[comp.id])

    out: List[Dict] = []
    for dim in DISPLAY_DIMENSIONS:
        rs = buckets[dim]
        scored = [combined_score(r) for r in rs]
        nonzero = [s for s in scored if s > 0]
        competent = sum(1 for r in rs if r.readiness_state in COMPETENT_STATES)
        out.append({
            "dimension": dim,
            "avg_score": round(sum(nonzero) / len(nonzero), 2) if nonzero else 0.0,
            "coverage": round(len(nonzero) / len(rs), 2) if rs else 0.0,
            "competent": competent,
            "total": len(rs),
        })
    return out
