"""Deterministic FAQ / RAG-lite matcher for help chat (CPU-only)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from sensorflow.help.knowledge import KnowledgeDoc, build_knowledge_index

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]{1,}", re.I)

# Soft boosts when the user names a page-ish keyword
_PAGE_ALIASES: Dict[str, str] = {
    "command center": "command",
    "commandcenter": "command",
    "megaeval": "command",
    "overview": "overview",
    "dataset": "datasets",
    "datasets": "datasets",
    "label generation": "label-generation",
    "autolabel": "label-generation",
    "rare event": "rare-events",
    "raremine": "raremine",
    "costumed": "raremine",
    "quality engine": "quality",
    "regression": "regression",
    "root cause": "rca",
    "rca": "rca",
    "triage": "triage",
    "hitl": "review",
    "human review": "review",
    "training": "training",
    "flywheel": "training",
    "models": "models",
    "evaluation record": "evaluation",
    "audit": "audit",
    "pipeline": "pipeline",
    "hill climb": "hillclimb",
    "hillclimb": "hillclimb",
    "vitis": "vitis",
    "ssam": "ssam",
    "odd": "safety-odd",
    "release gate": "safety-gates",
    "evidence package": "safety-evidence",
    "calibration": "safety-calibration",
    "discrepancy": "safety-discrepancy",
    "scenario db": "safety-scenarios",
    "semantic search": "safety-search",
    "seqeval": "seqeval",
    "sequential": "seqeval",
    "bev": "bevfusion",
    "bevfusion": "bevfusion",
    "retro": "retro",
    "closed loop": "closed-loop-lab",
    "launch readiness": "launch-readiness",
    "studio 2": "studio2",
    "studio2": "studio2",
    "legacy": "legacy",
    "production readiness": "production-readiness",
    "hardening": "production-readiness",
    "rotr": "rotr",
    "right of the road": "rotr",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _infer_page_hint(question: str, page_id: Optional[str]) -> Optional[str]:
    if page_id:
        return page_id
    q = (question or "").lower()
    # Longer aliases first
    for alias in sorted(_PAGE_ALIASES.keys(), key=len, reverse=True):
        if alias in q:
            return _PAGE_ALIASES[alias]
    return None


def score_doc(query_tokens: List[str], doc: KnowledgeDoc, page_hint: Optional[str]) -> float:
    if not query_tokens:
        return 0.0
    hay = tokenize(f"{doc.title} {doc.text}")
    if not hay:
        return 0.0
    hay_set = set(hay)
    hits = sum(1 for t in query_tokens if t in hay_set)
    # Prefer denser overlap on short queries
    score = hits / max(len(set(query_tokens)), 1)
    # Partial credit for substring matches in raw text
    raw = f"{doc.title} {doc.text}".lower()
    for t in query_tokens:
        if len(t) >= 4 and t in raw:
            score += 0.05
    if page_hint and doc.page_id == page_hint:
        score += 0.55
    if page_hint and doc.kind == "page" and doc.page_id == page_hint:
        score += 0.35
    # Soft-penalize long docs when a page is in focus so the page guide wins
    if page_hint and doc.kind == "docs":
        score *= 0.65
    if doc.kind == "faq":
        score += 0.05
        title_tokens = set(tokenize(doc.title))
        if title_tokens:
            title_hits = sum(1 for t in query_tokens if t in title_tokens)
            # Strong boost when the question echoes an FAQ title (e.g. "How do I load a dataset?")
            if title_hits >= max(2, len(title_tokens) // 2):
                score += 0.9
            elif title_hits >= 2:
                score += 0.45
    return score


def match_faq(
    question: str,
    *,
    page_id: Optional[str] = None,
    top_k: int = 3,
    min_score: float = 0.12,
) -> List[Tuple[KnowledgeDoc, float]]:
    """Return top matching knowledge docs with scores."""
    q_tokens = tokenize(question)
    page_hint = _infer_page_hint(question, page_id)
    scored: List[Tuple[KnowledgeDoc, float]] = []
    for doc in build_knowledge_index():
        s = score_doc(q_tokens, doc, page_hint)
        if s >= min_score:
            scored.append((doc, s))
    scored.sort(key=lambda x: (-x[1], x[0].id))
    return scored[:top_k]


def format_fallback_answer(
    question: str,
    matches: List[Tuple[KnowledgeDoc, float]],
    page_id: Optional[str] = None,
) -> Dict[str, object]:
    page_hint = _infer_page_hint(question, page_id)
    sources = [
        {
            "id": doc.id,
            "title": doc.title,
            "kind": doc.kind,
            "score": round(score, 3),
            "page_id": doc.page_id,
        }
        for doc, score in matches
    ]

    if not matches:
        answer = (
            "I could not find a close match in the local help index. "
            "Try asking about a specific page (Command Center, Triage, Human Review, ROTR…) "
            "or open Help → Pages / Tips. Docs live under hf-space/docs/."
        )
        return {"answer": answer, "sources": sources, "page_hint": page_hint, "provider": "faq_offline"}

    primary, _ = matches[0]
    # Prefer the FAQ answer body when available
    body = primary.text
    if primary.kind == "faq" and ". " in body:
        # knowledge stores "question. answer" — keep from first sentence break after keywords
        parts = body.split(". ", 1)
        body = parts[1] if len(parts) > 1 else body

    extras = []
    for doc, _score in matches[1:]:
        extras.append(f"- {doc.title}")

    answer = body
    if extras:
        answer += "\n\nRelated:\n" + "\n".join(extras)
    if page_hint and primary.page_id != page_hint:
        answer += f"\n\n(You are on `{page_hint}` — open About this page for that screen’s guide.)"

    return {
        "answer": answer.strip(),
        "sources": sources,
        "page_hint": page_hint,
        "provider": "faq_offline",
    }
