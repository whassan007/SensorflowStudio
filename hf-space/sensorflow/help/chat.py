"""Help chat: try optional Ollama, always fall back to FAQ matcher (CPU-safe)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from sensorflow.help.matcher import format_fallback_answer, match_faq


def _llm_endpoints() -> List[Dict[str, str]]:
    endpoints: List[Dict[str, str]] = []
    url = os.environ.get("SENSORFLOW_LLM_URL")
    if url:
        endpoints.append(
            {
                "url": url if url.endswith("/api/chat") else url.rstrip("/") + "/api/chat",
                "model": os.environ.get("SENSORFLOW_LLM_MODEL", "gemma3:latest"),
            }
        )
    # Local Ollama last — optional enrichment only
    endpoints.append({"url": "http://localhost:11434/api/chat", "model": "gemma3:latest"})
    return endpoints


def _build_prompt(question: str, context_blocks: List[str], page_id: Optional[str]) -> str:
    ctx = "\n\n".join(context_blocks[:4])
    page_line = f"Current UI page id: {page_id}\n" if page_id else ""
    return (
        "You are the Sensorflow Studio in-app help assistant. Answer briefly and accurately "
        "using ONLY the context below. If the context is insufficient, say what the user "
        "should open in the Help menu. Do not invent APIs or metrics.\n\n"
        f"{page_line}"
        f"CONTEXT:\n{ctx}\n\n"
        f"USER QUESTION: {question}\n\n"
        "Answer in plain language (2–6 short sentences). Mention relevant page names when useful."
    )


def answer_help_question(question: str, page_id: Optional[str] = None) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {
            "answer": "Ask a question about a Sensorflow Studio page or feature.",
            "sources": [],
            "provider": "faq_offline",
            "page_hint": page_id,
        }

    matches = match_faq(q, page_id=page_id, top_k=4)
    offline = format_fallback_answer(q, matches, page_id=page_id)

    context_blocks = [f"[{doc.kind}] {doc.title}: {doc.text[:700]}" for doc, _ in matches]
    if not context_blocks:
        return offline

    prompt = _build_prompt(q, context_blocks, page_id)
    for ep in _llm_endpoints():
        try:
            res = httpx.post(
                ep["url"],
                json={
                    "model": ep["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=8.0,
            )
            if res.status_code == 200:
                text = (res.json().get("message") or {}).get("content") or ""
                if text.strip():
                    return {
                        "answer": text.strip(),
                        "sources": offline["sources"],
                        "provider": ep["url"],
                        "page_hint": offline.get("page_hint"),
                    }
        except Exception:
            continue

    return offline
