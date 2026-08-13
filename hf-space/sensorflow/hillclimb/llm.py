"""LLM client wrapper with graceful degradation.

Follows the same pattern as sensorflow/evaluation/copilot.py: try local Ollama
endpoints, and if unreachable (or the response fails schema validation) the
caller falls back to the deterministic rule-based path. LLM output is NEVER
allowed to mutate state as free-form text — callers must validate against a
pydantic schema via `generate_json`.

Set HILLCLIMB_DISABLE_LLM=1 to force offline behavior (used in tests).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

OLLAMA_ENDPOINTS = [
    {"url": "http://localhost:11434/api/chat", "model": "gemma3:latest"},
    {"url": "http://dgx-spark.tail16d8d9.ts.net:11434/api/chat", "model": "gemma4:26b"},
]

T = TypeVar("T", bound=BaseModel)


def llm_enabled() -> bool:
    return os.environ.get("HILLCLIMB_DISABLE_LLM", "") != "1"


def generate_text(prompt: str, timeout: float = 20.0) -> Optional[str]:
    """Return LLM text or None when no endpoint is reachable."""
    if not llm_enabled():
        return None
    for ep in OLLAMA_ENDPOINTS:
        try:
            res = httpx.post(ep["url"], json={
                "model": ep["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }, timeout=timeout)
            if res.status_code == 200:
                text = res.json().get("message", {}).get("content", "")
                if text:
                    return text
        except Exception:
            continue
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from LLM output."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    brace = text.find("{")
    if brace >= 0:
        candidates.append(text[brace:text.rfind("}") + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def generate_json(prompt: str, schema: Type[T], timeout: float = 20.0) -> Optional[T]:
    """Ask the LLM for JSON and validate it against `schema`.

    Returns None on any failure (unreachable, unparseable, schema-invalid) so
    the caller falls back to its deterministic rule-based path.
    """
    text = generate_text(prompt, timeout=timeout)
    if not text:
        return None
    obj = _extract_json(text)
    if obj is None:
        return None
    try:
        return schema(**obj)
    except Exception:
        return None
