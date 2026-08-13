"""BaseAgent: the shared contract for every advisory agent.

Design (mirrors sensorflow.evaluation.copilot):
  * `_analyze(...)` is the deterministic rule-based core — always runs, always
    testable offline.
  * `maybe_llm_rationale(...)` optionally asks a local Ollama endpoint for a
    natural-language rationale; any failure is swallowed and the deterministic
    output stands alone. The LLM can only ever ADD prose; it cannot change
    structured outputs, confidences, or verdicts.
  * Exceptions inside `_analyze` are converted to a failed AgentResult with
    escalation required — an agent crash can therefore never silently pass a
    stage (the pipeline treats it as missing evidence -> INDETERMINATE-ward).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

from sensorflow.agentic.models import AgentEscalation, AgentResult, now_iso

# Local-only by default; kept short so offline runs fail fast.
OLLAMA_ENDPOINTS = [
    {"url": "http://localhost:11434/api/chat", "model": "gemma3:latest"},
]
LLM_TIMEOUT_S = 4.0


class BaseAgent:
    name: str = "base"
    version: str = "1.0.0"
    epistemic_status: str = "DERIVED"

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    # -- to be provided by subclasses -----------------------------------
    def _analyze(self, failure_id: str, **inputs) -> Tuple[Dict[str, Any], float, str,
                                                           AgentEscalation]:
        """Return (output, confidence, confidence_basis, escalation)."""
        raise NotImplementedError

    def _llm_prompt(self, failure_id: str, output: Dict[str, Any],
                    **inputs) -> Optional[str]:
        return None

    # -- uniform entry point ---------------------------------------------
    def run(self, failure_id: str, **inputs) -> AgentResult:
        started = now_iso()
        try:
            output, confidence, basis, escalation = self._analyze(failure_id, **inputs)
            status = "escalated" if escalation.required else "ok"
            result = AgentResult(
                agent=self.name, agent_version=self.version,
                failure_id=failure_id, status=status, output=output,
                confidence=confidence, confidence_basis=basis,
                epistemic_status=self.epistemic_status,
                escalation=escalation, started_at=started)
        except Exception as e:
            result = AgentResult(
                agent=self.name, agent_version=self.version,
                failure_id=failure_id, status="failed",
                output={"error": str(e)}, confidence=0.0,
                confidence_basis="agent raised an exception",
                escalation=AgentEscalation(
                    required=True,
                    reasons=[f"{self.name} failed: {e}"],
                    human_review_triggers=["agent_failure"]),
                failure_handling=("exception captured; stage marked failed; "
                                  "downstream policy treats this as missing "
                                  "evidence (fail-safe)"),
                started_at=started)
            result.finished_at = now_iso()
            return result

        if self.use_llm:
            prompt = self._llm_prompt(failure_id, result.output, **inputs)
            if prompt:
                text, provider = _try_ollama(prompt)
                if text:
                    result.llm_used = True
                    result.llm_provider = provider
                    result.llm_rationale = (
                        "LLM RATIONALE (advisory prose only; structured "
                        "outputs above are deterministic): " + text)
        result.finished_at = now_iso()
        return result


def _try_ollama(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    for ep in OLLAMA_ENDPOINTS:
        try:
            res = httpx.post(ep["url"], json={
                "model": ep["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }, timeout=LLM_TIMEOUT_S)
            if res.status_code == 200:
                text = res.json().get("message", {}).get("content", "")
                if text:
                    return text, ep["url"]
        except Exception:
            continue
    return None, None


def compact(obj: Any, limit: int = 3000) -> str:
    return json.dumps(obj, default=str)[:limit]


def no_escalation() -> AgentEscalation:
    return AgentEscalation()


def escalate(reasons: List[str], triggers: List[str]) -> AgentEscalation:
    return AgentEscalation(required=True, reasons=reasons,
                           human_review_triggers=triggers)
