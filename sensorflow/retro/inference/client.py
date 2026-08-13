"""Pluggable inference client: vllm | ollama | mock backends.

All backends implement the same contract:
    health()  -> BackendStatus (never raises)
    generate(prompt, system, max_tokens) -> InferenceResult with token counts
                                            and wall-clock latency
    validate() -> deterministic test-prompt round trip

The mock backend is the default for tests: fully deterministic, no network.
The ollama backend follows the platform's existing local pattern
(sensorflow/evaluation/copilot.py). The vllm backend targets any
OpenAI-compatible endpoint and is only exercisable on CUDA/ROCm hosts —
see sensorflow/retro/inference/vllm_server/.
"""

from __future__ import annotations

import hashlib
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field

VALIDATION_PROMPT = "Respond with exactly the token RETRO-OK and nothing else."
VALIDATION_EXPECT = "RETRO-OK"

DEFAULT_OLLAMA_URL = os.environ.get("RETRO_OLLAMA_URL", "http://localhost:11434")
DEFAULT_VLLM_URL = os.environ.get("RETRO_VLLM_URL", "http://localhost:8001/v1")


class InferenceResult(BaseModel):
    backend: str
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    notes: str = ""


class BackendStatus(BaseModel):
    backend: str
    available: bool
    endpoint: Optional[str] = None
    model: Optional[str] = None
    detail: str = ""
    health_latency_ms: Optional[float] = None


class ValidationResult(BaseModel):
    backend: str
    passed: bool
    expected: str
    got: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    detail: str = ""


class InferenceBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def health(self) -> BackendStatus: ...

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024) -> InferenceResult: ...

    def validate(self) -> ValidationResult:
        """Deterministic test-prompt inference validation with latency."""
        t0 = time.perf_counter()
        try:
            res = self.generate(VALIDATION_PROMPT, max_tokens=16)
        except Exception as exc:
            return ValidationResult(backend=self.name, passed=False,
                                    expected=VALIDATION_EXPECT, got="",
                                    latency_ms=(time.perf_counter() - t0) * 1000,
                                    prompt_tokens=0, completion_tokens=0,
                                    detail=f"generate failed: {exc}")
        passed = VALIDATION_EXPECT in res.text
        return ValidationResult(backend=self.name, passed=passed,
                                expected=VALIDATION_EXPECT, got=res.text.strip()[:200],
                                latency_ms=res.latency_ms,
                                prompt_tokens=res.prompt_tokens,
                                completion_tokens=res.completion_tokens,
                                detail="" if passed else
                                "model reply did not contain the expected token")


def _approx_tokens(text: str) -> int:
    """Deterministic whitespace-ish token approximation (mock backend only)."""
    return max(1, len(text.split())) if text else 0


class MockBackend(InferenceBackend):
    """Deterministic, offline. Replies are pure functions of the prompt.

    The retrospective agent does NOT parse mock replies for analysis content —
    in mock mode the analysis itself is produced by the deterministic scripted
    path (sensorflow/retro/agent). The mock backend exists to satisfy the
    client contract (health/generate/validate) without any model.
    """

    name = "mock"

    def health(self) -> BackendStatus:
        return BackendStatus(backend="mock", available=True, endpoint=None,
                             model="deterministic-mock",
                             detail="always available; replies are deterministic "
                                    "functions of the prompt", health_latency_ms=0.0)

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024) -> InferenceResult:
        t0 = time.perf_counter()
        if VALIDATION_EXPECT in prompt or "RETRO-OK" in prompt:
            text = VALIDATION_EXPECT
        else:
            digest = hashlib.sha256(((system or "") + "\n" + prompt).encode()).hexdigest()[:12]
            text = f"[mock-response {digest}] Deterministic placeholder reply."
        return InferenceResult(
            backend="mock", model="deterministic-mock", text=text,
            prompt_tokens=_approx_tokens((system or "") + " " + prompt),
            completion_tokens=_approx_tokens(text),
            latency_ms=(time.perf_counter() - t0) * 1000,
            notes="deterministic mock; token counts are whitespace approximations")


class OllamaBackend(InferenceBackend):
    """Local Ollama server (Metal/CPU on this machine — NOT vLLM numbers)."""

    name = "ollama"

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL,
                 model: Optional[str] = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("RETRO_OLLAMA_MODEL", "")
        self.timeout = timeout

    def _pick_model(self) -> Optional[str]:
        if self.model:
            return self.model
        try:
            res = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            models = [m.get("name") for m in res.json().get("models", [])]
            self.model = models[0] if models else None
        except Exception:
            self.model = None
        return self.model

    def health(self) -> BackendStatus:
        t0 = time.perf_counter()
        try:
            res = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            ms = (time.perf_counter() - t0) * 1000
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                return BackendStatus(backend="ollama", available=True,
                                     endpoint=self.base_url,
                                     model=self.model or (models[0] if models else None),
                                     detail=f"{len(models)} model(s) installed",
                                     health_latency_ms=round(ms, 1))
            return BackendStatus(backend="ollama", available=False,
                                 endpoint=self.base_url,
                                 detail=f"HTTP {res.status_code} from /api/tags")
        except Exception as exc:
            return BackendStatus(backend="ollama", available=False,
                                 endpoint=self.base_url,
                                 detail=f"unreachable: {exc.__class__.__name__}: {exc}")

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024) -> InferenceResult:
        model = self._pick_model()
        if not model:
            raise RuntimeError(f"no Ollama model available at {self.base_url}")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        res = httpx.post(f"{self.base_url}/api/chat",
                         json={"model": model, "messages": messages, "stream": False,
                               "options": {"num_predict": max_tokens}},
                         timeout=self.timeout)
        latency_ms = (time.perf_counter() - t0) * 1000
        res.raise_for_status()
        body = res.json()
        return InferenceResult(
            backend="ollama", model=model,
            text=body.get("message", {}).get("content", ""),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            latency_ms=round(latency_ms, 1),
            notes="Ollama on local CPU/Metal — these are NOT vLLM/GPU numbers")


class VLLMBackend(InferenceBackend):
    """OpenAI-compatible vLLM endpoint. Only runnable against a CUDA/ROCm host
    (see vllm_server/); on this macOS machine health() honestly reports it
    unavailable unless a remote endpoint is configured."""

    name = "vllm"

    def __init__(self, base_url: str = DEFAULT_VLLM_URL,
                 model: Optional[str] = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("RETRO_VLLM_MODEL", "")
        self.timeout = timeout

    def health(self) -> BackendStatus:
        t0 = time.perf_counter()
        try:
            res = httpx.get(f"{self.base_url}/models", timeout=3.0)
            ms = (time.perf_counter() - t0) * 1000
            if res.status_code == 200:
                models = [m.get("id") for m in res.json().get("data", [])]
                return BackendStatus(backend="vllm", available=True,
                                     endpoint=self.base_url,
                                     model=self.model or (models[0] if models else None),
                                     detail=f"OpenAI-compatible server, models: {models}",
                                     health_latency_ms=round(ms, 1))
            return BackendStatus(backend="vllm", available=False,
                                 endpoint=self.base_url,
                                 detail=f"HTTP {res.status_code} from /models")
        except Exception as exc:
            return BackendStatus(backend="vllm", available=False,
                                 endpoint=self.base_url,
                                 detail=f"unreachable ({exc.__class__.__name__}); vLLM "
                                        "cannot run on this macOS machine — point "
                                        "RETRO_VLLM_URL at a CUDA/ROCm host")

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024) -> InferenceResult:
        model = self.model
        if not model:
            res = httpx.get(f"{self.base_url}/models", timeout=5.0)
            data = res.json().get("data", [])
            if not data:
                raise RuntimeError(f"no models served at {self.base_url}")
            model = data[0]["id"]
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        res = httpx.post(f"{self.base_url}/chat/completions",
                         json={"model": model, "messages": messages,
                               "max_tokens": max_tokens, "temperature": 0.0},
                         timeout=self.timeout)
        latency_ms = (time.perf_counter() - t0) * 1000
        res.raise_for_status()
        body = res.json()
        usage = body.get("usage", {})
        return InferenceResult(
            backend="vllm", model=model,
            text=body["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=round(latency_ms, 1))


BACKENDS = {"mock": MockBackend, "ollama": OllamaBackend, "vllm": VLLMBackend}


def get_backend(name: str = "mock", **kwargs) -> InferenceBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)


def all_backend_statuses() -> List[BackendStatus]:
    return [get_backend(n).health() for n in ("mock", "ollama", "vllm")]
