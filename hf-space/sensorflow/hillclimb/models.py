"""Data model + JSON persistence for the Hill Climbing EM platform.

Everything persists as JSON documents under runs/hillclimb/ (one file per
collection). The store is intentionally simple: this is a single-user,
low-write-volume product surface, and file-backed JSON keeps the whole thing
inspectable and dependency-free.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------- enums


class Dimension(str, Enum):
    KNOWLEDGE = "Knowledge"
    TECHNICAL_REASONING = "Technical Reasoning"
    LEADERSHIP = "Leadership"
    EXECUTION = "Execution"


class ReadinessState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    LEARNING = "LEARNING"
    PRACTICING = "PRACTICING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    COMPETENT = "COMPETENT"
    STRONG = "STRONG"
    INTERVIEW_READY = "INTERVIEW_READY"


class JourneyState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    DIAGNOSTIC = "DIAGNOSTIC"
    LEARNING = "LEARNING"
    PRACTICE = "PRACTICE"
    ASSESSMENT = "ASSESSMENT"
    REMEDIATION = "REMEDIATION"
    REASSESS = "REASSESS"


# --------------------------------------------------------------------- models


class UserProfile(BaseModel):
    user_id: str = "default"
    name: str = "You"
    target_role: str = "Engineering Manager, ML/Perception"
    experience_years: float = 8.0
    focus_note: str = ""
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class EvaluationResult(BaseModel):
    """Structured AI-evaluation contract. Free-form LLM text NEVER mutates
    state directly — it must parse into this schema or the rule-based
    evaluator's result is used instead."""

    competency: str
    score: int = Field(ge=1, le=5, description="1=Novice .. 5=Expert")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list, description="Quotes from the user's own statements")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    misconceptions: List[str] = Field(default_factory=list)
    recommended_action: str = ""
    follow_up_question: str = ""
    evaluator: str = "rule_based"  # rule_based | llm


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    user_id: str = "default"
    competency_ids: List[str] = Field(default_factory=list)
    artifact_type: str = "exercise_attempt"  # exercise_attempt | star_story | design_submission | simulation_debrief | interview_transcript | diagnostic
    source: str = ""
    summary: str = ""
    quotes: List[str] = Field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    timestamp: str = Field(default_factory=now_iso)
    payload: Dict = Field(default_factory=dict)


class Attempt(BaseModel):
    attempt_id: str = Field(default_factory=lambda: new_id("att"))
    user_id: str = "default"
    exercise_id: str = ""
    competency_id: str = ""
    kind: str = "exercise"  # exercise | diagnostic | interview | design | simulation
    responses: Dict = Field(default_factory=dict)
    evaluation: Optional[EvaluationResult] = None
    timestamp: str = Field(default_factory=now_iso)


class CompetencyReadiness(BaseModel):
    competency_id: str
    knowledge_score: float = 0.0
    application_score: float = 0.0
    evidence_score: float = 0.0
    readiness_state: ReadinessState = ReadinessState.NOT_STARTED
    evidence_ids: List[str] = Field(default_factory=list)
    last_updated: str = Field(default_factory=now_iso)


class Journey(BaseModel):
    user_id: str = "default"
    state: JourneyState = JourneyState.NOT_STARTED
    current_phase: int = 1
    current_competency: Optional[str] = None
    remediation_target: Optional[str] = None
    history: List[Dict] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------- store


class Store:
    """JSON-file collection store rooted at runs/hillclimb/ by default."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or os.environ.get("HILLCLIMB_DATA_DIR", "runs/hillclimb"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Dict]] = {}

    def _path(self, collection: str) -> Path:
        return self.root / f"{collection}.json"

    def _load(self, collection: str) -> Dict[str, Dict]:
        if collection in self._cache:
            return self._cache[collection]
        path = self._path(collection)
        data: Dict[str, Dict] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
        self._cache[collection] = data
        return data

    def _flush(self, collection: str) -> None:
        path = self._path(collection)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache.get(collection, {}), indent=2, default=str))
        tmp.replace(path)

    def put(self, collection: str, key: str, value: Dict | BaseModel) -> None:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        with self._lock:
            data = self._load(collection)
            data[key] = value
            self._flush(collection)

    def get(self, collection: str, key: str) -> Optional[Dict]:
        with self._lock:
            return self._load(collection).get(key)

    def delete(self, collection: str, key: str) -> bool:
        with self._lock:
            data = self._load(collection)
            if key in data:
                del data[key]
                self._flush(collection)
                return True
            return False

    def all(self, collection: str) -> List[Dict]:
        with self._lock:
            return list(self._load(collection).values())

    def where(self, collection: str, **eq) -> List[Dict]:
        return [r for r in self.all(collection) if all(r.get(k) == v for k, v in eq.items())]


_store: Optional[Store] = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def reset_store(root: Optional[Path] = None) -> Store:
    """Replace the singleton (used by tests to isolate state)."""
    global _store
    _store = Store(root)
    return _store
