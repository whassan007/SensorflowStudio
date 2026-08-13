"""FastAPI router for in-app help (/api/help/*)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from sensorflow.help.chat import answer_help_question
from sensorflow.help.knowledge import list_page_guides

router = APIRouter(prefix="/api/help", tags=["help"])


class HelpChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    page_id: Optional[str] = None


class HelpSource(BaseModel):
    id: str
    title: str
    kind: str
    score: Optional[float] = None
    page_id: Optional[str] = None


class HelpChatResponse(BaseModel):
    answer: str
    sources: List[HelpSource] = Field(default_factory=list)
    provider: Optional[str] = None
    page_hint: Optional[str] = None


@router.post("/chat", response_model=HelpChatResponse)
def help_chat(req: HelpChatRequest) -> Dict[str, Any]:
    """Answer a Studio help question (Ollama optional; FAQ matcher always works)."""
    return answer_help_question(req.question, page_id=req.page_id)


@router.get("/guides")
def help_guides() -> Dict[str, Any]:
    """List compact page guides for the Help menu / chatbot."""
    return {"guides": list_page_guides()}
