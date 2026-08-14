"""In-app help: FAQ knowledge base + optional Ollama chat with offline fallback."""

from sensorflow.help.chat import answer_help_question
from sensorflow.help.matcher import match_faq

__all__ = ["answer_help_question", "match_faq"]
