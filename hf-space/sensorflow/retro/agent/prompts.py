"""Prompts for the LLM-backed synthesis stage (Ollama backend)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You are the Safety Synthesizer stage of a retrospective analyzer for an "
    "autonomous-vehicle perception platform. You interpret evidence and form "
    "hypotheses. You NEVER compute metrics, NEVER invent telemetry, and NEVER "
    "make launch determinations — deterministic code owns those. Separate "
    "fact from inference rigorously: anything not present in the supplied "
    "evidence must be listed as missing_evidence, not asserted."
)


def synthesis_prompt(failure_type: str, observed: List[Dict[str, Any]],
                     derived: List[Dict[str, Any]],
                     retrieved: List[Dict[str, Any]],
                     missing_fields: List[str]) -> str:
    return f"""FAILURE TYPE (deterministically derived): {failure_type}

TIER1 OBSERVED FACTS (from the artifact):
{json.dumps(observed, indent=1)[:3000]}

TIER2 DERIVED METRICS (deterministic calculators):
{json.dumps(derived, indent=1)[:2500]}

TIER3 RETRIEVED REQUIREMENTS (safety-case retrieval; synthetic-labeled corpus):
{json.dumps([{k: r[k] for k in ('doc_id', 'section', 'retrieved_text')}
             for r in retrieved], indent=1)[:2500]}

MISSING ARTIFACT FIELDS (must stay UNKNOWN): {missing_fields}

Respond with ONLY a JSON object:
{{
  "hypotheses": [
    {{"hypothesis": "<causal hypothesis, clearly inference>",
      "confidence": <0..1>,
      "supporting_evidence_keys": ["<keys from the observed/derived items>"],
      "missing_evidence": ["<what you would need to confirm>"]}}
  ],
  "proposed_severity": "BENIGN|DISRUPTIVE|CRITICAL|FATAL",
  "behavioral_consequence": "<one sentence, grounded in the derived metrics>",
  "additional_queries": ["<up to 2 follow-up safety-case retrieval queries>"]
}}
Base every claim on the supplied evidence. 2-4 hypotheses."""
