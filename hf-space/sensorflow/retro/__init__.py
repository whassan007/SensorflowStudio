"""Agentic Retrospective Safety Analyzer (retro).

Evidence-driven analysis of perception failures with a traceable chain:

    RAW FAILURE -> OBSERVED EVIDENCE -> DERIVED METRICS -> RETRIEVED
    ENGINEERING EVIDENCE -> AGENT HYPOTHESIS -> BEHAVIORAL ANALYSIS ->
    SAFETY POLICY -> RETROSPECTIVE SCORECARD -> HUMAN DECISION

Core rule: the LLM interprets, correlates, and hypothesizes; deterministic
code computes every metric and owns the safety boundary. Launch
determinations only ever pass through the deterministic policy gate
(sensorflow.retro.policy); INSUFFICIENT_EVIDENCE never becomes PASS.

Package layout:
    inference/   pluggable LLM backends (vllm | ollama | mock) + env/compat
    rag/         safety-case retrieval (synthetic, clearly-labeled corpus)
    tools/       MCP-style audited tool registry
    agent/       retrospective orchestrator (Log -> Evidence -> RAG -> Synth)
    scorecard.py tier-tagged evidence + RetrospectiveScorecard model
    policy.py    deterministic severity framework + launch gate
    api.py       FastAPI router mounted at /api/retro
"""

AGENT_VERSION = "retro-agent/1.0.0"
