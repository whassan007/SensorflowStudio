"""Retrospective agent: Log Agent -> Evidence Analyzer -> RAG Agent ->
Safety Synthesizer, with ReAct-style audited tool use.

Orchestration note: pydantic-ai installed cleanly in this environment, but
the orchestrator deliberately uses the internal typed loop (typed pydantic
stage contracts + the audited ToolRegistry) as the execution engine for BOTH
backends, because the deterministic mock path and the per-call audit
requirements must hold with no model present. The LLM (Ollama, when
reachable) contributes hypotheses, severity proposals, and follow-up
retrieval queries — never metrics or determinations.
"""
