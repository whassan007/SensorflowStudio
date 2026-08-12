---
name: llm-provider-orchestration
description: Abstract LLM providers behind a common interface with streaming, blocking, fallback, and failure handling.
---

# llm-provider-orchestration

## Purpose

Prevent application logic from becoming dependent on one model provider and keep inference resilient.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Define a provider-neutral chat and streaming contract.
- Register available providers and their capabilities.
- Attempt the preferred inference path.
- Detect timeout, provider, or streaming failures.
- Fall back according to an explicit priority ladder.
- Return provider/fallback metadata for observability.
- Keep prompts and application semantics independent of provider-specific APIs.

## Guardrails

- Never silently change providers without recording that fallback occurred.
- Keep provider credentials out of application code.
- Set explicit timeouts and token limits.
- Do not make provider-specific response formats leak into core application logic.

## Expected outputs

- provider-neutral inference client
- fallback ladder
- streaming/blocking interface
- inference telemetry

## Example

A streaming request fails at the primary provider. The orchestrator retries through the approved blocking or secondary-provider path and records that fallback was used.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
