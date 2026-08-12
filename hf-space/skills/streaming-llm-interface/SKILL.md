---
name: streaming-llm-interface
description: Build responsive LLM interfaces using incremental server-sent responses and progressive rendering.
---

# streaming-llm-interface

## Purpose

Reduce perceived latency and expose model progress while preserving a structured final response.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Open an SSE or equivalent streaming channel.
- Render valid partial text as it arrives.
- Keep transient stream state separate from the final structured response.
- Detect completion, timeout, cancellation, and malformed events.
- Render the final answer and associated actions once available.
- Provide a fallback path when streaming fails.

## Guardrails

- Never assume every streamed event is user-displayable text.
- Handle disconnects and cancellation.
- Do not lose the final structured action when the text stream is incomplete.
- Make thinking/loading/error states explicit.

## Expected outputs

- streaming endpoint
- incremental UI renderer
- final response contract
- fallback behavior

## Example

The user sees the answer progressively while the backend continues generating the final structured result and action metadata.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
