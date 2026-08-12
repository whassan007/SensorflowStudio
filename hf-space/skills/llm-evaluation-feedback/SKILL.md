---
name: llm-evaluation-feedback
description: Capture, aggregate, and use human feedback on LLM outputs to measure production quality.
---

# llm-evaluation-feedback

## Purpose

Turn subjective user feedback into an observable evaluation loop.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Attach feedback to a specific model response and context.
- Capture positive, negative, or structured review signals.
- Aggregate feedback by output type, model, prompt family, and time period where appropriate.
- Monitor quality trends and recurring failure patterns.
- Use findings to improve prompts, retrieval, schemas, or provider selection.
- Retain representative examples for regression testing.

## Guardrails

- Feedback must be traceable to the response being evaluated.
- Do not equate raw thumbs-up rate with complete model quality.
- Separate user preference from factual correctness when possible.
- Protect sensitive response/context data.

## Expected outputs

- feedback events
- quality dashboards
- failure taxonomy
- evaluation dataset
- regression cases

## Example

Negative feedback clusters around one response type. The team samples those cases, identifies a grounding failure, fixes the retrieval contract, and adds the cases to regression evaluation.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
