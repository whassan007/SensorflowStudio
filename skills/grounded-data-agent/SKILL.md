---
name: grounded-data-agent
description: Build agents that answer analytical questions using only retrieved, structured evidence and explicit data-grounding contracts.
---

# grounded-data-agent

## Purpose

Turn natural-language analytical questions into evidence-backed answers without allowing the model to invent facts or directly execute arbitrary SQL.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Parse the user's question into intent, entities, metrics, time range, and constraints.
- Resolve the request into a typed filter/action schema.
- Retrieve data through application-owned, parameterized queries or approved data services.
- Return compact aggregates and relevant evidence to the model.
- Generate the answer only from the supplied evidence.
- Quantify claims when the data supports them and explicitly state insufficiency when it does not.
- Optionally return a structured deep-link or follow-up action.

## Guardrails

- Never fabricate a metric, row, dimension value, or trend.
- Do not let the LLM generate executable SQL when an application-owned query layer can do the job.
- Keep factual grounding separate from presentation language.
- Validate all model-generated structured actions before execution.

## Expected outputs

- grounded answer
- evidence/aggregate payload
- validated action or deep link when applicable

## Example

User: 'Is category A getting worse over the last 12 weeks?' The agent resolves the time window and category, retrieves approved aggregates, compares the relevant metrics, and answers only from those values.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
