---
name: investigative-analytics-workspace
description: Design an AI-assisted workspace that connects conversational analysis, filters, aggregate views, and record-level evidence.
---

# investigative-analytics-workspace

## Purpose

Support an investigation from question to pattern to evidence to follow-up without forcing users to switch tools.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Present an AI command surface alongside structured analytical data.
- Let users apply filters through either UI controls or validated natural-language actions.
- Show aggregate trends and breakdowns.
- Allow drill-down from aggregate results to subsets and individual records.
- Keep the investigation context synchronized across views.
- Support follow-up questions using the current analytical context.

## Guardrails

- Every analytical visualization should have an identifiable data source.
- Do not hide the transition from aggregate evidence to individual records.
- Preserve investigation state when moving between views.
- Make loading, errors, and data freshness visible.

## Expected outputs

- investigation workspace
- linked aggregate/detail views
- persistent analytical context

## Example

A user notices an abnormal trend, selects the affected segment, inspects representative records, and asks the AI to explain the pattern using the same filtered context.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
