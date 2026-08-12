---
name: automated-report-generation
description: Generate structured analytical reports asynchronously from approved data and reusable report sections.
---

# automated-report-generation

## Purpose

Turn recurring analytical workflows into repeatable reports rather than manual copy/paste.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Define report sections and their required data contracts.
- Select the requested sections and analytical scope.
- Retrieve approved data and aggregates.
- Generate each section from grounded evidence.
- Run validation checks on generated content.
- Persist report/job status for asynchronous retrieval.
- Allow sections to be reordered, copied, or exported.

## Guardrails

- Reports must be grounded in approved data.
- Separate data retrieval from narrative generation.
- Expose job state and failures.
- Do not claim conclusions that are absent from the supplied evidence.

## Expected outputs

- report job
- sectioned analytical report
- job status
- Markdown/export representation

## Example

A user selects executive summary, trends, quality, and root-cause sections. The system runs the report asynchronously and makes completed sections available progressively.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
