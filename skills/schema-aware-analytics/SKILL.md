---
name: schema-aware-analytics
description: Discover, validate, and use data schemas dynamically so analytical applications can adapt to changing datasets.
---

# schema-aware-analytics

## Purpose

Separate analytical semantics from physical database column names and make the data layer resilient to schema variation.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Discover available tables, columns, types, and relevant metadata.
- Map business concepts to physical fields using explicit mappings or controlled heuristics.
- Validate required fields before executing an analytical request.
- Expose only approved fields and dimensions to the query layer and agent.
- Generate parameterized queries from validated schema information.
- Report missing or ambiguous mappings rather than silently guessing.

## Guardrails

- Never infer a field mapping and treat it as certain without validation.
- Keep identifiers separate from user-provided query values.
- Make schema assumptions observable.
- Fail clearly when required data is unavailable.

## Expected outputs

- validated schema map
- available dimensions/measures
- query-ready field contract
- data-gap report

## Example

A dataset changes its physical column name for a timestamp field. The schema layer detects the new field, validates its type, updates the semantic mapping, and keeps the analytical layer unchanged.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
