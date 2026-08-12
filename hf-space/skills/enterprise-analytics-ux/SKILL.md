---
name: enterprise-analytics-ux
description: Design enterprise analytical interfaces that make complex data, AI interactions, loading states, filters, and investigation context usable and reproducible.
---

# enterprise-analytics-ux

## Purpose

Provide an operational UX that remains understandable under large datasets, long-running queries, and AI uncertainty.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Establish a consistent application shell and navigation model.
- Use persistent, visible filters and shareable state.
- Provide skeletons and explicit loading/error states.
- Use progressive disclosure for dense analytical information.
- Make evidence and data provenance visible.
- Support keyboard and assistive-technology interaction.
- Keep dark/light themes and design tokens consistent.
- Deep-link analytical views so investigations are reproducible.

## Guardrails

- Do not hide important system state behind animation.
- Every major analytical surface needs clear loading, empty, error, and stale-data states.
- Use accessible semantics for dialogs, controls, tables, and navigation.
- Keep high-density dashboards visually hierarchical.
- Do not sacrifice evidence visibility for visual polish.

## Expected outputs

- enterprise analytics UX system
- design patterns
- accessibility checklist
- state/deep-link model
- loading/error conventions

## Example

A dense analytical dashboard combines persistent filters, KPI summaries, drill-down tables, progressive loading, keyboard-accessible detail views, and URL-addressable investigation state.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
