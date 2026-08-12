---
name: large-scale-analytics-engine
description: Build interactive analytical systems over large warehouse datasets using bounded queries, pre-aggregation, caching, and controlled refresh.
---

# large-scale-analytics-engine

## Purpose

Keep analytical UX responsive without repeatedly scanning very large fact tables.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Identify fact tables that are too large for per-request interactive aggregation.
- Define the metrics and dimensions required by the UI.
- Create derived or materialized aggregates for those access patterns.
- Use parameterized, bounded queries for interactive requests.
- Cache expensive but reusable results with explicit TTLs.
- Provide explicit refresh behavior for users who need newer data.
- Monitor query latency and revise aggregation strategy when access patterns change.

## Guardrails

- Do not make large raw tables the default backend for every chart.
- Prefer small purpose-built aggregates for repeated interactive queries.
- Do not hide stale-cache behavior.
- Keep expensive warehouse work out of the request path when practical.

## Expected outputs

- interactive query layer
- pre-aggregated datasets
- cache policy
- refresh strategy
- latency safeguards

## Example

A multi-hundred-million-row fact table feeds a weekly trend chart. Build a compact time/dimension aggregate and serve the chart from that aggregate instead of scanning the fact table per request.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
