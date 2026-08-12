---
name: ml-observability
description: Monitor deployed ML systems through quality, drift, agreement, sampling, and operational metrics.
---

# ml-observability

## Purpose

Detect degradation and understand whether an ML system remains reliable as data and operating conditions change.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Define model-quality metrics appropriate to the task.
- Track metrics across time and relevant cohorts.
- Compare predictions with trusted labels or reference sets.
- Monitor input/class distribution and sampling changes.
- Track drift and agreement metrics.
- Maintain a stable evaluation or golden set where applicable.
- Surface anomalies and data-quality gaps as first-class signals.

## Guardrails

- Do not treat a single metric as model health.
- Distinguish model degradation from label/data-pipeline changes.
- Record denominators and sample sizes.
- Flag insufficient history rather than manufacturing a trend.

## Expected outputs

- model health dashboard
- quality trends
- drift indicators
- agreement metrics
- evaluation-set monitoring

## Example

A model's headline metric is stable, but class mix and sampling distributions have shifted. The observability layer surfaces both signals so the team does not mistake metric stability for unchanged operating conditions.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
