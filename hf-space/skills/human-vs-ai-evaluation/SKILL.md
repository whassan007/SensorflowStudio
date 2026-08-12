---
name: human-vs-ai-evaluation
description: Measure automated decision quality against human decisions and quantify the transition from manual workflows to AI-assisted or automated workflows.
---

# human-vs-ai-evaluation

## Purpose

Determine whether automation is actually replacing work safely and effectively.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Define the human reference process and its limitations.
- Define equivalent automated outputs.
- Measure agreement, disagreement, quality, throughput, and latency.
- Segment results by relevant cohorts.
- Track performance over time.
- Investigate disagreement cases rather than hiding them in a single score.
- Use results to decide where automation should expand, pause, or remain human-reviewed.

## Guardrails

- Human labels are a reference, not automatically ground truth.
- Report disagreement rates with denominators.
- Separate quality improvement from throughput improvement.
- Track changes to the human process that could invalidate comparisons.

## Expected outputs

- human-vs-AI comparison
- agreement metrics
- automation effectiveness
- disagreement analysis
- migration decision signals

## Example

An automated classifier processes a workload previously reviewed manually. Compare its outputs with a controlled human sample, quantify agreement and errors, and track whether throughput gains preserve acceptable quality.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
