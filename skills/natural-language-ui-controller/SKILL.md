---
name: natural-language-ui-controller
description: Convert natural-language requests into validated UI navigation, filtering, and interaction commands.
---

# natural-language-ui-controller

## Purpose

Make an analytical application controllable through natural language without coupling the LLM directly to frontend implementation details.

## When to use

Use this skill when the task requires the capability described above. Keep domain-specific terminology out of the implementation unless the target system explicitly requires it.

## Workflow

- Classify the request as navigation/filtering, analysis, or a mixed request.
- Emit a constrained action object containing only supported tabs, filters, and actions.
- Validate the action against current dimension values and UI capabilities.
- Apply the action to application state.
- Refresh the affected data views.
- Expose the resulting state as a shareable URL when possible.

## Guardrails

- Use an explicit action schema.
- Reject unknown dimension values instead of guessing.
- Do not allow arbitrary JavaScript or UI commands from the model.
- Keep navigation and analytical answering as separate modes.
- Preserve existing filters only when the action contract explicitly permits it.

## Expected outputs

- validated UI action
- updated UI state
- optional shareable deep link

## Example

User: 'Show failed cases in the last eight weeks.' The controller selects the appropriate view and applies a validated status and time-window filter.

## Implementation principles

- Prefer boring, explicit interfaces over implicit agent behavior.
- Keep domain semantics configurable rather than embedded in the skill.
- Validate model-generated structured data before execution.
- Make data provenance, uncertainty, and failure states observable.
- Separate retrieval, reasoning, execution, and presentation.
