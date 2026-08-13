"""Agentic Launch Readiness & Misclassification Triage Subsystem.

Motivating failure: a pedestrian misclassified as a construction cone at an
observed ~0.01% rate. Before treating that number as real, the system must
establish denominator, sample size, confidence interval, statistical
significance, safety criticality and novelty.

Core architectural rule (enforced structurally, see policy.py and
agents/base.py): AI agents discover, analyze, summarize and RECOMMEND;
deterministic services own metrics, statistics, policy enforcement, launch
gates and audit. No LLM output ever directly authorizes a launch decision —
final authorization is deterministic policy + recorded human approval.

NOTE ON PROVENANCE: the referenced workflow image was not provided to the
implementing agent. The five-layer methodology (FAILURE DETECTION -> EVIDENCE
AGGREGATION -> FAILURE ANALYSIS -> LAUNCH DECISION -> LEARNING FLYWHEEL) was
reconstructed from the written specification and is marked as such.

Module map:
    models.py         shared pydantic schemas (FailureEvent, stages, enums)
    store.py          JSON store under runs/agentic/ + append-only audit log
    data.py           deterministic synthetic populations (scenes + rate log)
    evidence.py       Failure Evidence Graph (typed nodes, evidence statuses)
    snippets.py       reproducible failure snippet packages
    concentration.py  uniform-vs-concentrated distribution analysis
    policy.py         deterministic, hash-versioned stop-ship policy engine
    scorecard.py      AgenticSafetyScorecard (leadership retrospective)
    review.py         human-review decisions, mandatory triggers, governance
    flywheel.py       evaluation-suite creation + contamination guard
    pipeline.py       staged five-layer orchestrator (explicit state)
    worked_example.py deterministic pedestrian->cone walkthrough
    agents/           the eight agents (advisory only, LLM optional)
    api.py            REST surface under /api/agentic
"""

METHODOLOGY_PROVENANCE = (
    "Reconstructed from the written specification; the referenced workflow "
    "image was not provided to the implementing agent."
)
