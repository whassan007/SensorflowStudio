# Next-Gen Evaluation Platform — Rollout Plan (Phases 0-6)

Each phase lists what is ALREADY BUILT in this repository versus what
remains for a production deployment, with dependencies, risks, exit criteria
and rollback. The rollback story is uniform because of design-for-deletion:
`sensorflow/nextgen` is imported by `app_backend.py` in exactly one place;
removing that include and the `runs/nextgen/` directory removes the entire
platform without touching any other package.

## Phase 0 — Foundations: lineage + safety-informed metrics

* Built: `nextgen/lineage.py` (component versions, INVALID-on-missing
  policy), `nextgen/safety_metrics.py` (parameterized SCR region, risk
  weights, divergence demo at `GET /api/nextgen/metrics/divergence-demo`).
* Remaining: wire SCR into the existing release gates (`safety.gates`)
  as a first-class check; agree region parameters with the safety team.
* Dependencies: none beyond existing packages. Risks: mis-parameterized
  region (mitigation: parameters are explicit in every report).
* Exit criteria: every evaluation report shows recall AND SCR; a run
  without lineage is refused for launch (tested:
  `test_regression_lineage.py`).
* Rollback: stop consulting SCR; open-loop gates unchanged.

## Phase 1 — Compute dedup

* Built: `nextgen/cache.py` (content-addressed IR cache, version-bump =
  miss), `nextgen/compute.py` (measured benchmark,
  `POST /api/nextgen/compute/benchmark`, `GET /api/nextgen/compute/report`).
* Remaining: point the cache at the real preprocessing/backbone stages of
  the production stack; size the LRU cap against real artifact sizes.
* Dependencies: Phase 0 (versions in lineage records are the cache keys).
* Risks: semantic change without a version bump serves stale features —
  mitigated by making versions part of lineage, audited per run.
* Exit criteria: measured hit rate > 90% on repeat suites; measured savings
  ratio published per run. Rollback: bypass the cache (compute function is
  called directly); nothing else changes.

## Phase 2 — Launch gauntlet + scheduler

* Built: `nextgen/scheduler.py` (priority strata, budget, seqeval-delegated
  anytime-valid early stopping, catastrophic halt, related-strata promotion,
  critical-pass expansion; demonstrated on a 100k-unit synthetic gauntlet
  with real timings), `nextgen/regression.py` (statistical vs safety
  significance, kept separate).
* Remaining: replace synthetic strata with real scenario suites (safety
  suites from `safety.scenario_db`, historical regressions from
  `evaluation.regression`, ODD strata from `safety.odd`).
* Dependencies: Phases 0-1. Risks: mis-prioritized strata starve nominal
  coverage — mitigated by the budget floor and expansion policy.
* Exit criteria: a planted catastrophic safety regression halts the
  candidate within the safety-critical stratum (tested); budget respected.
* Rollback: run strata exhaustively (scheduler off) — costs compute only.

## Phase 3 — Counterfactual generation + validity gating

* Built: `nextgen/counterfactual.py` (15 deterministic transformations with
  provenance + seeds), `nextgen/worldmodel.py` (transformer interface +
  external-world-model stub), `nextgen/validity.py` (5 checks, 3 scores,
  weight caps), endpoints `POST /api/nextgen/counterfactuals/generate`,
  `POST /api/nextgen/counterfactuals/{id}/validate`.
* Remaining: recipe curation per ODD; calibration of realism scoring
  against real fleet data (today's reference is the synthetic scene pool,
  documented in `validity.check_distribution`).
* Dependencies: Phase 2 (gauntlet consumes gated scenarios with weights).
* Risks: invalid scenarios entering suites — mitigated twice (per-scenario
  cap + suite share cap, both tested); provenance loss — mitigated by
  labels carried through every report (tested).
* Exit criteria: gate rejects planted implausible scenarios (tested);
  low-fidelity suite share <= 25% by construction.
* Rollback: evaluation suites simply stop including COUNTERFACTUAL-labeled
  scenarios (the label makes exclusion a one-line filter).

## Phase 4 — Closed-loop behavioral evaluation

* Built: `nextgen/closedloop.py` (deterministic IDM + lane-keep stack,
  seeded perception failure model, behavioral metrics reusing
  `safety.ssam_ext`), `POST /api/nextgen/simulation/replay`.
* Remaining: swap the simplified planner for a shadow build of the real
  planner; validate behavioral metric distributions against drive logs.
* Dependencies: Phase 3 (scenarios), Phase 0 (labels).
* Risks: sim-to-real gap read as truth — mitigated by SIMULATED/
  COUNTERFACTUAL labels on every assessment and open-loop metrics attached
  to every report. Exit criteria: planted late-detection produces shorter
  TTC / smaller margin (tested); metrics reproducible per seed.
* Rollback: behavioral metrics become advisory-only (they are a separate
  section of the report; removing them removes no open-loop evidence).

## Phase 5 — Causal counterfactual replay

* Built: `nextgen/causal.py` (actual-vs-corrected replay, stepwise causal
  chain, METRIC_ONLY vs BEHAVIORALLY_CONSEQUENTIAL),
  `POST /api/nextgen/causal/replay`.
* Remaining: calibrate verdict thresholds against triage history; feed
  verdicts into `sensorflow/agentic` scorecards via
  `ScorecardBehavioralExtension`.
* Dependencies: Phase 4. Risks: verdict over-trust — mitigated: verdict
  never suppresses an open-loop regression, it only prioritizes.
* Exit criteria: cosmetic class flip => METRIC_ONLY and missed crossing
  pedestrian => BEHAVIORALLY_CONSEQUENTIAL on planted cases (tested).
* Rollback: verdicts advisory-only; triage falls back to metric deltas.

## Phase 6 — External world model integration (optional, gated)

* Built: the integration contract (`worldmodel.ExternalWorldModelAdapter`)
  — full ground truth out, determinism, GENERATED provenance labels, no
  validity-gate exemption. Deliberately NOT implemented.
* Remaining: everything behind the adapter; see
  `nextgen-worldmodel-generative-comparison.md` for the build/buy analysis.
* Dependencies: Phases 3-5 (the gate and replay are what make generated
  content safe to use). Risks: fidelity-validation burden dominates —
  the gate becomes the bottleneck and must be hardened first.
* Exit criteria: generated scenarios pass the same gate at >= the internal
  transformer's acceptance rate on matched recipes; lineage reproducibility
  holds (same seed, same scenario).
* Rollback: unplug the adapter; the deterministic transformer remains the
  only live implementation (today's state).
