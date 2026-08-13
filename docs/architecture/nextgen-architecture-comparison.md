# Next-Gen Evaluation Platform — Three-Way Architecture Comparison

Decision doc for how far to take the evaluation platform beyond open-loop
metrics. Every claim below is grounded in what actually exists in this
repository today.

## What exists today (the substrate all three options build on)

| Capability | Where it lives | State |
|---|---|---|
| Scenario/sensor/perception substrate | `sensorflow/bevfusion` (scenes, sensors, engines, evaluate) | complete |
| Anytime-valid sequential statistics | `sensorflow/seqeval` (sequential, units, paired, planner) | complete |
| Surrogate safety math (TTC/DRAC/PET/CSI) + release gates + ODD | `sensorflow/safety` (ssam_ext, gates, odd) | complete |
| Aggregate-scale evaluation, distribution shift, metric cube | `sensorflow/megaeval` (cube, analysis, runs) | complete |
| Open-loop label evaluation, regression tracking, triage | `sensorflow/evaluation` | complete |
| Divergence primitives (PSI/JS), RCA forensics | `sensorflow/rca` | complete |
| Agentic misclassification triage (Feature 1) | `sensorflow/agentic` | built concurrently |
| Counterfactuals, validity gate, closed loop, SCR, dedup, gauntlet | `sensorflow/nextgen` (this work) | built |

## The three architectures

**A — Minimal extension.** Keep the platform open-loop. Add safety-informed
metrics (`nextgen/safety_metrics.py`), the launch gauntlet scheduler
(`nextgen/scheduler.py` + `seqeval`), and compute dedup (`nextgen/cache.py`,
`nextgen/compute.py`). No counterfactuals, no closed loop. Roughly: today's
platform + Features 4-5 only.

**B — Agentic evaluation platform.** A plus the agentic triage layer
(`sensorflow/agentic`) and counterfactual generation WITH validity gating
(`nextgen/counterfactual.py`, `nextgen/validity.py`), but evaluation stays
open-loop: counterfactuals are scored with detection metrics, not behavior.
AI agents interpret evidence; deterministic policy decides.

**C — Full closed-loop platform.** B plus closed-loop behavioral evaluation
(`nextgen/closedloop.py`), causal counterfactual replay (`nextgen/causal.py`)
and behavioral verdicts feeding the launch recommendation
(`POST /api/nextgen/causal/replay`, `POST /api/nextgen/gauntlet/run`).

## Dimension table

| Dimension | A: Minimal | B: Agentic | C: Full closed-loop |
|---|---|---|---|
| Regression-detection power | Detects metric drops; cannot rank by consequence | + triage explains and clusters failures | + separates METRIC_ONLY from BEHAVIORALLY_CONSEQUENTIAL (measured, not judged) |
| Safety evidence quality | SCR + risk weighting (already deterministic) | + counterfactual coverage of rare/dangerous slices | + behavioral outcomes (TTC, min-separation, collision) under corrected-vs-actual perception |
| Statistical rigor | High — seqeval e-processes gate everything | Same (agents never decide; policy does) | Same; causal verdicts are deterministic thresholds on simulated physics |
| Compute cost | Lowest; dedup gives ~5-6x on shared backbones (measured in `/api/nextgen/compute/report`) | + generation cost per counterfactual (cheap here; large with a learned world model) | + 2x closed-loop sims per causal replay (still bounded: replay only gated-in scenarios) |
| Ops burden | Low: no new long-running services | Medium: scenario store + gate maintenance, agent prompt/policy versioning | Highest: planner/controller model stewardship, sim-fidelity audits, verdict-threshold governance |
| New failure modes | Metric misconfiguration | Invalid generated scenarios polluting suites (mitigated by `validity.py` weight caps) | Sim-to-real gap in the simplified planner; over-trusting behavioral verdicts (mitigated: verdicts carry data labels, open-loop always reported) |
| Time-to-value | Immediate | Weeks (curate recipes, tune gate) | Longer (trust-building against real disengagement data) |
| Team skill needs | Existing | + generative/validation skills | + vehicle-dynamics/planning literacy |
| Deletability | Trivial | Gate + generator are separable modules | Closed-loop stack deletable without touching open-loop paths (deliberate: `nextgen` imports downstream, nothing imports it) |
| 100k-gauntlet scaling | Yes (scheduler + dedup: ~66k of 100k units evaluated, ~0.1 s on synthetic units) | Yes (validity gate is O(scenario), amortized) | Causal replay reserved for candidates that survive the gauntlet (top-of-funnel stays cheap) |

## Recommendation

**Adopt C, staged through B, with A's components as the permanent foundation**
— which is exactly how the code is factored. The gauntlet + SCR + dedup (A)
are unconditional wins with no new risk surface and should gate every launch
immediately. Counterfactual generation (B) is only safe WITH the validity
gate and weight caps — low-fidelity scenarios are capped so they cannot
dominate launch decisions (`validity.apply_suite_weight_policy`, tested).
Closed-loop causal replay (C) is the only component that answers "does this
regression matter?", and its cost is bounded because it sits at the bottom of
the funnel: gauntlet filters candidates, validity gates scenarios, and only
the survivors get behavioral replay. The honest caveat: C's verdicts are only
as good as the planner model — ours is a deliberately simple IDM + lane-keep
stack (`closedloop.py` docstring), so C's verdicts inform launch decisions
alongside, never instead of, open-loop metrics. That complementarity is
enforced in code: every `BehavioralAssessment` carries `open_loop` metrics,
and every report carries data labels.
