# ADR: Next-Gen Evaluation Platform — Build / Reuse / Don't-Build

Status: accepted. Scope: Features 2-5 (`sensorflow/nextgen`) of the
next-generation AV perception evaluation vision; Feature 1 is
`sensorflow/agentic` (concurrent workstream).

## Decision drivers

Evidence before inference; open-loop AND closed-loop complementarity;
simulation labeled as evidence, never as reality; deterministic safety
policy; sublinear scaling to a 100k-scenario gauntlet; design-for-deletion.

## REUSE (verbatim, no forks)

| Need | Reused | Decision rationale |
|---|---|---|
| Scenario substrate | `bevfusion.scenes.SceneSequence` + `generate_sequences` | Counterfactuals are transformations of existing scenes (`nextgen/counterfactual.py` reconstructs world kinematics and re-renders with the same conventions). Everything that consumes sequences — both perception engines, `bevfusion.evaluate` — works on counterfactuals unchanged. |
| Sensor simulation | `bevfusion.sensors.simulate_camera/simulate_lidar` | Used twice: the validity gate's sensor-consistency check runs the real simulators on generated scenes (`validity.check_sensor`); the compute benchmark's backbone stage runs the full `fusion.build_modality_map -> fuse_maps -> decode_bev` pipeline (`compute.compute_backbone_features`). |
| Sequential statistics | `seqeval.sequential.PairedSequentialTest`, `seqeval.units.cluster_units` | ALL gauntlet early stopping delegates here (`scheduler.py`); the regression module's CIs are seqeval confidence sequences (`regression.analyze_stratum`). Zero statistical code was reimplemented. |
| Safety-region math | `safety.ssam_ext.projected_ttc / rect_gap / rects_overlap / collision_probability` | Closed-loop TTC/separation/collision (`closedloop.py`), SCR TTC criterion (`safety_metrics.in_safety_critical_region`), validity overlap checks. |
| Divergence stats | `rca.stats.psi_continuous / js_divergence_continuous` (guarded import + local fallback) | Distribution-similarity scoring in `validity.check_distribution`. |
| Distribution shift at scale | `megaeval.analysis.distribution_shift` | `POST /api/nextgen/distribution/analyze` delegates for megaeval runs. |
| Cache pattern | `seqeval.paired.PredictionCache` design | Pattern reused, code not force-fit: that cache is bound to megaeval populations/npz; `nextgen.cache.FeatureCache` stores arbitrary JSON IRs for the bevfusion pipeline with the same fingerprint-keyed invalidation-by-construction. |
| Scorecard shape | `agentic.models.AgenticSafetyScorecard` | Imported (guarded); our fields compose via `ScorecardBehavioralExtension.agentic_scorecard_id` instead of redefining their model. |

## BUILD (new, because nothing equivalent existed)

* `counterfactual.py` — deterministic scene transformer (15 transformations)
  with full provenance (source id, recipe, seed, generator version, label
  COUNTERFACTUAL).
* `validity.py` — 5-check gate (physical, temporal, sensor, identity,
  distribution) producing simulation_fidelity_score / counterfactual_validity
  / realism_confidence, plus per-scenario and suite-level weight caps.
* `closedloop.py` — IDM-longitudinal + lane-keep-lateral deterministic
  planner/controller with a seeded perception failure model; behavioral
  metrics.
* `causal.py` — actual-vs-corrected replay with the stepwise causal chain
  and METRIC_ONLY / BEHAVIORALLY_CONSEQUENTIAL verdict.
* `safety_metrics.py` — parameterized safety-critical region (reaction time,
  friction-limited braking, class lateral-speed encroachment; not naive
  stopping distance) + risk-weighted metrics + the recall-up/SCR-down
  demonstration.
* `scheduler.py` — priority gauntlet with budget, catastrophic halt, related-
  strata promotion, critical-pass expansion.
* `lineage.py` — reproducibility records; missing lineage => INVALID for
  launch.

## DON'T BUILD (explicit non-goals)

* A learned world model. `worldmodel.ExternalWorldModelAdapter` documents the
  integration contract (full GT out, determinism, provenance labels, no gate
  exemption); no proprietary API is assumed. See
  `nextgen-worldmodel-generative-comparison.md`.
* A production planner. The closed-loop stack measures perception's
  behavioral consequences; certifying planning behavior is a different
  product with different validation burdens.
* A new metrics store / dashboard framework — persisted JSON under
  `runs/nextgen/` matches every other package's convention.
* Photorealistic rendering. The evaluation questions here are geometric and
  statistical; pixels would add cost and a validation burden without changing
  any verdict this platform produces.

## EXPERIMENTAL (kept behind clear seams)

* `vitis` optical-flow temporal cross-check: optional import only.
* Extended environment conditions (fog/glare/wet) are OUR parameterization
  (`closedloop.PERCEPTION_CONDITION_EFFECTS`) since `bevfusion.sensors` does
  not model them; marked in code and in reports.
* The causal-verdict thresholds (`causal.PLANNER_DIFF_MPS2` etc.) are policy
  constants pending calibration against real disengagement data.

## Ops burden / safety benefit

Ops: no new services or daemons; everything is request-scoped FastAPI +
file persistence. The recurring burdens are (1) version stewardship —
`lineage.COMPONENT_VERSIONS` must be bumped when semantics change, or the
content-addressed cache serves stale IRs; (2) gate-threshold governance —
validity and causal thresholds are policy, and policy needs an owner.
Safety benefit: launch decisions gain three defenses that open-loop metrics
cannot provide — SCR catches "recall up where it doesn't matter, down where
it does" (`/api/nextgen/metrics/divergence-demo`), the gauntlet halts
catastrophic safety regressions after ~thousands, not 100k, of units, and
causal replay stops both over-reaction to cosmetic regressions and
under-reaction to behaviorally consequential ones.

## Minimum viable architecture

`safety_metrics.py` + `scheduler.py` + `seqeval` + `lineage.py` (Architecture
A of the comparison doc). That subset already changes launch decisions and
carries no simulation risk. Counterfactuals and closed-loop join only with
their gates: generation without `validity.py` or behavioral verdicts without
data labels are explicitly rejected configurations.
