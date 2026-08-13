"""Next-Generation AV Perception Evaluation platform (Features 2-5).

This package adds four capabilities on top of the existing evaluation
substrate, plus the architecture decision deliverables under
docs/architecture/:

2. Counterfactual simulation with validity gating
   (:mod:`counterfactual`, :mod:`worldmodel`, :mod:`validity`)
3. Closed-loop behavioral evaluation + causal counterfactual replay
   (:mod:`closedloop`, :mod:`causal`)
4. Safety-informed metrics (Safety-Critical Recall, risk-weighted metrics)
   (:mod:`safety_metrics`)
5. Compute dedup + intelligent launch-eval scheduling
   (:mod:`cache`, :mod:`compute`, :mod:`scheduler`, :mod:`regression`,
   :mod:`lineage`)

Design principles enforced in code
----------------------------------
* Evidence before inference: every metric, gate and verdict here is
  deterministic math on data; nothing is decided by a model's opinion.
* Open-loop AND closed-loop metrics are complementary: closed-loop behavioral
  reports always carry the open-loop metrics (mAP/recall/IoU) alongside; no
  report replaces them.
* Simulation is evidence, not reality: every datum carries a
  :class:`sensorflow.nextgen.models.DataLabel`
  (REAL / REPLAYED / SIMULATED / GENERATED / COUNTERFACTUAL) which is carried
  through every downstream report.
* Deterministic safety policy: gate thresholds are explicit, versioned
  parameters; the same inputs always produce the same launch recommendation.
* Scale without linear recompute: the content-addressed feature cache
  (:mod:`cache`) and the anytime-valid early-stopping scheduler
  (:mod:`scheduler`, delegating to :mod:`sensorflow.seqeval`) together keep a
  100k-scenario gauntlet sublinear in full-inference cost.
* Design for deletion / reuse > extend > new. Reuse map (each documented at
  the point of use):

  - sensorflow.bevfusion   scenes/sensors/engines/evaluate: the scenario,
                           sensor-simulation and perception substrate.
                           Counterfactuals are transformations of its
                           SceneSequence; closed-loop runs its sensor models.
  - sensorflow.seqeval     ALL sequential statistics: anytime-valid e-process
                           early stopping, cluster units, paired deltas.
  - sensorflow.safety      ssam_ext TTC/DRAC/collision-probability/stopping
                           math; gate structuring conventions; ODD strata.
  - sensorflow.megaeval    distribution-shift machinery (train-vs-eval mix)
                           and metric-cube aggregation patterns.
  - sensorflow.rca.stats   PSI/JS divergence primitives (imported read-only
                           with a guarded fallback).
  - sensorflow.vitis       optional optical-flow temporal cross-check
                           (guarded import; absent -> skipped, reported).
  - sensorflow.agentic     Feature 1 (built concurrently by another agent):
                           guarded import for the AgenticSafetyScorecard
                           shape; a compatible local definition is used when
                           the package is not importable.
"""
