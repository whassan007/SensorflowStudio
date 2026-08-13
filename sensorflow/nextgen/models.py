"""Pydantic schemas shared by the nextgen evaluation platform.

Every datum that can appear in a report carries a DataLabel; the label is
assigned at creation time and is never inferred downstream.

AgenticSafetyScorecard: Feature 1 (sensorflow.agentic) is being built
concurrently by another workstream. We import its scorecard when available so
the two subsystems share one shape; otherwise a compatible local definition
(the fields relevant to Features 2-5: behavioral_impact,
counterfactual_validated, closed_loop_validated) is used and marked as such.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataLabel(str, Enum):
    """Provenance of a datum. Simulation is evidence, not reality; the label
    says which kind of evidence and travels with every report."""

    REAL = "REAL"                      # captured on-vehicle
    REPLAYED = "REPLAYED"              # real log replayed open-loop
    SIMULATED = "SIMULATED"            # synthetic scenario, nominal params
    GENERATED = "GENERATED"            # model-generated content
    COUNTERFACTUAL = "COUNTERFACTUAL"  # transformation of a source scenario


class TransformationStep(BaseModel):
    """One step of a counterfactual recipe (deterministic, parameterized)."""

    kind: str                  # e.g. "environment.day_to_night"
    params: Dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Where a counterfactual scenario came from — carried through every
    evaluation report that touches the scenario."""

    source_scene_id: str
    recipe: List[TransformationStep] = Field(default_factory=list)
    seed: int = 0
    generator: str = "nextgen.worldmodel.DeterministicSceneTransformer"
    generator_version: str = "1.0"
    data_label: DataLabel = DataLabel.COUNTERFACTUAL
    created_at: str = Field(default_factory=utc_now)


class ValidityReport(BaseModel):
    """Output of the counterfactual validity gate (validity.py)."""

    scenario_id: str
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    simulation_fidelity_score: float = 0.0   # [0,1] physical/temporal/sensor
    counterfactual_validity: float = 0.0     # [0,1] gate-weighted composite
    realism_confidence: float = 0.0          # [0,1] distribution similarity
    accepted: bool = False
    evaluation_weight: float = 0.0           # cap applied for low fidelity
    weight_capped: bool = False
    reasons: List[str] = Field(default_factory=list)


class CounterfactualScenario(BaseModel):
    scenario_id: str
    provenance: Provenance
    n_frames: int = 0
    n_actors: int = 0
    environment: Dict[str, str] = Field(default_factory=dict)
    validity: Optional[ValidityReport] = None


class LineageRecord(BaseModel):
    """Full reproducibility record for one evaluation run. A run whose lineage
    is incomplete is marked INVALID for launch purposes (lineage.py)."""

    model_version: Optional[str] = None
    baseline_version: Optional[str] = None
    dataset_version: Optional[str] = None
    scenario_set_version: Optional[str] = None
    sensor_sim_version: Optional[str] = None
    simulation_version: Optional[str] = None
    feature_pipeline_version: Optional[str] = None
    metric_version: Optional[str] = None
    policy_version: Optional[str] = None
    seeds: Dict[str, int] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class EvaluationRun(BaseModel):
    run_id: str
    kind: str  # counterfactual_generation | closed_loop | causal_replay | gauntlet | safety_report
    status: str = "created"  # created | running | completed | halted | failed
    data_labels: List[DataLabel] = Field(default_factory=list)
    lineage: Optional[LineageRecord] = None
    lineage_valid: bool = False
    valid_for_launch: bool = False
    invalid_reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    params: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)


class BehavioralMetrics(BaseModel):
    """Closed-loop behavioral metrics for one scenario run (closedloop.py).
    All safety-region math reuses sensorflow.safety.ssam_ext."""

    detection_latency_s: Optional[float] = None   # first-relevant-object detect delay
    time_to_detection_s: Optional[float] = None   # sim time of first detection
    min_ttc_s: Optional[float] = None
    stopping_distance_m: Optional[float] = None
    max_deceleration_mps2: float = 0.0
    max_steering_rate_radps: float = 0.0
    planner_interventions: int = 0
    collision: bool = False
    collision_probability: float = 0.0
    min_separation_m: Optional[float] = None
    safety_margin_m: Optional[float] = None       # min_separation - required stopping gap
    final_speed_mps: float = 0.0


class BehavioralAssessment(BaseModel):
    scenario_id: str
    data_label: DataLabel
    perception_mode: str  # "actual" | "corrected"
    metrics: BehavioralMetrics
    open_loop: Dict[str, Any] = Field(default_factory=dict)   # complementary, never replaced
    trajectory: List[Dict[str, float]] = Field(default_factory=list)


CAUSAL_METRIC_ONLY = "METRIC_ONLY"
CAUSAL_BEHAVIORAL = "BEHAVIORALLY_CONSEQUENTIAL"


class CausalReplayResult(BaseModel):
    scenario_id: str
    data_label: DataLabel
    actual: BehavioralAssessment
    corrected: BehavioralAssessment
    diffs: Dict[str, Any] = Field(default_factory=dict)
    causal_chain: List[Dict[str, Any]] = Field(default_factory=list)
    verdict: str = CAUSAL_METRIC_ONLY


class DistributionShiftAssessment(BaseModel):
    run_id: str
    method: str
    shifts: List[Dict[str, Any]] = Field(default_factory=list)
    psi: Optional[float] = None
    js_divergence: Optional[float] = None
    magnitude: Optional[str] = None
    data_labels: List[DataLabel] = Field(default_factory=list)


class ComputeOptimizationResult(BaseModel):
    report_id: str
    n_scenarios: int
    n_models: int
    naive_full_inferences: int
    optimized_backbone_computes: int
    optimized_head_computes: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    naive_cost_s: float          # measured per-unit timings x naive counts
    optimized_cost_s: float      # actually measured wall time
    savings_ratio: float
    measured_backbone_s: float
    measured_head_s: float
    invalidation: str
    created_at: str = Field(default_factory=utc_now)


class LaunchRecommendation(BaseModel):
    run_id: str
    recommendation: str  # LAUNCH | DO_NOT_LAUNCH | INSUFFICIENT_EVIDENCE | INVALID
    blockers: List[str] = Field(default_factory=list)
    statistical_significance: Dict[str, Any] = Field(default_factory=dict)
    safety_significance: Dict[str, Any] = Field(default_factory=dict)  # never equated with statistical
    lineage_valid: bool = False
    data_labels: List[DataLabel] = Field(default_factory=list)
    policy_version: str = "nextgen-launch-policy-1.0"


# --------------------------------------------------------------- scorecard

try:  # Feature 1 is built concurrently in sensorflow/agentic (guarded import).
    from sensorflow.agentic.models import AgenticSafetyScorecard  # type: ignore # noqa: F401
    AGENTIC_SCORECARD_SOURCE = "sensorflow.agentic.models"
except Exception:  # pragma: no cover - depends on sibling workstream timing
    AgenticSafetyScorecard = None  # type: ignore[assignment]
    AGENTIC_SCORECARD_SOURCE = "unavailable (sensorflow.agentic not importable)"


class ScorecardBehavioralExtension(BaseModel):
    """The Features-2-5 contribution to the shared safety scorecard.

    The agentic package's AgenticSafetyScorecard (severity / exposure /
    policy_outcome / ...) does not itself carry closed-loop fields; this
    extension composes with it via agentic_scorecard_id rather than
    redefining it. behavioral_impact carries the causal-replay verdict
    (METRIC_ONLY / BEHAVIORALLY_CONSEQUENTIAL); the two *_validated flags say
    whether counterfactual validity gating and closed-loop replay were
    actually run for the case.
    """

    case_id: str
    agentic_scorecard_id: Optional[str] = None  # link to Feature 1's scorecard
    behavioral_impact: Optional[str] = None     # causal replay verdict
    counterfactual_validated: Optional[bool] = None
    closed_loop_validated: Optional[bool] = None
    data_label: DataLabel = DataLabel.SIMULATED
    details: Dict[str, Any] = Field(default_factory=dict)
