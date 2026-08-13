"""Shared pydantic schemas for the agentic launch-readiness subsystem.

Everything downstream (pipeline state, policy inputs, scorecards) is explicit,
typed, staged state — never free-form agent memory. Evidence values carry an
epistemic tag (OBSERVED / DERIVED / HYPOTHESIS / UNAVAILABLE) so an LLM
hypothesis can never masquerade as a measurement.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ------------------------------------------------------------------ enums

EvidenceStatus = Literal["OBSERVED", "DERIVED", "HYPOTHESIS", "UNAVAILABLE"]

EvidenceQuality = Literal["CONFIRMED", "LIKELY", "POSSIBLE", "INSUFFICIENT_EVIDENCE"]

Severity = Literal["S0", "S1", "S2", "S3", "S4", "S5"]

PolicyOutcome = Literal[
    "AUTOMATIC_STOP_SHIP",
    "LAUNCH_REVIEW_REQUIRED",
    "CONTINUE_INVESTIGATION",
    "NO_LAUNCH_IMPACT",
    "INDETERMINATE",
]

OptionCode = Literal[
    "STOP_SHIP",
    "OPTION_A_DELAY",
    "OPTION_B_MITIGATION",
    "OPTION_C_REDUCED_ODD",
    "EXPAND_EVALUATION",
    "HUMAN_SAFETY_REVIEW",
    "PROCEED",
]

FusionVerdict = Literal[
    "multi_modal_supported", "single_modality_only", "modality_conflict",
    "verification_failed",
]

STAGES = [
    "FAILURE_DETECTION",
    "EVIDENCE_AGGREGATION",
    "FAILURE_ANALYSIS",
    "LAUNCH_DECISION",
    "LEARNING_FLYWHEEL",
]

StageStatus = Literal["pending", "running", "complete", "failed", "blocked"]

FailureKind = Literal[
    "classification_flip", "localization_regression", "detection_regression",
    "confidence_calibration_regression", "temporal_regression",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------------ failure events


class FailureInstance(BaseModel):
    """One concrete occurrence of the failure pattern (scene-addressable)."""

    instance_id: str
    sequence_id: str
    frame_id: str
    frame_index: int
    object_instance_id: str
    gt_class: str
    predicted_class: str
    confidence: float
    distance_m: float
    construction_zone: bool = False
    time_of_day: str = "day"
    weather: str = "clear"
    geo_bucket: str = "unknown"
    occluded: bool = False
    has_planner_trace: bool = False


class DetectionBasis(BaseModel):
    """Deterministic measurements that caused the agent to emit the event."""

    method: str
    candidate_events: int
    baseline_events: int
    denominator: int
    candidate_rate: float
    baseline_rate: float
    metric_deltas: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class FailureEvent(BaseModel):
    """A candidate failure PATTERN (with member instances), the unit the
    five-layer pipeline analyzes."""

    failure_id: str
    kind: FailureKind
    title: str
    description: str = ""
    gt_class: Optional[str] = None
    predicted_class: Optional[str] = None
    detection_basis: DetectionBasis
    instances: List[FailureInstance] = Field(default_factory=list)
    baseline_model: str = "baseline-v1"
    candidate_model: str = "candidate-v2"
    dataset_fingerprint: str = ""
    population_id: str = ""
    status: str = "detected"          # detected|investigating|decided|closed
    severity: Optional[Severity] = None
    policy_outcome: Optional[PolicyOutcome] = None
    validated: bool = False           # set only by human review
    created_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ agent I/O


class AgentEscalation(BaseModel):
    required: bool = False
    reasons: List[str] = Field(default_factory=list)
    human_review_triggers: List[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """Uniform output envelope for every agent.

    `authority` is a structural constant: agents can only ever RECOMMEND.
    """

    agent: str
    agent_version: str
    failure_id: str
    status: Literal["ok", "failed", "escalated"] = "ok"
    authority: Literal["ADVISORY_ONLY"] = "ADVISORY_ONLY"
    output: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    confidence_basis: str = ""
    epistemic_status: EvidenceStatus = "DERIVED"
    escalation: AgentEscalation = Field(default_factory=AgentEscalation)
    failure_handling: Optional[str] = None
    llm_used: bool = False
    llm_provider: Optional[str] = None
    llm_rationale: Optional[str] = None
    started_at: str = Field(default_factory=now_iso)
    finished_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ evidence graph


class EvidenceNode(BaseModel):
    node_id: str
    node_type: Literal[
        "Object", "Environment", "Sensors", "Prediction", "GroundTruth",
        "Tracking", "Planner", "HistoricalSimilarity", "Frequency",
        "SafetyConsequence",
    ]
    status: EvidenceStatus
    summary: str
    fields: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    caveats: List[str] = Field(default_factory=list)


class EvidenceEdge(BaseModel):
    src: str
    dst: str
    relation: str


class EvidenceGraph(BaseModel):
    failure_id: str
    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)
    built_at: str = Field(default_factory=now_iso)

    def node(self, node_type: str) -> Optional[EvidenceNode]:
        for n in self.nodes:
            if n.node_type == node_type:
                return n
        return None


# ------------------------------------------------------------------ snippet


class TemporalFrameRef(BaseModel):
    frame_id: str
    frame_index: int
    role: Literal["before", "failure", "after"]
    gt_class: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    bbox_3d: Optional[List[float]] = None


class FailureSnippet(BaseModel):
    """Reproducible failure package: everything needed to replay/triage one
    concrete failure instance, plus temporal context."""

    snippet_id: str
    failure_id: str
    instance_id: str
    scene_id: str
    sequence_id: str
    timestamp_us: int
    frame_index: int
    model_version_candidate: str
    model_version_baseline: str
    prediction: Dict[str, Any]
    ground_truth: Dict[str, Any]
    confidence: float
    tracking_state: Dict[str, Any] = Field(default_factory=dict)
    planner_state: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    geo_bucket: str = "unknown"
    software_config: Dict[str, Any] = Field(default_factory=dict)
    hardware_config: Dict[str, Any] = Field(default_factory=dict)
    feature_pipeline_version: str = ""
    temporal_context: List[TemporalFrameRef] = Field(default_factory=list)
    similar_historical: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ statistics


class RateEstimate(BaseModel):
    events: int
    denominator: int
    rate: float
    wilson_ci: List[float]
    ci_method: str = "Wilson score (megaeval.sampling.wilson_interval)"


class StatisticalAssessment(BaseModel):
    """Output of the (mostly deterministic) statistical regression agent."""

    baseline: RateEstimate
    candidate: RateEstimate
    absolute_delta: float
    relative_delta: Optional[float]
    significant: bool
    significance_method: str
    exact_binomial_p: Optional[float] = None
    seqeval: Dict[str, Any] = Field(default_factory=dict)
    power_mde: Dict[str, Any] = Field(default_factory=dict)
    small_sample_flags: List[str] = Field(default_factory=list)
    rare_event_handling: str = ""


# ------------------------------------------------------------------ concentration


class StratumRisk(BaseModel):
    dimension: str
    stratum: str
    exposure: int
    exposure_share: float
    events: int
    stratum_rate: float
    baseline_rate: float
    relative_risk: Optional[float]
    odds_ratio: Optional[float]
    risk_difference: float
    rate_wilson_ci: List[float]
    small_sample_flag: bool = False


class ConcentrationAnalysis(BaseModel):
    failure_id: str
    determination: Literal["concentrated", "uniform", "insufficient_data"]
    concentrated_dimensions: List[str] = Field(default_factory=list)
    strata: List[StratumRisk] = Field(default_factory=list)
    method: str = ""


# ------------------------------------------------------------------ scorecard


class ScorecardField(BaseModel):
    """A scorecard value with explicit provenance."""

    value: Any
    tag: Literal["OBSERVED", "PREDICTED", "HYPOTHETICAL", "REQUIRED_EVIDENCE"]
    evidence_ref: str = ""


class AgenticSafetyScorecard(BaseModel):
    scorecard_id: str
    failure_id: str
    title: str
    failure_summary: ScorecardField
    frequency: ScorecardField
    exposure: ScorecardField
    severity: ScorecardField
    confidence: ScorecardField
    novelty: ScorecardField
    concentration: ScorecardField
    downstream_impact: ScorecardField
    mitigations: ScorecardField
    residual_risk: ScorecardField
    evidence_quality: EvidenceQuality
    policy_outcome: Optional[PolicyOutcome] = None
    recommended_option: Optional[OptionCode] = None
    policy_version: str = ""
    generated_at: str = Field(default_factory=now_iso)
    notes: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------ governance


class HumanReviewDecision(BaseModel):
    review_id: str
    failure_id: str
    reviewer: str
    decision: Literal[
        "confirm_failure", "reject_failure", "approve_launch",
        "block_launch", "approve_option", "request_more_evidence",
    ]
    approved_option: Optional[OptionCode] = None
    evidence_reviewed: List[str] = Field(default_factory=list)
    policy_version: str = ""
    rationale: str = ""
    override_reason: Optional[str] = None
    timestamp: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ flywheel


class SuiteMember(BaseModel):
    member_id: str
    sequence_id: str
    frame_id: str
    object_instance_id: str
    source_failure_id: str
    training_eligible: bool = False   # contamination guard default
    added_at: str = Field(default_factory=now_iso)


class EvaluationSuite(BaseModel):
    suite_id: str
    name: str
    version: int = 1
    creation_reason: str
    source_failures: List[str] = Field(default_factory=list)
    taxonomy_tags: List[str] = Field(default_factory=list)
    sampling_policy: str = ""
    coverage: Dict[str, Any] = Field(default_factory=dict)
    known_limitations: List[str] = Field(default_factory=list)
    approval_status: Literal["draft", "approved", "retired"] = "draft"
    members: List[SuiteMember] = Field(default_factory=list)
    contamination_guard: str = (
        "suite members are training-ineligible without an explicit recorded "
        "governance override"
    )
    governance_overrides: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ pipeline state


class StageRecord(BaseModel):
    stage: str
    status: StageStatus = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    detail: str = ""


class PipelineState(BaseModel):
    """Explicit staged state — deliberately NOT one autonomous agent
    controlling the chain. Each stage transition is orchestrated, persisted
    and audited by deterministic code."""

    failure_id: str
    stages: List[StageRecord] = Field(
        default_factory=lambda: [StageRecord(stage=s) for s in STAGES])
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict)
    statistical: Optional[StatisticalAssessment] = None
    concentration: Optional[ConcentrationAnalysis] = None
    policy_evaluation: Optional[Dict[str, Any]] = None
    scorecard_id: Optional[str] = None
    suite_ids: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)

    def stage_record(self, stage: str) -> StageRecord:
        for s in self.stages:
            if s.stage == stage:
                return s
        raise KeyError(stage)
