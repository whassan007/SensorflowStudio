"""Typed data contracts for the ROTR capability.

Design rules (docs/architecture/rotr-architecture.md §7):
* every persisted record carries a Provenance block;
* tri-state evidence is an enum, never a boolean;
* no defaults that could fabricate evidence — missing evidence is UNKNOWN.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- enums

CAUSAL_LAYERS = ["perception", "prediction", "planning", "localization",
                 "map", "control", "policy_rule", "data_label"]

EVIDENCE_STATUSES = ["SUPPORTED", "RULED_OUT", "UNKNOWN"]

CONSEQUENCE_CLASSES = ["NO_MATERIAL_CONSEQUENCE", "DEGRADED_COMFORT",
                       "PLANNER_INTERVENTION", "SAFETY_CRITICAL"]

DATASET_ROLES = ["TRAIN", "VALIDATION", "TEST", "REGRESSION", "LAUNCH",
                 "MONITORING"]
# Membership in these roles blocks training eligibility without an override.
PROTECTED_ROLES = {"REGRESSION", "LAUNCH"}

RIGHT_OF_WAY = ["HAS_ROW", "MUST_YIELD", "NONE"]
INTENTS = ["CROSSING", "MERGING", "YIELDING", "PROCEEDING", "WAITING"]

SURROGATE_CAVEAT = (
    "TTC/PET/DRAC are surrogate conflict measures: they do not alone "
    "establish collision risk (controllability, evasive options and actor "
    "reaction are unmodeled). They prioritize triage; they never decide a "
    "gate by themselves.")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- provenance


class Provenance(BaseModel):
    scenario_id: Optional[str] = None
    dataset_version: str = ""
    model_version: str = ""
    software_version: str = ""
    calibration_version: str = ""
    timestamp: str = Field(default_factory=now_iso)
    source: str = "SYNTHETIC"          # SYNTHETIC | LOG | COUNTERFACTUAL
    confidence: Optional[float] = None


# ---------------------------------------------------------------- world state


class EgoState(BaseModel):
    t: float
    x: float
    y: float
    yaw: float = 0.0
    speed: float = 0.0
    accel: float = 0.0
    lane_id: Optional[str] = None            # true lane association
    believed_x: Optional[float] = None       # localization view (None=UNKNOWN)
    believed_y: Optional[float] = None
    believed_lane_id: Optional[str] = None


class ActorObservation(BaseModel):
    """The stack's view of one actor at one frame (None = not detected)."""

    detected: bool = False
    x: Optional[float] = None
    y: Optional[float] = None
    class_name: Optional[str] = None


class ActorState(BaseModel):
    t: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    occluded: bool = False
    observation: Optional[ActorObservation] = None


class Actor(BaseModel):
    actor_id: str
    class_name: str                          # pedestrian|cyclist|vehicle|bus
    dims: List[float] = Field(default_factory=lambda: [4.5, 1.9, 1.6])
    right_of_way: str = "NONE"               # relative to ego at the conflict
    intent: str = "PROCEEDING"               # ground-truth intent
    predicted_intent: Optional[str] = None   # stack's prediction (None=UNKNOWN)
    states: List[ActorState] = Field(default_factory=list)


class Lane(BaseModel):
    lane_id: str
    center_y: float
    direction: str = "+x"
    permitted_maneuvers: List[str] = Field(
        default_factory=lambda: ["STRAIGHT", "LANE_CHANGE"])
    restricted_to: Optional[str] = None      # e.g. "bus" — None = unrestricted


class Crosswalk(BaseModel):
    crosswalk_id: str
    x_min: float
    x_max: float


class RoadContext(BaseModel):
    context_id: str = "ctx"
    intersection_type: str = "none"          # none|uncontrolled|controlled
    control: str = "none"                    # none|signal|stop_sign|yield_sign
    signal_state_for_ego: Optional[str] = None   # green|red (controlled only)
    intersection_x_min: Optional[float] = None
    intersection_x_max: Optional[float] = None
    stop_line_x: Optional[float] = None
    lanes: List[Lane] = Field(default_factory=list)
    crosswalks: List[Crosswalk] = Field(default_factory=list)
    speed_limit_mps: float = 13.4


class Environment(BaseModel):
    visibility: str = "clear"                # clear|low
    weather: str = "clear"                   # clear|rain
    lighting: str = "day"                    # day|dusk|night


class PlannedPoint(BaseModel):
    t: float
    x: float
    y: float
    v: float


class PlantedTruth(BaseModel):
    """Ground truth planted by the generator, used ONLY for evaluation."""

    kind: str
    is_violation_opportunity: bool = False
    committed: bool = False                  # did this stack commit it?
    expected_rule_id: Optional[str] = None
    cause_layer: Optional[str] = None
    notes: str = ""


class ROTRScenario(BaseModel):
    scenario_id: str
    bank_id: str = ""
    seed: int = 0
    description: str = ""
    environment: Environment = Field(default_factory=Environment)
    map_context: RoadContext = Field(default_factory=RoadContext)   # as mapped
    actual_context: RoadContext = Field(default_factory=RoadContext)  # as built
    ego: List[EgoState] = Field(default_factory=list)     # executed trajectory
    planned: List[PlannedPoint] = Field(default_factory=list)
    actors: List[Actor] = Field(default_factory=list)
    planted: PlantedTruth = Field(default_factory=lambda: PlantedTruth(kind=""))
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------- detection


class ROTRViolation(BaseModel):
    violation_id: str
    scenario_id: str
    rule_id: str
    rule_version: str
    description: str = ""
    actor_ids: List[str] = Field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0
    evidence: Dict = Field(default_factory=dict)
    confidence: float = 0.0
    taxonomy: Dict[str, str] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------- attribution


class LayerEvidence(BaseModel):
    layer: str
    status: str = "UNKNOWN"                  # SUPPORTED|RULED_OUT|UNKNOWN
    evidence: str = ""
    confidence: float = 0.0


class FailureAttribution(BaseModel):
    violation_id: str
    scenario_id: str
    layers: Dict[str, LayerEvidence] = Field(default_factory=dict)
    primary_layer: Optional[str] = None      # None = triage to HITL
    note: str = ""
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------- consequence


class SafetyAssessment(BaseModel):
    min_ttc_s: Optional[float] = None
    pet_s: Optional[float] = None
    min_clearance_m: Optional[float] = None
    stopping_distance_m: Optional[float] = None
    max_braking_mps2: float = 0.0
    max_lateral_deviation_m: float = 0.0
    collision: bool = False
    surrogate_caveat: str = SURROGATE_CAVEAT


class PlannerEvaluation(BaseModel):
    engine: str = ""                         # which replay planner produced this
    observed_trajectory: List[Dict] = Field(default_factory=list)
    corrected_trajectory: List[Dict] = Field(default_factory=list)
    max_position_divergence_m: float = 0.0
    max_speed_divergence_mps: float = 0.0
    corrected_max_braking_mps2: float = 0.0
    corrected_intervention: bool = False


class CounterfactualScenario(BaseModel):
    counterfactual_id: str
    violation_id: str
    scenario_id: str
    corrected_layers: List[str] = Field(default_factory=list)
    consequence_class: str = "NO_MATERIAL_CONSEQUENCE"
    planner_evaluation: PlannerEvaluation = Field(default_factory=PlannerEvaluation)
    observed_safety: SafetyAssessment = Field(default_factory=SafetyAssessment)
    corrected_safety: SafetyAssessment = Field(default_factory=SafetyAssessment)
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------- metrics


class RegressionResult(BaseModel):
    regression_id: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_model: str = ""
    candidate_model: str = ""
    metric_deltas: Dict = Field(default_factory=dict)
    six_outcomes: Dict[str, bool] = Field(default_factory=dict)
    primary_outcome: str = ""
    seqeval: Optional[Dict] = None           # delegation payload (None=unavailable)
    distribution_note: str = ""
    provenance: Provenance = Field(default_factory=Provenance)


# ---------------------------------------------------------------- governance


class HITLReview(BaseModel):
    review_id: str
    run_id: str
    violation_id: str
    cluster_id: Optional[str] = None
    status: str = "PENDING"                  # PENDING|VALIDATED|REJECTED
    action: Optional[str] = None
    actor: Optional[str] = None
    notes: str = ""
    provenance: Provenance = Field(default_factory=Provenance)


class TrainingCandidate(BaseModel):
    candidate_id: str
    run_id: str
    violation_id: str
    cluster_id: Optional[str] = None
    dataset_role: str = "REGRESSION"
    training_eligible: bool = False
    guard_state: str = "PROTECTED"           # PROTECTED|ELIGIBLE|OVERRIDDEN
    override: Optional[Dict] = None          # {actor, reason, timestamp}
    provenance: Provenance = Field(default_factory=Provenance)


class ReleaseGateResult(BaseModel):
    gate_id: str
    run_id: str
    policy_version: str
    outcome: str = "GO"                      # GO|NO_GO
    events: List[Dict] = Field(default_factory=list)
    agentic_advisory: Optional[Dict] = None  # forwarded policy-engine outcome
    provenance: Provenance = Field(default_factory=Provenance)
