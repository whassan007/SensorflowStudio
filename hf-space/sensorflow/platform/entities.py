"""Platform entity schemas with versioning / provenance.

Prefer these as the shared vocabulary across LabelEval + MegaEval.
Existing store models in evaluation.records remain the persistence layer for
label QA; MegaEval continues to use npz cubes. These Pydantic models are the
contract for APIs, evidence packages, and later lakehouse rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


class Provenance(BaseModel):
    """Lineage fields required on durable evaluation artifacts."""

    created_at: str = Field(default_factory=_now)
    created_by: str = "system"
    schema_version: str = "platform-1.0"
    parent_ids: List[str] = Field(default_factory=list)
    source_system: str = "sensorflow"  # labeleval | megaeval | pipeline | platform
    seed: Optional[int] = None
    code_version: str = "phase1"
    notes: str = ""


class VersionedEntity(BaseModel):
    version: str = "v1"
    provenance: Provenance = Field(default_factory=Provenance)


# ------------------------------------------------------------------ hierarchy


class DatasetEntity(VersionedEntity):
    dataset_id: str
    name: str = ""
    population_id: Optional[str] = None  # MegaEval link
    num_frames: int = 0
    gt_type: Optional[str] = None


class Container(VersionedEntity):
    """Scene/segment holding related objects (MegaEval container_id)."""

    container_id: str
    dataset_id: Optional[str] = None
    run_id: Optional[str] = None
    mega_container_id: Optional[int] = None
    drive_id: Optional[str] = None
    scene_id: Optional[str] = None
    dims: Dict[str, str] = Field(default_factory=dict)
    n_objects: int = 0


class Group(VersionedEntity):
    """Logical grouping: verification bucket OR cohort alias."""

    group_id: str
    kind: Literal["verification", "cohort", "custom"] = "custom"
    dataset_id: Optional[str] = None
    member_ids: List[str] = Field(default_factory=list)
    label: str = ""


class Drive(VersionedEntity):
    drive_id: str
    dataset_id: Optional[str] = None
    scene_ids: List[str] = Field(default_factory=list)
    container_ids: List[str] = Field(default_factory=list)
    description: str = ""


class SceneEntity(VersionedEntity):
    scene_id: str
    dataset_id: Optional[str] = None
    drive_id: Optional[str] = None
    sequence_ids: List[str] = Field(default_factory=list)
    description: str = ""


class ScenarioEntity(VersionedEntity):
    scenario_id: str
    scenario_type: str
    dataset_id: Optional[str] = None
    frame_ids: List[str] = Field(default_factory=list)
    description: str = ""


class FrameEntity(VersionedEntity):
    frame_id: str
    dataset_id: Optional[str] = None
    scene_id: Optional[str] = None
    sequence_id: Optional[str] = None
    container_id: Optional[str] = None
    index: int = 0
    timestamp_us: int = 0


class Sensor(VersionedEntity):
    sensor_id: str
    modality: Literal["lidar", "camera", "radar", "ultrasonic", "other"] = "lidar"
    frame_id: Optional[str] = None
    calibration_ref: Optional[str] = None


class ObjectEntity(VersionedEntity):
    object_id: str
    class_name: str = "vehicle"
    frame_id: Optional[str] = None
    container_id: Optional[str] = None
    track_id: Optional[str] = None
    bbox_3d: Optional[List[float]] = None


class Trajectory(VersionedEntity):
    trajectory_id: str
    track_id: Optional[str] = None
    object_ids: List[str] = Field(default_factory=list)
    frame_ids: List[str] = Field(default_factory=list)


class Label(VersionedEntity):
    """Alias contract for Annotation — hypothesis, never assumed correct."""

    label_id: str
    annotation_id: Optional[str] = None
    frame_id: Optional[str] = None
    object_id: Optional[str] = None
    class_name: str = ""
    confidence: float = 0.0
    status: str = "PENDING"
    model_version: Optional[str] = None


class AnnotationEntity(Label):
    """Explicit Annotation mirror for API clarity."""

    pass


# ------------------------------------------------------------------ models / eval


class ModelEntity(VersionedEntity):
    model_id: str
    name: str = ""
    status: str = "ready"


class ModelVersion(VersionedEntity):
    model_version_id: str
    model_id: str
    model_version: str
    checkpoint_ref: Optional[str] = None
    trained_on_dataset: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class Metric(VersionedEntity):
    metric_id: str
    name: str
    value: Optional[float] = None
    unit: str = "ratio"
    higher_is_better: bool = True
    support: Optional[int] = None
    level: str = "container"  # AggregateLevel value
    exact: bool = True


class Evaluation(VersionedEntity):
    evaluation_id: str
    level: str
    dataset_id: Optional[str] = None
    population_id: Optional[str] = None
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    label_version: Optional[str] = None
    scope_ref: Optional[str] = None  # container/cohort/drive id
    metrics: List[Metric] = Field(default_factory=list)
    status: str = "created"


class Cohort(VersionedEntity):
    cohort_id: str
    filters: Dict[str, List[str]] = Field(default_factory=dict)
    group_by: List[str] = Field(default_factory=list)
    n: int = 0
    run_id: Optional[str] = None


class Embedding(VersionedEntity):
    embedding_id: str
    ref_type: str = "container"  # container | object | frame
    ref_id: str
    dim: int = 0
    # TODO(Phase 3): vector payload / external vector-DB handle
    vector_ref: Optional[str] = None


class Anomaly(VersionedEntity):
    anomaly_id: str
    annotation_id: Optional[str] = None
    container_id: Optional[str] = None
    score: float = 0.0
    is_anomaly: bool = False
    detector_scores: Dict[str, float] = Field(default_factory=dict)


class Conflict(VersionedEntity):
    """Safety / label conflict placeholder — Phase 4 fills SSAM DRAC/DeltaS."""

    conflict_id: str
    kind: str = "label"  # label | trajectory | ssam_drac | ssam_deltas
    entity_ids: List[str] = Field(default_factory=list)
    severity: str = "medium"
    # TODO(Phase 4): full SSAM conflict engine fields
    detail: Dict[str, Any] = Field(default_factory=dict)


class ODDDefinition(VersionedEntity):
    odd_id: str
    name: str
    axes: Dict[str, List[str]] = Field(default_factory=dict)
    # TODO(Phase 5): combinatorial coverage rules
    description: str = ""


class ODDObservation(VersionedEntity):
    observation_id: str
    odd_id: str
    observed_axes: Dict[str, str] = Field(default_factory=dict)
    covered: bool = False
    support: int = 0
    run_id: Optional[str] = None


class QualityGateDef(VersionedEntity):
    gate_id: str
    name: str
    gate_type: Literal[
        "scenario", "coverage", "regression", "safety", "release", "quality", "launch"
    ]
    thresholds: Dict[str, float] = Field(default_factory=dict)
    config_path: Optional[str] = None
    enabled: bool = True


class GateResult(VersionedEntity):
    result_id: str
    gate_id: str
    gate_type: str
    passed: Optional[bool] = None  # None = not yet evaluated (skeleton)
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    scope_ref: Optional[str] = None
    ready: bool = False  # False until Phase N wires real checks
    message: str = ""


class HumanReviewEntity(VersionedEntity):
    review_id: str
    task_id: Optional[str] = None
    annotation_id: Optional[str] = None
    reviewer: str = "human-reviewer"
    action: Dict[str, Any] = Field(default_factory=dict)


class ComputeUsage(VersionedEntity):
    usage_id: str = Field(default_factory=lambda: _id("compute"))
    stage: str = ""
    units: int = 0
    run_id: str = ""
    evaluation_id: Optional[str] = None
    hardware: str = "cpu"


class EvidencePackage(VersionedEntity):
    """Exportable evaluation artifact — Phase 1 stub with real fields when available."""

    package_id: str = Field(default_factory=lambda: _id("evidence"))
    evaluation_id: str
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    population_id: Optional[str] = None
    model_versions: List[str] = Field(default_factory=list)
    label_version: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    gate_results: List[Dict[str, Any]] = Field(default_factory=list)
    compute_usage: List[Dict[str, Any]] = Field(default_factory=list)
    drill_down_ids: Dict[str, List[str]] = Field(default_factory=dict)
    placeholders: List[str] = Field(default_factory=list)
