"""Data model + JSON-backed store for the label evaluation platform.

Entities (spec §37): Dataset, Scene, Sequence, Frame, Sensor, Object, Annotation,
Track, Scenario, RareEvent, AnomalyDetection, ValidationResult, GroundTruth,
Grader, GraderComparison, RegressionResult, Benchmark, BenchmarkResult,
TriageDecision, ReviewTask, HumanReview, RelabelingAction, TrainingDataset,
TrainingJob, Model, ModelEvaluation, ProcessUsage, AuditEvent.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

STORE_DIR = Path("runs/labeleval")

GroundTruthType = Literal[
    "PSEUDO_GROUND_TRUTH",
    "VENDOR_GROUND_TRUTH",
    "HUMAN_VERIFIED_GROUND_TRUTH",
    "GOLD_STANDARD",
]

TriageStatus = Literal["AUTO_GRADED", "FLAGGED", "VERIFIED", "REJECTED", "PENDING"]

FAILURE_REASONS = [
    "LOW_IOU",
    "POSITION_ERROR",
    "ORIENTATION_ERROR",
    "INSUFFICIENT_POINT_SUPPORT",
    "ANOMALY",
    "GRADER_DISAGREEMENT",
    "TRACK_FRAGMENTATION",
    "ID_SWITCH",
    "MODEL_REGRESSION",
    "LOW_CONFIDENCE",
    "SENSOR_DISAGREEMENT",
]

CLASSES = ["pedestrian", "cyclist", "vehicle", "motorcycle", "truck"]
SAFETY_CRITICAL_CLASSES = {"pedestrian", "cyclist", "motorcycle"}

SCENARIO_TYPES = [
    "near_collision",
    "extreme_ttc",
    "extreme_pet",
    "vru_interaction",
    "unusual_object_behavior",
    "severe_occlusion",
    "sensor_failure",
    "nighttime_glare",
    "adverse_weather",
    "unusual_trajectory",
    "unexpected_road_geometry",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------------ sensor data


class EgoState(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    speed_mps: float = 0.0


class GroundTruthBox(BaseModel):
    gt_id: str
    class_name: str
    bbox_3d: List[float]  # [x, y, z, l, w, h, yaw]
    bbox_2d: Optional[List[float]] = None
    track_instance_id: str = ""
    gt_type: GroundTruthType = "VENDOR_GROUND_TRUTH"
    velocity: List[float] = Field(default_factory=lambda: [0.0, 0.0])


class Frame(BaseModel):
    frame_id: str
    dataset_id: str
    scene_id: str
    sequence_id: str
    index: int = 0
    timestamp_us: int = 0
    ego_pose: EgoState = Field(default_factory=EgoState)
    num_lidar_points: int = 0
    camera_width: int = 800
    camera_height: int = 450
    gt_boxes: List[GroundTruthBox] = Field(default_factory=list)
    scenario_tags: List[str] = Field(default_factory=list)
    weather: str = "clear"
    time_of_day: str = "day"


class Sequence(BaseModel):
    sequence_id: str
    scene_id: str
    dataset_id: str
    frame_ids: List[str] = Field(default_factory=list)


class Scene(BaseModel):
    scene_id: str
    dataset_id: str
    sequence_ids: List[str] = Field(default_factory=list)
    description: str = ""


class DatasetLineage(BaseModel):
    generated_from_model: Optional[str] = None
    corrected_by_review_batch: Optional[str] = None
    validated_by_policy: Optional[str] = None
    parent_dataset: Optional[str] = None


class Dataset(BaseModel):
    dataset_id: str
    name: str
    version: str = "v1"
    created_at: str = Field(default_factory=now_iso)
    seed: int = 7
    num_scenes: int = 0
    num_sequences: int = 0
    num_frames: int = 0
    num_annotations: int = 0
    gt_type: Optional[GroundTruthType] = "VENDOR_GROUND_TRUTH"
    gt_coverage: float = 0.0
    lineage: DatasetLineage = Field(default_factory=DatasetLineage)
    status: str = "created"
    generation_params: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ labels


class Annotation(BaseModel):
    """Auto-generated label: a hypothesis requiring evaluation. Never assumed correct."""

    annotation_id: str
    dataset_id: str
    frame_id: str
    object_id: str
    class_name: str
    confidence: float = 0.0
    bbox_2d: Optional[List[float]] = None  # [x, y, w, h] image px
    bbox_3d: Optional[List[float]] = None  # [x, y, z, l, w, h, yaw]
    mask: Optional[str] = None
    track_id: Optional[str] = None
    model: str = "synthlab-detector"
    model_version: str = "model-v1"
    source: str = "auto"
    status: TriageStatus = "PENDING"
    matched_gt_id: Optional[str] = None
    injected_errors: List[str] = Field(default_factory=list)


class Track(BaseModel):
    track_id: str
    dataset_id: str
    class_name: str = "vehicle"
    annotation_ids: List[str] = Field(default_factory=list)
    frame_ids: List[str] = Field(default_factory=list)


class Scenario(BaseModel):
    scenario_id: str
    dataset_id: str
    scenario_type: str
    frame_ids: List[str] = Field(default_factory=list)
    description: str = ""


class RareEvent(BaseModel):
    event_id: str
    dataset_id: str
    scenario_type: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    rarity_score: float = 0.0
    anomaly_score: float = 0.0
    confidence: float = 0.0
    evidence_frames: List[str] = Field(default_factory=list)
    sensor_evidence: Dict[str, str] = Field(default_factory=dict)
    description: str = ""
    verified: bool = False


# ------------------------------------------------------------------ evaluation evidence


class AnomalyDetection(BaseModel):
    annotation_id: str
    score: float = 0.0
    is_anomaly: bool = False
    detector_scores: Dict[str, float] = Field(default_factory=dict)
    normalized_scores: Dict[str, float] = Field(default_factory=dict)
    ensemble_strategy: str = "weighted_average"
    ensemble_score: float = 0.0
    decision_threshold: float = 0.7


class CheckLine(BaseModel):
    gate: str
    actual: Any = None
    threshold: Any = None
    passed: bool = True
    applicable: bool = True


class ValidationResult(BaseModel):
    annotation_id: str
    passed: bool = True
    checks: List[CheckLine] = Field(default_factory=list)
    iou_3d: Optional[float] = None
    position_error: Optional[float] = None
    orientation_error_deg: Optional[float] = None
    dimension_error: Optional[float] = None
    point_density: Optional[float] = None
    point_in_box_ratio: Optional[float] = None
    ground_contact_error: Optional[float] = None
    sensor_consistent: bool = True


class GraderComparison(BaseModel):
    annotation_id: str
    grader_count: int = 0
    graders: List[str] = Field(default_factory=list)
    class_votes: Dict[str, str] = Field(default_factory=dict)
    class_agreement: Optional[float] = None
    spatial_agreement: Optional[float] = None
    temporal_agreement: Optional[float] = None
    consensus: Optional[float] = None
    disagreement_types: List[str] = Field(default_factory=list)
    kappa_stats: Dict[str, float] = Field(default_factory=dict)


class TrackingEvidence(BaseModel):
    annotation_id: str
    id_switch: bool = False
    fragmentation: bool = False
    track_quality: Optional[float] = None


class TriageDecision(BaseModel):
    decision_id: str
    annotation_id: str
    status: TriageStatus = "PENDING"
    failure_reasons: List[str] = Field(default_factory=list)
    primary_failure_reason: Optional[str] = None
    gate_lines: List[CheckLine] = Field(default_factory=list)
    policy_id: str = ""
    policy_values: Dict[str, float] = Field(default_factory=dict)
    decided_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ review


class RelabelingAction(BaseModel):
    action: str
    corrected_bbox_3d: Optional[List[float]] = None
    corrected_class: Optional[str] = None
    merge_with_track_id: Optional[str] = None
    note: Optional[str] = None


class ReviewResolution(BaseModel):
    action: str
    corrected_bbox_3d: Optional[List[float]] = None
    corrected_class: Optional[str] = None
    revalidation_passed: Optional[bool] = None
    final_status: Optional[TriageStatus] = None
    reviewed_at: str = Field(default_factory=now_iso)


class ReviewTask(BaseModel):
    task_id: str
    annotation_id: str
    frame_id: str
    dataset_id: str
    failure_reasons: List[str] = Field(default_factory=list)
    primary_failure_reason: Optional[str] = None
    status: Literal["open", "in_review", "resolved"] = "open"
    created_at: str = Field(default_factory=now_iso)
    review_batch: str = ""
    resolution: Optional[ReviewResolution] = None


class HumanReview(BaseModel):
    review_id: str
    task_id: str
    annotation_id: str
    reviewer: str = "human-reviewer"
    action: RelabelingAction
    created_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ regression / benchmark


class RegressionMetricDelta(BaseModel):
    metric: str
    baseline: float
    current: float
    delta: float
    tolerance: float
    regressed: bool


class RegressionResult(BaseModel):
    result_id: str
    model_version: str
    baseline_version: Optional[str] = None
    dataset_version: str = ""
    run_id: str = ""
    date: str = Field(default_factory=now_iso)
    regression_detected: bool = False
    affected_classes: List[str] = Field(default_factory=list)
    affected_scenarios: List[str] = Field(default_factory=list)
    deltas: List[RegressionMetricDelta] = Field(default_factory=list)
    kinds: List[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    technique: str
    precision: float = 0.0
    recall: float = 0.0
    rare_recall: float = 0.0
    f1: float = 0.0
    box_error_3d: float = 0.0
    id_swap_rate: float = 0.0
    consensus: float = 0.0
    process_units: int = 0
    fp_rate: float = 0.0


class Benchmark(BaseModel):
    benchmark_id: str
    dataset_id: str = ""
    created_at: str = Field(default_factory=now_iso)
    rows: List[BenchmarkResult] = Field(default_factory=list)
    highlights: Dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------ training / models


class TrainingDataset(BaseModel):
    training_dataset_id: str
    source_dataset_id: str
    version: str
    num_verified_labels: int = 0
    lineage: DatasetLineage = Field(default_factory=DatasetLineage)
    created_at: str = Field(default_factory=now_iso)


class TrainingJob(BaseModel):
    job_id: str
    model_id: str
    model_version: str
    dataset_version: str
    status: Literal["queued", "running", "completed", "failed", "stopped"] = "queued"
    epoch: int = 0
    total_epochs: int = 10
    loss: float = 0.0
    rare_recall: float = 0.0
    safety_recall: float = 0.0
    process_units: int = 0
    logs: List[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=now_iso)
    quality_policy: str = ""
    configuration: Dict[str, Any] = Field(default_factory=dict)
    lineage: DatasetLineage = Field(default_factory=DatasetLineage)


class ModelEvaluation(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    map_3d: Optional[float] = None
    safety_critical_recall: Optional[float] = None
    rare_recall: Optional[float] = None


class Model(BaseModel):
    model_id: str
    model_version: str
    name: str = "synthlab-detector"
    trained_on_dataset: Optional[str] = None
    created_at: str = Field(default_factory=now_iso)
    status: str = "ready"
    metrics: ModelEvaluation = Field(default_factory=ModelEvaluation)
    regression_status: Literal["baseline", "improved", "regressed", "unknown"] = "unknown"


# ------------------------------------------------------------------ ops


class ProcessUsage(BaseModel):
    usage_id: str
    stage: str
    units: int = 0
    run_id: str = ""
    created_at: str = Field(default_factory=now_iso)


class AuditEvent(BaseModel):
    event_id: str
    timestamp: str = Field(default_factory=now_iso)
    actor: str = "system"
    action: str = ""
    entity_type: str = ""
    entity_id: str = ""
    detail: str = ""


class Alert(BaseModel):
    alert_id: str
    kind: str
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str = ""
    evidence_page: str = "overview"
    evidence_id: Optional[str] = None
    created_at: str = Field(default_factory=now_iso)


# ------------------------------------------------------------------ store

_COLLECTIONS = {
    "datasets": Dataset,
    "scenes": Scene,
    "sequences": Sequence,
    "frames": Frame,
    "annotations": Annotation,
    "tracks": Track,
    "scenarios": Scenario,
    "rare_events": RareEvent,
    "anomalies": AnomalyDetection,
    "validations": ValidationResult,
    "grader_comparisons": GraderComparison,
    "tracking_evidence": TrackingEvidence,
    "triage_decisions": TriageDecision,
    "review_tasks": ReviewTask,
    "human_reviews": HumanReview,
    "regressions": RegressionResult,
    "benchmarks": Benchmark,
    "training_datasets": TrainingDataset,
    "training_jobs": TrainingJob,
    "models": Model,
    "process_usage": ProcessUsage,
    "audit_events": AuditEvent,
    "alerts": Alert,
}

_KEY_FIELDS = {
    "datasets": "dataset_id",
    "scenes": "scene_id",
    "sequences": "sequence_id",
    "frames": "frame_id",
    "annotations": "annotation_id",
    "tracks": "track_id",
    "scenarios": "scenario_id",
    "rare_events": "event_id",
    "anomalies": "annotation_id",
    "validations": "annotation_id",
    "grader_comparisons": "annotation_id",
    "tracking_evidence": "annotation_id",
    "triage_decisions": "annotation_id",  # latest decision per annotation
    "review_tasks": "task_id",
    "human_reviews": "review_id",
    "regressions": "result_id",
    "benchmarks": "benchmark_id",
    "training_datasets": "training_dataset_id",
    "training_jobs": "job_id",
    "models": "model_id",
    "process_usage": "usage_id",
    "audit_events": "event_id",
    "alerts": "alert_id",
}


class EvalStore:
    """Thread-safe in-memory store with JSON persistence.

    Determinism and explainability matter more than storage tech (spec).
    """

    def __init__(self, base_dir: Path = STORE_DIR):
        self.base_dir = Path(base_dir)
        self.lock = threading.RLock()
        self.collections: Dict[str, Dict[str, BaseModel]] = {c: {} for c in _COLLECTIONS}
        self.meta: Dict[str, Any] = {}
        self._load()

    # -- persistence

    def _path(self) -> Path:
        return self.base_dir / "store.json"

    def _load(self) -> None:
        path = self._path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception:
            return
        with self.lock:
            for name, model_cls in _COLLECTIONS.items():
                items = raw.get(name, {})
                parsed: Dict[str, BaseModel] = {}
                for key, payload in items.items():
                    try:
                        parsed[key] = model_cls.model_validate(payload)
                    except Exception:
                        continue
                self.collections[name] = parsed
            self.meta = raw.get("_meta", {})

    def save(self) -> None:
        with self.lock:
            payload = {
                name: {k: v.model_dump() for k, v in items.items()}
                for name, items in self.collections.items()
            }
            payload["_meta"] = self.meta
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path().with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(self._path())

    # -- generic access

    def put(self, collection: str, item: BaseModel) -> None:
        key = getattr(item, _KEY_FIELDS[collection])
        with self.lock:
            self.collections[collection][key] = item

    def get(self, collection: str, key: str) -> Optional[Any]:
        with self.lock:
            return self.collections[collection].get(key)

    def all(self, collection: str) -> List[Any]:
        with self.lock:
            return list(self.collections[collection].values())

    def where(self, collection: str, **filters: Any) -> List[Any]:
        with self.lock:
            items = list(self.collections[collection].values())
        out = []
        for it in items:
            if all(getattr(it, k, None) == v for k, v in filters.items()):
                out.append(it)
        return out

    def delete(self, collection: str, key: str) -> None:
        with self.lock:
            self.collections[collection].pop(key, None)

    def clear(self, collection: str) -> None:
        with self.lock:
            self.collections[collection] = {}

    # -- audit helper

    def audit(self, action: str, entity_type: str, entity_id: str, detail: str = "", actor: str = "system") -> None:
        evt = AuditEvent(
            event_id=new_id("audit"),
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            detail=detail,
        )
        self.put("audit_events", evt)


_STORE: Optional[EvalStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> EvalStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = EvalStore()
        return _STORE


def reset_store(base_dir: Optional[Path] = None) -> EvalStore:
    """Used by tests to get a fresh isolated store."""
    global _STORE
    with _STORE_LOCK:
        _STORE = EvalStore(base_dir or STORE_DIR)
        return _STORE
