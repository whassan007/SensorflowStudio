"""Pydantic data model + JSON-backed store for the rare-event miner.

Follows the record/store pattern of sensorflow.evaluation.records (thread-safe
in-memory collections with JSON persistence under runs/raremine/), and the
lineage/report shapes of sensorflow.megaeval.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

STORE_DIR = Path("runs/raremine")

# ------------------------------------------------------------------ vocabularies

COSTUME_TYPES = [
    "mascot",
    "inflatable",
    "animal",
    "character",
    "robot_armor",
    "oversized",
    "large_prop",
]

CONFOUNDER_TYPES = [
    "mascot_statue",
    "inflatable_decoration",
    "roadside_advertisement",
    "mannequin",
    "sign",
    "construction_object",
]

MODALITIES = [
    "rgb",
    "multi_camera",
    "lidar_projection",
    "lidar_intensity",
    "point_cloud",
    "fusion_view",
    "temporal_sequence",
    "baseline_predictions",
    "gt_annotations",
]

# Modalities that may carry visual/sensor evidence (predictions & GT are data
# availability flags, not evidence sources for the proposer).
EVIDENCE_MODALITIES = [
    "rgb",
    "multi_camera",
    "lidar_projection",
    "lidar_intensity",
    "point_cloud",
    "fusion_view",
    "temporal_sequence",
]

SilhouetteDeviation = Literal["LOW", "MODERATE", "HIGH", "EXTREME"]
OcclusionLevel = Literal["NONE", "PARTIAL", "HEAVY", "EXTREME"]
OcclusionSource = Literal["NONE", "COSTUME_INDUCED", "ENVIRONMENTAL", "COMBINED"]
Difficulty = Literal["EASY", "MODERATE", "HARD", "EXTREME"]
Priority = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
EvidenceQuality = Literal["LOW", "MEDIUM", "HIGH"]

LIGHTING = ["day", "night"]
WEATHER = ["clear", "rain", "fog"]
CONTEXTS = ["sidewalk", "crosswalk", "road_edge", "event_area"]
SAFETY_CONTEXTS = {"crosswalk", "road_edge"}

FAILURE_MODES = [
    "FALSE_NEGATIVE",
    "MISCLASSIFICATION_VEHICLE",
    "MISCLASSIFICATION_ANIMAL",
    "MISCLASSIFICATION_BACKGROUND",
    "LOCALIZATION_ERROR",
    "LOW_CONFIDENCE",
    "TRACKING_FAILURE",
]

DESTINATIONS = [
    "NO_ACTION",
    "REVIEW_QUEUE",
    "RARE_EVENT_DATASET",
    "HARD_EXAMPLE_DATASET",
    "REGRESSION_EVALUATION_SET",
    "SAFETY_CRITICAL_EVALUATION_SET",
    "TRAINING_CANDIDATE",
]

# Anything landing in these sets is protected: leakage into training requires
# an explicit, audited governance override.
PROTECTED_EVAL_DESTINATIONS = {
    "REGRESSION_EVALUATION_SET",
    "SAFETY_CRITICAL_EVALUATION_SET",
}

STAGES = [
    "RAW",
    "MINED",
    "DEDUPLICATED",
    "AUTO_VALIDATED",
    "HUMAN_VALIDATED",
    "CURATED",
    "ARCHIVED",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------------ scene bank


class SceneObject(BaseModel):
    """One object in a scene. `truth_*` fields are the planted ground truth
    used ONLY by validation/statistics — the miner consumes `observables`."""

    object_id: str
    track_id: str = ""
    # planted truth (hidden from the miner)
    truth_kind: Literal["normal_pedestrian", "costumed_pedestrian", "confounder"]
    truth_costume_type: Optional[str] = None  # COSTUME_TYPES or CONFOUNDER_TYPES
    truth_silhouette_deviation: SilhouetteDeviation = "LOW"
    truth_occlusion_env: OcclusionLevel = "NONE"
    truth_occlusion_costume: OcclusionLevel = "NONE"
    truth_is_moving: bool = True
    # shared scene-truth attributes (also observable)
    distance_m: float = 15.0
    context: str = "sidewalk"
    # schematic geometry for the UI (BEV metres + normalized 2D bbox)
    position: List[float] = Field(default_factory=lambda: [10.0, 0.0])
    bbox_2d: List[float] = Field(default_factory=lambda: [0.4, 0.4, 0.1, 0.25])
    # per-modality observable features; keys MUST be available scene modalities
    observables: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class BaselinePrediction(BaseModel):
    prediction_id: str
    object_id: Optional[str] = None  # None => spurious detection
    predicted_class: str = "pedestrian"  # vehicle | animal | background | none
    confidence: float = 0.9
    # offset between predicted box and the sensed object cluster (observable)
    localization_offset_m: float = 0.1
    track_stable: bool = True
    # truth-only fields, used by quantitative validation — never by the miner
    iou_with_gt: Optional[float] = None
    planted_failure: Optional[str] = None


class GtBox(BaseModel):
    object_id: str
    class_name: str = "pedestrian"
    is_costumed: bool = False
    costume_type: Optional[str] = None


class Scene(BaseModel):
    scene_id: str
    bank_id: str
    sequence_id: str
    frame_index: int = 0
    lighting: str = "day"
    weather: str = "clear"
    modalities: Dict[str, bool] = Field(default_factory=dict)
    objects: List[SceneObject] = Field(default_factory=list)
    baseline_predictions: List[BaselinePrediction] = Field(default_factory=list)
    gt_boxes: List[GtBox] = Field(default_factory=list)
    near_duplicate_of: Optional[str] = None  # scene_id of the original frame


class SceneBank(BaseModel):
    bank_id: str
    seed: int = 7
    created_at: str = Field(default_factory=now_iso)
    num_scenes: int = 0
    num_sequences: int = 0
    num_planted_rare: int = 0  # track-level planted costumed pedestrians
    num_confounders: int = 0
    generation_params: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ candidates


class Evidence(BaseModel):
    modality: str  # one of EVIDENCE_MODALITIES
    description: str


class AlternativeHypothesis(BaseModel):
    hypothesis: str
    confidence: float
    status: Literal["RETAINED", "REJECTED"]
    reason: str


class TemporalValidation(BaseModel):
    available: bool = False
    status: Literal["VALIDATED", "NOT_AVAILABLE", "INCONCLUSIVE"] = "NOT_AVAILABLE"
    evidence: List[str] = Field(default_factory=list)


class ObservedModelBehavior(BaseModel):
    """ONLY populated when baseline predictions were supplied for the scene.
    Strictly separated from `predicted_failure_mode` (the miner's forecast)."""

    failure_observed: bool = False
    failure_modes: List[str] = Field(default_factory=list)
    details: List[str] = Field(default_factory=list)


class Candidate(BaseModel):
    candidate_id: str
    bank_id: str
    scene_id: str
    sequence_id: str
    object_id: str
    track_id: str = ""
    frame_index: int = 0
    run_id: str = ""

    edge_case_detected: bool = False
    insufficient_visual_evidence: bool = False
    event_type: str = "none"  # costumed_pedestrian | none
    costume_type: List[str] = Field(default_factory=list)

    # THREE separate confidences — never collapsed into one number.
    confidence_human_identity: float = 0.0
    confidence_costume: float = 0.0
    confidence_rare_event: float = 0.0

    visual_evidence: List[Evidence] = Field(default_factory=list)
    human_identity_evidence: List[Evidence] = Field(default_factory=list)
    alternative_hypotheses: List[AlternativeHypothesis] = Field(default_factory=list)

    silhouette_deviation: SilhouetteDeviation = "LOW"
    occlusion_level: OcclusionLevel = "NONE"
    occlusion_source: OcclusionSource = "NONE"
    visible_human_features: List[str] = Field(default_factory=list)
    occluded_human_features: List[str] = Field(default_factory=list)
    unusual_geometry: List[str] = Field(default_factory=list)

    temporal_validation: TemporalValidation = Field(default_factory=TemporalValidation)
    location: Dict[str, Any] = Field(default_factory=dict)

    perception_difficulty: Difficulty = "EASY"
    difficulty_evidence: List[str] = Field(default_factory=list)

    observed_model_behavior: Optional[ObservedModelBehavior] = None
    predicted_failure_mode: Optional[str] = None

    curation_priority: Priority = "LOW"
    priority_reason: str = ""
    recommended_dataset_destination: str = "NO_ACTION"
    requires_human_validation: bool = False
    human_validation_reason: str = ""
    evidence_quality: EvidenceQuality = "LOW"
    rank_in_scene: int = 1


class RepresentativeFrames(BaseModel):
    best_evidence: Optional[str] = None  # scene_id
    worst_case: Optional[str] = None
    model_failure: Optional[str] = None
    minimal_set: List[str] = Field(default_factory=list)


class TrackCandidate(BaseModel):
    """One consolidated candidate per (sequence, track): the unit that flows
    through the validation pipeline."""

    track_candidate_id: str
    bank_id: str
    run_id: str = ""
    sequence_id: str
    track_id: str
    object_id: str
    frame_count: int = 1
    duration_frames: int = 1
    frame_candidate_ids: List[str] = Field(default_factory=list)
    representative: Candidate  # highest-evidence frame-level proposal
    representative_frames: RepresentativeFrames = Field(default_factory=RepresentativeFrames)
    max_visibility: EvidenceQuality = "LOW"
    max_difficulty: Difficulty = "EASY"

    # lifecycle
    stage: str = "MINED"
    duplicate_of: Optional[str] = None
    diversity_selected: bool = False
    auto_validation: Optional[Dict[str, Any]] = None
    human_validation: Optional[Dict[str, Any]] = None
    destination: str = "REVIEW_QUEUE"


class LineageRecord(BaseModel):
    lineage_id: str
    track_candidate_id: str
    source_frame_id: str
    source_sequence_id: str
    dataset_version: str = "scenebank-v1"
    curation_timestamp: str = Field(default_factory=now_iso)
    curator: str = "raremine-pipeline"
    validation_status: str = "PENDING"  # PENDING | AUTO_COHERENT | AUTO_INCOHERENT | APPROVED | REJECTED
    training_eligible: bool = False
    evaluation_eligible: bool = False
    protected_evaluation: bool = False
    destination: str = "REVIEW_QUEUE"
    governance_overrides: List[Dict[str, str]] = Field(default_factory=list)


class MiningRun(BaseModel):
    run_id: str
    bank_id: str
    created_at: str = Field(default_factory=now_iso)
    config: Dict[str, Any] = Field(default_factory=dict)
    num_frame_candidates: int = 0
    num_track_candidates: int = 0
    num_duplicates: int = 0
    num_diversity_selected: int = 0


class AuditEvent(BaseModel):
    event_id: str
    timestamp: str = Field(default_factory=now_iso)
    actor: str = "system"
    action: str = ""
    entity_type: str = ""
    entity_id: str = ""
    detail: str = ""


# ------------------------------------------------------------------ store

_COLLECTIONS = {
    "banks": SceneBank,
    "scenes": Scene,
    "runs": MiningRun,
    "candidates": Candidate,
    "track_candidates": TrackCandidate,
    "lineage": LineageRecord,
    "audit_events": AuditEvent,
}

_KEY_FIELDS = {
    "banks": "bank_id",
    "scenes": "scene_id",
    "runs": "run_id",
    "candidates": "candidate_id",
    "track_candidates": "track_candidate_id",
    "lineage": "lineage_id",
    "audit_events": "event_id",
}


class RareMineStore:
    """Thread-safe in-memory store with JSON persistence (EvalStore pattern)."""

    def __init__(self, base_dir: Path = STORE_DIR):
        self.base_dir = Path(base_dir)
        self.lock = threading.RLock()
        self.collections: Dict[str, Dict[str, BaseModel]] = {c: {} for c in _COLLECTIONS}
        self.meta: Dict[str, Any] = {}
        self._load()

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
                parsed: Dict[str, BaseModel] = {}
                for key, payload in raw.get(name, {}).items():
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
        return [it for it in items
                if all(getattr(it, k, None) == v for k, v in filters.items())]

    def clear(self, collection: str) -> None:
        with self.lock:
            self.collections[collection] = {}

    def audit(self, action: str, entity_type: str, entity_id: str,
              detail: str = "", actor: str = "system") -> None:
        self.put("audit_events", AuditEvent(
            event_id=new_id("rmaudit"), actor=actor, action=action,
            entity_type=entity_type, entity_id=entity_id, detail=detail))


_STORE: Optional[RareMineStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> RareMineStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = RareMineStore()
        return _STORE


def reset_store(base_dir: Optional[Path] = None) -> RareMineStore:
    """Used by tests to get a fresh isolated store."""
    global _STORE
    with _STORE_LOCK:
        _STORE = RareMineStore(base_dir or STORE_DIR)
        return _STORE
