"""Production-grade data contracts with mandatory provenance.

The audit (F-002, F-004, F-014: mock-presented-as-real; label-provenance
conflation) found that the platform blurs MODEL PREDICTION vs AUTO-LABEL vs
VLM output vs HUMAN label vs CERTIFIED ground truth in several paths. These
contracts make provenance a first-class, mandatory field so downstream code
can refuse inputs whose provenance is insufficient (e.g. a quality gate must
never benchmark against AUTO_LABEL "ground truth").

Adoption path (existing packages are NOT rewritten now):
1. New code (hardening, future packages) imports these contracts directly.
2. Existing records are mapped through the `adapt_*` helpers below at package
   boundaries; the adapters make the implicit provenance of each legacy record
   type explicit.
3. At each package's next breaking release, its internal records inherit from
   or are replaced by these contracts (tracked in the remediation plan).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LabelProvenance(str, Enum):
    """Where a label came from, ordered roughly by trust.

    The distinction the legacy code blurs: a MODEL_PREDICTION is a hypothesis
    to be evaluated; CERTIFIED_GROUND_TRUTH is calibrated, human-audited truth
    suitable for gating launches. Nothing in between may silently substitute
    for the level above it.
    """

    MODEL_PREDICTION = "MODEL_PREDICTION"
    AUTO_LABEL = "AUTO_LABEL"
    VLM_INFERENCE = "VLM_INFERENCE"
    HUMAN_LABEL = "HUMAN_LABEL"
    CERTIFIED_GROUND_TRUTH = "CERTIFIED_GROUND_TRUTH"


#: Trust ordering for gate checks (higher = more trustworthy).
PROVENANCE_RANK: Dict[LabelProvenance, int] = {
    LabelProvenance.MODEL_PREDICTION: 0,
    LabelProvenance.AUTO_LABEL: 1,
    LabelProvenance.VLM_INFERENCE: 1,
    LabelProvenance.HUMAN_LABEL: 3,
    LabelProvenance.CERTIFIED_GROUND_TRUTH: 4,
}


class DataOrigin(str, Enum):
    """Whether the underlying sensor data is real or simulated."""

    REAL_SENSOR = "REAL_SENSOR"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC_FALLBACK = "SYNTHETIC_FALLBACK"  # substituted when real data missing
    REPLAYED = "REPLAYED"


class Provenance(BaseModel):
    """Common provenance block attached to every contract."""

    label_provenance: LabelProvenance
    data_origin: DataOrigin = DataOrigin.REAL_SENSOR
    source_system: str = ""            # e.g. "perception_automator", "vendor:acme"
    model_version: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    lineage_ids: List[str] = Field(default_factory=list)  # parent record ids
    notes: str = ""


# ------------------------------------------------------------------ sensor / scene


class SensorFrame(BaseModel):
    frame_id: str
    sequence_id: str
    dataset_version: str
    timestamp_us: int
    data_origin: DataOrigin
    sensor_uris: Dict[str, str] = Field(default_factory=dict)  # sensor name -> URI
    calibration_version: str = ""
    ego_speed_mps: float = 0.0
    weather: str = "unknown"
    lighting: str = "unknown"


class ScenarioEvidence(BaseModel):
    """One piece of evidence supporting a scenario classification — kept
    decomposed so routing/review can explain WHY a scenario was flagged."""

    kind: str                      # e.g. "ttc_conflict", "novelty_knn", "detector_anomaly"
    value: float
    threshold: Optional[float] = None
    frame_ids: List[str] = Field(default_factory=list)
    detail: Dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    scenario_id: str
    dataset_version: str
    scenario_type: str
    stratum: Optional[str] = None          # sampling stratum (see sampling.py)
    frame_ids: List[str] = Field(default_factory=list)
    evidence: List[ScenarioEvidence] = Field(default_factory=list)
    provenance: Provenance
    sampling_weight: float = 1.0           # inverse-inclusion-probability weight


# ------------------------------------------------------------------ objects / tracks


class ObjectProposal(BaseModel):
    proposal_id: str
    frame_id: str
    class_name: str
    bbox_3d: List[float]                   # [x, y, z, l, w, h, yaw]
    confidence: float = 0.0
    provenance: Provenance
    mask_ref: Optional[str] = None


class Track(BaseModel):
    track_id: str
    sequence_id: str
    class_name: str
    frame_ids: List[str] = Field(default_factory=list)
    proposal_ids: List[str] = Field(default_factory=list)
    provenance: Provenance


class GroundTruthObject(BaseModel):
    gt_id: str
    frame_id: str
    class_name: str
    bbox_3d: List[float]
    instance_id: str = ""
    provenance: Provenance                 # must be HUMAN_LABEL or CERTIFIED_GROUND_TRUTH
    certified: bool = False

    def is_gate_grade(self) -> bool:
        """True iff this object may back a launch/quality gate."""
        return PROVENANCE_RANK[self.provenance.label_provenance] >= \
            PROVENANCE_RANK[LabelProvenance.HUMAN_LABEL]


# ------------------------------------------------------------------ decisions / results


class ValidationResult(BaseModel):
    target_id: str
    checks: List[Dict[str, Any]] = Field(default_factory=list)  # name/passed/value/threshold
    passed: bool
    provenance: Provenance


class SafetyAssessment(BaseModel):
    target_id: str
    measures: Dict[str, Optional[float]] = Field(default_factory=dict)  # ttc/pet/drac...
    severity: str = "none"
    threshold_config_version: str = ""     # from hardening.safety_config
    simulated: bool = False
    provenance: Provenance


class TriageDecision(BaseModel):
    target_id: str
    decision: str                          # ACCEPT | HITL | QUARANTINE
    explanation: List[str] = Field(default_factory=list)
    evidence: Dict[str, float] = Field(default_factory=dict)
    priority: float = 0.0
    provenance: Provenance


class EvaluationResult(BaseModel):
    run_id: str
    dataset_version: str
    model_version: str
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    sample_sizes: Dict[str, int] = Field(default_factory=dict)  # metric -> n
    seed: Optional[int] = None
    provenance: Provenance


class RegressionResult(BaseModel):
    run_id: str
    baseline_run_id: str
    metric_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    regression_detected: bool
    method: str = "newcombe_ci"            # or "seqeval_eprocess"
    provenance: Provenance


# ------------------------------------------------------------------ versions / cache


class DatasetVersion(BaseModel):
    dataset_version: str
    parent_version: Optional[str] = None
    label_provenance: LabelProvenance
    data_origin: DataOrigin
    num_frames: int = 0
    content_hash: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    split_assignments: Dict[str, str] = Field(default_factory=dict)  # unit -> train/val/test


class ModelVersion(BaseModel):
    model_version: str
    training_dataset_version: Optional[str] = None
    weights_hash: str = ""
    config_hash: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class FeatureCacheEntry(BaseModel):
    """See cache_manifest.py for the full key derivation."""

    cache_key: str
    manifest: Dict[str, str] = Field(default_factory=dict)  # dependency -> version/hash
    checksum: str = ""
    size_bytes: int = 0
    created_at: str = Field(default_factory=utc_now_iso)
    last_accessed_at: str = Field(default_factory=utc_now_iso)


# ------------------------------------------------------------------ adapters
#
# Map existing packages' records into these contracts, making implicit
# provenance explicit. Read-only: they never mutate the source records.


def adapt_labeleval_annotation(ann: Any) -> ObjectProposal:
    """evaluation.records.Annotation -> ObjectProposal.

    Annotations in labeleval are auto-generated hypotheses (their docstring
    says so), so provenance is AUTO_LABEL unless a human review verified them.
    """
    verified = getattr(ann, "status", "") == "VERIFIED"
    return ObjectProposal(
        proposal_id=ann.annotation_id,
        frame_id=ann.frame_id,
        class_name=ann.class_name,
        bbox_3d=list(ann.bbox_3d or [0.0] * 7),
        confidence=float(ann.confidence),
        mask_ref=getattr(ann, "mask", None),
        provenance=Provenance(
            label_provenance=(LabelProvenance.HUMAN_LABEL if verified
                              else LabelProvenance.AUTO_LABEL),
            data_origin=DataOrigin.SYNTHETIC,  # labeleval datasets are synthetic
            source_system="evaluation.records.Annotation",
            model_version=getattr(ann, "model_version", None),
            lineage_ids=[ann.dataset_id],
        ),
    )


_GT_TYPE_TO_PROVENANCE: Dict[str, LabelProvenance] = {
    "PSEUDO_GROUND_TRUTH": LabelProvenance.AUTO_LABEL,
    "VENDOR_GROUND_TRUTH": LabelProvenance.HUMAN_LABEL,
    "HUMAN_VERIFIED_GROUND_TRUTH": LabelProvenance.HUMAN_LABEL,
    "GOLD_STANDARD": LabelProvenance.CERTIFIED_GROUND_TRUTH,
}


def adapt_labeleval_gt_box(box: Any, frame_id: str = "",
                           data_origin: DataOrigin = DataOrigin.SYNTHETIC) -> GroundTruthObject:
    """evaluation.records.GroundTruthBox -> GroundTruthObject.

    Maps the 4-level labeleval GroundTruthType onto LabelProvenance. Note
    PSEUDO_GROUND_TRUTH maps to AUTO_LABEL — is_gate_grade() correctly
    refuses it for launch gates.
    """
    gt_type = str(getattr(box, "gt_type", "VENDOR_GROUND_TRUTH"))
    prov = _GT_TYPE_TO_PROVENANCE.get(gt_type, LabelProvenance.AUTO_LABEL)
    return GroundTruthObject(
        gt_id=box.gt_id,
        frame_id=frame_id,
        class_name=box.class_name,
        bbox_3d=list(box.bbox_3d),
        instance_id=getattr(box, "track_instance_id", ""),
        certified=prov == LabelProvenance.CERTIFIED_GROUND_TRUTH,
        provenance=Provenance(
            label_provenance=prov,
            data_origin=data_origin,
            source_system="evaluation.records.GroundTruthBox",
        ),
    )


def adapt_pipeline_proposal(obj: Any, frame_id: str, used_gt_fallback: bool) -> ObjectProposal:
    """sensorflow.schemas.unified_frame.Object3D -> ObjectProposal.

    `used_gt_fallback` MUST be True when the legacy perception pipeline built
    the proposal from GT-derived fallback masks (audit F-004) — those
    proposals are AUTO_LABELs contaminated by GT and must never be evaluated
    against the same GT.
    """
    return ObjectProposal(
        proposal_id=f"{frame_id}:{getattr(obj, 'track_id', '') or id(obj)}",
        frame_id=frame_id,
        class_name=obj.class_name,
        bbox_3d=list(obj.bbox_3d),
        confidence=float(obj.confidence),
        mask_ref=getattr(obj, "sam_mask_ref", None),
        provenance=Provenance(
            label_provenance=LabelProvenance.AUTO_LABEL,
            data_origin=DataOrigin.REAL_SENSOR,
            source_system="perception_automator",
            notes="GT_CONTAMINATED: built from ground-truth fallback masks"
            if used_gt_fallback else "",
        ),
    )


def content_hash(payload: Any) -> str:
    """Stable content hash for contract payloads (sorted-key JSON, sha256)."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
