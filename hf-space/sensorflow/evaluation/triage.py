"""Strict, deterministic, configurable triage gate (spec §19-§21).

Combines ML anomaly score, regression status, grader consensus, geometric
validation and track validation under a quality policy. Auto-release only when
ALL applicable gates pass. Every decision stores the policy id + values and
per-gate lines with actual vs threshold (explainability, spec §39).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.evaluation.records import (
    Annotation,
    AnomalyDetection,
    CheckLine,
    EvalStore,
    GraderComparison,
    TrackingEvidence,
    TriageDecision,
    ValidationResult,
    new_id,
)


class QualityPolicy(BaseModel):
    policy_id: str = "quality-policy-v1"
    min_iou_3d: float = 0.80
    max_position_error_m: float = 0.50
    max_orientation_error_deg: float = 10.0
    min_points_in_box: int = 12
    min_consensus: float = 0.90
    max_anomaly_score: float = 0.90
    min_track_quality: float = 0.60
    min_confidence: float = 0.40
    require_sensor_consistency: bool = True
    require_geometric_validation: bool = True
    block_on_model_regression: bool = False

    def values(self) -> Dict[str, float]:
        return {k: (float(v) if not isinstance(v, bool) else float(v))
                for k, v in self.model_dump().items() if k != "policy_id"}


POLICY_PATH = Path("runs/labeleval/quality_policy.json")


def load_policy(path: Path = POLICY_PATH) -> QualityPolicy:
    if path.exists():
        try:
            with open(path) as f:
                return QualityPolicy.model_validate(json.load(f))
        except Exception:
            pass
    return QualityPolicy()


def save_policy(policy: QualityPolicy, path: Path = POLICY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy.model_dump(), f, indent=2)


# Gate failure -> failure reason mapping (spec §21).
def triage_annotation(
    store: EvalStore,
    ann: Annotation,
    validation: Optional[ValidationResult],
    anomaly: Optional[AnomalyDetection],
    grading: Optional[GraderComparison],
    tracking: Optional[TrackingEvidence],
    policy: QualityPolicy,
    model_regressed: bool = False,
) -> TriageDecision:
    lines: List[CheckLine] = []
    reasons: List[str] = []

    def gate(name: str, actual, threshold, passed: bool, applicable: bool = True, reason: Optional[str] = None):
        lines.append(CheckLine(gate=name, actual=actual, threshold=threshold,
                               passed=passed or not applicable, applicable=applicable))
        if applicable and not passed and reason and reason not in reasons:
            reasons.append(reason)

    # --- geometric gates (only applicable when a reference exists)
    has_ref = validation is not None and validation.iou_3d is not None
    gate("iou_3d", validation.iou_3d if has_ref else "no reference", policy.min_iou_3d,
         bool(has_ref and validation.iou_3d >= policy.min_iou_3d), applicable=has_ref, reason="LOW_IOU")
    gate("position_error_m", validation.position_error if has_ref else "no reference",
         policy.max_position_error_m,
         bool(has_ref and validation.position_error <= policy.max_position_error_m),
         applicable=has_ref, reason="POSITION_ERROR")
    gate("orientation_error_deg", validation.orientation_error_deg if has_ref else "no reference",
         policy.max_orientation_error_deg,
         bool(has_ref and validation.orientation_error_deg <= policy.max_orientation_error_deg),
         applicable=has_ref, reason="ORIENTATION_ERROR")

    # Unmatched labels (no reference GT at all) are inherently suspect: the
    # independent evidence must carry them; flag as anomaly-checked FP risk.
    if ann.matched_gt_id is None:
        gate("has_reference_gt", False, True, False, applicable=True, reason="ANOMALY")

    # --- point support
    n_pts = None
    if validation is not None:
        pts_line = next((c for c in validation.checks if c.gate == "points_in_box"), None)
        n_pts = pts_line.actual if pts_line else None
    gate("points_in_box", n_pts, policy.min_points_in_box,
         bool(isinstance(n_pts, (int, float)) and n_pts >= policy.min_points_in_box),
         applicable=n_pts is not None, reason="INSUFFICIENT_POINT_SUPPORT")

    # --- geometric validation verdict; map each failing check to its reason
    CHECK_REASONS = {
        "points_in_box": "INSUFFICIENT_POINT_SUPPORT",
        "point_density_per_m3": "INSUFFICIENT_POINT_SUPPORT",
        "box_occupancy": "INSUFFICIENT_POINT_SUPPORT",
        "iou_3d_vs_reference": "LOW_IOU",
        "position_error_m": "POSITION_ERROR",
        "centroid_consistency_m": "POSITION_ERROR",
        "ground_contact_error_m": "POSITION_ERROR",
        "orientation_error_deg": "ORIENTATION_ERROR",
        "box_dimensions_vs_class_prior": "LOW_IOU",
        "dimension_error": "LOW_IOU",
        "sensor_consistency_cam_lidar": "SENSOR_DISAGREEMENT",
    }
    if validation is not None and not validation.passed:
        for c in validation.checks:
            if c.applicable and not c.passed:
                r = CHECK_REASONS.get(c.gate)
                if r and r not in reasons:
                    reasons.append(r)
    gate("geometric_validation", "passed" if (validation and validation.passed) else "failed",
         "passed", bool(validation and validation.passed),
         applicable=policy.require_geometric_validation, reason=None)

    # --- sensor consistency
    gate("sensor_consistency", "consistent" if (validation is None or validation.sensor_consistent) else "disagreement",
         "consistent", bool(validation is None or validation.sensor_consistent),
         applicable=policy.require_sensor_consistency and validation is not None, reason="SENSOR_DISAGREEMENT")

    # --- anomaly
    gate("anomaly_score", round(anomaly.score, 4) if anomaly else None, policy.max_anomaly_score,
         bool(anomaly is None or not anomaly.is_anomaly),
         applicable=anomaly is not None, reason="ANOMALY")

    # --- grader consensus
    gate("grader_consensus", grading.consensus if grading else None, policy.min_consensus,
         bool(grading is not None and grading.consensus is not None and grading.consensus >= policy.min_consensus),
         applicable=grading is not None, reason="GRADER_DISAGREEMENT")

    # --- tracking
    if tracking is not None:
        gate("id_switch", tracking.id_switch, False, not tracking.id_switch, reason="ID_SWITCH")
        gate("track_fragmentation", tracking.fragmentation, False, not tracking.fragmentation,
             reason="TRACK_FRAGMENTATION")
        gate("track_quality", tracking.track_quality, policy.min_track_quality,
             bool(tracking.track_quality is None or tracking.track_quality >= policy.min_track_quality),
             applicable=tracking.track_quality is not None, reason="TRACK_FRAGMENTATION")

    # --- confidence
    gate("detection_confidence", ann.confidence, policy.min_confidence,
         ann.confidence >= policy.min_confidence, reason="LOW_CONFIDENCE")

    # --- model regression
    gate("model_regression", "regressed" if model_regressed else "stable", "stable",
         not model_regressed, applicable=policy.block_on_model_regression, reason="MODEL_REGRESSION")

    all_pass = all(l.passed for l in lines if l.applicable)
    status = "AUTO_GRADED" if all_pass else "FLAGGED"

    decision = TriageDecision(
        decision_id=new_id("dec"),
        annotation_id=ann.annotation_id,
        status=status,
        failure_reasons=reasons,
        primary_failure_reason=reasons[0] if reasons else None,
        gate_lines=lines,
        policy_id=policy.policy_id,
        policy_values=policy.values(),
    )
    store.put("triage_decisions", decision)
    return decision
