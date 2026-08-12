"""Strict Quality Validation Engine: deterministic geometric plausibility checks.

Compares LiDAR point cloud vs bounding-box proposal (spec §8) and computes
per-3D-box metrics with configurable tolerances (spec §9): 3D IoU, position
error, orientation error, dimension error, point density, point-in-box ratio,
ground contact error, plus sensor-consistency checks.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from sensorflow.evaluation.records import (
    Annotation,
    CheckLine,
    EvalStore,
    Frame,
    ValidationResult,
)
from sensorflow.evaluation.synthetic import (
    CLASS_DIMS,
    frame_points,
    points_in_box,
    project_bbox_2d,
)
from sensorflow.metrics.perception_3d import bev_iou

DEFAULT_TOLERANCES: Dict[str, float] = {
    "min_points_in_box": 12,
    "min_point_density": 0.4,       # points per m^3
    "min_box_occupancy": 0.05,      # fraction of nearby points captured
    "max_dim_ratio": 1.6,           # vs class prior
    "min_dim_ratio": 0.55,
    "max_centroid_offset_m": 1.2,   # points centroid vs box center (BEV)
    "max_ground_contact_error_m": 0.5,
    "max_orientation_track_dev_deg": 45.0,
    "min_iou_3d": 0.5,              # plausibility floor (triage gate is stricter)
    "max_position_error_m": 1.0,
    "max_orientation_error_deg": 30.0,
    "max_dimension_error": 0.5,
}


def iou_3d(box_a: List[float], box_b: List[float]) -> float:
    """BEV IoU extended with vertical overlap."""
    biou = bev_iou(box_a, box_b)
    za0, za1 = box_a[2] - box_a[5] / 2, box_a[2] + box_a[5] / 2
    zb0, zb1 = box_b[2] - box_b[5] / 2, box_b[2] + box_b[5] / 2
    inter_h = max(0.0, min(za1, zb1) - max(za0, zb0))
    union_h = max(za1, zb1) - min(za0, zb0)
    if union_h <= 0:
        return 0.0
    return biou * (inter_h / union_h) * (union_h / max(box_a[5], box_b[5]))  # approx volumetric


def validate_annotation(
    store: EvalStore,
    ann: Annotation,
    frame: Frame,
    points: Optional[np.ndarray] = None,
    tolerances: Optional[Dict[str, float]] = None,
) -> ValidationResult:
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)

    checks: List[CheckLine] = []
    result = ValidationResult(annotation_id=ann.annotation_id)

    if not ann.bbox_3d:
        result.passed = False
        result.checks = [CheckLine(gate="has_bbox_3d", actual=False, threshold=True, passed=False)]
        store.put("validations", result)
        return result

    if points is None:
        points = frame_points(store, frame)

    x, y, z, l, w, h, yaw = ann.bbox_3d
    volume = max(l * w * h, 1e-3)

    # --- point support
    mask = points_in_box(points, ann.bbox_3d)
    n_in = int(mask.sum())
    density = n_in / volume
    result.point_density = round(density, 3)

    checks.append(CheckLine(gate="points_in_box", actual=n_in,
                            threshold=tol["min_points_in_box"],
                            passed=n_in >= tol["min_points_in_box"]))
    checks.append(CheckLine(gate="point_density_per_m3", actual=round(density, 2),
                            threshold=tol["min_point_density"],
                            passed=density >= tol["min_point_density"]))

    # --- box occupancy: fraction of points near the box that fall inside it
    near = points[(np.abs(points[:, 0] - x) < max(l, 4)) & (np.abs(points[:, 1] - y) < max(w, 4))]
    above_ground = near[near[:, 2] > 0.15] if len(near) else near
    occupancy = n_in / max(len(above_ground), 1)
    result.point_in_box_ratio = round(min(occupancy, 1.0), 4)
    checks.append(CheckLine(gate="box_occupancy", actual=round(occupancy, 3),
                            threshold=tol["min_box_occupancy"],
                            passed=occupancy >= tol["min_box_occupancy"]))

    # --- dimension plausibility vs class prior
    exp = CLASS_DIMS.get(ann.class_name)
    if exp:
        ratio = float(np.mean([l / exp[0], w / exp[1], h / exp[2]]))
        ok = tol["min_dim_ratio"] <= ratio <= tol["max_dim_ratio"]
        checks.append(CheckLine(gate="box_dimensions_vs_class_prior", actual=round(ratio, 3),
                                threshold=f"[{tol['min_dim_ratio']}, {tol['max_dim_ratio']}]", passed=ok))
    else:
        checks.append(CheckLine(gate="box_dimensions_vs_class_prior", actual="unknown_class",
                                threshold="known class", passed=True, applicable=False))

    # --- centroid consistency: LiDAR centroid vs box center
    if n_in >= 3:
        centroid = points[mask].mean(axis=0)
        offset = math.hypot(centroid[0] - x, centroid[1] - y)
        checks.append(CheckLine(gate="centroid_consistency_m", actual=round(offset, 3),
                                threshold=tol["max_centroid_offset_m"],
                                passed=offset <= tol["max_centroid_offset_m"]))
    else:
        checks.append(CheckLine(gate="centroid_consistency_m", actual="insufficient points",
                                threshold=tol["max_centroid_offset_m"], passed=False, applicable=n_in > 0))

    # --- object-to-ground relationship
    ground_err = abs((z - h / 2) - 0.0)
    result.ground_contact_error = round(ground_err, 3)
    checks.append(CheckLine(gate="ground_contact_error_m", actual=round(ground_err, 3),
                            threshold=tol["max_ground_contact_error_m"],
                            passed=ground_err <= tol["max_ground_contact_error_m"]))

    # --- sensor consistency: LiDAR box should have a camera projection when
    #     visible; a projectable box missing a 2D detection is a disagreement.
    expected_2d = project_bbox_2d(ann.bbox_3d)
    sensor_ok = not (expected_2d is not None and ann.bbox_2d is None)
    result.sensor_consistent = sensor_ok
    checks.append(CheckLine(gate="sensor_consistency_cam_lidar",
                            actual="2d_present" if ann.bbox_2d else "2d_missing",
                            threshold="camera+lidar agree", passed=sensor_ok,
                            applicable=expected_2d is not None))

    # --- reference comparison (only when a reference ground truth exists)
    gt = next((g for g in frame.gt_boxes if g.gt_id == ann.matched_gt_id), None)
    if gt is not None:
        iou = iou_3d(ann.bbox_3d, gt.bbox_3d)
        pos_err = math.hypot(x - gt.bbox_3d[0], y - gt.bbox_3d[1])
        yaw_err = abs(yaw - gt.bbox_3d[6]) % (2 * math.pi)
        yaw_err = min(yaw_err, 2 * math.pi - yaw_err)
        yaw_err_deg = math.degrees(yaw_err)
        dim_err = float(np.mean([abs(ann.bbox_3d[i] - gt.bbox_3d[i]) / max(gt.bbox_3d[i], 0.1) for i in (3, 4, 5)]))

        result.iou_3d = round(iou, 4)
        result.position_error = round(pos_err, 3)
        result.orientation_error_deg = round(yaw_err_deg, 2)
        result.dimension_error = round(dim_err, 4)

        checks.append(CheckLine(gate="iou_3d_vs_reference", actual=round(iou, 3),
                                threshold=tol["min_iou_3d"], passed=iou >= tol["min_iou_3d"]))
        checks.append(CheckLine(gate="position_error_m", actual=round(pos_err, 3),
                                threshold=tol["max_position_error_m"],
                                passed=pos_err <= tol["max_position_error_m"]))
        checks.append(CheckLine(gate="orientation_error_deg", actual=round(yaw_err_deg, 1),
                                threshold=tol["max_orientation_error_deg"],
                                passed=yaw_err_deg <= tol["max_orientation_error_deg"]))
        checks.append(CheckLine(gate="dimension_error", actual=round(dim_err, 3),
                                threshold=tol["max_dimension_error"],
                                passed=dim_err <= tol["max_dimension_error"]))
    else:
        checks.append(CheckLine(gate="iou_3d_vs_reference", actual="no reference GT",
                                threshold=tol["min_iou_3d"], passed=True, applicable=False))

    result.checks = checks
    result.passed = all(c.passed for c in checks if c.applicable)
    store.put("validations", result)
    return result
