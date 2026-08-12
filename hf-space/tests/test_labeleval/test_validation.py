"""Geometric validation engine tests."""

import math
from pathlib import Path

import pytest

from sensorflow.evaluation import synthetic
from sensorflow.evaluation.records import Annotation, reset_store
from sensorflow.evaluation.validation import iou_3d, validate_annotation


@pytest.fixture()
def env(tmp_path):
    store = reset_store(tmp_path)
    ds = synthetic.generate_dataset(store, num_sequences=1, frames_per_sequence=5, seed=11)
    frames = sorted(store.where("frames", dataset_id=ds.dataset_id), key=lambda f: f.index)
    frame = next(f for f in frames if f.gt_boxes and "sensor_failure" not in f.scenario_tags)
    return store, ds, frame


def _ann(ds, frame, bbox, cls, gt_id=None, bbox_2d="auto", conf=0.9):
    return Annotation(
        annotation_id="ann-test-1",
        dataset_id=ds.dataset_id,
        frame_id=frame.frame_id,
        object_id="obj",
        class_name=cls,
        confidence=conf,
        bbox_3d=bbox,
        bbox_2d=synthetic.project_bbox_2d(bbox) if bbox_2d == "auto" else bbox_2d,
        matched_gt_id=gt_id,
    )


def test_iou_3d_identity():
    box = [10, 2, 0.8, 4.5, 1.9, 1.6, 0.3]
    assert iou_3d(box, box) == pytest.approx(1.0, abs=1e-6)


def test_iou_3d_disjoint():
    a = [10, 2, 0.8, 4.5, 1.9, 1.6, 0.0]
    b = [40, -10, 0.8, 4.5, 1.9, 1.6, 0.0]
    assert iou_3d(a, b) == 0.0


def test_accurate_box_passes_all_checks(env):
    store, ds, frame = env
    gt = max(frame.gt_boxes, key=lambda g: g.bbox_3d[3] * g.bbox_3d[4])  # largest object
    ann = _ann(ds, frame, list(gt.bbox_3d), gt.class_name, gt.gt_id)
    res = validate_annotation(store, ann, frame)
    assert res.passed, [c.model_dump() for c in res.checks if not c.passed]
    assert res.iou_3d == pytest.approx(1.0, abs=1e-4)
    assert res.position_error == pytest.approx(0.0, abs=1e-6)


def test_box_in_empty_space_fails_point_support(env):
    store, ds, frame = env
    bbox = [55.0, 18.0, 0.8, 4.5, 1.9, 1.6, 0.0]  # far corner, no object there
    ann = _ann(ds, frame, bbox, "vehicle")
    res = validate_annotation(store, ann, frame)
    failing = {c.gate for c in res.checks if c.applicable and not c.passed}
    assert not res.passed
    assert {"points_in_box", "point_density_per_m3"} & failing


def test_implausible_dimensions_fail_class_prior(env):
    store, ds, frame = env
    gt = frame.gt_boxes[0]
    x, y, z, l, w, h, yaw = gt.bbox_3d
    ann = _ann(ds, frame, [x, y, z, l * 2.5, w * 2.5, h, yaw], gt.class_name, gt.gt_id)
    res = validate_annotation(store, ann, frame)
    failing = {c.gate for c in res.checks if c.applicable and not c.passed}
    assert "box_dimensions_vs_class_prior" in failing


def test_floating_box_fails_ground_contact(env):
    store, ds, frame = env
    gt = frame.gt_boxes[0]
    x, y, z, l, w, h, yaw = gt.bbox_3d
    ann = _ann(ds, frame, [x, y, z + 2.0, l, w, h, yaw], gt.class_name, gt.gt_id)
    res = validate_annotation(store, ann, frame)
    failing = {c.gate for c in res.checks if c.applicable and not c.passed}
    assert "ground_contact_error_m" in failing


def test_missing_camera_detection_fails_sensor_consistency(env):
    store, ds, frame = env
    gt = next((g for g in frame.gt_boxes if synthetic.project_bbox_2d(g.bbox_3d)), None)
    if gt is None:
        pytest.skip("no projectable GT in this frame")
    ann = _ann(ds, frame, list(gt.bbox_3d), gt.class_name, gt.gt_id, bbox_2d=None)
    res = validate_annotation(store, ann, frame)
    assert not res.sensor_consistent
    failing = {c.gate for c in res.checks if c.applicable and not c.passed}
    assert "sensor_consistency_cam_lidar" in failing


def test_wrong_orientation_measured_against_reference(env):
    store, ds, frame = env
    gt = max(frame.gt_boxes, key=lambda g: g.bbox_3d[3])
    x, y, z, l, w, h, yaw = gt.bbox_3d
    ann = _ann(ds, frame, [x, y, z, l, w, h, yaw + math.radians(60)], gt.class_name, gt.gt_id)
    res = validate_annotation(store, ann, frame)
    assert res.orientation_error_deg == pytest.approx(60.0, abs=1.0)
    failing = {c.gate for c in res.checks if c.applicable and not c.passed}
    assert "orientation_error_deg" in failing
