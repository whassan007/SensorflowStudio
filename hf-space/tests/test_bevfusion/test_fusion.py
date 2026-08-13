"""BEV fusion behavior: complementarity mechanics with hand-planted evidence."""

import math

from sensorflow.bevfusion.fusion import fuse_frame
from sensorflow.bevfusion.sensors import Detection


def _lidar(x, y, cls="cyclist", cls_conf=0.4, conf=0.9, dims=(0.7, 0.7, 1.75)):
    return Detection(modality="lidar", x=x, y=y, z=dims[2] / 2,
                     cov=[[0.0144, 0.0], [0.0, 0.0144]], dims=list(dims), yaw=0.3,
                     class_name=cls, class_confidence=cls_conf, confidence=conf)


def _camera(x, y, cls="pedestrian", cls_conf=0.9, conf=0.85, dims=(0.7, 0.7, 1.75),
            sigma_along=1.4, sigma_cross=0.12):
    theta = math.atan2(y, x)
    c, s = math.cos(theta), math.sin(theta)
    sa2, sc2 = sigma_along**2, sigma_cross**2
    cov = [[c*c*sa2 + s*s*sc2, c*s*(sa2 - sc2)],
           [c*s*(sa2 - sc2), s*s*sa2 + c*c*sc2]]
    return Detection(modality="camera", x=x, y=y, z=dims[2] / 2, cov=cov,
                     dims=list(dims), yaw=0.3, class_name=cls,
                     class_confidence=cls_conf, confidence=conf)


def test_fusion_recovers_planted_camera_miss_from_lidar_evidence():
    """Camera saw nothing (night/occlusion); LiDAR evidence alone must decode."""
    boxes = fuse_frame(camera_dets=[], lidar_dets=[_lidar(30.0, 2.0)])
    assert len(boxes) == 1
    box = boxes[0]
    assert math.hypot(box["bbox_3d"][0] - 30.0, box["bbox_3d"][1] - 2.0) < 0.5
    assert box["confidence"] >= 0.30


def test_camera_class_flows_to_lidar_anchored_detection():
    """Camera depth is 3 m off but its class evidence smears along the ray and
    labels the LiDAR-anchored cell; geometry snaps to LiDAR (inverse-variance)."""
    true_x, true_y = 20.0, 0.0
    cam = _camera(23.0, 0.0, cls="pedestrian", cls_conf=0.9)   # +3 m range error
    lid = _lidar(true_x, true_y, cls="cyclist", cls_conf=0.4)  # weak wrong class
    boxes = fuse_frame([cam], [lid])
    # Ray-aware NMS must collapse the camera's range ghost into one object.
    assert len(boxes) == 1
    box = boxes[0]
    assert math.hypot(box["bbox_3d"][0] - true_x, box["bbox_3d"][1] - true_y) < 0.5, \
        "fused position must be LiDAR-anchored, not the camera's depth estimate"
    assert box["class_name"] == "pedestrian", \
        "camera semantics must win over the weak LiDAR template class"


def test_camera_only_detection_survives_when_lidar_missing():
    """Long-range/sparse LiDAR: the camera's evidence alone keeps existence."""
    cam = _camera(55.0, -4.0, conf=0.8, sigma_along=3.85, sigma_cross=0.22)
    boxes = fuse_frame([cam], [])
    assert len(boxes) == 1
    assert math.hypot(boxes[0]["bbox_3d"][0] - 55.0, boxes[0]["bbox_3d"][1] + 4.0) < 1.0


def test_weak_single_modality_clutter_is_suppressed_at_decode():
    """Low-confidence rain clutter stays below the fused decode threshold."""
    clutter = _lidar(20.4, 3.4, conf=0.25, cls_conf=0.25)
    assert fuse_frame([], [clutter]) == []


def test_both_modalities_beat_camera_alone_on_position():
    true_x, true_y = 40.0, 5.0
    cam = _camera(43.0, 5.375, sigma_along=2.8, sigma_cross=0.16)  # ~3 m range error
    lid = _lidar(40.1, 5.05)
    fused = fuse_frame([cam], [lid])
    cam_only = fuse_frame([cam], [])
    assert len(fused) == 1 and len(cam_only) == 1
    err_fused = math.hypot(fused[0]["bbox_3d"][0] - true_x, fused[0]["bbox_3d"][1] - true_y)
    err_cam = math.hypot(cam_only[0]["bbox_3d"][0] - true_x, cam_only[0]["bbox_3d"][1] - true_y)
    assert err_fused < 0.4
    assert err_fused < err_cam / 3
