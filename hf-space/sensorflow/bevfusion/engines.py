"""The two auto-label engines under comparison.

perception-v1-camera (baseline)
    Camera-primary label generation: monocular detections are lifted to 3D
    directly (noisy range along the ray, ground-plane z) and associated
    frame-to-frame with a greedy nearest-neighbour matcher — no motion model,
    no gap tolerance. Expected weaknesses (all measured, none hardcoded):
    misses under occlusion and at night, position error growing with range,
    ID switches/fragmentation whenever detection drops for even one frame or
    the depth estimate jumps.

perception-v3-bevfusion (candidate)
    Camera + LiDAR fused in the shared BEV grid (fusion.py), decoded to boxes,
    then tracked with BEV Hungarian association + masklet propagation
    (masklet.py). The improvement comes from the fusion math: LiDAR existence
    recovers camera misses, camera semantics label LiDAR geometry, and
    inverse-variance weighting gives near-LiDAR position accuracy.
"""

from __future__ import annotations

import math
from typing import Dict, List

from sensorflow.bevfusion.geometry import BEVGrid
from sensorflow.bevfusion.fusion import build_modality_map, decode_bev, fuse_maps
from sensorflow.bevfusion.masklet import BEVMaskletTracker
from sensorflow.bevfusion.scenes import SceneSequence
from sensorflow.bevfusion.sensors import (
    Detection, camera_rng, lidar_rng, simulate_camera, simulate_lidar,
)
from sensorflow.schemas.unified_frame import Object3D

BASELINE_ENGINE = "perception-v1-camera"
FUSED_ENGINE = "perception-v3-bevfusion"


class FrameToFrameTracker:
    """Deliberately naive baseline association: greedy NN against the previous
    frame only. Any dropout or large position jitter spawns a new identity."""

    def __init__(self, gate: float = 3.0):
        self.gate = gate
        self.prev: List[Dict] = []  # [{track_id, x, y}]
        self.next_id = 1

    def update(self, boxes: List[Dict]) -> List[Dict]:
        used = set()
        for box in sorted(boxes, key=lambda b: -b.get("confidence", 0.0)):
            best, best_d = None, self.gate
            for i, p in enumerate(self.prev):
                if i in used:
                    continue
                d = math.hypot(box["bbox_3d"][0] - p["x"], box["bbox_3d"][1] - p["y"])
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                box["track_id"] = self.prev[best]["track_id"]
            else:
                box["track_id"] = self.next_id
                self.next_id += 1
        self.prev = [{"track_id": b["track_id"], "x": b["bbox_3d"][0], "y": b["bbox_3d"][1]}
                     for b in boxes]
        return boxes


def _det_to_box(det: Detection) -> Dict:
    return {
        "bbox_3d": [det.x, det.y, det.z, det.dims[0], det.dims[1], det.dims[2], det.yaw],
        "class_name": det.class_name,
        "class_confidence": det.class_confidence,
        "confidence": det.confidence,
        "propagated": False,
    }


def run_baseline(seq: SceneSequence, seed: int, seq_index: int) -> Dict[str, List[Dict]]:
    """Camera-only labels per frame. Uses the same camera detections (same rng
    stream) as the fused engine, so the comparison isolates the fusion."""
    rng = camera_rng(seed, seq_index)
    tracker = FrameToFrameTracker(gate=3.0)
    out: Dict[str, List[Dict]] = {}
    for frame in seq.frames:
        boxes = [_det_to_box(d) for d in simulate_camera(frame, seq, rng)]
        out[frame.frame_id] = tracker.update(boxes)
    return out


def run_fused(seq: SceneSequence, seed: int, seq_index: int,
              grid: BEVGrid = None):
    """BEV-fused labels per frame, tracked with masklet propagation."""
    grid = grid or BEVGrid()
    cam_gen = camera_rng(seed, seq_index)
    lid_gen = lidar_rng(seed, seq_index)
    tracker = BEVMaskletTracker(bounds=(grid.x_min, grid.x_max, grid.y_min, grid.y_max))
    out: Dict[str, List[Dict]] = {}
    stats = {"camera_detections": 0, "lidar_detections": 0, "fused_boxes": 0}

    for frame in seq.frames:
        cam_dets = simulate_camera(frame, seq, cam_gen)
        lid_dets = simulate_lidar(frame, seq, lid_gen)
        stats["camera_detections"] += len(cam_dets)
        stats["lidar_detections"] += len(lid_dets)

        cam_map = build_modality_map(cam_dets, grid, "camera")
        lid_map = build_modality_map(lid_dets, grid, "lidar")
        decoded = decode_bev(fuse_maps(cam_map, lid_map))
        stats["fused_boxes"] += len(decoded)

        proposals = [Object3D(bbox_3d=b["bbox_3d"], class_name=b["class_name"],
                              confidence=b["confidence"]) for b in decoded]
        tracked = tracker.update(frame.frame_id, proposals)
        propagated_ids = getattr(tracker, "last_propagated", set())
        out[frame.frame_id] = [{
            "bbox_3d": t.bbox_3d, "class_name": t.class_name,
            "confidence": t.confidence, "track_id": t.track_id,
            "propagated": t.track_id in propagated_ids,
        } for t in tracked]
    return out, stats
