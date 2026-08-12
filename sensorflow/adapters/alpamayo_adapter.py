"""Normalize Alpamayo / PhysicalAI samples into UnifiedSequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sensorflow.adapters.base import VendorAdapter
from sensorflow.schemas.taxonomy_axes import assign_taxonomy_axes
from sensorflow.schemas.unified_frame import (
    CameraView,
    EgoPose,
    FusedFrame,
    GroundTruthObject,
    LidarData,
    UnifiedSequence,
)


def _synthetic_lidar_path(output_dir: Path, frame_id: str, annotations: List[Dict]) -> tuple[str, int]:
    """Generate a minimal synthetic LiDAR point cloud for local dev."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lidar_path = output_dir / f"{frame_id}_lidar.bin"
    points = []
    for i, ann in enumerate(annotations):
        box = ann.get("box", [100, 100, 50, 50])
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        x = (cx - 320) * 0.05
        y = (240 - cy) * 0.05
        z = 0.5 + i * 0.1
        for _ in range(50):
            noise = np.random.randn(3) * 0.1
            points.append([x + noise[0], y + noise[1], z + noise[2]])
    if not points:
        points = np.random.randn(100, 3).astype(np.float32) * 2
    else:
        points = np.array(points, dtype=np.float32)
    points.tofile(lidar_path)
    return str(lidar_path), len(points)


def _box2d_to_bbox3d(box: List[float], conf: float) -> List[float]:
    """Convert 2D box [x,y,w,h] to stub 3D bbox [x,y,z,l,w,h,yaw]."""
    cx = box[0] + box[2] / 2
    cy = box[1] + box[3] / 2
    x = (cx - 320) * 0.05
    y = (240 - cy) * 0.05
    l = max(box[2] * 0.05, 1.0)
    w = max(box[3] * 0.05, 0.5)
    return [x, y, 0.5, l, w, 1.5, 0.0]


DEFAULT_ALPAMAYO_SAMPLES = {
    "physical_ai": {
        "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
        "frame_index": 4209,
        "views": {
            "front": "https://images.unsplash.com/photo-1506015391300-4802dc74de2e?w=800",
            "left": "https://images.unsplash.com/photo-1519074002996-a69e7ac46a42?w=800",
            "right": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=800",
        },
        "telemetry": {"lat": 37.774929, "lon": -122.419416, "speed_kmh": 35.8},
        "annotations": [
            {"id": 1, "label": "pedestrian", "box": [120, 240, 60, 120], "conf": 0.89},
            {"id": 2, "label": "car", "box": [340, 220, 150, 100], "conf": 0.95},
            {"id": 3, "label": "traffic_light", "box": [500, 80, 30, 60], "conf": 0.92},
        ],
    },
    "nurec": {
        "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
        "frame_index": 781,
        "views": {
            "front": "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=800",
        },
        "telemetry": {"lat": 37.783312, "lon": -122.416733, "speed_kmh": 48.2},
        "annotations": [
            {"id": 1, "label": "car", "box": [300, 180, 180, 140], "conf": 0.94},
            {"id": 2, "label": "truck", "box": [50, 150, 220, 180], "conf": 0.81},
        ],
    },
}


class AlpamayoAdapter(VendorAdapter):
    def load(self, source: Dict[str, Any], sequence_id: str) -> UnifiedSequence:
        data = source if source else DEFAULT_ALPAMAYO_SAMPLES["physical_ai"]
        output_dir = Path("runs/pipeline") / sequence_id / "lidar"
        speed = data.get("telemetry", {}).get("speed_kmh", 35.0)
        annotations = data.get("annotations", [])

        frames: List[FusedFrame] = []
        num_frames = max(3, len(annotations))
        for i in range(num_frames):
            frame_id = f"frame_{i:04d}"
            lidar_path, num_points = _synthetic_lidar_path(output_dir, frame_id, annotations)
            cameras = {
                name: CameraView(image_path=url)
                for name, url in data.get("views", {}).items()
            }
            gt = []
            for ann in annotations:
                axes = assign_taxonomy_axes(ann["label"], speed_kmh=speed)
                gt.append(GroundTruthObject(
                    instance_id=f"gt_{ann['id']}",
                    class_name=ann["label"],
                    confidence=ann.get("conf", 1.0),
                    bbox_3d=_box2d_to_bbox3d(ann["box"], ann.get("conf", 1.0)),
                    taxonomy_axes=axes,
                ))
            frames.append(FusedFrame(
                frame_id=frame_id,
                timestamp_us=i * 100_000,
                lidar=LidarData(path=lidar_path, num_points=num_points),
                cameras=cameras,
                ego_pose=EgoPose(speed_kmh=speed, x=i * 1.1, y=i * 5.2),
                ground_truth=gt,
            ))

        return UnifiedSequence(
            sequence_id=sequence_id,
            vendor="alpamayo",
            frames=frames,
            taxonomy_manifest={"source_dataset": data.get("dataset", "alpamayo")},
        )

    @classmethod
    def load_from_file(cls, path: Path, sequence_id: str) -> UnifiedSequence:
        with open(path) as f:
            return cls().load(json.load(f), sequence_id)
