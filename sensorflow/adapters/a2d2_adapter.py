"""Normalize Audi A2D2 samples into UnifiedSequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sensorflow.adapters.base import VendorAdapter
from sensorflow.adapters.stub_images import stub_camera_path, write_stub_camera_png
from sensorflow.schemas.taxonomy_axes import assign_taxonomy_axes
from sensorflow.schemas.unified_frame import (
    CameraView,
    EgoPose,
    FusedFrame,
    GroundTruthObject,
    LidarData,
    UnifiedSequence,
)

# Audi A2D2 class names used in the public dataset catalog.
A2D2_CLASSES = (
    "car",
    "pedestrian",
    "truck",
    "bicycle",
    "traffic_sign",
    "signal",
)


def _default_a2d2_frames() -> List[Dict[str, Any]]:
    """Built-in shard when A2D2 HDF5/PNG sources are unavailable."""
    return [
        {
            "frame_id": "a2d2_0000",
            "timestamp_us": 0,
            "lidar_points": 2200,
            "ego_speed_kmh": 52.0,
            "image_path": "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800",
            "labels": [
                {"instance_id": "a2d2_1", "class_name": "car",
                 "center_x": 12.0, "center_y": 1.5, "center_z": 0.6,
                 "length": 4.4, "width": 1.8, "height": 1.5, "heading": 0.05},
                {"instance_id": "a2d2_2", "class_name": "pedestrian",
                 "center_x": 8.0, "center_y": -2.0, "center_z": 0.9,
                 "length": 0.6, "width": 0.5, "height": 1.7, "heading": -0.3},
                {"instance_id": "a2d2_3", "class_name": "traffic_sign",
                 "center_x": 18.0, "center_y": 3.0, "center_z": 1.8,
                 "length": 0.3, "width": 0.8, "height": 1.2, "heading": 0.0},
            ],
        },
        {
            "frame_id": "a2d2_0001",
            "timestamp_us": 100_000,
            "lidar_points": 2150,
            "ego_speed_kmh": 51.2,
            "image_path": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=800",
            "labels": [
                {"instance_id": "a2d2_1", "class_name": "car",
                 "center_x": 11.5, "center_y": 1.4, "center_z": 0.6,
                 "length": 4.4, "width": 1.8, "height": 1.5, "heading": 0.04},
                {"instance_id": "a2d2_2", "class_name": "pedestrian",
                 "center_x": 7.6, "center_y": -1.8, "center_z": 0.9,
                 "length": 0.6, "width": 0.5, "height": 1.7, "heading": -0.2},
                {"instance_id": "a2d2_4", "class_name": "bicycle",
                 "center_x": 14.0, "center_y": -3.0, "center_z": 0.7,
                 "length": 1.6, "width": 0.5, "height": 1.2, "heading": 1.1},
            ],
        },
        {
            "frame_id": "a2d2_0002",
            "timestamp_us": 200_000,
            "lidar_points": 2300,
            "ego_speed_kmh": 49.8,
            "image_path": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800",
            "labels": [
                {"instance_id": "a2d2_1", "class_name": "car",
                 "center_x": 11.0, "center_y": 1.3, "center_z": 0.6,
                 "length": 4.4, "width": 1.8, "height": 1.5, "heading": 0.02},
                {"instance_id": "a2d2_5", "class_name": "truck",
                 "center_x": 22.0, "center_y": 0.5, "center_z": 1.2,
                 "length": 8.5, "width": 2.5, "height": 3.2, "heading": 0.0},
                {"instance_id": "a2d2_6", "class_name": "signal",
                 "center_x": 16.0, "center_y": 2.5, "center_z": 3.0,
                 "length": 0.4, "width": 0.4, "height": 1.0, "heading": 0.0},
            ],
        },
    ]


def _write_synthetic_lidar(path: Path, num_points: int, labels: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    for lbl in labels:
        cx, cy, cz = lbl["center_x"], lbl["center_y"], lbl["center_z"]
        for _ in range(max(20, num_points // max(len(labels), 1) // 10)):
            points.append([
                cx + np.random.randn() * 0.2,
                cy + np.random.randn() * 0.2,
                cz + np.random.randn() * 0.1,
            ])
    if not points:
        points = np.random.randn(num_points, 3).astype(np.float32)
    else:
        points = np.array(points, dtype=np.float32)
    points.tofile(path)


DEFAULT_A2D2_SAMPLES = {
    "ingolstadt": {
        "dataset": "audi/a2d2",
        "frames": _default_a2d2_frames(),
    },
}


class A2D2Adapter(VendorAdapter):
    """Audi A2D2 adapter — stub frames when HDF5/PNG lake is unavailable."""

    def load(self, source: Dict[str, Any], sequence_id: str) -> UnifiedSequence:
        from sensorflow.adapters.vendor_media import (
            frames_from_media_root,
            media_available,
            resolve_vendor_root,
        )

        source = source or {}
        root = resolve_vendor_root(source, "source_path", "root", "path")
        if root is not None and media_available(root):
            frames = frames_from_media_root(
                sequence_id=sequence_id,
                vendor="a2d2",
                root=root,
                max_frames=source.get("max_frames"),
            )
            return UnifiedSequence(
                sequence_id=sequence_id,
                vendor="a2d2",
                frames=frames,
                taxonomy_manifest={
                    "source_dataset": source.get("dataset", "audi/a2d2"),
                    "source_path": str(root.resolve()),
                    "demo_stub": False,
                    "total_frames": len(frames),
                },
            )

        shard_dir = Path("runs/pipeline/a2d2_shards")
        shard_path = shard_dir / f"{source.get('shard_id', 'default')}.json"

        if shard_path.exists():
            with open(shard_path) as f:
                frames_data = json.load(f)
        elif source.get("frames"):
            frames_data = source["frames"]
        else:
            frames_data = DEFAULT_A2D2_SAMPLES["ingolstadt"]["frames"]
            shard_dir.mkdir(parents=True, exist_ok=True)
            with open(shard_path, "w") as f:
                json.dump(frames_data, f, indent=2)

        frames: List[FusedFrame] = []
        lidar_dir = Path("runs/pipeline") / sequence_id / "lidar"

        for shard in frames_data:
            frame_id = shard["frame_id"]
            lidar_path = lidar_dir / f"{frame_id}_lidar.bin"
            labels = shard.get("labels", [])
            num_points = shard.get("lidar_points", 2048)
            _write_synthetic_lidar(lidar_path, num_points, labels)

            gt = []
            speed = shard.get("ego_speed_kmh", 50.0)
            for lbl in labels:
                class_name = lbl.get("class_name", "car")
                if class_name not in A2D2_CLASSES:
                    class_name = "car"
                axes = assign_taxonomy_axes(class_name, speed_kmh=speed)
                gt.append(GroundTruthObject(
                    instance_id=lbl["instance_id"],
                    class_name=class_name,
                    confidence=1.0,
                    bbox_3d=[
                        lbl["center_x"], lbl["center_y"], lbl["center_z"],
                        lbl["length"], lbl["width"], lbl["height"], lbl["heading"],
                    ],
                    taxonomy_axes=axes,
                ))

            image_path = shard.get("image_path") or ""
            if not image_path or str(image_path).startswith("http"):
                cam_file = stub_camera_path(sequence_id, frame_id, "front")
                write_stub_camera_png(
                    cam_file,
                    seed=sum(ord(c) for c in frame_id) % 991,
                    label=f"a2d2:{frame_id}",
                )
                image_path = str(cam_file)

            frames.append(FusedFrame(
                frame_id=frame_id,
                timestamp_us=shard.get("timestamp_us", 0),
                lidar=LidarData(path=str(lidar_path), num_points=num_points),
                cameras={"front": CameraView(image_path=image_path)},
                ego_pose=EgoPose(speed_kmh=speed),
                ground_truth=gt,
            ))

        demo_stub = bool(source.get("demo_stub", True)) if not source.get("frames") else bool(
            source.get("demo_stub", False)
        )
        if shard_path.exists() and not source.get("frames") and len(frames_data) <= 3:
            demo_stub = True

        return UnifiedSequence(
            sequence_id=sequence_id,
            vendor="a2d2",
            frames=frames,
            taxonomy_manifest={
                "source_dataset": source.get("dataset", "audi/a2d2"),
                "shard": str(shard_path),
                "demo_stub": demo_stub,
                "stub_note": (
                    "Built-in Audi A2D2 sample (~3 frames), not a full AV video lake. "
                    "Set a2d2_root to a real A2D2 PNG/HDF5 export folder when available."
                ) if demo_stub else None,
                "total_frames": len(frames),
            },
        )

    @classmethod
    def load_from_file(cls, path: Path, sequence_id: str) -> UnifiedSequence:
        with open(path) as f:
            return cls().load(json.load(f), sequence_id)
