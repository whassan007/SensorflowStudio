"""Normalize Waymo data into UnifiedSequence."""

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

WAYMO_TYPE_MAP = {
    1: "vehicle",
    2: "pedestrian",
    3: "sign",
    4: "cyclist",
}


def _default_waymo_shard() -> List[Dict[str, Any]]:
    """Built-in shard when Waymo SDK and external files are unavailable."""
    return [
        {
            "frame_id": "waymo_0000",
            "timestamp_us": 0,
            "lidar_points": 2048,
            "ego_speed_kmh": 45.0,
            "labels": [
                {"instance_id": "wgt_1", "type": 1, "center_x": 10.0, "center_y": 2.0,
                 "center_z": 0.5, "length": 4.5, "width": 1.8, "height": 1.5, "heading": 0.1},
                {"instance_id": "wgt_2", "type": 2, "center_x": 15.0, "center_y": -1.5,
                 "center_z": 0.9, "length": 0.5, "width": 0.5, "height": 1.7, "heading": 1.2},
            ],
        },
        {
            "frame_id": "waymo_0001",
            "timestamp_us": 100_000,
            "lidar_points": 2100,
            "ego_speed_kmh": 45.5,
            "labels": [
                {"instance_id": "wgt_1", "type": 1, "center_x": 10.5, "center_y": 2.1,
                 "center_z": 0.5, "length": 4.5, "width": 1.8, "height": 1.5, "heading": 0.1},
                {"instance_id": "wgt_2", "type": 2, "center_x": 15.2, "center_y": -1.4,
                 "center_z": 0.9, "length": 0.5, "width": 0.5, "height": 1.7, "heading": 1.1},
            ],
        },
        {
            "frame_id": "waymo_0002",
            "timestamp_us": 200_000,
            "lidar_points": 1980,
            "ego_speed_kmh": 46.0,
            "labels": [
                {"instance_id": "wgt_1", "type": 1, "center_x": 11.0, "center_y": 2.0,
                 "center_z": 0.5, "length": 4.5, "width": 1.8, "height": 1.5, "heading": 0.0},
            ],
        },
    ]


def _write_synthetic_lidar(path: Path, num_points: int, labels: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    for lbl in labels:
        cx, cy, cz = lbl["center_x"], lbl["center_y"], lbl["center_z"]
        for _ in range(max(20, num_points // max(len(labels), 1) // 10)):
            points.append([cx + np.random.randn() * 0.2, cy + np.random.randn() * 0.2, cz + np.random.randn() * 0.1])
    if not points:
        points = np.random.randn(num_points, 3).astype(np.float32)
    else:
        points = np.array(points, dtype=np.float32)
    points.tofile(path)


class WaymoAdapter(VendorAdapter):
    def load(self, source: Dict[str, Any], sequence_id: str) -> UnifiedSequence:
        shard_dir = Path("runs/pipeline/waymo_shards")
        shard_path = shard_dir / f"{source.get('shard_id', 'default')}.json"
        used_builtin_stub = False

        if source.get("frames"):
            shards = source["frames"]
        elif shard_path.exists():
            with open(shard_path) as f:
                shards = json.load(f)
            used_builtin_stub = len(shards) <= 3 and not source.get("tfrecord")
        else:
            shards = _default_waymo_shard()
            used_builtin_stub = True
            shard_dir.mkdir(parents=True, exist_ok=True)
            with open(shard_path, "w") as f:
                json.dump(shards, f, indent=2)

        frames: List[FusedFrame] = []
        lidar_dir = Path("runs/pipeline") / sequence_id / "lidar"

        for shard in shards:
            frame_id = shard["frame_id"]
            lidar_path = lidar_dir / f"{frame_id}_lidar.bin"
            labels = shard.get("labels", [])
            num_points = shard.get("lidar_points", 2048)
            _write_synthetic_lidar(lidar_path, num_points, labels)

            gt = []
            speed = shard.get("ego_speed_kmh", 40.0)
            for lbl in labels:
                class_name = WAYMO_TYPE_MAP.get(lbl.get("type", 1), "vehicle")
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

            frames.append(FusedFrame(
                frame_id=frame_id,
                timestamp_us=shard.get("timestamp_us", 0),
                lidar=LidarData(path=str(lidar_path), num_points=num_points),
                cameras={"front": CameraView(image_path=shard.get("image_path", ""))},
                ego_pose=EgoPose(speed_kmh=speed),
                ground_truth=gt,
            ))

        demo_stub = bool(source.get("demo_stub", used_builtin_stub))
        manifest: Dict[str, Any] = {
            "shard": str(shard_path),
            "demo_stub": demo_stub,
            "total_frames": len(frames),
        }
        if demo_stub:
            manifest["stub_note"] = (
                "Built-in Waymo sample (~3 frames), not a full AV video lake. "
                "Enable Local frames/video on Ingest and set Images Path to load a real sequence."
            )

        return UnifiedSequence(
            sequence_id=sequence_id,
            vendor="waymo",
            frames=frames,
            taxonomy_manifest=manifest,
        )

    def load_tfrecord(self, tfrecord_path: Path, sequence_id: str) -> UnifiedSequence:
        """Load from Waymo TFRecord when SDK is available."""
        try:
            from waymo_open_dataset import dataset_pb2 as open_dataset  # type: ignore
            import tensorflow as tf  # type: ignore
        except ImportError:
            return self.load({}, sequence_id)

        frames = []
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
        for i, raw_record in enumerate(dataset.take(10)):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytes(raw_record.numpy()))
            labels = []
            for obj in frame.laser_labels:
                labels.append({
                    "instance_id": str(obj.id),
                    "type": obj.type,
                    "center_x": obj.box.center_x,
                    "center_y": obj.box.center_y,
                    "center_z": obj.box.center_z,
                    "length": obj.box.length,
                    "width": obj.box.width,
                    "height": obj.box.height,
                    "heading": obj.box.heading,
                })
            frames.append({
                "frame_id": f"waymo_{i:04d}",
                "timestamp_us": int(frame.timestamp_micros),
                "lidar_points": len(frame.lasers[0].ri_return1.range_values) if frame.lasers else 2048,
                "ego_speed_kmh": 40.0,
                "labels": labels,
            })
        return self.load({"frames": frames}, sequence_id)
