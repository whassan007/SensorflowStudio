"""Shared helpers to load real camera media for vendor adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sensorflow.adapters.local_adapter import DEFAULT_MAX_FRAMES, discover_media
from sensorflow.adapters.stub_images import stub_camera_path, write_stub_camera_png
from sensorflow.schemas.unified_frame import (
    CameraView,
    EgoPose,
    FusedFrame,
    LidarData,
)


def _write_placeholder_lidar(path: Path, seed: int = 0) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 2.0, size=(128, 3)).astype(np.float32)
    points.tofile(path)
    return len(points)


def resolve_vendor_root(source: Dict[str, Any], *keys: str) -> Optional[Path]:
    for key in keys:
        raw = source.get(key)
        if raw:
            path = Path(str(raw)).expanduser()
            if path.exists():
                return path
    return None


def media_available(path: Path) -> bool:
    images, videos = discover_media(path)
    return bool(images or videos)


def frames_from_media_root(
    *,
    sequence_id: str,
    vendor: str,
    root: Path,
    max_frames: Optional[int] = None,
) -> List[FusedFrame]:
    """Build FusedFrames from a local image/video root with vendor-prefixed IDs."""
    from sensorflow.adapters.local_adapter import _decode_video_frames

    images, videos = discover_media(root)
    frame_images = list(images)
    decode_dir = Path("runs/pipeline") / sequence_id / "decoded_frames"
    limit = DEFAULT_MAX_FRAMES if max_frames is None else max_frames

    for video in videos:
        remaining = None if limit is None else max(0, limit - len(frame_images))
        if remaining == 0:
            break
        decoded = _decode_video_frames(
            video,
            decode_dir / video.stem,
            remaining,
            start_index=len(frame_images),
        )
        frame_images.extend(decoded)

    if not frame_images:
        raise FileNotFoundError(
            f"No images or videos under {root} for vendor {vendor}. "
            f"Point the {vendor} dataset root at a folder of frames or a video file."
        )

    if limit is not None:
        frame_images = frame_images[:limit]

    lidar_dir = Path("runs/pipeline") / sequence_id / "lidar"
    frames: List[FusedFrame] = []
    for i, img_path in enumerate(frame_images):
        frame_id = f"{vendor}_{i:04d}"
        lidar_path = lidar_dir / f"{frame_id}_lidar.bin"
        num_points = _write_placeholder_lidar(lidar_path, seed=i)
        # Prefer copying a stub-sized preview under the run so browse stays under runs/
        # when the source is outside allowed roots — keep original path (allowed via data/).
        frames.append(
            FusedFrame(
                frame_id=frame_id,
                timestamp_us=i * 100_000,
                lidar=LidarData(path=str(lidar_path), num_points=num_points),
                cameras={"front": CameraView(image_path=str(img_path.resolve()))},
                ego_pose=EgoPose(speed_kmh=0.0, x=float(i) * 0.5),
                ground_truth=[],
            )
        )
    return frames


def ensure_stub_camera(sequence_id: str, frame_id: str, camera: str = "front", seed: int = 0) -> str:
    path = stub_camera_path(sequence_id, frame_id, camera)
    write_stub_camera_png(path, seed=seed, label=f"{sequence_id}:{frame_id}:{camera}")
    return str(path)
