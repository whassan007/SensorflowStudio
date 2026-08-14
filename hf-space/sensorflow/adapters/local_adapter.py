"""Load local image folders or video files into UnifiedSequence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sensorflow.adapters.base import VendorAdapter
from sensorflow.schemas.unified_frame import (
    CameraView,
    EgoPose,
    FusedFrame,
    LidarData,
    UnifiedSequence,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# High enough for full demo sequences; callers can raise or set None for uncapped.
DEFAULT_MAX_FRAMES = 10_000


def discover_media(
    path: Path,
    *,
    dataset_type: Optional[str] = None,
) -> Tuple[List[Path], List[Path]]:
    """Return (image_files, video_files) under path (file or directory)."""
    from sensorflow.av_media_filter import (
        IMAGE_EXTS as AV_IMAGE_EXTS,
        VIDEO_EXTS as AV_VIDEO_EXTS,
        partition_media_files,
    )

    if path.is_file():
        ext = path.suffix.lower()
        if ext in AV_IMAGE_EXTS:
            images, _ = partition_media_files([path], root=path.parent, dataset_type=dataset_type)
            return images, []
        if ext in AV_VIDEO_EXTS:
            _, excluded = partition_media_files([path], root=path.parent, dataset_type=dataset_type, allow_videos=True)
            return [], [] if excluded else [path]
        return [], []

    if not path.is_dir():
        return [], []

    raw_images: List[Path] = []
    raw_videos: List[Path] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in AV_IMAGE_EXTS:
            raw_images.append(p)
        elif ext in AV_VIDEO_EXTS:
            raw_videos.append(p)

    images, _ = partition_media_files(raw_images, root=path, dataset_type=dataset_type)
    videos, _ = partition_media_files(
        raw_videos,
        root=path,
        dataset_type=dataset_type,
        allow_videos=True,
    )
    return images, videos


def _write_placeholder_lidar(path: Path, seed: int = 0) -> int:
    """Minimal LiDAR cloud so downstream 3D lift still has points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 2.0, size=(128, 3)).astype(np.float32)
    points.tofile(path)
    return len(points)


def _decode_video_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: Optional[int],
    start_index: int = 0,
) -> List[Path]:
    """Decode video to JPEG frames under output_dir. Returns written paths."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required to decode video files for local ingest"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    idx = 0
    limit = max_frames if max_frames is not None else DEFAULT_MAX_FRAMES
    try:
        while len(written) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            out = output_dir / f"vid_{start_index + idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            written.append(out)
            idx += 1
    finally:
        cap.release()
    return written


class LocalSequenceAdapter(VendorAdapter):
    """Normalize a local image directory or video into UnifiedSequence."""

    def load(self, source: Dict[str, Any], sequence_id: str) -> UnifiedSequence:
        raw_path = source.get("source_path") or source.get("path") or "data"
        path = Path(str(raw_path)).expanduser()
        max_frames = source.get("max_frames", DEFAULT_MAX_FRAMES)
        if max_frames is not None:
            max_frames = int(max_frames)

        if not path.exists():
            raise FileNotFoundError(
                f"Local source path not found: {path}. "
                "Set Dataset Configuration → Images Path to a folder of frames "
                "or a video file, then re-run ingest with Local enabled."
            )

        images, videos = discover_media(path, dataset_type=source.get("dataset_type"))
        frame_images = list(images)

        decode_dir = Path("runs/pipeline") / sequence_id / "decoded_frames"
        for i, video in enumerate(videos):
            remaining = None if max_frames is None else max(0, max_frames - len(frame_images))
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
                f"No images or videos found under {path}. "
                f"Supported images: {sorted(IMAGE_EXTS)}; videos: {sorted(VIDEO_EXTS)}."
            )

        if max_frames is not None:
            frame_images = frame_images[:max_frames]

        lidar_dir = Path("runs/pipeline") / sequence_id / "lidar"
        frames: List[FusedFrame] = []
        for i, img_path in enumerate(frame_images):
            frame_id = f"local_{i:06d}"
            lidar_path = lidar_dir / f"{frame_id}_lidar.bin"
            num_points = _write_placeholder_lidar(lidar_path, seed=i)
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

        return UnifiedSequence(
            sequence_id=sequence_id,
            vendor="local",
            frames=frames,
            taxonomy_manifest={
                "source_path": str(path.resolve()),
                "demo_stub": False,
                "total_frames": len(frames),
                "images_discovered": len(images),
                "videos_discovered": len(videos),
                "max_frames": max_frames,
            },
        )
