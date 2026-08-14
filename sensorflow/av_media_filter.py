"""Heuristic filters for driving / AV pipeline image and video discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Path segments that almost never contain drivable-scene media.
EXCLUDED_DIR_SEGMENTS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "docs",
    "doc",
    "documentation",
    "marketing",
    "assets",
    "static/icons",
    "icons",
    "badges",
    "screenshots",
}

# Filename stems that are commonly README / branding / test junk.
JUNK_NAME_STEMS = {
    "logo",
    "icon",
    "favicon",
    "avatar",
    "profile",
    "readme",
    "banner",
    "badge",
    "placeholder",
    "sample",
    "example",
    "demo",
    "test",
    "screenshot",
    "thumbnail",
    "thumb",
}

# Substrings in relative paths that suggest AV camera frames.
AV_PATH_MARKERS = (
    "camera",
    "cameras",
    "images",
    "frames",
    "frame",
    "front",
    "rear",
    "left",
    "right",
    "rgb",
    "video",
    "sequence",
    "segment",
    "train",
    "val",
    "training",
    "validation",
    "lidar",
    "sensor",
    "recording",
    "drive",
    "driving",
    "scene",
)

OFFICIAL_AV_DATASETS = frozenset({"alpamayo", "waymo", "a2d2"})

MIN_DRIVING_IMAGE_BYTES = 10 * 1024  # icons / tiny README assets

EXCLUSION_CATEGORIES = (
    "wrong_extension",
    "excluded_path",
    "junk_filename",
    "too_small",
    "non_av_layout",
)


def _normalize_parts(path: Path) -> Tuple[str, ...]:
    return tuple(part.lower() for part in path.parts)


def _relative_to_root(path: Path, root: Optional[Path]) -> Path:
    if root is None:
        return path
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _has_excluded_dir(path: Path) -> bool:
    parts = _normalize_parts(path)
    for i, part in enumerate(parts):
        if part in EXCLUDED_DIR_SEGMENTS:
            return True
        if part == "assets" and i + 1 < len(parts) and parts[i + 1] in {
            "marketing",
            "branding",
            "icons",
            "logos",
        }:
            return True
    return False


def _junk_filename(path: Path) -> bool:
    stem = path.stem.lower()
    if stem in JUNK_NAME_STEMS:
        return True
    if re.match(r"^(logo|icon|favicon|readme|banner|badge)[-_]?\d*$", stem):
        return True
    return False


def _looks_like_av_layout(relative: Path) -> bool:
    rel_str = str(relative).lower().replace("\\", "/")
    if any(marker in rel_str for marker in AV_PATH_MARKERS):
        return True
    # Nested structure (e.g. data/sequences/001/front.jpg) — not a loose root file.
    return len(relative.parts) >= 2


def classify_media_file(
    path: Path,
    *,
    root: Optional[Path] = None,
    dataset_type: Optional[str] = None,
    allow_videos: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Return (accepted, exclusion_category)."""
    if not path.is_file():
        return False, "non_media"

    ext = path.suffix.lower()
    allowed = IMAGE_EXTS | (VIDEO_EXTS if allow_videos else set())
    if ext not in allowed:
        return False, "wrong_extension"

    if _has_excluded_dir(path):
        return False, "excluded_path"

    if _junk_filename(path):
        return False, "junk_filename"

    if ext in IMAGE_EXTS:
        try:
            size = path.stat().st_size
        except OSError:
            return False, "non_media"
        if size < MIN_DRIVING_IMAGE_BYTES:
            return False, "too_small"

    relative = _relative_to_root(path, root)
    dtype = (dataset_type or "local").lower()
    if dtype in OFFICIAL_AV_DATASETS and not _looks_like_av_layout(relative):
        return False, "non_av_layout"

    return True, None


def partition_media_files(
    paths: Iterable[Path],
    *,
    root: Optional[Path] = None,
    dataset_type: Optional[str] = None,
    allow_videos: bool = True,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Split paths into accepted driving media and excluded entries with reasons."""
    accepted: List[Path] = []
    excluded: List[Dict[str, Any]] = []
    for path in paths:
        ok, reason = classify_media_file(
            path,
            root=root,
            dataset_type=dataset_type,
            allow_videos=allow_videos,
        )
        if ok:
            accepted.append(path)
        else:
            excluded.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "reason": reason or "excluded_path",
                }
            )
    accepted.sort(key=lambda p: str(p).lower())
    return accepted, excluded


def iter_candidate_files(
    root: Path,
    *,
    allow_videos: bool = True,
) -> List[Path]:
    """Collect image/video candidates under root before AV heuristics."""
    exts = set(IMAGE_EXTS)
    if allow_videos:
        exts |= VIDEO_EXTS

    if root.is_file():
        return [root] if root.suffix.lower() in exts else []

    if not root.is_dir():
        return []

    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def scan_driving_media(
    source_path: str | Path,
    *,
    dataset_type: Optional[str] = None,
    allow_videos: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Scan a path and return accepted media plus exclusion breakdown."""
    root = Path(source_path)
    if not root.is_absolute():
        root = Path.cwd() / root

    if not root.exists():
        return {
            "source_path": str(source_path),
            "exists": False,
            "accepted": [],
            "excluded": [],
            "count": 0,
            "excluded_count": 0,
            "excluded_by_reason": {},
        }

    candidates = iter_candidate_files(root, allow_videos=allow_videos)
    scan_root = root if root.is_dir() else root.parent
    accepted, excluded = partition_media_files(
        candidates,
        root=scan_root,
        dataset_type=dataset_type,
        allow_videos=allow_videos,
    )

    if limit is not None:
        accepted = accepted[:limit]

    by_reason: Dict[str, int] = {}
    for item in excluded:
        reason = item.get("reason") or "excluded_path"
        by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "source_path": str(source_path),
        "exists": True,
        "accepted": accepted,
        "excluded": excluded,
        "count": len(accepted),
        "excluded_count": len(excluded),
        "excluded_by_reason": by_reason,
    }


def summarize_exclusions(excluded: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in excluded:
        reason = item.get("reason") or "excluded_path"
        counts[reason] = counts.get(reason, 0) + 1
    return counts
