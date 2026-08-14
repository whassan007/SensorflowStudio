"""Read-only helpers for browsing pipeline run artifacts under runs/pipeline/."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

PIPELINE_ROOT = Path("runs/pipeline")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_allowed(path: Path, roots: List[Path]) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            root_res = root.resolve()
            resolved.relative_to(root_res)
            return resolved
        except ValueError:
            continue
    raise PermissionError(f"Path not under allowed roots: {path}")


def allowed_roots() -> List[Path]:
    roots = [PIPELINE_ROOT, Path("runs"), Path("data")]
    return [r for r in roots if True]


def list_sequences() -> List[Dict[str, Any]]:
    root = PIPELINE_ROOT
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        manifest = child / "manifest.json"
        proposals = child / "proposals"
        entry: Dict[str, Any] = {
            "sequence_id": child.name,
            "manifest_exists": manifest.exists(),
            "has_proposals": proposals.exists() and any(proposals.glob("*.json")),
            "has_tracks": (child / "tracks.json").exists(),
            "path": str(child),
        }
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                entry["frames"] = len(data.get("frames", []))
                entry["vendor"] = data.get("vendor", "unknown")
                tax = data.get("taxonomy_manifest") or {}
                entry["demo_stub"] = bool(tax.get("demo_stub"))
            except Exception:
                entry["frames"] = None
        out.append(entry)
    return out


def artifacts_summary(sequence_id: str) -> Dict[str, Any]:
    base = PIPELINE_ROOT / sequence_id
    manifest = base / "manifest.json"
    proposals_dir = base / "proposals"
    tracks = base / "tracks.json"
    benchmark = base / "benchmark" / "metric_card.json"
    quality = base / "benchmark" / "quality_report.json"

    summary: Dict[str, Any] = {
        "sequence_id": sequence_id,
        "exists": base.exists(),
        "base_path": str(base),
        "manifest_path": str(manifest) if manifest.exists() else None,
        "proposals_dir": str(proposals_dir) if proposals_dir.exists() else None,
        "tracks_path": str(tracks) if tracks.exists() else None,
        "benchmark_path": str(benchmark) if benchmark.exists() else None,
        "quality_report_path": str(quality) if quality.exists() else None,
        "frames": 0,
        "proposal_files": 0,
        "vendor": None,
        "demo_stub": None,
        "browsable": False,
    }
    if not base.exists():
        return summary

    if proposals_dir.exists():
        summary["proposal_files"] = len(list(proposals_dir.glob("*.json")))

    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            summary["frames"] = len(data.get("frames", []))
            summary["vendor"] = data.get("vendor")
            tax = data.get("taxonomy_manifest") or {}
            summary["demo_stub"] = bool(tax.get("demo_stub"))
            summary["taxonomy_manifest"] = tax
            summary["browsable"] = summary["frames"] > 0
        except Exception as exc:
            summary["error"] = f"Failed to read manifest: {exc}"

    return summary


def _image_url(image_path: str) -> Optional[str]:
    if not image_path:
        return None
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    return f"/api/pipeline/file?path={quote(image_path, safe='')}"


def _local_image_exists(image_path: Optional[str]) -> bool:
    if not image_path or str(image_path).startswith("http"):
        return False
    raw = Path(image_path)
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    try:
        return candidate.is_file()
    except OSError:
        return False


def _camera_preview(cam_name: str, cam: Any) -> Dict[str, Any]:
    img = cam.get("image_path") if isinstance(cam, dict) else None
    exists = _local_image_exists(img) if img and not str(img).startswith("http") else bool(img)
    preview = _image_url(img) if img and exists else None
    # Remote http(s) URLs are treated as available for preview.
    if img and str(img).startswith("http"):
        preview = _image_url(img)
        exists = True
    return {
        "name": cam_name,
        "image_path": img,
        "preview_url": preview,
        "local_file": bool(img and not str(img).startswith("http")),
        "available": bool(preview),
    }


def _proposal_count(sequence_id: str, frame_id: str) -> int:
    prop_path = PIPELINE_ROOT / sequence_id / "proposals" / f"{frame_id}.json"
    if not prop_path.exists():
        return 0
    try:
        data = json.loads(prop_path.read_text())
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data.get("proposals") or data.get("objects") or [])
    except Exception:
        return 0
    return 0


def list_frames(sequence_id: str) -> Dict[str, Any]:
    summary = artifacts_summary(sequence_id)
    manifest = PIPELINE_ROOT / sequence_id / "manifest.json"
    if not manifest.exists():
        return {
            "status": "ok",
            "sequence_id": sequence_id,
            "frames": [],
            "browsable": False,
            "empty_reason": "No manifest.json — run 3D Ingest first.",
            **{k: summary[k] for k in ("vendor", "demo_stub", "proposal_files", "base_path")},
        }

    data = json.loads(manifest.read_text())
    frames_out: List[Dict[str, Any]] = []
    for frame in data.get("frames", []):
        cameras = frame.get("cameras") or {}
        camera_previews = [
            _camera_preview(cam_name, cam) for cam_name, cam in cameras.items()
        ]
        primary = next((c for c in camera_previews if c.get("preview_url")), None)
        available_cams = [c for c in camera_previews if c.get("available")]
        fid = frame.get("frame_id", "")
        frames_out.append(
            {
                "frame_id": fid,
                "timestamp_us": frame.get("timestamp_us"),
                "cameras": camera_previews,
                "preview_url": primary["preview_url"] if primary else None,
                "available_camera_count": len(available_cams),
                "proposal_count": _proposal_count(sequence_id, fid),
                "gt_count": len(frame.get("ground_truth") or []),
            }
        )

    return {
        "status": "ok",
        "sequence_id": sequence_id,
        "frames": frames_out,
        "browsable": len(frames_out) > 0,
        "empty_reason": None if frames_out else "Manifest has zero frames.",
        "vendor": data.get("vendor"),
        "demo_stub": bool((data.get("taxonomy_manifest") or {}).get("demo_stub")),
        "proposal_files": summary["proposal_files"],
        "base_path": summary["base_path"],
        "has_tracks": (PIPELINE_ROOT / sequence_id / "tracks.json").exists(),
    }


def get_frame(sequence_id: str, frame_id: str) -> Dict[str, Any]:
    manifest = PIPELINE_ROOT / sequence_id / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Manifest not found")

    data = json.loads(manifest.read_text())
    frame = next((f for f in data.get("frames", []) if f.get("frame_id") == frame_id), None)
    if frame is None:
        raise FileNotFoundError(f"Frame {frame_id} not found in sequence {sequence_id}")

    cameras = frame.get("cameras") or {}
    camera_previews = [
        _camera_preview(cam_name, cam) for cam_name, cam in cameras.items()
    ]
    primary = next((c for c in camera_previews if c.get("preview_url")), None)

    proposals: Any = []
    prop_path = PIPELINE_ROOT / sequence_id / "proposals" / f"{frame_id}.json"
    if prop_path.exists():
        raw = json.loads(prop_path.read_text())
        if isinstance(raw, list):
            proposals = raw
        elif isinstance(raw, dict):
            proposals = raw.get("proposals") or raw.get("objects") or raw

    return {
        "status": "ok",
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "frame": frame,
        "cameras": camera_previews,
        "preview_url": primary["preview_url"] if primary else None,
        "proposals": proposals,
        "proposal_count": len(proposals) if isinstance(proposals, list) else 0,
        "proposals_path": str(prop_path) if prop_path.exists() else None,
        "manifest_path": str(manifest),
    }


def resolve_file(path_str: str) -> Tuple[Path, str]:
    """Return (absolute_path, media_type) for a path under allowed roots."""
    raw = Path(path_str)
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    resolved = _resolve_allowed(candidate, allowed_roots())
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"File not found: {path_str}")
    media_type, _ = mimetypes.guess_type(str(resolved))
    return resolved, media_type or "application/octet-stream"


def scan_source_images(source_path: str, limit: int = 48) -> Dict[str, Any]:
    """List local images under Dataset Configuration Images Path for validation."""
    path = Path(source_path)
    if not path.exists():
        return {
            "status": "ok",
            "source_path": source_path,
            "exists": False,
            "browsable": False,
            "count": 0,
            "images": [],
            "empty_reason": f"Path does not exist: {source_path}",
        }

    images: List[Path] = []
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        images = [path]
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
                if len(images) >= max(limit, 1):
                    break
    else:
        return {
            "status": "ok",
            "source_path": source_path,
            "exists": True,
            "browsable": False,
            "count": 0,
            "images": [],
            "empty_reason": "Not an image file or directory of images.",
        }

    # Full count for directories (cap walk cost for huge trees by reusing discovered list when small)
    total = len(images)
    if path.is_dir() and total >= limit:
        total = sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    previews = []
    for img in images[:limit]:
        previews.append(
            {
                "name": img.name,
                "path": str(img),
                "preview_url": _image_url(str(img.resolve())),
            }
        )

    return {
        "status": "ok",
        "source_path": source_path,
        "exists": True,
        "browsable": total > 0,
        "count": total,
        "images": previews,
        "empty_reason": None if total else "No images found under this path.",
    }
