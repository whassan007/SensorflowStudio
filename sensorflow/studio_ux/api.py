"""FastAPI router for studio-UX support (/api/studio-ux/*).

Endpoints:
    GET  /layouts/{key}   fetch a persisted UI layout/preset (JSON blob)
    PUT  /layouts/{key}   persist a UI layout/preset
    GET  /bev/replay      deterministically re-generate one BEV-fusion scene
                          sequence (same generators/engines as /api/bevfusion,
                          read-only) and return per-frame GT boxes, camera and
                          LiDAR detections with covariances, fused+tracked
                          boxes (masklet-propagated flagged) and the camera
                          baseline boxes — the data behind the interactive
                          top-down BEV canvas.

This module only *reads* from other subsystems (pure functions on synthetic
scenes); the only writes are its own layout files under runs/studio_ux/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/studio-ux", tags=["studio-ux"])

LAYOUTS_DIR = Path("runs/studio_ux/layouts")
_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def _layout_path(key: str) -> Path:
    if not _KEY_RE.match(key):
        raise HTTPException(422, "layout key must match [A-Za-z0-9._-]{1,120}")
    return LAYOUTS_DIR / f"{key}.json"


@router.get("/layouts/{key}")
def get_layout(key: str) -> Dict[str, Any]:
    path = _layout_path(key)
    if not path.exists():
        return {"key": key, "value": None}
    try:
        with open(path) as f:
            return {"key": key, "value": json.load(f)}
    except Exception:
        return {"key": key, "value": None}


class LayoutBody(BaseModel):
    value: Any


@router.put("/layouts/{key}")
def put_layout(key: str, body: LayoutBody) -> Dict[str, Any]:
    path = _layout_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(body.value, f)
    return {"key": key, "saved": True}


# ------------------------------------------------------------------ BEV replay


def _det_dict(det) -> Dict[str, Any]:
    return {
        "modality": det.modality,
        "x": round(det.x, 3),
        "y": round(det.y, 3),
        "cov": [[round(v, 5) for v in row] for row in det.cov],
        "dims": [round(v, 3) for v in det.dims],
        "yaw": round(det.yaw, 4),
        "class_name": det.class_name,
        "confidence": round(det.confidence, 3),
    }


@router.get("/bev/replay")
def bev_replay(seed: int = Query(7),
               n_sequences: int = Query(6, ge=1, le=24),
               frames_per_sequence: int = Query(24, ge=8, le=60),
               sequence: int = Query(0, ge=0)) -> Dict[str, Any]:
    """Deterministic replay of one scene sequence through both engines."""
    from sensorflow.bevfusion.engines import FrameToFrameTracker
    from sensorflow.bevfusion.fusion import build_modality_map, decode_bev, fuse_maps
    from sensorflow.bevfusion.geometry import BEVGrid
    from sensorflow.bevfusion.masklet import BEVMaskletTracker
    from sensorflow.bevfusion.scenes import generate_sequences
    from sensorflow.bevfusion.sensors import camera_rng, lidar_rng, simulate_camera, simulate_lidar
    from sensorflow.schemas.unified_frame import Object3D

    if sequence >= n_sequences:
        raise HTTPException(422, f"sequence must be < n_sequences ({n_sequences})")

    sequences = generate_sequences(n_sequences=n_sequences,
                                   frames_per_sequence=frames_per_sequence,
                                   seed=seed)
    seq = sequences[sequence]

    grid = BEVGrid()
    cam_gen = camera_rng(seed, sequence)
    lid_gen = lidar_rng(seed, sequence)
    base_gen = camera_rng(seed, sequence)  # baseline uses the same camera stream
    masklet = BEVMaskletTracker(bounds=(grid.x_min, grid.x_max, grid.y_min, grid.y_max))
    base_tracker = FrameToFrameTracker(gate=3.0)

    frames: List[Dict[str, Any]] = []
    for frame in seq.frames:
        cam_dets = simulate_camera(frame, seq, cam_gen)
        lid_dets = simulate_lidar(frame, seq, lid_gen)

        # fused engine (mirrors engines.run_fused, but keeps the raw detections)
        cam_map = build_modality_map(cam_dets, grid, "camera")
        lid_map = build_modality_map(lid_dets, grid, "lidar")
        decoded = decode_bev(fuse_maps(cam_map, lid_map))
        proposals = [Object3D(bbox_3d=b["bbox_3d"], class_name=b["class_name"],
                              confidence=b["confidence"]) for b in decoded]
        tracked = masklet.update(frame.frame_id, proposals)
        propagated_ids = getattr(masklet, "last_propagated", set())
        fused = [{
            "bbox_3d": [round(v, 3) for v in t.bbox_3d],
            "class_name": t.class_name,
            "confidence": round(float(t.confidence or 0.0), 3),
            "track_id": t.track_id,
            "propagated": t.track_id in propagated_ids,
        } for t in tracked]

        # camera-only baseline (mirrors engines.run_baseline; identical rng stream)
        base_boxes = [{
            "bbox_3d": [d.x, d.y, d.z, d.dims[0], d.dims[1], d.dims[2], d.yaw],
            "class_name": d.class_name,
            "confidence": d.confidence,
            "propagated": False,
        } for d in simulate_camera(frame, seq, base_gen)]
        base_boxes = base_tracker.update(base_boxes)
        baseline = [{
            "bbox_3d": [round(float(v), 3) for v in b["bbox_3d"]],
            "class_name": b["class_name"],
            "confidence": round(float(b["confidence"]), 3),
            "track_id": b["track_id"],
            "propagated": False,
        } for b in base_boxes]

        frames.append({
            "frame_id": frame.frame_id,
            "index": frame.index,
            "gt": [{
                "instance_id": g.instance_id,
                "class_name": g.class_name,
                "bbox_3d": g.bbox_3d,
                "occluded": g.occluded,
                "distance": g.distance,
            } for g in frame.gt],
            "camera": [_det_dict(d) for d in cam_dets],
            "lidar": [_det_dict(d) for d in lid_dets],
            "fused": fused,
            "baseline": baseline,
        })

    return {
        "sequence_id": seq.sequence_id,
        "sequence_index": sequence,
        "n_sequences": n_sequences,
        "time_of_day": seq.time_of_day,
        "weather": seq.weather,
        "params": {"n_sequences": n_sequences,
                   "frames_per_sequence": frames_per_sequence, "seed": seed},
        "frames": frames,
    }
