"""Render bevfusion synthetic scenes to schematic sensor images.

The bevfusion package (imported read-only) provides deterministic scenes
with 3D ground truth, planted occlusion windows, and day/night/rain
conditions. This module rasterizes those scenes into two image spaces:

* BEV top-down grayscale images — one pixel maps linearly to ego-frame
  meters, so dense optical flow in BEV pixel space is directly a motion
  field in the ego frame (used by HIL detection and temporal profiling).
* Front-camera stereo pairs — pinhole projection with a horizontal baseline,
  so block-matching disparity converts back to metric depth (used by the
  stereo consistency check).

These are schematic renderings (intensity-coded boxes plus sensor noise),
not photorealistic frames; they exist so the acceleration backends process
images whose ground truth is exactly known.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sensorflow.bevfusion.scenes import SceneFrame, SceneSequence

# BEV raster geometry: ego-frame x in [0, 80] m forward, y in [-25, 25] m.
BEV_H, BEV_W = 192, 256
X_MIN, X_MAX = 0.0, 80.0
Y_MIN, Y_MAX = -25.0, 25.0

CLASS_INTENSITY = {"vehicle": 0.85, "truck": 0.95, "pedestrian": 0.55,
                   "cyclist": 0.65, "motorcycle": 0.75}

# Front stereo camera model.
FOCAL_PX = 200.0
BASELINE_M = 0.5
CAM_H, CAM_W = 128, 256
CAM_HEIGHT_M = 1.6


def ego_to_bev_px(x: float, y: float) -> Tuple[float, float]:
    """Ego meters -> (row, col) in the BEV raster. Row 0 is the far field."""
    row = (X_MAX - x) / (X_MAX - X_MIN) * (BEV_H - 1)
    col = (y - Y_MIN) / (Y_MAX - Y_MIN) * (BEV_W - 1)
    return row, col


def bev_px_to_ego(row: float, col: float) -> Tuple[float, float]:
    x = X_MAX - row / (BEV_H - 1) * (X_MAX - X_MIN)
    y = Y_MIN + col / (BEV_W - 1) * (Y_MAX - Y_MIN)
    return x, y


BEV_M_PER_PX_X = (X_MAX - X_MIN) / (BEV_H - 1)   # meters per row
BEV_M_PER_PX_Y = (Y_MAX - Y_MIN) / (BEV_W - 1)   # meters per col


def _condition_params(seq: SceneSequence) -> Dict:
    scale = 1.0
    noise = 0.012
    if seq.time_of_day == "night":
        scale = 0.45     # dimmer returns -> fewer effective signal bits
        noise = 0.02
    if seq.weather == "rain":
        scale *= 0.8
        noise += 0.015
    return {"intensity_scale": scale, "noise_sigma": noise}


def render_bev_frame(frame: SceneFrame, seq: SceneSequence, seed: int,
                     include_occluded: bool = True) -> np.ndarray:
    """Grayscale BEV raster (BEV_H, BEV_W) in [0, 1] with per-frame noise.

    Noise is seeded from (seed, frame index) so BOTH backends see the exact
    same source pixels — every ref-vs-vitis difference downstream is caused
    by the backend, never by the renderer.
    """
    params = _condition_params(seq)
    img = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    for gt in frame.gt:
        if gt.occluded and not include_occluded:
            continue
        x, y, _, l, w, _, _ = gt.bbox_3d
        r, c = ego_to_bev_px(x, y)
        hl = max(1.5, l / 2.0 / BEV_M_PER_PX_X)
        hw = max(1.5, w / 2.0 / BEV_M_PER_PX_Y)
        r0, r1 = int(max(0, r - hl)), int(min(BEV_H, r + hl + 1))
        c0, c1 = int(max(0, c - hw)), int(min(BEV_W, c + hw + 1))
        inten = CLASS_INTENSITY.get(gt.class_name, 0.7) * params["intensity_scale"]
        if gt.occluded:
            inten *= 0.55  # partially attenuated return
        img[r0:r1, c0:c1] = np.maximum(img[r0:r1, c0:c1], inten)
    rng = np.random.default_rng((seed * 1_000_003 + frame.index * 7919) % (2**63))
    img += rng.normal(0.0, params["noise_sigma"], img.shape).astype(np.float32)
    img += 0.04  # ground-return floor
    return np.clip(img, 0.0, 1.0)


def render_bev_rgb(frame: SceneFrame, seq: SceneSequence, seed: int) -> np.ndarray:
    """Color BEV raster (class-hued boxes) used as ISP/augmentation source."""
    gray = render_bev_frame(frame, seq, seed)
    hue = {"vehicle": (1.0, 0.85, 0.6), "truck": (1.0, 0.7, 0.4),
           "pedestrian": (0.6, 1.0, 0.7), "cyclist": (0.6, 0.85, 1.0),
           "motorcycle": (0.9, 0.7, 1.0)}
    rgb = np.stack([gray, gray, gray], axis=-1)
    for gt in frame.gt:
        x, y, _, l, w, _, _ = gt.bbox_3d
        r, c = ego_to_bev_px(x, y)
        hl = max(1.5, l / 2.0 / BEV_M_PER_PX_X)
        hw = max(1.5, w / 2.0 / BEV_M_PER_PX_Y)
        r0, r1 = int(max(0, r - hl)), int(min(BEV_H, r + hl + 1))
        c0, c1 = int(max(0, c - hw)), int(min(BEV_W, c + hw + 1))
        tint = np.array(hue.get(gt.class_name, (1.0, 1.0, 1.0)), dtype=np.float32)
        rgb[r0:r1, c0:c1] *= tint
    return np.clip(rgb, 0.0, 1.0)


def render_stereo_pair(frame: SceneFrame, seq: SceneSequence,
                       seed: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Front-camera stereo pair + per-object projection records.

    Left/right pinhole cameras separated by BASELINE_M; the right image sees
    each object shifted left by disparity = FOCAL_PX * BASELINE_M / depth.
    Returns (left, right, objects) where objects carry
    {instance_id, class_name, depth_m, u, v, disparity_px}.
    """
    params = _condition_params(seq)
    cx, cy = CAM_W / 2.0, CAM_H / 2.0
    left = np.zeros((CAM_H, CAM_W), dtype=np.float32)
    right = np.zeros((CAM_H, CAM_W), dtype=np.float32)
    objects: List[Dict] = []
    # Draw far to near so nearer objects overwrite (correct occlusion order).
    for gt in sorted(frame.gt, key=lambda g: -g.bbox_3d[0]):
        if gt.occluded:
            continue
        x, y, z, l, w, h, _ = gt.bbox_3d
        if x < 3.0:
            continue
        depth = float(x)
        u = FOCAL_PX * (-y) / depth + cx
        v = FOCAL_PX * (CAM_HEIGHT_M - z) / depth + cy
        half_w = max(1.5, FOCAL_PX * (w / 2.0) / depth)
        half_h = max(1.5, FOCAL_PX * (h / 2.0) / depth)
        disparity = FOCAL_PX * BASELINE_M / depth
        inten = CLASS_INTENSITY.get(gt.class_name, 0.7) * params["intensity_scale"]
        # Textured fill so block matching has gradients to lock onto.
        rng_tex = np.random.default_rng(abs(hash(gt.instance_id)) % (2**63))
        for img, du in ((left, 0.0), (right, -disparity)):
            u0 = int(max(0, u + du - half_w))
            u1 = int(min(CAM_W, u + du + half_w + 1))
            v0 = int(max(0, v - half_h))
            v1 = int(min(CAM_H, v + half_h + 1))
            if u1 <= u0 or v1 <= v0:
                continue
            tex = rng_tex.uniform(0.7, 1.0, (v1 - v0, u1 - u0)).astype(np.float32)
            # Same texture crop in both views (rng reset below).
            img[v0:v1, u0:u1] = inten * tex
            rng_tex = np.random.default_rng(abs(hash(gt.instance_id)) % (2**63))
        if 0 <= u < CAM_W and 0 <= v < CAM_H:
            objects.append({"instance_id": gt.instance_id,
                            "class_name": gt.class_name,
                            "depth_m": round(depth, 3),
                            "u": round(u, 2), "v": round(v, 2),
                            "disparity_px": round(disparity, 3)})
    rng = np.random.default_rng((seed * 999_983 + frame.index * 104_729) % (2**63))
    for img in (left, right):
        img += rng.normal(0.0, params["noise_sigma"], img.shape).astype(np.float32)
        img += 0.03
    return (np.clip(left, 0.0, 1.0), np.clip(right, 0.0, 1.0), objects)
