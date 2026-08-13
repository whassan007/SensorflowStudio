"""Camera and LiDAR detection simulators with modality-specific failure modes.

These stand in for the learned detection backbones of a real BEVFusion stack.
Detections are sampled from ground truth with honest, physically motivated
error models — the complementarity that the fusion exploits is real:

Camera ("perception-v1-camera" front end):
    - misses under occlusion (line of sight blocked), at night (with a
      distance ramp: dark + small pixels), lightly in rain;
    - monocular depth ambiguity: range estimate noisy (~7% of range), bearing
      precise -> anisotropic BEV covariance elongated along the viewing ray;
    - good class discrimination (94% correct, confusions within size groups);
    - occasional false positives (glare phantoms, more at night).

LiDAR:
    - accurate range/geometry (~0.1-0.2 m position, ~3.5% dims);
    - degraded at long distance (sparse returns beyond ~45 m) and in heavy
      rain (spray attenuation growing with distance); mild degradation under
      partial occlusion (fewer returns, but the sensor sits higher and sees
      in 3D, so much less than the camera);
    - weak class discrimination: geometry-template guess with low confidence
      (semantics is the camera's job);
    - unaffected by night — this is the key complementarity for night recall.

All randomness is seeded per (seed, sequence, modality): deterministic.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from sensorflow.bevfusion.geometry import (
    backproject, make_camera_extrinsic, make_intrinsics, project_to_image,
    CAM_W, CAM_H,
)
from sensorflow.bevfusion.scenes import CLASS_DIMS, CLASSES, SceneFrame, SceneSequence

# Class confusion groups (similar appearance / similar footprint).
_CONFUSION = {
    "vehicle": ["truck"],
    "truck": ["vehicle"],
    "pedestrian": ["cyclist"],
    "cyclist": ["pedestrian", "motorcycle"],
    "motorcycle": ["cyclist"],
}

DEFAULT_K = make_intrinsics()
DEFAULT_CAM_TO_EGO = make_camera_extrinsic(tx=0.2, ty=0.0, tz=1.5)


class Detection(BaseModel):
    """A single-modality detection expressed in the ego/BEV frame."""

    modality: str  # "camera" | "lidar"
    x: float
    y: float
    z: float = 0.0
    cov: List[List[float]] = Field(default_factory=lambda: [[0.04, 0.0], [0.0, 0.04]])
    dims: List[float] = Field(default_factory=lambda: [4.5, 1.9, 1.6])  # l, w, h
    yaw: float = 0.0
    class_name: str = "vehicle"
    class_confidence: float = 0.5
    confidence: float = 0.5
    source_instance_id: Optional[str] = None  # debug/tests only; engines never read it


def _ray_cov(x: float, y: float, sigma_along: float, sigma_cross: float,
             origin_x: float = 0.2, origin_y: float = 0.0) -> List[List[float]]:
    """BEV covariance elongated along the sensor->object ray."""
    theta = math.atan2(y - origin_y, x - origin_x)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    cov = rot @ np.diag([sigma_along ** 2, sigma_cross ** 2]) @ rot.T
    return cov.tolist()


# ------------------------------------------------------------------ camera


def simulate_camera(frame: SceneFrame, seq: SceneSequence, rng: np.random.Generator,
                    cam_to_ego: np.ndarray = DEFAULT_CAM_TO_EGO,
                    K: np.ndarray = DEFAULT_K) -> List[Detection]:
    night = seq.time_of_day == "night"
    rain = seq.weather == "rain"
    dets: List[Detection] = []

    for gt in frame.gt:
        gx, gy, gz, l, w, h, yaw = gt.bbox_3d
        d = gt.distance
        p_miss = 0.04 + 0.10 * min(1.0, d / 70.0)
        if gt.occluded:
            p_miss += 0.82
        if night:
            p_miss += 0.35 + 0.25 * min(1.0, d / 60.0)
        if rain:
            p_miss += 0.06
        if rng.random() < min(p_miss, 0.97):
            continue

        proj = project_to_image(np.array([gx, gy, gz]), cam_to_ego, K)
        if proj is None:
            continue
        u, v, r_true = proj
        u += float(rng.normal(0, 0.4))
        v += float(rng.normal(0, 0.4))
        if not (0 <= u <= CAM_W and 0 <= v <= CAM_H):
            continue

        # Monocular depth: multiplicative range noise along the precise ray.
        r_est = r_true * float(1.0 + rng.normal(0, 0.07))
        p_ego = backproject(u, v, max(r_est, 1.0), cam_to_ego, K)
        dims = [max(0.3, dim * float(1.0 + rng.normal(0, 0.10))) for dim in (l, w, h)]

        if rng.random() < 0.94:
            cls = gt.class_name
        else:
            cls = str(rng.choice(_CONFUSION[gt.class_name]))
        sigma_along = max(0.35, 0.07 * d)
        sigma_cross = max(0.12, 0.004 * d)
        dets.append(Detection(
            modality="camera",
            x=float(p_ego[0]), y=float(p_ego[1]), z=dims[2] / 2,  # ground-plane assumption
            cov=_ray_cov(float(p_ego[0]), float(p_ego[1]), sigma_along, sigma_cross),
            dims=dims,
            yaw=float(yaw + rng.normal(0, math.radians(8))),
            class_name=cls,
            class_confidence=float(np.clip(rng.normal(0.88, 0.05), 0.5, 0.99)),
            confidence=float(np.clip(0.88 - d / 250.0 + rng.normal(0, 0.05), 0.30, 0.99)),
            source_instance_id=gt.instance_id,
        ))

    # False positives: glare/texture phantoms, more frequent at night.
    if rng.random() < (0.10 if night else 0.05):
        fx = float(rng.uniform(8, 70))
        fy = float(rng.uniform(-15, 15))
        cls = str(rng.choice(CLASSES))
        l, w, h = CLASS_DIMS[cls]
        d = math.hypot(fx, fy)
        dets.append(Detection(
            modality="camera", x=fx, y=fy, z=h / 2,
            cov=_ray_cov(fx, fy, max(0.35, 0.07 * d), max(0.12, 0.004 * d)),
            dims=[l, w, h], yaw=float(rng.uniform(-math.pi, math.pi)),
            class_name=cls, class_confidence=0.55,
            confidence=float(rng.uniform(0.35, 0.55)),
        ))
    return dets


# ------------------------------------------------------------------ lidar


def _template_class(dims: List[float], rng: np.random.Generator) -> str:
    """Weak geometric classification: nearest size template, often confused.

    Real LiDAR-only detectors get geometry right but semantics wrong; we model
    that as a 55% template guess / 45% confusion-group draw.
    """
    best, best_err = "vehicle", float("inf")
    for cls, (tl, tw, th) in CLASS_DIMS.items():
        err = (math.log(dims[0] / tl) ** 2 + math.log(dims[1] / tw) ** 2
               + math.log(dims[2] / th) ** 2)
        if err < best_err:
            best, best_err = cls, err
    if rng.random() < 0.55:
        return best
    return str(rng.choice(_CONFUSION[best] + [best]))


def simulate_lidar(frame: SceneFrame, seq: SceneSequence,
                   rng: np.random.Generator) -> List[Detection]:
    rain = seq.weather == "rain"
    dets: List[Detection] = []

    for gt in frame.gt:
        gx, gy, gz, l, w, h, yaw = gt.bbox_3d
        d = gt.distance
        p_miss = 0.02 + 0.60 * max(0.0, (d - 45.0) / 35.0)
        if rain:
            p_miss += 0.03 + 0.25 * max(0.0, (d - 25.0) / 50.0)
        if gt.occluded:
            p_miss += 0.15  # partial returns: far milder than the camera's 0.82
        if rng.random() < min(p_miss, 0.97):
            continue

        sigma = 0.10 + 0.0015 * d
        dims = [max(0.3, dim * float(1.0 + rng.normal(0, 0.035))) for dim in (l, w, h)]
        dets.append(Detection(
            modality="lidar",
            x=float(gx + rng.normal(0, sigma)),
            y=float(gy + rng.normal(0, sigma)),
            z=float(gz + rng.normal(0, sigma)),
            cov=[[sigma ** 2, 0.0], [0.0, sigma ** 2]],
            dims=dims,
            yaw=float(yaw + rng.normal(0, math.radians(3))),
            class_name=_template_class(dims, rng),
            class_confidence=float(np.clip(rng.normal(0.40, 0.05), 0.2, 0.6)),
            confidence=float(np.clip(0.93 - 0.5 * max(0.0, (d - 40.0) / 40.0)
                                     - (0.08 if rain else 0.0) + rng.normal(0, 0.03),
                                     0.30, 0.99)),
            source_instance_id=gt.instance_id,
        ))

    # Rain clutter (spray) false positives: weak, usually below decode threshold.
    if rain and rng.random() < 0.06:
        fx, fy = float(rng.uniform(5, 40)), float(rng.uniform(-12, 12))
        dets.append(Detection(
            modality="lidar", x=fx, y=fy, z=0.5,
            cov=[[0.04, 0.0], [0.0, 0.04]],
            dims=[0.8, 0.8, 1.0], yaw=0.0,
            class_name="pedestrian", class_confidence=0.25,
            confidence=float(rng.uniform(0.30, 0.45)),
        ))
    return dets


def camera_rng(seed: int, seq_index: int) -> np.random.Generator:
    return np.random.default_rng(seed * 100003 + seq_index * 7919 + 1)


def lidar_rng(seed: int, seq_index: int) -> np.random.Generator:
    return np.random.default_rng(seed * 100003 + seq_index * 7919 + 2)
