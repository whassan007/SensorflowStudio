"""Calibrated camera geometry + metric BEV grid.

Coordinate conventions (matching the rest of the platform, e.g.
sensorflow/evaluation/synthetic.py): ego frame has x forward (m), y left (m),
z up (m), ground plane at z=0.

The camera uses the standard optical convention (z forward, x right, y down);
`make_camera_extrinsic` builds the 6-DoF camera-to-ego transform, mirroring
the Calibration.camera_to_ego 4x4 matrices in
sensorflow/schemas/unified_frame.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Default intrinsics, consistent with the existing synthetic generator
# (sensorflow/evaluation/synthetic.py: CAM_W, CAM_H, CAM_F).
CAM_W, CAM_H, CAM_F = 800, 450, 500.0


def make_intrinsics(fx: float = CAM_F, fy: float = CAM_F,
                    cx: float = CAM_W / 2, cy: float = CAM_H / 2) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def make_camera_extrinsic(tx: float = 0.2, ty: float = 0.0, tz: float = 1.5,
                          yaw: float = 0.0) -> np.ndarray:
    """4x4 camera-to-ego transform for a forward-looking camera.

    Camera axes expressed in the ego frame (before yaw): optical axis
    z_cam -> +x_ego, x_cam (image right) -> -y_ego, y_cam (image down) -> -z_ego.
    """
    base = np.array([
        [0.0, 0.0, 1.0],   # ego x row: from cam z
        [-1.0, 0.0, 0.0],  # ego y row: from -cam x
        [0.0, -1.0, 0.0],  # ego z row: from -cam y
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = rz @ base
    T[:3, 3] = [tx, ty, tz]
    return T


def project_to_image(point_ego: np.ndarray, cam_to_ego: np.ndarray,
                     K: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Ego-frame point -> (u, v, range). None if behind the camera."""
    R, t = cam_to_ego[:3, :3], cam_to_ego[:3, 3]
    p_cam = R.T @ (np.asarray(point_ego, dtype=float) - t)
    if p_cam[2] < 0.5:  # behind or too close to the optical center
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    rng = float(np.linalg.norm(p_cam))
    return float(u), float(v), rng


def backproject(u: float, v: float, range_est: float, cam_to_ego: np.ndarray,
                K: np.ndarray) -> np.ndarray:
    """Pixel + estimated range along the viewing ray -> ego-frame 3D point.

    This is the monocular "lift" step: the bearing (pixel) is precise, the
    range is whatever the depth estimate says — hence the along-ray
    uncertainty that the BEV fusion exploits.
    """
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d_cam /= np.linalg.norm(d_cam)
    R, t = cam_to_ego[:3, :3], cam_to_ego[:3, 3]
    return R @ (d_cam * range_est) + t


@dataclass(frozen=True)
class BEVGrid:
    """Metric BEV grid over the ego frame (x forward, y left)."""

    x_min: float = 0.0
    x_max: float = 80.0
    y_min: float = -25.0
    y_max: float = 25.0
    cell: float = 0.5

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.cell))

    def index(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        ix = int((x - self.x_min) / self.cell)
        iy = int((y - self.y_min) / self.cell)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None

    def center(self, ix: int, iy: int) -> Tuple[float, float]:
        return (self.x_min + (ix + 0.5) * self.cell,
                self.y_min + (iy + 0.5) * self.cell)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max
