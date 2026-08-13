"""BEV rasterization, per-cell modality fusion, and box decoding.

This is the numpy analogue of BEVFusion's shared-BEV-space fusion:

1. Each modality's detections are splatted into a metric BEV feature map
   (:class:`ModalityBEV`) as Gaussian evidence. The splat covariance is the
   detection's position covariance (floored at the cell size for existence so
   tight LiDAR kernels are not lost between cell centers). Three kinds of
   evidence are accumulated per cell:

   - existence: noisy-OR of (confidence x kernel) — "is something here?";
   - geometry: inverse-variance weighted votes for the detection's continuous
     mean/dims/yaw (weight = confidence x kernel / worst-axis variance), so
     LiDAR dominates position wherever it has evidence and there is no grid
     quantization in the decoded centers;
   - class: reliability-weighted class histograms. Camera class evidence is
     splatted with an inflated along-ray kernel: monocular semantics is
     positionally vague along the ray, which is precisely what lets a camera
     label flow to the LiDAR-anchored cell of the same object even when the
     camera's depth estimate is metres off.

2. :func:`fuse_maps` fuses per cell: noisy-OR for existence, summed
   inverse-variance geometry accumulators, summed weighted class histograms
   (camera weight 1.0 vs LiDAR 0.35 — camera is the semantics expert).

3. :func:`decode_bev` extracts local existence maxima above a threshold,
   applies euclidean NMS plus ray-aware NMS (two peaks on the same bearing
   within the monocular depth-ambiguity band are one object — the fused,
   LiDAR-anchored peak wins over the camera's displaced ghost), then decodes
   each surviving peak's neighbourhood into a 3D box with class + confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import maximum_filter

from sensorflow.bevfusion.geometry import BEVGrid
from sensorflow.bevfusion.scenes import CLASSES
from sensorflow.bevfusion.sensors import Detection

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Modality reliability weights for class evidence (camera >> lidar).
CLASS_WEIGHT = {"camera": 1.0, "lidar": 0.35}
# Camera class evidence is smeared further along the ray (see module docstring).
CAMERA_CLASS_SIGMA_SCALE = 2.0


@dataclass
class ModalityBEV:
    """Per-modality BEV feature map built by splatting detections."""

    grid: BEVGrid
    modality: str = "camera"
    miss: np.ndarray = field(default=None)          # prod(1 - c*g): existence complement
    sum_w: np.ndarray = field(default=None)
    sum_wx: np.ndarray = field(default=None)
    sum_wy: np.ndarray = field(default=None)
    sum_wz: np.ndarray = field(default=None)
    sum_wl: np.ndarray = field(default=None)
    sum_ww: np.ndarray = field(default=None)
    sum_wh: np.ndarray = field(default=None)
    sum_wsin: np.ndarray = field(default=None)
    sum_wcos: np.ndarray = field(default=None)
    class_hist: np.ndarray = field(default=None)    # (nx, ny, n_classes)

    def __post_init__(self):
        nx, ny = self.grid.nx, self.grid.ny
        self.miss = np.ones((nx, ny))
        for name in ("sum_w", "sum_wx", "sum_wy", "sum_wz", "sum_wl", "sum_ww",
                     "sum_wh", "sum_wsin", "sum_wcos"):
            setattr(self, name, np.zeros((nx, ny)))
        self.class_hist = np.zeros((nx, ny, len(CLASSES)))

    @property
    def existence(self) -> np.ndarray:
        return 1.0 - self.miss

    # -- splatting

    def _window(self, mx: float, my: float, ex: float, ey: float):
        g = self.grid
        ix0 = max(0, int((mx - ex - g.x_min) / g.cell))
        ix1 = min(g.nx, int((mx + ex - g.x_min) / g.cell) + 1)
        iy0 = max(0, int((my - ey - g.y_min) / g.cell))
        iy1 = min(g.ny, int((my + ey - g.y_min) / g.cell) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            return None
        xs = g.x_min + (np.arange(ix0, ix1) + 0.5) * g.cell
        ys = g.y_min + (np.arange(iy0, iy1) + 0.5) * g.cell
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return (slice(ix0, ix1), slice(iy0, iy1)), X - mx, Y - my

    @staticmethod
    def _kernel(dx: np.ndarray, dy: np.ndarray, cov: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(cov)
        q = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
        return np.exp(-0.5 * q)

    def splat(self, det: Detection) -> None:
        cov = np.array(det.cov, dtype=float)
        # Floor the existence/geometry kernel at ~the cell size so tight LiDAR
        # Gaussians are not lost between cell centers (grid smoothing only:
        # geometry votes still use the true covariance for their weights).
        floor = (0.8 * self.grid.cell) ** 2
        cov_splat = cov + np.eye(2) * floor
        ex = 3.0 * math.sqrt(cov_splat[0, 0])
        ey = 3.0 * math.sqrt(cov_splat[1, 1])
        win = self._window(det.x, det.y, ex, ey)
        if win is None:
            return
        sl, dx, dy = win
        g = self._kernel(dx, dy, cov_splat)

        # Existence: noisy-OR accumulation.
        self.miss[sl] *= (1.0 - det.confidence * g)

        # Geometry: inverse-variance weighted votes for the *continuous* mean.
        # The kernel is normalized to unit mass so each detection contributes
        # exactly one measurement of total weight conf/sigma_eff^2 no matter
        # how widely its kernel is spread — proper inverse-variance fusion.
        eigvals = np.linalg.eigvalsh(cov)
        sigma_eff_sq = float(max(eigvals[-1], 1e-4))  # worst-axis variance
        g_mass = float(g.sum())
        if g_mass <= 1e-12:
            return
        w = det.confidence * (g / g_mass) / sigma_eff_sq
        self.sum_w[sl] += w
        self.sum_wx[sl] += w * det.x
        self.sum_wy[sl] += w * det.y
        self.sum_wz[sl] += w * det.z
        self.sum_wl[sl] += w * det.dims[0]
        self.sum_ww[sl] += w * det.dims[1]
        self.sum_wh[sl] += w * det.dims[2]
        self.sum_wsin[sl] += w * math.sin(det.yaw)
        self.sum_wcos[sl] += w * math.cos(det.yaw)

        # Class: reliability-weighted histogram; camera semantics smeared
        # further along the ray so labels reach the LiDAR-anchored cell.
        cls_idx = CLASS_INDEX.get(det.class_name)
        if cls_idx is None:
            return
        if self.modality == "camera":
            cov_cls = cov * (CAMERA_CLASS_SIGMA_SCALE ** 2) + np.eye(2) * floor
        else:
            cov_cls = cov * 4.0 + np.eye(2) * floor
        exc = 3.0 * math.sqrt(cov_cls[0, 0])
        eyc = 3.0 * math.sqrt(cov_cls[1, 1])
        winc = self._window(det.x, det.y, exc, eyc)
        if winc is None:
            return
        slc, dxc, dyc = winc
        gc = self._kernel(dxc, dyc, cov_cls)
        wc = det.confidence * det.class_confidence * CLASS_WEIGHT.get(self.modality, 0.5)
        self.class_hist[slc[0], slc[1], cls_idx] += wc * gc


def build_modality_map(dets: List[Detection], grid: BEVGrid, modality: str) -> ModalityBEV:
    bev = ModalityBEV(grid=grid, modality=modality)
    for det in dets:
        if det.modality == modality:
            bev.splat(det)
    return bev


@dataclass
class FusedBEV:
    grid: BEVGrid
    existence: np.ndarray
    sum_w: np.ndarray
    sum_wx: np.ndarray
    sum_wy: np.ndarray
    sum_wz: np.ndarray
    sum_wl: np.ndarray
    sum_ww: np.ndarray
    sum_wh: np.ndarray
    sum_wsin: np.ndarray
    sum_wcos: np.ndarray
    class_hist: np.ndarray


def fuse_maps(cam: ModalityBEV, lidar: ModalityBEV) -> FusedBEV:
    """Per-cell confidence-weighted fusion of the two modality feature maps."""
    return FusedBEV(
        grid=cam.grid,
        existence=1.0 - cam.miss * lidar.miss,  # noisy-OR across modalities
        sum_w=cam.sum_w + lidar.sum_w,
        sum_wx=cam.sum_wx + lidar.sum_wx,
        sum_wy=cam.sum_wy + lidar.sum_wy,
        sum_wz=cam.sum_wz + lidar.sum_wz,
        sum_wl=cam.sum_wl + lidar.sum_wl,
        sum_ww=cam.sum_ww + lidar.sum_ww,
        sum_wh=cam.sum_wh + lidar.sum_wh,
        sum_wsin=cam.sum_wsin + lidar.sum_wsin,
        sum_wcos=cam.sum_wcos + lidar.sum_wcos,
        class_hist=cam.class_hist + lidar.class_hist,
    )


def decode_bev(fused: FusedBEV, threshold: float = 0.30,
               nms_radius_m: float = 2.0, gather_radius_m: float = 2.0,
               ray_nms_bearing_deg: float = 1.5,
               ray_nms_range_m: float = 6.0) -> List[Dict]:
    """Decode fused BEV cells into 3D boxes with class + confidence."""
    grid = fused.grid
    e = fused.existence
    local_max = (e >= maximum_filter(e, size=5)) & (e >= threshold)
    ixs, iys = np.nonzero(local_max)
    if ixs.size == 0:
        return []
    order = np.argsort(-e[ixs, iys])
    peaks = [(int(ixs[i]), int(iys[i])) for i in order]

    accepted: List[Dict] = []
    gather_cells = max(1, int(round(gather_radius_m / grid.cell)))
    for ix, iy in peaks:
        cx, cy = grid.center(ix, iy)
        suppress = False
        for box in accepted:
            bx, by = box["_peak_xy"]
            if math.hypot(cx - bx, cy - by) < nms_radius_m:
                suppress = True
                break
            # Ray-aware NMS: same bearing, within the monocular depth
            # ambiguity band -> the weaker peak is the camera's range ghost.
            db = abs(math.atan2(cy, cx) - math.atan2(by, bx))
            db = min(db, 2 * math.pi - db)
            if math.degrees(db) < ray_nms_bearing_deg and \
                    abs(math.hypot(cx, cy) - math.hypot(bx, by)) < ray_nms_range_m:
                suppress = True
                break
        if suppress:
            continue

        x0, x1 = max(0, ix - gather_cells), min(grid.nx, ix + gather_cells + 1)
        y0, y1 = max(0, iy - gather_cells), min(grid.ny, iy + gather_cells + 1)
        sl = (slice(x0, x1), slice(y0, y1))
        w = float(fused.sum_w[sl].sum())
        if w <= 1e-9:
            continue
        x = float(fused.sum_wx[sl].sum() / w)
        y = float(fused.sum_wy[sl].sum() / w)
        z = float(fused.sum_wz[sl].sum() / w)
        l = float(fused.sum_wl[sl].sum() / w)
        wd = float(fused.sum_ww[sl].sum() / w)
        h = float(fused.sum_wh[sl].sum() / w)
        yaw = float(math.atan2(fused.sum_wsin[sl].sum(), fused.sum_wcos[sl].sum()))
        hist = fused.class_hist[sl].sum(axis=(0, 1))
        hist_total = float(hist.sum())
        cls_idx = int(np.argmax(hist)) if hist_total > 0 else 0
        accepted.append({
            "bbox_3d": [x, y, z, l, wd, h, yaw],
            "class_name": CLASSES[cls_idx],
            "class_confidence": round(float(hist[cls_idx] / hist_total), 4) if hist_total > 0 else 0.0,
            "confidence": round(float(e[ix, iy]), 4),
            "_peak_xy": (cx, cy),
        })

    for box in accepted:
        box.pop("_peak_xy", None)
    return accepted


def fuse_frame(camera_dets: List[Detection], lidar_dets: List[Detection],
               grid: Optional[BEVGrid] = None, threshold: float = 0.30) -> List[Dict]:
    """Convenience: full rasterize -> fuse -> decode for one frame."""
    grid = grid or BEVGrid()
    cam_map = build_modality_map(camera_dets, grid, "camera")
    lidar_map = build_modality_map(lidar_dets, grid, "lidar")
    return decode_bev(fuse_maps(cam_map, lidar_map), threshold=threshold)
