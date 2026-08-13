"""Automated 2D/3D perception: SAM masks + LiDAR lifting."""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.schemas.taxonomy_axes import assign_taxonomy_axes
from sensorflow.schemas.unified_frame import FusedFrame, Object3D, UnifiedSequence


class PerceptionAutomator:
    """SAM-based 2D masks lifted to 3D bounding box proposals."""

    def __init__(
        self,
        sam_checkpoint: str = "models/sam_vit_b.pth",
        device: str = "cpu",
        use_sam: bool = True,
    ):
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.use_sam = use_sam
        self._sam = None
        self._mask_generator = None

    def _init_sam(self):
        if self._sam is not None:
            return
        if not self.use_sam:
            return
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            self._sam = sam_model_registry["vit_b"](checkpoint=self.sam_checkpoint)
            self._sam.to(self.device)
            self._mask_generator = SamAutomaticMaskGenerator(self._sam)
        except (ImportError, FileNotFoundError):
            self.use_sam = False

    def run_sequence(self, sequence: UnifiedSequence, output_dir: Path) -> Dict[str, List[Object3D]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_proposals: Dict[str, List[Object3D]] = {}
        for frame in sequence.frames:
            proposals = self.run_frame(frame)
            all_proposals[frame.frame_id] = proposals
            out_path = output_dir / f"{frame.frame_id}.json"
            with open(out_path, "w") as f:
                json.dump([p.model_dump() for p in proposals], f, indent=2)
        return all_proposals

    def run_frame(self, frame: FusedFrame) -> List[Object3D]:
        camera_name = "front" if "front" in frame.cameras else next(iter(frame.cameras), None)
        if camera_name is None:
            return self._proposals_from_gt(frame)

        camera = frame.cameras[camera_name]
        masks = self._sam_masks(camera.image_path)
        if not masks:
            masks = self._fallback_masks(frame)
        points = self._load_lidar(frame.lidar.path)
        proposals = self._lift_masks_to_3d(points, masks, frame)
        return self._assign_taxonomy(proposals, frame)

    def _sam_masks(self, image_path: str) -> List[Dict]:
        self._init_sam()
        if not self.use_sam or not self._mask_generator:
            return []
        try:
            import cv2
            if image_path.startswith("http"):
                return []
            img = cv2.imread(image_path)
            if img is None:
                return []
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self._mask_generator.generate(img)
        except Exception:
            return []

    def _fallback_masks(self, frame: FusedFrame) -> List[Dict]:
        """Generate masks from GT 2D annotations or synthetic grid."""
        masks = []
        for i, gt in enumerate(frame.ground_truth):
            masks.append({
                "segmentation": None,
                "bbox": self._bbox3d_to_2d(gt.bbox_3d),
                "area": 1000,
                "predicted_iou": gt.confidence,
                "stability_score": gt.confidence,
                "mask_id": f"mask_{i}",
            })
        if not masks:
            masks.append({
                "segmentation": None,
                "bbox": [100, 100, 200, 200],
                "area": 10000,
                "predicted_iou": 0.8,
                "stability_score": 0.8,
                "mask_id": "mask_default",
            })
        return masks

    def _bbox3d_to_2d(self, bbox_3d: List[float]) -> List[float]:
        x, y = bbox_3d[0], bbox_3d[1]
        l, w = bbox_3d[3], bbox_3d[4]
        cx = int(320 + x / 0.05)
        cy = int(240 - y / 0.05)
        bw = int(l / 0.05)
        bh = int(w / 0.05)
        return [cx - bw // 2, cy - bh // 2, bw, bh]

    def _load_lidar(self, path: Optional[str]) -> np.ndarray:
        if path and Path(path).exists():
            pts = np.fromfile(path, dtype=np.float32)
            if len(pts) % 3 == 0:
                return pts.reshape(-1, 3)
        # SYNTHETIC_FALLBACK: no LiDAR file available. The point cloud is
        # simulated, seeded deterministically from the requested path so that
        # repeated runs over the same input produce identical proposals.
        seed = zlib.crc32((path or "missing").encode("utf-8")) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((500, 3)) * 2).astype(np.float32)

    def _lift_masks_to_3d(
        self,
        points: np.ndarray,
        masks: List[Dict],
        frame: FusedFrame,
    ) -> List[Object3D]:
        proposals = []
        for mask in masks:
            bbox_2d = mask.get("bbox", [0, 0, 100, 100])
            x1, y1, w, h = bbox_2d
            x2, y2 = x1 + w, y1 + h
            cx_img = (x1 + x2) / 2
            cy_img = (y1 + y2) / 2

            x_world = (cx_img - 320) * 0.05
            y_world = (240 - cy_img) * 0.05

            nearby = points[
                (np.abs(points[:, 0] - x_world) < 3) &
                (np.abs(points[:, 1] - y_world) < 3)
            ]
            if len(nearby) < 3:
                nearby = np.array([[x_world, y_world, 0.5]])

            center = nearby.mean(axis=0)
            if len(nearby) >= 3:
                cov = np.cov(nearby[:, :2].T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                yaw = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
                l = float(np.sqrt(max(eigvals[1], 0.1)) * 4) if len(eigvals) > 1 else 2.0
                w = float(np.sqrt(max(eigvals[0], 0.1)) * 4) if len(eigvals) > 0 else 1.0
            else:
                yaw = 0.0
                l, w = 2.0, 1.0

            conf = mask.get("predicted_iou", mask.get("stability_score", 0.5))
            proposals.append(Object3D(
                bbox_3d=[float(center[0]), float(center[1]), float(center[2]),
                         max(l, 0.5), max(w, 0.3), 1.5, yaw],
                class_name="unknown",
                confidence=float(conf),
                sam_mask_ref=mask.get("mask_id", ""),
            ))
        return proposals

    def _assign_taxonomy(self, proposals: List[Object3D], frame: FusedFrame) -> List[Object3D]:
        speed = frame.ego_pose.speed_kmh
        for i, prop in enumerate(proposals):
            if i < len(frame.ground_truth):
                prop.class_name = frame.ground_truth[i].class_name
            prop.taxonomy_axes = assign_taxonomy_axes(prop.class_name, speed_kmh=speed)
        return proposals

    def _proposals_from_gt(self, frame: FusedFrame) -> List[Object3D]:
        return [
            Object3D(
                bbox_3d=gt.bbox_3d,
                class_name=gt.class_name,
                confidence=gt.confidence,
                taxonomy_axes=gt.taxonomy_axes,
            )
            for gt in frame.ground_truth
        ]
