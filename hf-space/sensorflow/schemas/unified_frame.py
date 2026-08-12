"""OpenLABEL-inspired unified frame/sequence schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from sensorflow.schemas.taxonomy_axes import TaxonomyAxes


class EgoPose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    speed_kmh: float = 0.0


class LidarData(BaseModel):
  path: Optional[str] = None
  num_points: int = 0
  format: str = "bin"


class CameraView(BaseModel):
    image_path: str
    intrinsics: List[List[float]] = Field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    extrinsics: List[List[float]] = Field(default_factory=lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class Calibration(BaseModel):
    lidar_to_ego: List[List[float]] = Field(default_factory=lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    camera_to_ego: Dict[str, List[List[float]]] = Field(default_factory=dict)


class Object3D(BaseModel):
    bbox_3d: List[float] = Field(description="[x, y, z, l, w, h, yaw]")
    class_name: str = "unknown"
    confidence: float = 0.0
    track_id: Optional[int] = None
    sam_mask_ref: Optional[str] = None
    taxonomy_axes: TaxonomyAxes = Field(default_factory=TaxonomyAxes)
    object_id: Optional[str] = None


class GroundTruthObject(Object3D):
    instance_id: str = ""


class FusedFrame(BaseModel):
    frame_id: str
    timestamp_us: int = 0
    lidar: LidarData = Field(default_factory=LidarData)
    cameras: Dict[str, CameraView] = Field(default_factory=dict)
    ego_pose: EgoPose = Field(default_factory=EgoPose)
    proposals: List[Object3D] = Field(default_factory=list)
    ground_truth: List[GroundTruthObject] = Field(default_factory=list)


class UnifiedSequence(BaseModel):
    sequence_id: str
    vendor: Literal["alpamayo", "waymo", "mixed"] = "mixed"
    frames: List[FusedFrame] = Field(default_factory=list)
    calibration: Calibration = Field(default_factory=Calibration)
    taxonomy_manifest: Dict[str, Any] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "UnifiedSequence":
        with open(path) as f:
            return cls.model_validate(json.load(f))
