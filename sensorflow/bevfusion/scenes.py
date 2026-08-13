"""Deterministic synthetic multi-sensor scenes with known 3D ground truth.

Mirrors the conventions of the existing evaluation generator
(sensorflow/evaluation/synthetic.py: ego drives +x at 10 m/s, objects have
world-frame kinematics, class dimensions from CLASS_DIMS) but is purpose-built
for the sensor-simulation study: every ground-truth box carries a per-frame
camera-occlusion flag (planted contiguous occlusion windows) so that both
sensor simulators and the cohort evaluation can reason about occlusion.

Conditions cycle deterministically across sequences (day/clear, night/clear,
day/rain) so every cohort is well represented at any n_sequences >= 3.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from pydantic import BaseModel, Field

# Reuse the platform's class dimension priors rather than duplicating them.
from sensorflow.evaluation.synthetic import CLASS_DIMS, CLASS_POOL

CLASSES = list(CLASS_DIMS.keys())
SAFETY_CLASSES = {"pedestrian", "cyclist", "motorcycle"}

DT = 0.1
EGO_SPEED = 10.0


class GTBox(BaseModel):
    instance_id: str
    class_name: str
    bbox_3d: List[float]  # [x, y, z, l, w, h, yaw] ego frame
    occluded: bool = False  # camera line-of-sight blocked this frame
    distance: float = 0.0


class SceneFrame(BaseModel):
    frame_id: str
    index: int
    gt: List[GTBox] = Field(default_factory=list)


class SceneSequence(BaseModel):
    sequence_id: str
    time_of_day: str = "day"    # day | night
    weather: str = "clear"      # clear | rain
    frames: List[SceneFrame] = Field(default_factory=list)


def generate_sequences(n_sequences: int = 6, frames_per_sequence: int = 24,
                       seed: int = 7) -> List[SceneSequence]:
    """Generate deterministic sequences with planted camera-occlusion windows."""
    sequences: List[SceneSequence] = []
    for qi in range(n_sequences):
        rng = np.random.default_rng(seed * 100003 + qi * 7919)
        time_of_day = "night" if qi % 3 == 1 else "day"
        weather = "rain" if qi % 3 == 2 else "clear"
        seq = SceneSequence(sequence_id=f"bev-seq-{seed}-{qi}",
                            time_of_day=time_of_day, weather=weather)

        num_objects = int(rng.integers(7, 11))
        objects = []
        for oi in range(num_objects):
            cls = CLASS_POOL[int(rng.integers(0, len(CLASS_POOL)))]
            l, w, h = CLASS_DIMS[cls]
            # Rejection-sample spawn positions so objects start >= 5 m apart
            # (keeps BEV peaks separable; documented decode limitation).
            for _ in range(30):
                x0 = float(rng.uniform(10, 68))
                y0 = float(rng.uniform(-13, 13))
                if all(math.hypot(x0 - o["x"], y0 - o["y"]) >= 5.0 for o in objects):
                    break
            heading = (float(rng.uniform(-math.pi, math.pi)) if cls == "pedestrian"
                       else float(rng.choice([0.0, math.pi, math.pi / 2, -math.pi / 2])))
            speed = {"pedestrian": 1.4, "cyclist": 4.5, "motorcycle": 9.0,
                     "vehicle": 8.0, "truck": 7.0}[cls] * float(rng.uniform(0.6, 1.2))
            objects.append({
                "iid": f"{seq.sequence_id}-obj-{oi}", "cls": cls, "dims": (l, w, h),
                "x": x0, "y": y0,
                "vx": speed * math.cos(heading), "vy": speed * math.sin(heading),
                "yaw": heading,
            })

        # Plant camera-occlusion windows: 2 objects per sequence lose camera
        # line of sight for a contiguous 2-3 frame window mid-sequence.
        occlusion = {}
        if objects and frames_per_sequence >= 10:
            occluded_idx = rng.choice(len(objects), size=min(2, len(objects)), replace=False)
            for k, oi in enumerate(occluded_idx):
                start = int(rng.integers(4, max(5, frames_per_sequence - 6)))
                length = int(rng.integers(2, 4))
                occlusion[objects[int(oi)]["iid"]] = (start, start + length)

        for fi in range(frames_per_sequence):
            ego_x = EGO_SPEED * DT * fi
            frame = SceneFrame(frame_id=f"{seq.sequence_id}-f{fi:04d}", index=fi)
            for obj in objects:
                wx = obj["x"] + obj["vx"] * DT * fi
                wy = obj["y"] + obj["vy"] * DT * fi
                rel_x, rel_y = wx - ego_x, wy
                if rel_x < 3.0 or rel_x > 78.0 or abs(rel_y) > 22.0:
                    continue
                l, w, h = obj["dims"]
                win = occlusion.get(obj["iid"])
                frame.gt.append(GTBox(
                    instance_id=obj["iid"],
                    class_name=obj["cls"],
                    bbox_3d=[round(rel_x, 3), round(rel_y, 3), round(h / 2, 3),
                             l, w, h, round(obj["yaw"], 4)],
                    occluded=bool(win and win[0] <= fi < win[1]),
                    distance=round(math.hypot(rel_x, rel_y), 3),
                ))
            seq.frames.append(frame)
        sequences.append(seq)
    return sequences


def gt_tracks(sequences: List[SceneSequence]) -> List[dict]:
    """Ground-truth tracks in the shape expected by sensorflow.metrics.temporal_mot."""
    by_instance: dict = {}
    for seq in sequences:
        for frame in seq.frames:
            for gt in frame.gt:
                entry = by_instance.setdefault(
                    gt.instance_id,
                    {"instance_id": gt.instance_id, "track_id": gt.instance_id,
                     "class_name": gt.class_name, "frames": []})
                entry["frames"].append({"frame_id": frame.frame_id, "bbox_3d": gt.bbox_3d})
    return list(by_instance.values())
