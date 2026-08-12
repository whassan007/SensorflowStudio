"""Resource profiling in deterministic process_units and compute_cycles."""

from __future__ import annotations

from typing import Dict, List

from sensorflow.schemas.unified_frame import FusedFrame


COMPLEXITY_FACTORS = {
    "sam": 1.0,
    "lift_3d": 0.3,
    "tracker": 0.1,
    "benchmark": 0.05,
}


def compute_resource_profile(
    frames: List[FusedFrame],
    stages_run: List[str] = None,
) -> Dict[str, int]:
    """
    Compute deterministic resource cost without wall-clock time.
    process_units = frames * cameras * (lidar_points / 1000)
    compute_cycles = process_units * sum(complexity_factors for stages)
    """
    stages_run = stages_run or ["sam", "lift_3d", "tracker", "benchmark"]
    total_points = sum(f.lidar.num_points for f in frames)
    num_cameras = max((len(f.cameras) for f in frames), default=1)
    num_frames = len(frames)

    process_units = int(num_frames * num_cameras * max(total_points / max(num_frames, 1) / 1000, 1))
    factor_sum = sum(COMPLEXITY_FACTORS.get(s, 0.1) for s in stages_run)
    compute_cycles = int(process_units * factor_sum)

    return {
        "process_units": process_units,
        "compute_cycles": compute_cycles,
        "num_frames": num_frames,
        "num_cameras": num_cameras,
        "total_lidar_points": total_points,
    }
