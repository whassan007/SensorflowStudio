"""Multi-sensor extrinsic calibration validation (Deepen-style safety gate).

Models a 6-DoF extrinsic transform (rotation + translation) for a LiDAR ->
camera sensor pair and validates it by projecting LiDAR points of labeled
objects into the camera frame and measuring projection consistency against the
objects' 2D boxes:

- reprojection residual: the labeled 3D box corners are projected through the
  assumed extrinsic; residual = projected-box center vs labeled 2D box center (px)
- inlier ratio: fraction of the object's projected LiDAR points inside the
  (padded) 2D box

Diagnosis separates two failure modes:
- MISCALIBRATED       systematic bias — residual vectors share a common
                      direction/magnitude across objects (extrinsic drift);
                      the rotation offset is estimated from the bias.
- PERCEPTION_FAILURE  random disagreement — a minority of objects have large
                      residuals in inconsistent directions while the global
                      bias stays small (bad 2D detections, not calibration);
                      flagged objects carry the platform's existing
                      SENSOR_DISAGREEMENT failure reason.

SIMULATED: scenes are synthetic (deterministic given seed). Calibrated scenes
are generated with the true extrinsic; the validator then either uses a
perturbed extrinsic (miscalibration injection) or tampers a fraction of the 2D
boxes (perception-failure injection) to prove the detector works. Results feed
the Scenario Quality Gate (gates.py) and the labeleval alerting store.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from sensorflow.safety.store import read_json, write_json

# Pinhole intrinsics matching the platform's synthetic camera (800x450, f=500).
INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 400.0, "cy": 225.0,
              "width": 800, "height": 450}

# True LiDAR->camera extrinsic. LiDAR frame: x fwd / y left / z up.
# Camera frame: x right / y down / z fwd.  R0 maps axes; t0 is the lever arm (m).
R0 = np.array([[0.0, -1.0, 0.0],
               [0.0, 0.0, -1.0],
               [1.0, 0.0, 0.0]])
T0 = np.array([0.06, -0.12, 0.15])

CLASS_DIMS = {  # l, w, h (m) — mirrors the labeleval synthetic priors
    "vehicle": (4.5, 1.9, 1.6),
    "truck": (8.0, 2.5, 3.1),
    "pedestrian": (0.7, 0.7, 1.75),
    "cyclist": (1.8, 0.6, 1.7),
}

DEFAULT_THRESHOLDS = {
    "bias_px": 6.0,             # systematic residual => miscalibration suspect
    "bias_to_scatter_ratio": 1.5,
    "outlier_px": 14.0,         # per-object residual => perception suspect
    "min_outlier_fraction": 0.10,
    "min_inlier_ratio": 0.75,   # mean over objects for a healthy pair
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rotation_matrix(roll_deg: float = 0.0, pitch_deg: float = 0.0,
                    yaw_deg: float = 0.0) -> np.ndarray:
    """Intrinsic XYZ Euler rotation (applied in the camera frame)."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_extrinsic(rotation: Optional[np.ndarray] = None,
                   translation: Optional[np.ndarray] = None) -> Dict:
    return {"R": (rotation if rotation is not None else R0),
            "t": (translation if translation is not None else T0)}


def perturb_extrinsic(extrinsic: Dict, rotation_offset_deg: float = 0.0,
                      axis: str = "pitch", translation_offset_m: float = 0.0) -> Dict:
    kw = {f"{axis}_deg": rotation_offset_deg}
    r_off = rotation_matrix(**kw)
    t_off = np.array([translation_offset_m, 0.0, 0.0])
    return {"R": r_off @ extrinsic["R"], "t": extrinsic["t"] + t_off}


def project_points(points_lidar: np.ndarray, extrinsic: Dict) -> np.ndarray:
    """Project Nx3 LiDAR points to Nx2 pixel coordinates."""
    cam = points_lidar @ extrinsic["R"].T + extrinsic["t"]
    z = np.clip(cam[:, 2], 0.1, None)
    u = INTRINSICS["fx"] * cam[:, 0] / z + INTRINSICS["cx"]
    v = INTRINSICS["fy"] * cam[:, 1] / z + INTRINSICS["cy"]
    return np.stack([u, v], axis=1)


# ------------------------------------------------------------------ scene generation


def generate_scene(num_objects: int = 14, seed: int = 3) -> Dict:
    """Deterministic calibrated scene: 3D objects with LiDAR point clusters and
    2D boxes produced by projecting through the TRUE extrinsic."""
    rng = np.random.default_rng(seed)
    classes = list(CLASS_DIMS)
    objects = []
    attempts = 0
    while len(objects) < num_objects and attempts < num_objects * 20:
        attempts += 1
        cls = classes[int(rng.integers(0, len(classes)))]
        l, w, h = CLASS_DIMS[cls]
        center = np.array([float(rng.uniform(8, 38)),
                           float(rng.uniform(-9, 9)),
                           h / 2])
        n_pts = int(np.clip(240 * 12.0 / center[0], 40, 260))
        pts = center + rng.uniform([-l / 2, -w / 2, -h / 2], [l / 2, w / 2, h / 2],
                                   size=(n_pts, 3))
        corners = center + np.array([[sx * l / 2, sy * w / 2, sz * h / 2]
                                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        uv = project_points(corners, make_extrinsic())
        u0, v0 = uv.min(axis=0)
        u1, v1 = uv.max(axis=0)
        if u0 < 0 or v0 < 0 or u1 > INTRINSICS["width"] or v1 > INTRINSICS["height"]:
            continue  # keep only fully-visible objects
        # 2D label noise: real camera labels are not pixel-perfect.
        jitter = rng.normal(0.0, 1.2, size=4)
        objects.append({
            "object_id": f"cal-obj-{len(objects)}",
            "class_name": cls,
            "center_lidar": center.tolist(),
            "corners_lidar": corners,
            "points_lidar": pts,
            "bbox_2d": [float(u0 + jitter[0]), float(v0 + jitter[1]),
                        float(u1 - u0 + jitter[2]), float(v1 - v0 + jitter[3])],
        })
    return {"seed": seed, "objects": objects, "true_extrinsic": make_extrinsic(),
            "simulated": True}


# ------------------------------------------------------------------ validation


def validate_scene(scene: Dict, assumed_extrinsic: Dict,
                   thresholds: Optional[Dict] = None) -> Dict:
    """Project each object's LiDAR points with the assumed extrinsic and measure
    consistency against its 2D box; classify the failure mode."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    per_object: List[Dict] = []
    residuals = []
    for obj in scene["objects"]:
        corner_uv = project_points(np.asarray(obj["corners_lidar"]), assumed_extrinsic)
        proj_center = (corner_uv.min(axis=0) + corner_uv.max(axis=0)) / 2
        uv = project_points(np.asarray(obj["points_lidar"]), assumed_extrinsic)
        bx, by, bw, bh = obj["bbox_2d"]
        box_center = np.array([bx + bw / 2, by + bh / 2])
        residual = proj_center - box_center
        pad_x, pad_y = 0.15 * bw, 0.15 * bh
        inside = ((uv[:, 0] >= bx - pad_x) & (uv[:, 0] <= bx + bw + pad_x)
                  & (uv[:, 1] >= by - pad_y) & (uv[:, 1] <= by + bh + pad_y))
        inlier_ratio = float(inside.mean())
        r_mag = float(np.linalg.norm(residual))
        residuals.append(residual)
        per_object.append({
            "object_id": obj["object_id"],
            "class_name": obj["class_name"],
            "residual_px": [round(float(residual[0]), 2), round(float(residual[1]), 2)],
            "residual_magnitude_px": round(r_mag, 2),
            "inlier_ratio": round(inlier_ratio, 4),
            "flagged": False,
            "failure_reason": None,
        })

    residuals = np.array(residuals)
    mean_res = residuals.mean(axis=0)
    bias = float(np.linalg.norm(mean_res))
    scatter = float(np.linalg.norm(residuals - mean_res, axis=1).mean())
    mean_inlier = float(np.mean([o["inlier_ratio"] for o in per_object]))
    outliers = [o for o in per_object if o["residual_magnitude_px"] > th["outlier_px"]]
    outlier_fraction = len(outliers) / max(len(per_object), 1)

    systematic = bias > th["bias_px"] and bias > th["bias_to_scatter_ratio"] * scatter
    if systematic:
        status = "MISCALIBRATED"
        for o in per_object:
            o["flagged"] = True
            o["failure_reason"] = "MISCALIBRATION"
    elif outlier_fraction >= th["min_outlier_fraction"]:
        status = "PERCEPTION_FAILURE"
        for o in outliers:
            o["flagged"] = True
            # existing platform failure reason (records.FAILURE_REASONS)
            o["failure_reason"] = "SENSOR_DISAGREEMENT"
    else:
        status = "CALIBRATED"

    est_rot_deg = math.degrees(math.atan(bias / INTRINSICS["fx"]))
    checks = [
        {"name": "systematic_bias_px", "actual": round(bias, 2),
         "threshold": th["bias_px"], "passed": bias <= th["bias_px"]},
        {"name": "outlier_fraction", "actual": round(outlier_fraction, 4),
         "threshold": th["min_outlier_fraction"],
         "passed": outlier_fraction < th["min_outlier_fraction"]},
        {"name": "mean_inlier_ratio", "actual": round(mean_inlier, 4),
         "threshold": th["min_inlier_ratio"],
         "passed": mean_inlier >= th["min_inlier_ratio"]},
    ]
    return {
        "status": status,
        "passed": status == "CALIBRATED",
        "sensor_pair": "lidar_top->camera_front",
        "checks": checks,
        "metrics": {
            "bias_px": round(bias, 3),
            "bias_vector_px": [round(float(mean_res[0]), 3), round(float(mean_res[1]), 3)],
            "scatter_px": round(scatter, 3),
            "mean_inlier_ratio": round(mean_inlier, 4),
            "outlier_fraction": round(outlier_fraction, 4),
            "estimated_rotation_offset_deg": round(est_rot_deg, 3),
            "num_objects": len(per_object),
        },
        "per_object": per_object,
        "thresholds": th,
        "diagnosis": {
            "MISCALIBRATED": "systematic bias shared across objects — extrinsic drift",
            "PERCEPTION_FAILURE": "random disagreement on a minority of objects — "
                                  "detection defects, mapped to SENSOR_DISAGREEMENT",
            "CALIBRATED": "projection consistent within thresholds",
        }[status],
        "simulated": True,
    }


def run_validation(mode: str = "clean", rotation_offset_deg: float = 2.0,
                   translation_offset_m: float = 0.0, tamper_fraction: float = 0.25,
                   num_objects: int = 14, seed: int = 3,
                   thresholds: Optional[Dict] = None, persist: bool = True) -> Dict:
    """Generate a calibrated scene, optionally inject a fault, and validate.

    Modes: clean | miscalibrated (extrinsic perturbed by rotation_offset_deg)
    | perception_failure (a fraction of 2D boxes tampered)."""
    scene = generate_scene(num_objects=num_objects, seed=seed)
    assumed = make_extrinsic()
    if mode == "miscalibrated":
        assumed = perturb_extrinsic(assumed, rotation_offset_deg=rotation_offset_deg,
                                    axis="pitch", translation_offset_m=translation_offset_m)
    elif mode == "perception_failure":
        rng = np.random.default_rng(seed + 77)
        n_tamper = max(1, int(round(tamper_fraction * len(scene["objects"]))))
        idx = rng.choice(len(scene["objects"]), size=n_tamper, replace=False)
        for i in idx:
            box = scene["objects"][int(i)]["bbox_2d"]
            box[0] += float(rng.choice([-1, 1]) * rng.uniform(25, 45))
            box[1] += float(rng.choice([-1, 1]) * rng.uniform(20, 35))
    elif mode != "clean":
        raise ValueError(f"unknown mode {mode!r}")

    result = validate_scene(scene, assumed, thresholds)
    result["mode"] = mode
    result["injected"] = {
        "rotation_offset_deg": rotation_offset_deg if mode == "miscalibrated" else 0.0,
        "translation_offset_m": translation_offset_m if mode == "miscalibrated" else 0.0,
        "tamper_fraction": tamper_fraction if mode == "perception_failure" else 0.0,
    }
    result["seed"] = seed
    result["created_at"] = _now()

    if persist:
        slim = {k: v for k, v in result.items() if k != "per_object"}
        slim["per_object"] = [o for o in result["per_object"] if o["flagged"]][:25]
        write_json(slim, "calibration", "status.json")
        _emit_alert(result)
    return result


def latest_status() -> Optional[Dict]:
    return read_json("calibration", "status.json")


def _emit_alert(result: Dict) -> None:
    """Surface failures in the existing labeleval alerting store (best-effort)."""
    if result["status"] == "CALIBRATED":
        return
    try:
        from sensorflow.evaluation.records import Alert, get_store, new_id
        store = get_store()
        store.put("alerts", Alert(
            alert_id=new_id("alert"), kind="calibration",
            severity="critical" if result["status"] == "MISCALIBRATED" else "warning",
            message=(f"Calibration validation: {result['status']} — "
                     f"bias {result['metrics']['bias_px']}px, "
                     f"outliers {result['metrics']['outlier_fraction']*100:.0f}%, "
                     f"{result['diagnosis']}"),
            evidence_page="safety", evidence_id="calibration"))
        store.save()
    except Exception:
        pass
