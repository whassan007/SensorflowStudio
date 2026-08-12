"""Synthetic sensor data + auto-label generator with known error injection.

Generates a deterministic dataset (scenes -> sequences -> frames -> GT boxes,
ego motion, LiDAR point clouds) and then auto-generated labels (hypotheses)
with injected, recorded defects so every evaluation subsystem can be verified:
FALSE_POSITIVE, FALSE_NEGATIVE, BAD_3D_BOX, WRONG_ORIENTATION, WRONG_POSITION,
ID_SWITCH, TRACK_FRAGMENTATION, LOW_POINT_DENSITY, SENSOR_DISAGREEMENT,
GRADER_DISAGREEMENT, LOW_CONFIDENCE, RARE_EVENT scenarios, MODEL_REGRESSION.

All geometry is in the ego frame per frame: x forward (m), y left (m), z up (m).
Ground plane at z=0. Point clouds are re-derived deterministically from the
dataset seed, so nothing heavy is persisted.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensorflow.evaluation.records import (
    Annotation,
    Dataset,
    DatasetLineage,
    EvalStore,
    Frame,
    GroundTruthBox,
    EgoState,
    Scene,
    Scenario,
    Sequence,
    Track,
    new_id,
)

CLASS_DIMS = {
    "vehicle": (4.5, 1.9, 1.6),
    "truck": (8.0, 2.5, 3.1),
    "pedestrian": (0.7, 0.7, 1.75),
    "cyclist": (1.8, 0.6, 1.7),
    "motorcycle": (2.2, 0.8, 1.4),
}
CLASS_POOL = ["vehicle", "vehicle", "vehicle", "truck", "pedestrian", "pedestrian", "cyclist", "motorcycle"]

CAM_W, CAM_H, CAM_F, CAM_Z = 800, 450, 500.0, 1.5

ERROR_TYPES = [
    "FALSE_POSITIVE",
    "FALSE_NEGATIVE",
    "BAD_3D_BOX",
    "WRONG_ORIENTATION",
    "WRONG_POSITION",
    "ID_SWITCH",
    "TRACK_FRAGMENTATION",
    "LOW_POINT_DENSITY",
    "SENSOR_DISAGREEMENT",
    "GRADER_DISAGREEMENT",
    "LOW_CONFIDENCE",
]


def project_bbox_2d(bbox_3d: List[float]) -> Optional[List[float]]:
    x, y, z, l, w, h, _ = bbox_3d
    if x < 3.0:
        return None
    u = CAM_W / 2 - (y / x) * CAM_F
    v = CAM_H / 2 + ((CAM_Z - z) / x) * CAM_F
    bw = max(6.0, CAM_F * max(l, w) / x)
    bh = max(6.0, CAM_F * h / x)
    if u + bw / 2 < 0 or u - bw / 2 > CAM_W:
        return None
    return [round(u - bw / 2, 1), round(v - bh / 2, 1), round(bw, 1), round(bh, 1)]


# ------------------------------------------------------------------ dataset generation


def generate_dataset(
    store: EvalStore,
    name: str = "synthetic-perception",
    num_sequences: int = 6,
    frames_per_sequence: int = 30,
    seed: int = 7,
    version: str = "v1",
    generated_from_model: Optional[str] = "model-v1",
) -> Dataset:
    rng = np.random.default_rng(seed)
    dataset_id = new_id("ds")
    dataset = Dataset(
        dataset_id=dataset_id,
        name=name,
        version=version,
        seed=seed,
        gt_type="VENDOR_GROUND_TRUTH",
        lineage=DatasetLineage(generated_from_model=generated_from_model),
        generation_params={
            "num_sequences": num_sequences,
            "frames_per_sequence": frames_per_sequence,
            "sparse_gt": [],
            "injected": {},
        },
        status="generated",
    )

    sparse_gt: List[List[str]] = []
    scenario_frames: Dict[str, List[str]] = {}

    num_scenes = max(1, num_sequences // 2)
    scenes = []
    for si in range(num_scenes):
        scene = Scene(scene_id=f"{dataset_id}-scene-{si}", dataset_id=dataset_id, description=f"Synthetic scene {si}")
        scenes.append(scene)

    frame_count = 0
    for qi in range(num_sequences):
        scene = scenes[qi % num_scenes]
        seq_id = f"{dataset_id}-seq-{qi}"
        sequence = Sequence(sequence_id=seq_id, scene_id=scene.scene_id, dataset_id=dataset_id)
        scene.sequence_ids.append(seq_id)

        # Sequence-level conditions (rare-event scenario diversity).
        weather = "rain" if qi % 5 == 2 else "clear"
        time_of_day = "night" if qi % 5 == 1 else "day"

        # Object population: world-frame kinematics; ego drives +x at 10 m/s.
        num_objects = int(rng.integers(5, 9))
        objects = []
        for oi in range(num_objects):
            cls = CLASS_POOL[int(rng.integers(0, len(CLASS_POOL)))]
            l, w, h = CLASS_DIMS[cls]
            x0 = float(rng.uniform(12, 65))
            y0 = float(rng.uniform(-12, 12))
            heading = float(rng.uniform(-math.pi, math.pi)) if cls == "pedestrian" else float(rng.choice([0.0, math.pi, math.pi / 2, -math.pi / 2]))
            speed = {"pedestrian": 1.4, "cyclist": 4.5, "motorcycle": 9.0, "vehicle": 8.0, "truck": 7.0}[cls] * float(rng.uniform(0.6, 1.2))
            objects.append({
                "iid": f"{seq_id}-obj-{oi}",
                "cls": cls,
                "dims": (l, w, h),
                "x": x0, "y": y0,
                "vx": speed * math.cos(heading),
                "vy": speed * math.sin(heading),
                "yaw": heading,
            })

        # Rare-event injections per sequence.
        if qi % 3 == 0:
            # Near-collision: pedestrian cutting across the ego path, converging.
            objects.append({
                "iid": f"{seq_id}-obj-nearmiss",
                "cls": "pedestrian",
                "dims": CLASS_DIMS["pedestrian"],
                "x": 40.0, "y": 8.0, "vx": 0.0, "vy": -2.2, "yaw": -math.pi / 2,
            })
        if qi % 3 == 1:
            # Unusual trajectory: vehicle moving erratically against traffic.
            objects.append({
                "iid": f"{seq_id}-obj-erratic",
                "cls": "vehicle",
                "dims": CLASS_DIMS["vehicle"],
                "x": 55.0, "y": -3.0, "vx": -14.0, "vy": 1.5, "yaw": math.pi,
            })

        dt = 0.1
        ego_speed = 10.0
        sensor_failure_frames = set()
        if qi % 3 == 2 and frames_per_sequence >= 10:
            sensor_failure_frames = {frames_per_sequence // 2, frames_per_sequence // 2 + 1}

        for fi in range(frames_per_sequence):
            frame_id = f"{dataset_id}-f{frame_count:05d}"
            ego_x = ego_speed * dt * fi
            tags: List[str] = []
            if time_of_day == "night":
                tags.append("night_glare")
            if weather == "rain":
                tags.append("adverse_weather")
            if fi in sensor_failure_frames:
                tags.append("sensor_failure")

            gt_boxes: List[GroundTruthBox] = []
            for obj in objects:
                wx = obj["x"] + obj["vx"] * dt * fi
                wy = obj["y"] + obj["vy"] * dt * fi
                rel_x = wx - ego_x
                rel_y = wy
                if rel_x < 1.0 or rel_x > 80 or abs(rel_y) > 25:
                    continue
                l, w, h = obj["dims"]
                gt = GroundTruthBox(
                    gt_id=f"{frame_id}-{obj['iid']}",
                    class_name=obj["cls"],
                    bbox_3d=[round(rel_x, 3), round(rel_y, 3), round(h / 2, 3), l, w, h, round(obj["yaw"], 4)],
                    track_instance_id=obj["iid"],
                    gt_type="VENDOR_GROUND_TRUTH",
                    velocity=[round(obj["vx"] - ego_speed, 3), round(obj["vy"], 3)],
                )
                gt.bbox_2d = project_bbox_2d(gt.bbox_3d)
                gt_boxes.append(gt)

                # TTC tagging for approaching objects near the ego path.
                rvx = obj["vx"] - ego_speed
                if rvx < -0.5 and abs(rel_y) < 2.5:
                    ttc = rel_x / (-rvx)
                    if ttc < 3.0:
                        tags.append("extreme_ttc")
                        if ttc < 1.8:
                            tags.append("near_collision")
                if obj["cls"] in ("pedestrian", "cyclist") and rel_x < 15 and abs(rel_y) < 5:
                    tags.append("vru_interaction")

            # Occlusion: mark ~8% of frames; their first GT object is point-sparse.
            occluded = rng.random() < 0.08
            if occluded and gt_boxes:
                tags.append("severe_occlusion")
                sparse_gt.append([frame_id, gt_boxes[0].gt_id])

            frame = Frame(
                frame_id=frame_id,
                dataset_id=dataset_id,
                scene_id=scene.scene_id,
                sequence_id=seq_id,
                index=frame_count,
                timestamp_us=int(fi * dt * 1e6) + qi * 10_000_000,
                ego_pose=EgoState(x=round(ego_x, 2), speed_mps=ego_speed),
                gt_boxes=gt_boxes,
                scenario_tags=sorted(set(tags)),
                weather=weather,
                time_of_day=time_of_day,
                num_lidar_points=0,  # filled below deterministically
                camera_width=CAM_W,
                camera_height=CAM_H,
            )
            frame.num_lidar_points = _expected_point_count(dataset, frame, sparse_gt)
            sequence.frame_ids.append(frame_id)
            store.put("frames", frame)
            frame_count += 1

        store.put("sequences", sequence)

    for scene in scenes:
        store.put("scenes", scene)

    dataset.generation_params["sparse_gt"] = sparse_gt
    dataset.num_scenes = num_scenes
    dataset.num_sequences = num_sequences
    dataset.num_frames = frame_count
    # Frame-level reference coverage; replaced by annotation-level coverage
    # once labels are generated.
    frames_with_gt = sum(1 for f in store.where("frames", dataset_id=dataset_id) if f.gt_boxes)
    dataset.gt_coverage = round(frames_with_gt / max(frame_count, 1), 4)
    store.put("datasets", dataset)

    # Scenario records (mined tags -> Scenario entities).
    tag_frames: Dict[str, List[str]] = {}
    for fr in store.where("frames", dataset_id=dataset_id):
        for t in fr.scenario_tags:
            tag_frames.setdefault(t, []).append(fr.frame_id)
    for tag, fids in tag_frames.items():
        store.put("scenarios", Scenario(
            scenario_id=f"{dataset_id}-scn-{tag}",
            dataset_id=dataset_id,
            scenario_type=tag,
            frame_ids=sorted(fids),
            description=f"Frames exhibiting {tag.replace('_', ' ')}",
        ))
    scenario_frames.update(tag_frames)

    store.audit("dataset_generated", "Dataset", dataset_id, f"{frame_count} frames, {num_sequences} sequences")
    return dataset


def _expected_point_count(dataset: Dataset, frame: Frame, sparse_gt: List[List[str]]) -> int:
    if "sensor_failure" in frame.scenario_tags:
        return 60
    n = 600
    sparse_ids = {g for f, g in sparse_gt if f == frame.frame_id}
    for gt in frame.gt_boxes:
        n += 8 if gt.gt_id in sparse_ids else _object_point_count(gt)
    return n


def _object_point_count(gt: GroundTruthBox) -> int:
    x, y = gt.bbox_3d[0], gt.bbox_3d[1]
    dist = max(3.0, math.hypot(x, y))
    l, w, h = gt.bbox_3d[3], gt.bbox_3d[4], gt.bbox_3d[5]
    base = l * w * h * 100
    return int(max(6, min(800, base * (12.0 / dist) ** 1.3)))


def frame_points(store: EvalStore, frame: Frame) -> np.ndarray:
    """Deterministically re-derive the LiDAR point cloud for a frame (ego frame)."""
    dataset = store.get("datasets", frame.dataset_id)
    seed = (dataset.seed if dataset else 0) * 100003 + frame.index
    rng = np.random.default_rng(seed)
    sparse = set()
    if dataset:
        sparse = {g for f, g in dataset.generation_params.get("sparse_gt", []) if f == frame.frame_id}

    if "sensor_failure" in frame.scenario_tags:
        ground = rng.uniform([0, -20, -0.05], [70, 20, 0.05], size=(60, 3))
        return ground.astype(np.float32)

    clouds = [rng.uniform([0, -20, -0.05], [70, 20, 0.05], size=(600, 3))]
    for gt in frame.gt_boxes:
        n = 8 if gt.gt_id in sparse else _object_point_count(gt)
        x, y, z, l, w, h, yaw = gt.bbox_3d
        local = rng.uniform([-l / 2, -w / 2, -h / 2], [l / 2, w / 2, h / 2], size=(n, 3))
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pts = local @ rot.T + np.array([x, y, z])
        clouds.append(pts)
    return np.concatenate(clouds).astype(np.float32)


def points_in_box(points: np.ndarray, bbox_3d: List[float]) -> np.ndarray:
    """Boolean mask of points inside an oriented 3D box."""
    x, y, z, l, w, h, yaw = bbox_3d
    shifted = points - np.array([x, y, z])
    c, s = math.cos(-yaw), math.sin(-yaw)
    rx = shifted[:, 0] * c - shifted[:, 1] * s
    ry = shifted[:, 0] * s + shifted[:, 1] * c
    rz = shifted[:, 2]
    return (np.abs(rx) <= l / 2) & (np.abs(ry) <= w / 2) & (np.abs(rz) <= h / 2)


# ------------------------------------------------------------------ label generation


DEFAULT_ERROR_RATES = {
    "FALSE_POSITIVE": 0.03,
    "FALSE_NEGATIVE": 0.03,
    "BAD_3D_BOX": 0.02,
    "WRONG_ORIENTATION": 0.02,
    "WRONG_POSITION": 0.02,
    "LOW_CONFIDENCE": 0.02,
    "SENSOR_DISAGREEMENT": 0.015,
    "GRADER_DISAGREEMENT": 0.02,
}


def generate_labels(
    store: EvalStore,
    dataset: Dataset,
    model_version: str = "model-v1",
    error_rates: Optional[Dict[str, float]] = None,
    degrade_classes: Optional[List[str]] = None,
    seed_offset: int = 1,
) -> List[Annotation]:
    """Create auto-generated labels (hypotheses) from GT with injected defects."""
    rates = dict(DEFAULT_ERROR_RATES)
    if error_rates:
        rates.update(error_rates)
    degrade_classes = degrade_classes or []
    rng = np.random.default_rng(dataset.seed * 7919 + seed_offset)

    frames = sorted(store.where("frames", dataset_id=dataset.dataset_id), key=lambda f: f.index)
    sparse = {(f, g) for f, g in dataset.generation_params.get("sparse_gt", [])}

    annotations: List[Annotation] = []
    injected_counts: Dict[str, int] = {e: 0 for e in ERROR_TYPES}
    track_ann: Dict[str, List[str]] = {}

    for frame in frames:
        for gt in frame.gt_boxes:
            drop_rate = rates["FALSE_NEGATIVE"] + (0.30 if gt.class_name in degrade_classes else 0.0)
            if rng.random() < drop_rate:
                injected_counts["FALSE_NEGATIVE"] += 1
                continue

            x, y, z, l, w, h, yaw = gt.bbox_3d
            # Nominal detector noise, proportional to object size (large boxes
            # tolerate more absolute error at equal IoU).
            sigma_pos = 0.012 * max(l, w)
            x += float(rng.normal(0, sigma_pos))
            y += float(rng.normal(0, sigma_pos))
            z += float(rng.normal(0, 0.008 * h))
            l *= float(rng.normal(1, 0.012))
            w *= float(rng.normal(1, 0.012))
            h *= float(rng.normal(1, 0.012))
            yaw += float(rng.normal(0, math.radians(1.2)))
            dist = math.hypot(x, y)
            conf = float(np.clip(0.97 - dist / 200 + rng.normal(0, 0.04), 0.5, 0.99))

            errors: List[str] = []
            if (frame.frame_id, gt.gt_id) in sparse:
                errors.append("LOW_POINT_DENSITY")
            r = rng.random()
            if r < rates["WRONG_POSITION"]:
                x += float(rng.choice([-1, 1])) * float(rng.uniform(1.5, 3.0))
                y += float(rng.choice([-1, 1])) * float(rng.uniform(1.0, 2.5))
                errors.append("WRONG_POSITION")
            elif r < rates["WRONG_POSITION"] + rates["BAD_3D_BOX"]:
                factor = float(rng.choice([0.45, 1.9, 2.3]))
                l *= factor
                w *= factor
                errors.append("BAD_3D_BOX")
            elif r < rates["WRONG_POSITION"] + rates["BAD_3D_BOX"] + rates["WRONG_ORIENTATION"]:
                yaw += math.radians(float(rng.uniform(30, 90)) * float(rng.choice([-1, 1])))
                errors.append("WRONG_ORIENTATION")
            if rng.random() < rates["LOW_CONFIDENCE"]:
                conf = float(rng.uniform(0.12, 0.35))
                errors.append("LOW_CONFIDENCE")
            if rng.random() < rates["SENSOR_DISAGREEMENT"]:
                errors.append("SENSOR_DISAGREEMENT")
            if rng.random() < rates["GRADER_DISAGREEMENT"]:
                errors.append("GRADER_DISAGREEMENT")
            for e in errors:
                injected_counts[e] += 1

            bbox_3d = [round(v, 4) for v in [x, y, z, l, w, h, yaw]]
            ann = Annotation(
                annotation_id=new_id("ann"),
                dataset_id=dataset.dataset_id,
                frame_id=frame.frame_id,
                object_id=gt.track_instance_id,
                class_name=gt.class_name,
                confidence=round(conf, 4),
                bbox_3d=bbox_3d,
                bbox_2d=None if "SENSOR_DISAGREEMENT" in errors else project_bbox_2d(bbox_3d),
                track_id=f"trk-{gt.track_instance_id}",
                model="synthlab-detector",
                model_version=model_version,
                matched_gt_id=gt.gt_id,
                injected_errors=errors,
            )
            annotations.append(ann)
            track_ann.setdefault(ann.track_id, []).append(ann.annotation_id)

        # False positives: labels with no supporting GT.
        if rng.random() < rates["FALSE_POSITIVE"] * 6:
            cls = CLASS_POOL[int(rng.integers(0, len(CLASS_POOL)))]
            l, w, h = CLASS_DIMS[cls]
            bx = float(rng.uniform(10, 60))
            by = float(rng.uniform(-15, 15))
            bbox_3d = [round(bx, 3), round(by, 3), round(h / 2, 3), l, w, h, round(float(rng.uniform(-math.pi, math.pi)), 3)]
            ann = Annotation(
                annotation_id=new_id("ann"),
                dataset_id=dataset.dataset_id,
                frame_id=frame.frame_id,
                object_id=f"fp-{frame.frame_id}",
                class_name=cls,
                confidence=round(float(rng.uniform(0.4, 0.75)), 4),
                bbox_3d=bbox_3d,
                bbox_2d=project_bbox_2d(bbox_3d),
                track_id=None,
                model_version=model_version,
                matched_gt_id=None,
                injected_errors=["FALSE_POSITIVE"],
            )
            annotations.append(ann)
            injected_counts["FALSE_POSITIVE"] += 1

    # Track-level errors: ID switches and fragmentation on long tracks.
    long_tracks = [t for t, ids in track_ann.items() if len(ids) >= 10]
    ann_by_id = {a.annotation_id: a for a in annotations}
    if len(long_tracks) >= 2:
        ta, tb = long_tracks[0], long_tracks[1]
        for aid in track_ann[ta][len(track_ann[ta]) // 2:]:
            ann_by_id[aid].track_id = tb
            ann_by_id[aid].injected_errors.append("ID_SWITCH")
            injected_counts["ID_SWITCH"] += 1
    if len(long_tracks) >= 3:
        tc = long_tracks[2]
        frag_id = f"{tc}-frag"
        for aid in track_ann[tc][len(track_ann[tc]) // 2:]:
            ann_by_id[aid].track_id = frag_id
            ann_by_id[aid].injected_errors.append("TRACK_FRAGMENTATION")
            injected_counts["TRACK_FRAGMENTATION"] += 1

    for ann in annotations:
        store.put("annotations", ann)

    # Track entities from final label track ids.
    final_tracks: Dict[str, List[Annotation]] = {}
    for a in annotations:
        if a.track_id:
            final_tracks.setdefault(a.track_id, []).append(a)
    for tid, anns in final_tracks.items():
        store.put("tracks", Track(
            track_id=tid,
            dataset_id=dataset.dataset_id,
            class_name=anns[0].class_name,
            annotation_ids=[a.annotation_id for a in anns],
            frame_ids=[a.frame_id for a in anns],
        ))

    dataset.num_annotations = len(annotations)
    dataset.generation_params["injected"] = {k: v for k, v in injected_counts.items() if v > 0}
    matched = len([a for a in annotations if a.matched_gt_id])
    dataset.gt_coverage = round(matched / len(annotations), 4) if annotations else 0.0
    dataset.status = "labeled"
    store.put("datasets", dataset)
    store.audit("labels_generated", "Dataset", dataset.dataset_id,
                f"{len(annotations)} labels by {model_version}; injected: {dataset.generation_params['injected']}")
    return annotations


# ------------------------------------------------------------------ feature extraction


def annotation_features(store: EvalStore, dataset_id: str) -> Tuple[np.ndarray, List[str]]:
    """Feature matrix for anomaly detection over all labels of a dataset."""
    anns = sorted(store.where("annotations", dataset_id=dataset_id), key=lambda a: a.annotation_id)
    frames = {f.frame_id: f for f in store.where("frames", dataset_id=dataset_id)}
    cache: Dict[str, np.ndarray] = {}

    feats, ids = [], []
    class_index = {c: i for i, c in enumerate(CLASS_DIMS)}
    for a in anns:
        if not a.bbox_3d:
            continue
        frame = frames.get(a.frame_id)
        if frame is None:
            continue
        if a.frame_id not in cache:
            cache[a.frame_id] = frame_points(store, frame)
        pts = cache[a.frame_id]
        x, y, z, l, w, h, yaw = a.bbox_3d
        mask = points_in_box(pts, a.bbox_3d)
        n_in = int(mask.sum())
        volume = max(l * w * h, 1e-3)
        density = n_in / volume
        exp_l, exp_w, exp_h = CLASS_DIMS.get(a.class_name, (2, 2, 2))
        dim_ratio = (l / exp_l + w / exp_w + h / exp_h) / 3
        onehot = [0.0] * len(class_index)
        onehot[class_index.get(a.class_name, 0)] = 1.0
        feats.append([
            a.confidence,
            math.hypot(x, y),
            z,
            l, w, h,
            volume,
            dim_ratio,
            abs(math.sin(yaw)),
            float(n_in),
            density,
            float(a.bbox_2d is None),
            *onehot,
        ])
        ids.append(a.annotation_id)
    return (np.array(feats, dtype=np.float64) if feats else np.zeros((0, 12 + len(class_index)))), ids
