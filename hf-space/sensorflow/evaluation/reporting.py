"""Aggregated reporting: quality metrics, funnel, groups, haystack, benchmark.

Ground-truth integrity (spec §41): precision/recall are only computed when a
reference exists; model confidence is never presented as precision; recall is
never inferred from anomaly scores.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from sensorflow.evaluation import synthetic
from sensorflow.evaluation.process_units import usage_summary
from sensorflow.evaluation.records import (
    Benchmark,
    BenchmarkResult,
    EvalStore,
    SAFETY_CRITICAL_CLASSES,
    new_id,
)

TP_IOU = 0.5


# ------------------------------------------------------------------ matching


def _match_stats(store: EvalStore, dataset_id: str, frame_ids: Optional[set] = None,
                 class_name: Optional[str] = None) -> Dict:
    """TP/FP/FN over annotations vs reference GT (only where reference exists)."""
    frames = store.where("frames", dataset_id=dataset_id)
    if frame_ids is not None:
        frames = [f for f in frames if f.frame_id in frame_ids]
    fid_set = {f.frame_id for f in frames}

    anns = [a for a in store.where("annotations", dataset_id=dataset_id)
            if a.frame_id in fid_set and a.status != "REJECTED"]
    if class_name:
        anns = [a for a in anns if a.class_name == class_name]

    gt_total = 0
    gt_hit: Dict[str, bool] = {}
    for f in frames:
        for g in f.gt_boxes:
            if class_name and g.class_name != class_name:
                continue
            gt_total += 1
            gt_hit[g.gt_id] = False

    tp = fp = 0
    for a in anns:
        v = store.get("validations", a.annotation_id)
        iou = v.iou_3d if v else None
        if a.matched_gt_id and a.matched_gt_id in gt_hit and iou is not None and iou >= TP_IOU:
            if not gt_hit[a.matched_gt_id]:
                gt_hit[a.matched_gt_id] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = sum(1 for hit in gt_hit.values() if not hit)
    return {"tp": tp, "fp": fp, "fn": fn, "gt_total": gt_total, "n_annotations": len(anns)}


def _prf(stats: Dict) -> Dict[str, Optional[float]]:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    if stats["gt_total"] == 0 and tp + fp == 0:
        return {"precision": None, "recall": None, "f1": None}
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    r = lambda v: round(v, 4) if v is not None else None
    return {"precision": r(precision), "recall": r(recall), "f1": r(f1)}


# ------------------------------------------------------------------ metrics


def quality_metrics(store: EvalStore, dataset_id: Optional[str]) -> Dict:
    dataset = store.get("datasets", dataset_id) if dataset_id else None
    empty = {
        "dataset_id": dataset_id, "gt_available": False, "gt_type": None, "gt_coverage": 0.0,
        "global": {k: None for k in (
            "precision", "recall", "f1", "map_3d", "safety_critical_recall", "mean_iou_3d",
            "mean_position_error", "mean_orientation_error_deg", "idf1", "id_swap_rate",
            "fragmentation_rate", "grader_consensus")} | {"anomaly_rate": 0.0},
        "per_class": [], "per_scenario": [],
    }
    if dataset is None:
        return empty

    anns = [a for a in store.where("annotations", dataset_id=dataset_id) if a.status != "REJECTED"]
    if not anns:
        return empty

    gt_available = dataset.gt_coverage > 0

    stats = _match_stats(store, dataset_id)
    prf = _prf(stats) if gt_available else {"precision": None, "recall": None, "f1": None}

    # Safety-critical recall (reference required).
    saf_tp = saf_fn = 0
    for cls in SAFETY_CRITICAL_CLASSES:
        s = _match_stats(store, dataset_id, class_name=cls)
        saf_tp += s["tp"]
        saf_fn += s["fn"]
    safety_recall = round(saf_tp / (saf_tp + saf_fn), 4) if gt_available and (saf_tp + saf_fn) > 0 else None

    # Geometry means over matched labels.
    ious, pos_errs, yaw_errs = [], [], []
    anomaly_flags, consensus_vals = [], []
    switches = frags = tracked = 0
    for a in anns:
        v = store.get("validations", a.annotation_id)
        if v:
            if v.iou_3d is not None:
                ious.append(v.iou_3d)
            if v.position_error is not None:
                pos_errs.append(v.position_error)
            if v.orientation_error_deg is not None:
                yaw_errs.append(v.orientation_error_deg)
        an = store.get("anomalies", a.annotation_id)
        if an:
            anomaly_flags.append(1.0 if an.is_anomaly else 0.0)
        g = store.get("grader_comparisons", a.annotation_id)
        if g and g.consensus is not None:
            consensus_vals.append(g.consensus)
        t = store.get("tracking_evidence", a.annotation_id)
        if t and a.matched_gt_id:
            tracked += 1
            switches += 1 if t.id_switch else 0
            frags += 1 if t.fragmentation else 0

    id_swap_rate = round(switches / tracked, 4) if tracked else None
    frag_rate = round(frags / tracked, 4) if tracked else None
    # IDF1 approximation from identity error rates (full IDF1 needs global
    # bipartite matching; this monotone surrogate is used consistently).
    idf1 = None
    if id_swap_rate is not None and frag_rate is not None:
        idf1 = round(max(0.0, 1.0 - id_swap_rate - 0.5 * frag_rate), 4)

    map_3d = _map_3d(store, dataset_id) if gt_available else None

    glob = {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "map_3d": map_3d,
        "safety_critical_recall": safety_recall,
        "mean_iou_3d": round(float(np.mean(ious)), 4) if ious else None,
        "mean_position_error": round(float(np.mean(pos_errs)), 4) if pos_errs else None,
        "mean_orientation_error_deg": round(float(np.mean(yaw_errs)), 4) if yaw_errs else None,
        "idf1": idf1,
        "id_swap_rate": id_swap_rate,
        "fragmentation_rate": frag_rate,
        "anomaly_rate": round(float(np.mean(anomaly_flags)), 4) if anomaly_flags else 0.0,
        "grader_consensus": round(float(np.mean(consensus_vals)), 4) if consensus_vals else None,
    }

    per_class = []
    for cls in ("pedestrian", "cyclist", "vehicle", "motorcycle", "truck"):
        s = _match_stats(store, dataset_id, class_name=cls)
        if s["gt_total"] == 0 and s["n_annotations"] == 0:
            continue
        p = _prf(s) if gt_available else {"precision": None, "recall": None, "f1": None}
        per_class.append({"class_name": cls, **p, "support": s["gt_total"]})

    per_scenario = []
    for scn in store.where("scenarios", dataset_id=dataset_id):
        fids = set(scn.frame_ids)
        s = _match_stats(store, dataset_id, frame_ids=fids)
        p = _prf(s) if gt_available else {"precision": None, "recall": None, "f1": None}
        per_scenario.append({"scenario": scn.scenario_type, **p, "support": s["gt_total"]})

    return {
        "dataset_id": dataset_id,
        "gt_available": gt_available,
        "gt_type": dataset.gt_type if gt_available else None,
        "gt_coverage": dataset.gt_coverage,
        "global": glob,
        "per_class": per_class,
        "per_scenario": per_scenario,
    }


def _map_3d(store: EvalStore, dataset_id: str, max_frames: int = 40) -> Optional[float]:
    """Mean AP over IoU thresholds {0.5, 0.7}, averaged over sampled frames."""
    from sensorflow.metrics.perception_3d import compute_map_mar

    frames = sorted(store.where("frames", dataset_id=dataset_id), key=lambda f: f.index)
    if not frames:
        return None
    step = max(1, len(frames) // max_frames)
    sampled = frames[::step]
    anns_by_frame: Dict[str, List] = defaultdict(list)
    for a in store.where("annotations", dataset_id=dataset_id):
        if a.status != "REJECTED" and a.bbox_3d:
            anns_by_frame[a.frame_id].append(a.bbox_3d)
    aps = []
    for f in sampled:
        gts = [g.bbox_3d for g in f.gt_boxes]
        preds = anns_by_frame.get(f.frame_id, [])
        if not gts and not preds:
            continue
        m = compute_map_mar(preds, gts)
        aps.append(m["map_3d"])
    return round(float(np.mean(aps)), 4) if aps else None


# ------------------------------------------------------------------ counters / funnel


def counters(store: EvalStore, dataset_id: Optional[str] = None) -> Dict[str, int]:
    anns = store.where("annotations", dataset_id=dataset_id) if dataset_id else store.all("annotations")
    frames = store.where("frames", dataset_id=dataset_id) if dataset_id else store.all("frames")
    evaluated = sum(1 for a in anns if store.get("triage_decisions", a.annotation_id) is not None)
    decisions = [store.get("triage_decisions", a.annotation_id) for a in anns]
    auto_graded = sum(1 for d in decisions if d is not None and d.status == "AUTO_GRADED")
    open_tasks = [t for t in store.all("review_tasks")
                  if t.status != "resolved" and (dataset_id is None or t.dataset_id == dataset_id)]
    events = store.where("rare_events", dataset_id=dataset_id) if dataset_id else store.all("rare_events")
    return {
        "frames_processed": len(frames),
        "auto_labeled": len(anns),
        "evaluated": evaluated,
        "auto_graded": auto_graded,
        "flagged": sum(1 for a in anns if a.status == "FLAGGED"),
        "in_hitl": len(open_tasks),
        "verified": sum(1 for a in anns if a.status == "VERIFIED"),
        "rejected": sum(1 for a in anns if a.status == "REJECTED"),
        "rare_events": len(events),
    }


def funnel(store: EvalStore, dataset_id: Optional[str] = None) -> Dict:
    c = counters(store, dataset_id)
    anns = store.where("annotations", dataset_id=dataset_id) if dataset_id else store.all("annotations")
    validated = sum(1 for a in anns if store.get("validations", a.annotation_id) is not None)
    tds = store.all("training_datasets")
    if dataset_id:
        tds = [t for t in tds if t.source_dataset_id == dataset_id]
    in_training = sum(t.num_verified_labels for t in tds)

    tasks = [t for t in store.all("review_tasks") if dataset_id is None or t.dataset_id == dataset_id]
    resolved = [t for t in tasks if t.status == "resolved"]
    relabeled = [t for t in resolved if t.resolution and t.resolution.action in
                 ("correct", "approve", "merge_tracks", "split_track")]
    revalidated = [t for t in relabeled if t.resolution.revalidation_passed is not None]
    re_verified = [t for t in relabeled if t.resolution.final_status == "VERIFIED"]

    base = max(c["auto_labeled"], 1)

    def stage(name: str, count: int) -> Dict:
        return {"stage": name, "count": count, "pct_of_input": round(100.0 * count / base, 1)}

    main = [
        stage("RAW SENSOR DATA (frames)", c["frames_processed"]),
        stage("AUTO-GENERATED LABELS", c["auto_labeled"]),
        stage("EVALUATED", c["evaluated"]),
        stage("QUALITY VALIDATION", validated),
        stage("AUTO-GRADED", c["auto_graded"]),
        stage("VERIFIED", c["verified"]),
        stage("TRAINING", in_training),
    ]
    side = [
        stage("FLAGGED", c["flagged"] + len(resolved)),
        stage("HITL REVIEW", len(tasks)),
        stage("RE-LABELING", len(relabeled)),
        stage("RE-VALIDATION", len(revalidated)),
        stage("VERIFIED (post-HITL)", len(re_verified)),
    ]
    return {"main_path": main, "side_path": side}


# ------------------------------------------------------------------ groups


GROUP_NAMES = ["verified", "non_verified", "hitl", "rejected"]


def _group_members(store: EvalStore, dataset_id: str) -> Dict[str, List]:
    anns = store.where("annotations", dataset_id=dataset_id)
    open_hitl = {t.annotation_id for t in store.all("review_tasks")
                 if t.dataset_id == dataset_id and t.status != "resolved"}
    groups: Dict[str, List] = {g: [] for g in GROUP_NAMES}
    for a in anns:
        if a.status == "VERIFIED":
            groups["verified"].append(a)
        elif a.status == "REJECTED":
            groups["rejected"].append(a)
        elif a.annotation_id in open_hitl:
            groups["hitl"].append(a)
        else:
            groups["non_verified"].append(a)
    return groups


def quality_groups(store: EvalStore, dataset_id: str) -> Dict:
    groups = _group_members(store, dataset_id)
    total = sum(len(v) for v in groups.values())
    out = []
    for name in GROUP_NAMES:
        members = groups[name]
        out.append({
            "group_id": f"{dataset_id}:{name}",
            "dataset_id": dataset_id,
            "name": name,
            "count": len(members),
            "pct": round(100.0 * len(members) / max(total, 1), 1),
        })
    return {
        "dataset_id": dataset_id,
        "total": total,
        "verification_rate": round(len(groups["verified"]) / max(total, 1), 4),
        "groups": out,
    }


def group_detail(store: EvalStore, group_id: str) -> Optional[Dict]:
    if ":" not in group_id:
        return None
    dataset_id, name = group_id.rsplit(":", 1)
    groups = _group_members(store, dataset_id)
    if name not in groups:
        return None
    members = groups[name]
    total = sum(len(v) for v in groups.values())

    ious, cons, anom, tq = [], [], [], []
    reason_counts: Dict[str, int] = defaultdict(int)
    tp = fp = 0
    for a in members:
        v = store.get("validations", a.annotation_id)
        if v and v.iou_3d is not None:
            ious.append(v.iou_3d)
            if v.iou_3d >= TP_IOU:
                tp += 1
            else:
                fp += 1
        elif v:
            fp += 1
        g = store.get("grader_comparisons", a.annotation_id)
        if g and g.consensus is not None:
            cons.append(g.consensus)
        an = store.get("anomalies", a.annotation_id)
        if an:
            anom.append(an.score)
        t = store.get("tracking_evidence", a.annotation_id)
        if t and t.track_quality is not None:
            tq.append(t.track_quality)
        d = store.get("triage_decisions", a.annotation_id)
        if d:
            for r in d.failure_reasons:
                reason_counts[r] += 1

    dataset = store.get("datasets", dataset_id)
    gt_available = dataset is not None and dataset.gt_coverage > 0
    precision = round(tp / (tp + fp), 4) if gt_available and (tp + fp) > 0 else None
    # Group-level recall vs all reference objects claimed by this group's labels.
    matched_gt = {a.matched_gt_id for a in members if a.matched_gt_id}
    recall = round(tp / max(len(matched_gt), 1), 4) if gt_available and matched_gt else None
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    mean = lambda arr: round(float(np.mean(arr)), 4) if arr else None
    return {
        "group_id": group_id,
        "dataset_id": dataset_id,
        "name": name,
        "count": len(members),
        "pct": round(100.0 * len(members) / max(total, 1), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_3d": mean(ious),
        "mean_consensus": mean(cons),
        "mean_anomaly_score": mean(anom),
        "tracking_quality": mean(tq),
        "failure_reason_counts": dict(reason_counts),
        "annotation_ids": [a.annotation_id for a in members][:500],
    }


# ------------------------------------------------------------------ haystack


def haystack(store: EvalStore, dataset_id: Optional[str] = None) -> List[Dict]:
    """2D embedding of all observations: normals, anomalies, rare events,
    detected FP/FN and verified labels. Clicking a point opens its evidence."""
    if dataset_id is None:
        datasets = sorted(store.all("datasets"), key=lambda d: d.created_at)
        if not datasets:
            return []
        dataset_id = datasets[-1].dataset_id

    X, ids = synthetic.annotation_features(store, dataset_id)
    if len(ids) == 0:
        return []
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    _, _, vt = np.linalg.svd(Xs, full_matrices=False)
    proj = Xs @ vt[:2].T
    # scale to [2, 98]
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    coords = 2 + 96 * (proj - lo) / span

    anns = {a.annotation_id: a for a in store.where("annotations", dataset_id=dataset_id)}
    rare_frames: Dict[str, str] = {}
    for e in store.where("rare_events", dataset_id=dataset_id):
        for fid in e.evidence_frames:
            rare_frames[fid] = e.event_id

    points: List[Dict] = []
    frame_positions: Dict[str, List[np.ndarray]] = defaultdict(list)
    for j, aid in enumerate(ids):
        a = anns[aid]
        an = store.get("anomalies", aid)
        d = store.get("triage_decisions", aid)
        v = store.get("validations", aid)
        frame_positions[a.frame_id].append(coords[j])
        if a.matched_gt_id is None and a.status in ("FLAGGED", "REJECTED"):
            cat = "false_positive"
        elif v and v.iou_3d is not None and v.iou_3d < TP_IOU and a.status in ("FLAGGED", "REJECTED"):
            cat = "false_positive"
        elif an and an.is_anomaly:
            cat = "anomaly"
        elif a.status == "VERIFIED" and a.source != "auto":
            cat = "verified"
        elif a.status == "VERIFIED":
            cat = "normal"
        elif d is not None and d.status == "FLAGGED":
            cat = "anomaly" if (an and an.score > 0.7) else "normal"
        else:
            cat = "normal"
        points.append({
            "id": aid,
            "x": round(float(coords[j][0]), 2),
            "y": round(float(coords[j][1]), 2),
            "category": cat,
            "anomaly_score": round(float(an.score), 4) if an else 0.0,
            "class_name": a.class_name,
            "frame_id": a.frame_id,
            "kind": "annotation",
        })

    # Missed objects (false negatives): reference GT never matched by a label.
    matched = {a.matched_gt_id for a in anns.values() if a.matched_gt_id}
    rng = np.random.default_rng(13)
    for f in store.where("frames", dataset_id=dataset_id):
        for g in f.gt_boxes:
            if g.gt_id in matched:
                continue
            base = np.mean(frame_positions[f.frame_id], axis=0) if frame_positions.get(f.frame_id) else np.array([50.0, 50.0])
            jitter = rng.normal(0, 3, 2)
            points.append({
                "id": f"fn-{g.gt_id}",
                "x": round(float(np.clip(base[0] + jitter[0], 1, 99)), 2),
                "y": round(float(np.clip(base[1] + jitter[1], 1, 99)), 2),
                "category": "false_negative",
                "anomaly_score": 0.0,
                "class_name": g.class_name,
                "frame_id": f.frame_id,
                "kind": "annotation",
            })

    # Rare events as first-class points near their evidence cluster.
    for e in store.where("rare_events", dataset_id=dataset_id):
        pos = [np.mean(frame_positions[fid], axis=0) for fid in e.evidence_frames if frame_positions.get(fid)]
        base = np.mean(pos, axis=0) if pos else np.array([90.0, 90.0])
        points.append({
            "id": e.event_id,
            "x": round(float(np.clip(base[0] + 1.5, 1, 99)), 2),
            "y": round(float(np.clip(base[1] + 1.5, 1, 99)), 2),
            "category": "rare_event",
            "anomaly_score": e.anomaly_score,
            "class_name": e.scenario_type,
            "frame_id": e.evidence_frames[0] if e.evidence_frames else "",
            "kind": "rare_event",
        })
    return points


# ------------------------------------------------------------------ evaluation record


def evaluation_record(store: EvalStore, annotation_id: str) -> Optional[Dict]:
    a = store.get("annotations", annotation_id)
    if a is None:
        return None
    frame = store.get("frames", a.frame_id)
    gt = None
    if frame is not None and a.matched_gt_id:
        gt = next((g for g in frame.gt_boxes if g.gt_id == a.matched_gt_id), None)
    v = store.get("validations", annotation_id)
    an = store.get("anomalies", annotation_id)
    g = store.get("grader_comparisons", annotation_id)
    t = store.get("tracking_evidence", annotation_id)
    d = store.get("triage_decisions", annotation_id)

    return {
        "annotation_id": annotation_id,
        "dataset_id": a.dataset_id,
        "frame_id": a.frame_id,
        "object_class": a.class_name,
        "model_version": a.model_version,
        "ground_truth_id": a.matched_gt_id,
        "ground_truth_type": gt.gt_type if gt else None,
        "detection": {"confidence": a.confidence},
        "geometry": {
            "iou_3d": v.iou_3d if v else None,
            "position_error": v.position_error if v else None,
            "orientation_error_deg": v.orientation_error_deg if v else None,
            "dimension_error": v.dimension_error if v else None,
            "point_density": v.point_density if v else None,
            "point_in_box_ratio": v.point_in_box_ratio if v else None,
            "ground_contact_error": v.ground_contact_error if v else None,
        },
        "tracking": {
            "id_switch": t.id_switch if t else False,
            "fragmentation": t.fragmentation if t else False,
            "track_quality": t.track_quality if t else None,
        },
        "anomaly": {
            "score": an.score if an else 0.0,
            "is_anomaly": an.is_anomaly if an else False,
            "detector_scores": an.detector_scores if an else {},
            "normalized_scores": an.normalized_scores if an else {},
            "ensemble_strategy": an.ensemble_strategy if an else "weighted_average",
            "decision_threshold": an.decision_threshold if an else 0.9,
        },
        "grading": {
            "consensus": g.consensus if g else None,
            "class_agreement": g.class_agreement if g else None,
            "spatial_agreement": g.spatial_agreement if g else None,
            "temporal_agreement": g.temporal_agreement if g else None,
            "grader_count": g.grader_count if g else 0,
            "disagreement_types": g.disagreement_types if g else [],
        },
        "validation": {
            "passed": v.passed if v else False,
            "checks": [c.model_dump() for c in v.checks] if v else [],
        },
        "decision": d.model_dump() if d else None,
        "injected_errors": a.injected_errors,
    }


# ------------------------------------------------------------------ frame summary


def frame_summary(store: EvalStore, frame_id: str, max_points: int = 900) -> Optional[Dict]:
    f = store.get("frames", frame_id)
    if f is None:
        return None
    pts = synthetic.frame_points(store, f)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    anns = store.where("annotations", frame_id=frame_id)
    return {
        "frame_id": f.frame_id,
        "sequence_id": f.sequence_id,
        "scene_id": f.scene_id,
        "timestamp_us": f.timestamp_us,
        "ego_pose": {"x": f.ego_pose.x, "y": f.ego_pose.y, "z": f.ego_pose.z,
                     "yaw": f.ego_pose.yaw, "speed_mps": f.ego_pose.speed_mps},
        "num_lidar_points": f.num_lidar_points,
        "camera": {"width": f.camera_width, "height": f.camera_height},
        "annotations": [a.model_dump() for a in anns],
        "gt_boxes": [{
            "gt_id": g.gt_id, "class_name": g.class_name, "bbox_3d": g.bbox_3d,
            "bbox_2d": g.bbox_2d, "gt_type": g.gt_type,
        } for g in f.gt_boxes],
        "lidar_points_bev": [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)] for p in pts],
    }


# ------------------------------------------------------------------ benchmark of techniques


TECH_PU_FACTORS = {
    "knn": 0.4, "lof": 0.5, "isolation_forest": 0.7, "ocsvm": 0.9, "dbscan": 0.5,
    "autoencoder": 1.4, "vae": 1.1, "gan": 1.0, "few_shot": 0.8,
    "ensemble_majority_vote": 3.2, "ensemble_weighted_average": 3.2, "ensemble_meta_classifier": 3.6,
}


def benchmark_techniques(store: EvalStore, dataset_id: Optional[str], anomaly_config: Dict) -> Benchmark:
    """Comparative benchmark: each technique + the three ensembles, evaluated
    against the injected-defect ground truth of the synthetic dataset."""
    from sensorflow.evaluation.detectors import AnomalyEnsemble, normalize_scores

    if dataset_id is None:
        datasets = sorted(store.all("datasets"), key=lambda d: d.created_at)
        if not datasets:
            raise ValueError("No dataset available to benchmark")
        dataset_id = datasets[-1].dataset_id
    dataset = store.get("datasets", dataset_id)

    X, ids = synthetic.annotation_features(store, dataset_id)
    if len(ids) < 20:
        raise ValueError("Not enough labels to benchmark")
    anns = {a.annotation_id: a for a in store.where("annotations", dataset_id=dataset_id)}
    y = np.array([1.0 if anns[i].injected_errors else 0.0 for i in ids])
    rare_fids = {fid for e in store.where("rare_events", dataset_id=dataset_id) for fid in e.evidence_frames}
    in_rare = np.array([1.0 if anns[i].frame_id in rare_fids else 0.0 for i in ids])
    id_sw = np.array([1.0 if "ID_SWITCH" in anns[i].injected_errors else 0.0 for i in ids])
    pos_err = np.array([
        (store.get("validations", i).position_error or 0.0) if store.get("validations", i) else 0.0
        for i in ids])
    consensus = np.array([
        (store.get("grader_comparisons", i).consensus or 0.0) if store.get("grader_comparisons", i) else 0.0
        for i in ids])

    cfg = dict(anomaly_config)
    # Enable everything for a complete benchmark table.
    full_cfg = {
        **cfg,
        "detectors": {k: {**v, "enabled": True} for k, v in cfg.get("detectors", {}).items()},
        "deep": {**cfg.get("deep", {}),
                 "autoencoder": {**cfg.get("deep", {}).get("autoencoder", {}), "enabled": True},
                 "vae": {**cfg.get("deep", {}).get("vae", {}), "enabled": True},
                 "gan": {"enabled": True}},
        "advanced": {**cfg.get("advanced", {}),
                     "few_shot": {**cfg.get("advanced", {}).get("few_shot", {}), "enabled": True}},
    }
    ensemble = AnomalyEnsemble(full_cfg, seed=dataset.seed if dataset else 7)
    _, raw, norm = ensemble.run(X, supervision=y)

    contamination = max(y.mean(), 0.02)
    k = int(round(contamination * len(y)))

    def evaluate(scores: np.ndarray, technique: str) -> BenchmarkResult:
        order = np.argsort(-scores)
        detected = np.zeros(len(scores))
        detected[order[:k]] = 1.0
        tp = float((detected * y).sum())
        fp = float((detected * (1 - y)).sum())
        fn = float(((1 - detected) * y).sum())
        tn = float(((1 - detected) * (1 - y)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rare_mask = (in_rare * y) > 0
        rare_recall = float((detected[rare_mask]).mean()) if rare_mask.any() else recall
        missed = detected == 0
        box_err = float(pos_err[missed & (y > 0)].mean()) if (missed & (y > 0)).any() else 0.0
        residual_swaps = float(id_sw[missed].sum() / max(id_sw.sum(), 1))
        cons = float(consensus[missed].mean()) if missed.any() else 0.0
        pu = int(len(ids) * TECH_PU_FACTORS.get(technique, 1.0))
        return BenchmarkResult(
            technique=technique,
            precision=round(precision, 4), recall=round(recall, 4),
            rare_recall=round(rare_recall, 4), f1=round(f1, 4),
            box_error_3d=round(box_err, 4), id_swap_rate=round(residual_swaps, 4),
            consensus=round(cons, 4), process_units=pu,
            fp_rate=round(fp / (fp + tn) if fp + tn else 0.0, 4),
        )

    rows = [evaluate(normalize_scores(raw[name]), name) for name in raw]
    for strategy in ("majority_vote", "weighted_average", "meta_classifier"):
        e2 = AnomalyEnsemble({**full_cfg, "advanced": {**full_cfg["advanced"], "ensemble_strategy": strategy}},
                             seed=dataset.seed if dataset else 7)
        s, _, _ = e2.run(X, supervision=y)
        rows.append(evaluate(np.asarray(s), f"ensemble_{strategy}"))

    highlights = {
        "best_rare_recall": max(rows, key=lambda r: r.rare_recall).technique,
        "best_safety_recall": max(rows, key=lambda r: r.recall).technique,
        "lowest_fp_rate": min(rows, key=lambda r: r.fp_rate).technique,
        "lowest_process_units": min(rows, key=lambda r: r.process_units).technique,
        "lowest_tracking_error": min(rows, key=lambda r: r.id_swap_rate).technique,
    }
    bench = Benchmark(benchmark_id=new_id("bench"), dataset_id=dataset_id, rows=rows, highlights=highlights)
    store.put("benchmarks", bench)
    store.audit("benchmark_run", "Benchmark", bench.benchmark_id, f"{len(rows)} techniques on {dataset_id}")
    return bench


# ------------------------------------------------------------------ overview


def overview(store: EvalStore, pipeline) -> Dict:
    dataset_id = pipeline.active_dataset
    c = counters(store, dataset_id)
    m = quality_metrics(store, dataset_id)["global"]
    total = max(c["auto_labeled"], 1)
    pu = usage_summary(store)
    models = sorted(store.all("models"), key=lambda x: x.created_at)
    return {
        "counters": c,
        "metrics": m,
        "verification_rate": round(c["verified"] / total, 4),
        "automation_rate": round(c["auto_graded"] / total, 4),
        "process_units_total": pu["total"],
        "active_dataset": dataset_id,
        "active_model": models[-1].model_version if models else "model-v1",
    }
