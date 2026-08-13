"""Auto-labeling discrepancy mining (continuous-learning loop).

Industry concept: production AV stacks run a weaker real-time ("online")
perception pass in the vehicle and a stronger non-causal ("offline") auto-label
pass over the same logs. Diffing the two is one of the highest-yield scenario
mining signals — anything the online stack missed but the offline stack found
is a candidate training example / regression scenario.

This module simulates the online pass deterministically over an existing
labeleval dataset (the stored auto-labels play the role of the strong offline
pass), diffs the two, and converts discrepancies into queryable scenario-mining
records:

- MISSED_ONLINE:  offline labeled the object, online did not detect it
- CLASS_MISMATCH: both detected it, classes differ
- GEOMETRY_DELTA: both detected it, geometry disagrees (center offset / IoU)
- OFFLINE_MISS:   online detected an object the offline pass dropped (rare)

High-severity discrepancies feed the existing rare-event store; all of them
feed the scenario database (source="discrepancy") and cohort summaries.

SIMULATED: the online pass is a seeded degradation model over ground truth
(higher miss rates, class confusion, geometry noise), not a real detector.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from sensorflow.evaluation.records import (
    SAFETY_CRITICAL_CLASSES,
    EvalStore,
    RareEvent,
    new_id,
)
from sensorflow.metrics.perception_3d import bev_iou
from sensorflow.safety.store import read_json, write_json

DEFAULT_ONLINE_PROFILE = {
    "extra_miss_rate": 0.12,        # online misses this fraction on top of offline
    "miss_rate_by_class": {"pedestrian": 0.08, "cyclist": 0.08},  # additional VRU misses
    "class_confusion_rate": 0.05,
    "position_noise_m": 0.55,
    "yaw_noise_deg": 6.0,
    "night_miss_multiplier": 1.6,   # online degrades harder at night/rain
    "rain_miss_multiplier": 1.4,
}

GEOMETRY_CENTER_THRESHOLD_M = 0.75
GEOMETRY_IOU_THRESHOLD = 0.5

_CONFUSABLE = {
    "vehicle": "truck", "truck": "vehicle", "pedestrian": "cyclist",
    "cyclist": "pedestrian", "motorcycle": "cyclist",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def simulate_online_pass(store: EvalStore, dataset_id: str,
                         profile: Optional[Dict] = None, seed: int = 101) -> Dict[str, Dict]:
    """Deterministic weaker 'online' perception pass over the dataset's GT.

    Returns {gt_id: detection dict} for every ground-truth object the online
    stack detected. SIMULATED degradation model, clearly marked.
    """
    p = {**DEFAULT_ONLINE_PROFILE, **(profile or {})}
    rng = np.random.default_rng(seed)
    detections: Dict[str, Dict] = {}
    frames = sorted(store.where("frames", dataset_id=dataset_id), key=lambda f: f.index)
    for frame in frames:
        cond_mult = 1.0
        if frame.time_of_day == "night":
            cond_mult *= p["night_miss_multiplier"]
        if frame.weather not in ("clear", ""):
            cond_mult *= p["rain_miss_multiplier"]
        for gt in frame.gt_boxes:
            miss = p["extra_miss_rate"] + p["miss_rate_by_class"].get(gt.class_name, 0.0)
            miss = min(1.0, miss * cond_mult) if miss < 1.0 else 1.0
            if rng.random() < miss:
                continue  # online miss
            x, y, z, l, w, h, yaw = gt.bbox_3d
            x += float(rng.normal(0, p["position_noise_m"]))
            y += float(rng.normal(0, p["position_noise_m"]))
            yaw += float(rng.normal(0, math.radians(p["yaw_noise_deg"])))
            cls = gt.class_name
            if rng.random() < p["class_confusion_rate"]:
                cls = _CONFUSABLE.get(cls, cls)
            detections[gt.gt_id] = {
                "gt_id": gt.gt_id, "frame_id": frame.frame_id, "class_name": cls,
                "bbox_3d": [round(v, 4) for v in (x, y, z, l, w, h, yaw)],
                "confidence": round(float(rng.uniform(0.35, 0.85)), 3),
            }
    return detections


def mine(store: EvalStore, dataset_id: str, profile: Optional[Dict] = None,
         seed: int = 101, persist: bool = True) -> Dict:
    """Diff the simulated online pass against the stored offline auto-labels."""
    dataset = store.get("datasets", dataset_id)
    if dataset is None:
        raise KeyError(f"Unknown dataset {dataset_id}")

    online = simulate_online_pass(store, dataset_id, profile, seed)
    frames = {f.frame_id: f for f in store.where("frames", dataset_id=dataset_id)}
    offline_by_gt = {a.matched_gt_id: a
                     for a in store.where("annotations", dataset_id=dataset_id)
                     if a.matched_gt_id}

    discrepancies: List[Dict] = []
    cohort_totals: Dict[str, int] = {}
    cohort_hits: Dict[str, int] = {}

    for frame in frames.values():
        for gt in frame.gt_boxes:
            cohort = f"{gt.class_name}/{frame.weather}/{frame.time_of_day}"
            cohort_totals[cohort] = cohort_totals.get(cohort, 0) + 1
            off = offline_by_gt.get(gt.gt_id)
            on = online.get(gt.gt_id)
            dtype = details = None
            if off is not None and on is None:
                dtype = "MISSED_ONLINE"
                details = {"offline_confidence": off.confidence}
            elif off is not None and on is not None:
                if on["class_name"] != off.class_name:
                    dtype = "CLASS_MISMATCH"
                    details = {"offline_class": off.class_name,
                               "online_class": on["class_name"]}
                elif off.bbox_3d:
                    center_d = math.hypot(on["bbox_3d"][0] - off.bbox_3d[0],
                                          on["bbox_3d"][1] - off.bbox_3d[1])
                    iou = bev_iou(on["bbox_3d"], off.bbox_3d)
                    if center_d > GEOMETRY_CENTER_THRESHOLD_M or iou < GEOMETRY_IOU_THRESHOLD:
                        dtype = "GEOMETRY_DELTA"
                        details = {"center_delta_m": round(center_d, 3),
                                   "bev_iou": round(iou, 4)}
            elif off is None and on is not None:
                dtype = "OFFLINE_MISS"
                details = {"online_class": on["class_name"]}
            if dtype is None:
                continue
            cohort_hits[cohort] = cohort_hits.get(cohort, 0) + 1
            safety = gt.class_name in SAFETY_CRITICAL_CLASSES
            severity = ("critical" if dtype == "MISSED_ONLINE" and safety
                        else "high" if dtype == "MISSED_ONLINE" or safety
                        else "medium")
            discrepancies.append({
                "discrepancy_id": new_id("disc"),
                "dataset_id": dataset_id,
                "frame_id": frame.frame_id,
                "gt_id": gt.gt_id,
                "type": dtype,
                "class_name": gt.class_name,
                "weather": frame.weather,
                "time_of_day": frame.time_of_day,
                "safety_critical": safety,
                "severity": severity,
                "details": details,
            })

    by_type: Dict[str, int] = {}
    for d in discrepancies:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1

    cohorts = []
    for cohort, total in sorted(cohort_totals.items()):
        hits = cohort_hits.get(cohort, 0)
        cls, weather, tod = cohort.split("/")
        cohorts.append({"cohort": cohort, "class": cls, "weather": weather,
                        "time_of_day": tod, "objects": total, "discrepancies": hits,
                        "discrepancy_rate": round(hits / total, 4)})
    cohorts.sort(key=lambda c: -c["discrepancy_rate"])

    total_objects = sum(cohort_totals.values())
    report = {
        "dataset_id": dataset_id,
        "seed": seed,
        "created_at": _now(),
        "online_profile": {**DEFAULT_ONLINE_PROFILE, **(profile or {})},
        "totals": {
            "objects": total_objects,
            "online_detections": len(online),
            "offline_annotations": len(offline_by_gt),
            "discrepancies": len(discrepancies),
            "discrepancy_rate": round(len(discrepancies) / max(total_objects, 1), 4),
        },
        "by_type": by_type,
        "cohorts": cohorts,
        "discrepancies": discrepancies,
        "simulated": True,
        "method": "deterministic simulated online pass vs stored offline auto-labels; "
                  "diff over shared ground-truth ids",
    }

    if persist:
        write_json(report, "discrepancy", f"{dataset_id}.json")
        _feed_stores(store, report)
    return report


def _feed_stores(store: EvalStore, report: Dict) -> None:
    """Feed high-severity discrepancies into the rare-event store, all of them
    into the scenario database, and record process units (best-effort)."""
    critical = [d for d in report["discrepancies"] if d["severity"] == "critical"]
    for d in critical[:40]:
        store.put("rare_events", RareEvent(
            event_id=new_id("evt"),
            dataset_id=d["dataset_id"],
            scenario_type="online_perception_miss",
            severity="critical",
            rarity_score=0.85,
            anomaly_score=0.0,
            confidence=0.9,
            evidence_frames=[d["frame_id"]],
            sensor_evidence={"discrepancy": f"{d['type']} on {d['class_name']} "
                                            f"({d['weather']}/{d['time_of_day']})"},
            description=f"Online stack missed a {d['class_name']} the offline "
                        f"auto-label pass found (simulated online pass)",
        ))
    try:
        from sensorflow.evaluation.process_units import ProcessMeter
        ProcessMeter(store, run_id=report["dataset_id"]).record(
            "discrepancy_mining", report["totals"]["objects"], factor=0.3)
    except Exception:
        pass
    store.save()
    try:
        from sensorflow.safety import scenario_db
        scenario_db.get_db().add_from_discrepancies(report)
    except Exception:
        pass


def latest_summary(dataset_id: str) -> Optional[Dict]:
    """Persisted mining report without the raw per-object rows."""
    report = read_json("discrepancy", f"{dataset_id}.json")
    if report is None:
        return None
    slim = {k: v for k, v in report.items() if k != "discrepancies"}
    slim["sample_discrepancies"] = report["discrepancies"][:25]
    return slim
