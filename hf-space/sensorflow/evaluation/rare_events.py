"""Rare event detection & scenario mining.

Finds rare/high-value safety scenarios (spec §15): near collisions, extreme
TTC/PET, VRU interactions, unusual object behavior, severe occlusions, sensor
failures, nighttime glare, adverse weather, unusual trajectories, unexpected
road geometry. Produces RareEvent records with rarity/anomaly scores and
per-sensor evidence. Reuses the SSAM-style TTC/PET surrogate-safety logic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from sensorflow.evaluation.records import EvalStore, Frame, RareEvent, new_id

SEVERITY_BY_TYPE = {
    "near_collision": "critical",
    "extreme_ttc": "high",
    "extreme_pet": "high",
    "vru_interaction": "high",
    "sensor_failure": "high",
    "unusual_object_behavior": "medium",
    "unusual_trajectory": "medium",
    "severe_occlusion": "medium",
    "nighttime_glare": "low",
    "adverse_weather": "low",
    "unexpected_road_geometry": "medium",
}


def _frame_ttc(frame: Frame) -> Optional[float]:
    """Minimum time-to-collision across GT objects approaching the ego path."""
    ttcs = []
    for gt in frame.gt_boxes:
        x, y = gt.bbox_3d[0], gt.bbox_3d[1]
        vx, vy = gt.velocity[0], gt.velocity[1]
        if abs(y) < 2.5 and vx < -0.3:
            ttcs.append(x / (-vx))
        # Laterally converging object crossing the ego corridor.
        if vy != 0 and abs(y) > 1.0 and np.sign(vy) == -np.sign(y):
            t_cross = abs(y) / abs(vy)
            x_at_cross = x + vx * t_cross
            if 0 < x_at_cross < 12 and t_cross < 6.0:
                ttcs.append(t_cross)
    return min(ttcs) if ttcs else None


def _frame_pet(frame: Frame) -> Optional[float]:
    """Post-encroachment-time surrogate: time gap between an object leaving the
    ego corridor and the ego arriving at that point."""
    pets = []
    ego_speed = max(frame.ego_pose.speed_mps, 0.1)
    for gt in frame.gt_boxes:
        x, y = gt.bbox_3d[0], gt.bbox_3d[1]
        vy = gt.velocity[1]
        if abs(y) < 1.5 and abs(vy) > 0.5:  # object currently inside corridor, exiting laterally
            t_exit = (1.5 - abs(y)) / abs(vy)
            t_ego_arrival = x / ego_speed
            pet = t_ego_arrival - t_exit
            if pet > 0:
                pets.append(pet)
    return min(pets) if pets else None


def detect_rare_events(
    store: EvalStore,
    dataset_id: str,
    anomaly_scores_by_frame: Optional[Dict[str, float]] = None,
) -> List[RareEvent]:
    """Mine sequences for rare scenarios; group contiguous frames per type."""
    frames = sorted(store.where("frames", dataset_id=dataset_id), key=lambda f: f.index)
    anomaly_scores_by_frame = anomaly_scores_by_frame or {}

    # Derive per-frame scenario hits (tags + kinematic measures).
    hits: List[Dict] = []
    for fr in frames:
        frame_hits: Dict[str, Dict[str, str]] = {}
        ttc = _frame_ttc(fr)
        pet = _frame_pet(fr)
        if ttc is not None and ttc < 3.0:
            frame_hits["extreme_ttc"] = {"lidar": f"TTC={ttc:.2f}s from range-rate", "telemetry": "converging range"}
            if ttc < 1.8:
                frame_hits["near_collision"] = {"lidar": f"TTC={ttc:.2f}s inside ego corridor",
                                                "camera": "object crossing ego path"}
        if pet is not None and pet < 1.5:
            frame_hits["extreme_pet"] = {"telemetry": f"PET={pet:.2f}s corridor encroachment"}
        for tag in fr.scenario_tags:
            if tag == "vru_interaction":
                frame_hits["vru_interaction"] = {"camera": "VRU within 15m of ego path", "lidar": "VRU point cluster"}
            elif tag == "sensor_failure":
                frame_hits["sensor_failure"] = {"lidar": f"only {fr.num_lidar_points} returns (expected >600)"}
            elif tag == "severe_occlusion":
                frame_hits["severe_occlusion"] = {"lidar": "object with <10 supporting points"}
            elif tag == "night_glare":
                frame_hits["nighttime_glare"] = {"camera": "night scene, glare-prone exposure"}
            elif tag == "adverse_weather":
                frame_hits["adverse_weather"] = {"camera": "rain droplets on lens", "lidar": "rain-scatter noise"}
        # Unusual trajectory / behavior: wrong-way or erratic objects.
        for gt in fr.gt_boxes:
            if gt.velocity[0] < -20:  # closing far faster than ego speed => oncoming in our lane region
                frame_hits["unusual_trajectory"] = {"lidar": f"object closing at {-gt.velocity[0]:.1f} m/s",
                                                    "map": "movement against traffic direction"}
                frame_hits["unusual_object_behavior"] = {"telemetry": "erratic heading vs lane geometry"}
        if frame_hits:
            hits.append({"frame": fr, "hits": frame_hits})

    # Scenario frequency -> rarity.
    type_counts: Dict[str, int] = defaultdict(int)
    for h in hits:
        for t in h["hits"]:
            type_counts[t] += 1
    total_frames = max(len(frames), 1)

    # Group contiguous frames of same sequence+type into one event.
    events: List[RareEvent] = []
    open_events: Dict[str, Dict] = {}
    for h in hits:
        fr: Frame = h["frame"]
        for t, evidence in h["hits"].items():
            key = f"{fr.sequence_id}:{t}"
            if key in open_events and fr.index - open_events[key]["last_index"] <= 2:
                open_events[key]["frames"].append(fr.frame_id)
                open_events[key]["last_index"] = fr.index
                open_events[key]["evidence"].update(evidence)
            else:
                if key in open_events:
                    events.append(_finalize(store, dataset_id, open_events.pop(key),
                                            type_counts, total_frames, anomaly_scores_by_frame))
                open_events[key] = {"type": t, "frames": [fr.frame_id], "last_index": fr.index,
                                    "evidence": dict(evidence)}
    for ev in open_events.values():
        events.append(_finalize(store, dataset_id, ev, type_counts, total_frames, anomaly_scores_by_frame))

    for e in events:
        store.put("rare_events", e)
    store.audit("rare_events_detected", "Dataset", dataset_id, f"{len(events)} events")
    return events


def _finalize(
    store: EvalStore,
    dataset_id: str,
    ev: Dict,
    type_counts: Dict[str, int],
    total_frames: int,
    anomaly_scores_by_frame: Dict[str, float],
) -> RareEvent:
    t = ev["type"]
    freq = type_counts[t] / total_frames
    rarity = float(np.clip(1.0 - freq * 4.0, 0.05, 1.0))
    scores = [anomaly_scores_by_frame.get(fid, 0.0) for fid in ev["frames"]]
    anomaly = float(np.mean(scores)) if scores else 0.0
    return RareEvent(
        event_id=new_id("evt"),
        dataset_id=dataset_id,
        scenario_type=t,
        severity=SEVERITY_BY_TYPE.get(t, "medium"),
        rarity_score=round(rarity, 4),
        anomaly_score=round(anomaly, 4),
        confidence=round(float(np.clip(0.55 + 0.4 * rarity, 0, 0.99)), 3),
        evidence_frames=ev["frames"],
        sensor_evidence=ev["evidence"],
        description=f"{t.replace('_', ' ').title()} spanning {len(ev['frames'])} frame(s)",
    )
