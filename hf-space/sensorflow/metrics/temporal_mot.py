"""Temporal MOT metrics: ID swaps and track fragmentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def compute_id_swap_rate(
    pred_tracks: List[Dict],
    gt_tracks: List[Dict],
) -> float:
    """
    Count ID swaps: when GT instance_id is stable but pred track_id changes.
    pred_tracks/gt_tracks: [{track_id/instance_id, frames: [{frame_id, bbox_3d}]}]
    """
    swaps = 0
    total_updates = 0

    gt_by_instance: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for gt in gt_tracks:
        iid = gt.get("instance_id", gt.get("track_id", ""))
        for fr in gt.get("frames", []):
            gt_by_instance[iid][fr["frame_id"]] = fr["bbox_3d"]

    pred_by_track: Dict[int, Dict[str, List[float]]] = defaultdict(dict)
    for pt in pred_tracks:
        tid = pt["track_id"]
        for fr in pt.get("frames", []):
            pred_by_track[tid][fr["frame_id"]] = fr["bbox_3d"]

    for iid, gt_frames in gt_by_instance.items():
        frame_ids = sorted(gt_frames.keys())
        if len(frame_ids) < 2:
            continue
        prev_tid = None
        for fid in frame_ids:
            best_tid = None
            best_dist = float("inf")
            for tid, pframes in pred_by_track.items():
                if fid in pframes:
                    dist = _center_dist(pframes[fid], gt_frames[fid])
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
            if best_tid is not None:
                total_updates += 1
                if prev_tid is not None and best_tid != prev_tid:
                    swaps += 1
                prev_tid = best_tid

    return swaps / total_updates if total_updates > 0 else 0.0


def compute_track_fragmentation_rate(
    pred_tracks: List[Dict],
    gt_tracks: List[Dict],
    match_threshold_m: float = 2.0,
) -> float:
    """Fraction of GT instances matched by more than one pred track."""
    fragmented = 0
    total_gt = len(gt_tracks)
    if total_gt == 0:
        return 0.0

    for gt in gt_tracks:
        iid = gt.get("instance_id", gt.get("track_id", ""))
        gt_frames = {fr["frame_id"]: fr["bbox_3d"] for fr in gt.get("frames", [])}
        matched_tracks = set()
        for pt in pred_tracks:
            pframes = {fr["frame_id"]: fr["bbox_3d"] for fr in pt.get("frames", [])}
            overlap = 0
            for fid, gt_box in gt_frames.items():
                if fid in pframes:
                    if _center_dist(pframes[fid], gt_box) < match_threshold_m:
                        overlap += 1
            if overlap >= 2:
                matched_tracks.add(pt["track_id"])
        if len(matched_tracks) > 1:
            fragmented += 1

    return fragmented / total_gt


def _center_dist(a: List[float], b: List[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
