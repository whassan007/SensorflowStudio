"""Quality gate benchmarking against vendor ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sensorflow.metrics.perception_3d import (
    bev_iou,
    compute_map_mar_by_frame,
    mean_orientation_error,
    mean_position_error,
)
from sensorflow.metrics.resource_profile import compute_resource_profile
from sensorflow.metrics.temporal_mot import (
    compute_id_swap_rate,
    compute_track_fragmentation_rate,
)
from sensorflow.schemas.unified_frame import UnifiedSequence


DEFAULT_THRESHOLDS = {
    "map_3d": 0.65,
    "orientation_error_deg": 5.0,
    "id_swap_rate": 0.02,
    "track_fragmentation_rate": 0.05,
    "position_error_m": 2.0,
}


class QualityGate:
    """Benchmark automated tracks against vendor ground truth."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

    def evaluate(
        self,
        sequence: UnifiedSequence,
        pred_tracks: List[Dict],
        stages_run: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Matching is scoped per frame: a prediction may only match ground
        # truth from the same timestamp. Pooling boxes across all frames
        # (the previous behavior) allowed physically impossible cross-frame
        # matches, inflating mAP/mAR and corrupting error statistics.
        preds_by_frame, gts_by_frame = self._collect_boxes_by_frame(sequence, pred_tracks)
        all_preds = [b for boxes in preds_by_frame.values() for b in boxes]
        all_gts = [b for boxes in gts_by_frame.values() for b in boxes]
        map_metrics = compute_map_mar_by_frame(preds_by_frame, gts_by_frame)

        matched_pairs = []
        for fid in sorted(set(preds_by_frame) | set(gts_by_frame)):
            matched_pairs.extend(self._match_for_errors(
                preds_by_frame.get(fid, []), gts_by_frame.get(fid, [])))
        orient_err = mean_orientation_error(matched_pairs)
        pos_err = mean_position_error(matched_pairs)

        gt_tracks = self._build_gt_tracks(sequence)
        id_swap = compute_id_swap_rate(pred_tracks, gt_tracks)
        fragmentation = compute_track_fragmentation_rate(pred_tracks, gt_tracks)

        resource = compute_resource_profile(sequence.frames, stages_run)

        metric_card = {
            "map_3d": map_metrics["map_3d"],
            "mar_3d": map_metrics["mar_3d"],
            "mean_iou_3d": map_metrics["mean_iou_3d"],
            "orientation_error_deg": orient_err,
            "position_error_m": pos_err,
            "id_swap_rate": id_swap,
            "track_fragmentation_rate": fragmentation,
            "process_units": resource["process_units"],
            "compute_cycles": resource["compute_cycles"],
        }

        failures = self._check_thresholds(metric_card)
        passed = len(failures) == 0

        quality_report = {
            "total_predictions": len(all_preds),
            "total_ground_truth": len(all_gts),
            "quality_score": self._quality_score(metric_card),
            "metric_card": metric_card,
            "passed": passed,
            "failures": failures,
            "issue_summary": {
                "high_orientation_error": 1 if orient_err > self.thresholds["orientation_error_deg"] else 0,
                "high_position_error": 1 if pos_err > self.thresholds["position_error_m"] else 0,
                "id_swaps": 1 if id_swap > self.thresholds["id_swap_rate"] else 0,
                "fragmentation": 1 if fragmentation > self.thresholds["track_fragmentation_rate"] else 0,
            },
            "recommendations": self._recommendations(metric_card, failures),
        }

        return {"metric_card": metric_card, "quality_report": quality_report, "passed": passed}

    def _collect_boxes(
        self,
        sequence: UnifiedSequence,
        pred_tracks: List[Dict],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        all_preds, all_gts = [], []
        for track in pred_tracks:
            for fr in track.get("frames", []):
                all_preds.append(fr["bbox_3d"])
        for frame in sequence.frames:
            for gt in frame.ground_truth:
                all_gts.append(gt.bbox_3d)
        return all_preds, all_gts

    def _collect_boxes_by_frame(
        self,
        sequence: UnifiedSequence,
        pred_tracks: List[Dict],
    ) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
        preds_by_frame: Dict[str, List[List[float]]] = {}
        gts_by_frame: Dict[str, List[List[float]]] = {}
        for track in pred_tracks:
            for fr in track.get("frames", []):
                preds_by_frame.setdefault(fr["frame_id"], []).append(fr["bbox_3d"])
        for frame in sequence.frames:
            for gt in frame.ground_truth:
                gts_by_frame.setdefault(frame.frame_id, []).append(gt.bbox_3d)
        return preds_by_frame, gts_by_frame

    def _match_for_errors(
        self,
        preds: List[List[float]],
        gts: List[List[float]],
    ) -> List[Tuple[List[float], List[float]]]:
        pairs = []
        used_gt = set()
        for pred in preds:
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gts):
                if gi in used_gt:
                    continue
                iou = bev_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou > 0.3:
                pairs.append((pred, gts[best_gi]))
                used_gt.add(best_gi)
        return pairs

    def _build_gt_tracks(self, sequence: UnifiedSequence) -> List[Dict]:
        by_instance: Dict[str, List[Dict]] = {}
        for frame in sequence.frames:
            for gt in frame.ground_truth:
                by_instance.setdefault(gt.instance_id, []).append({
                    "frame_id": frame.frame_id,
                    "bbox_3d": gt.bbox_3d,
                })
        return [
            {"instance_id": iid, "frames": frames}
            for iid, frames in by_instance.items()
        ]

    def _check_thresholds(self, metrics: Dict[str, float]) -> List[Dict]:
        failures = []
        checks = [
            ("map_3d", metrics["map_3d"], self.thresholds["map_3d"], ">="),
            ("orientation_error_deg", metrics["orientation_error_deg"], self.thresholds["orientation_error_deg"], "<="),
            ("id_swap_rate", metrics["id_swap_rate"], self.thresholds["id_swap_rate"], "<="),
            ("track_fragmentation_rate", metrics["track_fragmentation_rate"], self.thresholds["track_fragmentation_rate"], "<="),
            ("position_error_m", metrics["position_error_m"], self.thresholds["position_error_m"], "<="),
        ]
        for name, value, threshold, op in checks:
            failed = (value < threshold) if op == ">=" else (value > threshold)
            if failed:
                failures.append({
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "operator": op,
                })
        return failures

    def _quality_score(self, metrics: Dict[str, float]) -> float:
        def safe(v, default=0.0):
            import math
            return default if v is None or (isinstance(v, float) and math.isnan(v)) else v

        score = (
            safe(metrics.get("map_3d", 0)) * 40 +
            (1 - min(safe(metrics.get("orientation_error_deg", 0)) / 45, 1)) * 20 +
            (1 - min(safe(metrics.get("id_swap_rate", 0)) / 0.1, 1)) * 20 +
            (1 - min(safe(metrics.get("track_fragmentation_rate", 0)) / 0.2, 1)) * 20
        )
        return round(min(100, max(0, score)), 1)

    def _recommendations(self, metrics: Dict, failures: List[Dict]) -> List[str]:
        recs = []
        if metrics["orientation_error_deg"] > self.thresholds["orientation_error_deg"]:
            recs.append("Improve 3D orientation estimation via tighter SAM mask-to-LiDAR alignment.")
        if metrics["id_swap_rate"] > self.thresholds["id_swap_rate"]:
            recs.append("Increase tracker velocity penalty or reduce association gate during occlusions.")
        if metrics["track_fragmentation_rate"] > self.thresholds["track_fragmentation_rate"]:
            recs.append("Extend max_age for ghost tracks and review occlusion handling.")
        if metrics["map_3d"] < self.thresholds["map_3d"]:
            recs.append("Retrain or tune perception model; verify LiDAR-camera calibration.")
        if not recs:
            recs.append("All quality gate thresholds met. Proceed to launch gate.")
        return recs

    def save_results(self, sequence_id: str, results: Dict[str, Any]) -> Path:
        out_dir = Path("runs/pipeline") / sequence_id / "benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metric_card.json", "w") as f:
            json.dump(results["metric_card"], f, indent=2)
        with open(out_dir / "quality_report.json", "w") as f:
            json.dump(results["quality_report"], f, indent=2)
        return out_dir
