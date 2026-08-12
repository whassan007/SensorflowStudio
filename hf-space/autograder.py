#!/usr/bin/env python3
"""2D prediction quality autograder."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def compute_iou(box_a: Dict, box_b: Dict) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def grade_predictions(predictions: List[Dict], conf_threshold: float = 0.5) -> Dict:
    issues = []
    issue_summary = Counter()
    images = set(p["image"] for p in predictions)
    classes = [p["class_name"] for p in predictions]

    for i, pred in enumerate(predictions):
        if pred["confidence"] < conf_threshold:
            issue_summary["low_confidence"] += 1
            issues.append({
                "severity": "WARNING",
                "type": "low_confidence",
                "image": pred["image"],
                "detection_id": i,
                "confidence": pred["confidence"],
                "threshold": conf_threshold,
                "message": f"Confidence {pred['confidence']:.2f} below threshold",
            })

        bbox = pred["bbox"]
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
        if area < 100:
            issue_summary["small_detection"] += 1
            issues.append({
                "severity": "INFO",
                "type": "small_detection",
                "image": pred["image"],
                "detection_id": i,
                "confidence": pred["confidence"],
                "threshold": 100,
                "message": f"Small bounding box area: {area:.0f}px",
            })

    for i, pred_a in enumerate(predictions):
        for j, pred_b in enumerate(predictions):
            if j <= i or pred_a["image"] != pred_b["image"]:
                continue
            if pred_a["class_name"] != pred_b["class_name"]:
                iou = compute_iou(pred_a["bbox"], pred_b["bbox"])
                if iou > 0.5:
                    issue_summary["overlapping_different_class"] += 1
                    issues.append({
                        "severity": "WARNING",
                        "type": "overlapping_different_class",
                        "image": pred_a["image"],
                        "detection_id": i,
                        "confidence": pred_a["confidence"],
                        "threshold": 0.5,
                        "message": f"Overlap between {pred_a['class_name']} and {pred_b['class_name']}",
                    })

    class_counts = Counter(classes)
    if class_counts:
        dominant = class_counts.most_common(1)[0]
        if dominant[1] / len(classes) > 0.8:
            issue_summary["class_imbalance"] += 1

    total_issues = sum(issue_summary.values())
    quality_score = max(0, 100 - total_issues * 5)

    return {
        "total_predictions": len(predictions),
        "total_images": len(images),
        "quality_score": quality_score,
        "issues_by_severity": dict(Counter(i["severity"] for i in issues)),
        "issue_summary": dict(issue_summary),
        "flagged_issues": issues[:50],
        "recommendations": _recommendations(issue_summary),
    }


def _recommendations(summary: Counter) -> List[str]:
    recs = []
    if summary.get("low_confidence"):
        recs.append("Tune confidence threshold or retrain with more diverse samples.")
    if summary.get("overlapping_different_class"):
        recs.append("Review overlapping class labels; consider NMS tuning.")
    if summary.get("small_detection"):
        recs.append("Verify small bounding boxes are valid annotations.")
    if summary.get("class_imbalance"):
        recs.append("Balance class frequency in training dataset.")
    if not recs:
        recs.append("No significant issues detected.")
    return recs


def main():
    parser = argparse.ArgumentParser(description="Grade 2D predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    with open(pred_path) as f:
        predictions = json.load(f)

    report = grade_predictions(predictions, args.conf)
    output_path = Path(args.output) if args.output else pred_path.parent / "quality_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Quality score: {report['quality_score']}/100 -> {output_path}")


if __name__ == "__main__":
    main()
