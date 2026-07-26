#!/usr/bin/env python3
"""
Auto-grader for YOLO predictions and label quality evaluation.
Flags suspicious detections, regressions, and data quality issues.
Usage: python autograder.py --predictions predictions.json --conf-threshold 0.5
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import statistics

class YOLOAutoGrader:
    """Quality evaluation system for YOLO predictions and annotations."""
    
    def __init__(self, conf_threshold=0.5, iou_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.issues = []
        self.stats = defaultdict(int)
        
    def compute_iou(self, box1, box2):
        """Compute Intersection over Union between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1["x1"], box1["y1"], box1["x2"], box1["y2"]
        x2_min, y2_min, x2_max, y2_max = box2["x1"], box2["y1"], box2["x2"], box2["y2"]
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        inter = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def box_area(self, box):
        """Compute bounding box area."""
        return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    
    def grade_predictions(self, predictions):
        """Grade predictions for quality issues."""
        
        # Group by image
        by_image = defaultdict(list)
        for pred in predictions:
            by_image[pred["image"]].append(pred)
        
        # Check each image
        for img_name, preds in by_image.items():
            self._grade_image(img_name, preds)
        
        return self.issues, self.stats
    
    def _grade_image(self, img_name, predictions):
        """Grade predictions for a single image."""
        
        for i, pred in enumerate(predictions):
            # Issue 1: Low confidence detection
            if pred["confidence"] < self.conf_threshold:
                self.issues.append({
                    "severity": "WARNING",
                    "type": "low_confidence",
                    "image": img_name,
                    "detection_id": i,
                    "confidence": pred["confidence"],
                    "threshold": self.conf_threshold,
                    "message": f"Detection confidence {pred['confidence']:.3f} below threshold {self.conf_threshold}",
                })
                self.stats["low_confidence"] += 1
            
            # Issue 2: Unusual box size
            area = self.box_area(pred["bbox"])
            if area < 100:  # Very small detections
                self.issues.append({
                    "severity": "INFO",
                    "type": "small_detection",
                    "image": img_name,
                    "detection_id": i,
                    "area": area,
                    "message": f"Very small detection (area={area:.0f}px²). May be noise or label error.",
                })
                self.stats["small_detection"] += 1
        
        # Issue 3: Overlapping detections of different classes
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                if predictions[i]["class_id"] != predictions[j]["class_id"]:
                    iou = self.compute_iou(predictions[i]["bbox"], predictions[j]["bbox"])
                    if iou > 0.5:
                        self.issues.append({
                            "severity": "WARNING",
                            "type": "overlapping_different_class",
                            "image": img_name,
                            "detection_ids": [i, j],
                            "class_1": predictions[i]["class_name"],
                            "class_2": predictions[j]["class_name"],
                            "iou": iou,
                            "message": f"High IoU ({iou:.3f}) between different classes. Possible mislabel.",
                        })
                        self.stats["overlapping_different_class"] += 1
        
        # Issue 4: Class distribution anomalies
        class_counts = defaultdict(int)
        for pred in predictions:
            class_counts[pred["class_name"]] += 1
        
        total = len(predictions)
        for cls_name, count in class_counts.items():
            pct = count / total if total > 0 else 0
            if pct > 0.8:  # One class dominates
                self.issues.append({
                    "severity": "INFO",
                    "type": "class_imbalance",
                    "image": img_name,
                    "class": cls_name,
                    "percentage": pct * 100,
                    "message": f"Class '{cls_name}' represents {pct*100:.1f}% of detections. Check for label bias.",
                })
                self.stats["class_imbalance"] += 1
    
    def generate_report(self, predictions):
        """Generate quality report."""
        
        issues, stats = self.grade_predictions(predictions)
        
        report = {
            "total_predictions": len(predictions),
            "total_images": len(set(p["image"] for p in predictions)),
            "quality_score": self._compute_quality_score(predictions, issues),
            "issues_by_severity": self._count_by_severity(issues),
            "issue_summary": stats,
            "flagged_issues": issues[:50],  # Top 50 issues
            "recommendations": self._generate_recommendations(issues, stats),
        }
        
        return report
    
    def _compute_quality_score(self, predictions, issues):
        """Compute 0-100 quality score."""
        if not predictions:
            return 100.0
        
        critical = sum(1 for i in issues if i["severity"] == "ERROR")
        warning = sum(1 for i in issues if i["severity"] == "WARNING")
        
        score = 100.0
        score -= critical * 5
        score -= warning * 1
        
        return max(0, min(100, score))
    
    def _count_by_severity(self, issues):
        """Count issues by severity."""
        counts = defaultdict(int)
        for issue in issues:
            counts[issue["severity"]] += 1
        return dict(counts)
    
    def _generate_recommendations(self, issues, stats):
        """Generate actionable recommendations."""
        recs = []
        
        if stats.get("low_confidence", 0) > len(issues) * 0.1:
            recs.append("High number of low-confidence detections. Consider retraining or adjusting confidence threshold.")
        
        if stats.get("overlapping_different_class", 0) > 5:
            recs.append("Multiple overlapping detections of different classes detected. Review training data for label consistency.")
        
        if stats.get("small_detection", 0) > len(issues) * 0.2:
            recs.append("Many small detections present. Verify these are real objects and not labeling artifacts.")
        
        if not recs:
            recs.append("Dataset quality appears good. Continue monitoring in production.")
        
        return recs

def main():
    parser = argparse.ArgumentParser(description="Grade YOLO predictions for label quality")
    parser.add_argument("--predictions", default="runs/infer/predictions.json", 
                        help="Path to predictions.json")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Confidence threshold for flagging detections")
    parser.add_argument("--output", default="runs/infer/quality_report.json",
                        help="Output path for quality report")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO Auto-Grader: Label Quality Evaluation")
    print("=" * 60)

    # Load predictions
    pred_file = Path(args.predictions)
    if not pred_file.exists():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        return

    with open(pred_file) as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    # Grade
    grader = YOLOAutoGrader(conf_threshold=args.conf_threshold)
    report = grader.generate_report(predictions)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Quality Report Summary")
    print("=" * 60)
    print(f"Total Predictions: {report['total_predictions']}")
    print(f"Total Images: {report['total_images']}")
    print(f"Quality Score: {report['quality_score']:.1f}/100")
    print(f"Issues by Severity: {report['issues_by_severity']}")
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    print(f"\nDetailed report saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
