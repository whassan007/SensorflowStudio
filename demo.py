#!/usr/bin/env python3
"""
Quick demo of the auto-labeling pipeline using a pretrained YOLOv8 model.
Downloads a test image and runs the full pipeline: inference -> auto-grading.
No training required.
Usage: python demo.py
"""

import json
from pathlib import Path
import urllib.request
from ultralytics import YOLO
from autograder import YOLOAutoGrader

def download_test_image():
    """Download a sample image for testing."""
    url = "https://ultralytics.com/images/zidane.jpg"
    save_path = Path("data/test.jpg")
    save_path.parent.mkdir(exist_ok=True)
    
    if not save_path.exists():
        print(f"Downloading test image from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ Test image saved to {save_path}")
    return save_path

def run_demo():
    """Run complete demo: inference + auto-grading."""
    
    print("\n" + "=" * 60)
    print("YOLO Auto-Labeler Demo")
    print("=" * 60)
    
    # 1. Download test image
    print("\n[1/3] Setting up test data...")
    test_image = download_test_image()
    print(f"✓ Using test image: {test_image}")
    
    # 2. Run inference with pretrained model
    print("\n[2/3] Running inference...")
    model = YOLO("yolov8m.pt")  # Pretrained model (auto-downloads)
    results = model.predict(source=str(test_image), conf=0.25, verbose=False)
    
    # Extract predictions
    predictions = []
    for result in results:
        for box in result.boxes:
            pred = {
                "image": Path(result.path).name,
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x1": float(box.xyxy[0, 0]),
                    "y1": float(box.xyxy[0, 1]),
                    "x2": float(box.xyxy[0, 2]),
                    "y2": float(box.xyxy[0, 3]),
                },
            }
            predictions.append(pred)
    
    print(f"✓ Found {len(predictions)} detections")
    
    # 3. Auto-grade predictions
    print("\n[3/3] Auto-grading predictions...")
    grader = YOLOAutoGrader(conf_threshold=0.5)
    report = grader.generate_report(predictions)
    
    # Save report
    output_path = Path("runs/demo/quality_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Quality report saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total Predictions: {report['total_predictions']}")
    print(f"Quality Score: {report['quality_score']:.1f}/100")
    print(f"Issues Found: {sum(report['issues_by_severity'].values())}")
    print(f"\nIssues by Severity:")
    for severity, count in report['issues_by_severity'].items():
        print(f"  • {severity}: {count}")
    
    if report['flagged_issues']:
        print(f"\nSample Issues (showing 3 of {len(report['flagged_issues'])}):")
        for issue in report['flagged_issues'][:3]:
            print(f"  • [{issue['severity']}] {issue['type']}: {issue['message']}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "=" * 60)
    print("✓ Demo complete! Pipeline is working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train on your dataset: python train.py --epochs 25")
    print("2. Run inference: python infer.py --source /path/to/images")
    print("3. Grade quality: python autograder.py")
    print("\nSee README.md for full documentation.")

if __name__ == "__main__":
    run_demo()
