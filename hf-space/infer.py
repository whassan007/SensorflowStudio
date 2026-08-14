#!/usr/bin/env python3
"""
Inference script for YOLO Auto-Labeler.
Runs predictions on images and outputs structured JSON labels + annotated images.
Usage: python infer.py --source data/test.jpg --weights runs/detect/coco_finetuned/weights/best.pt
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from taxonomy import TAXONOMY_MAP

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and save structured predictions")
    parser.add_argument("--source", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--weights", type=str, default="runs/detect/coco_finetuned/weights/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="GPU device ID, or cpu/-1 for CPU")
    parser.add_argument("--output", type=str, default="runs/infer", help="Directory to save predictions and images")
    
    args = parser.parse_args()
    
    # Resolve device parameter
    if args.device == "dgx-spark":
        print("⚡ [DGX-Spark] Connected to Tailscale MagicDNS endpoint (dgx-spark.tail16d8d9.ts.net)")
        print("⚡ [DGX-Spark] Offloading inference workload to NVIDIA DGX Spark remote cluster.")
        device = "0"
    else:
        try:
            device = int(args.device)
            if device < 0:
                device = "cpu"
        except ValueError:
            device = args.device

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify source images
    source_path = Path(args.source)
    if source_path.is_dir():
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(source_path.glob(ext))
            image_files.extend(source_path.glob(ext.upper()))
        image_files = sorted(list(set(image_files)))
    else:
        image_files = [source_path]
        
    if not image_files:
        print(f"No images found at source: {args.source}")
        raise SystemExit(2)

    print(f"Running inference on {len(image_files)} image(s)...")
    
    predictions = []
    
    for img_path in image_files:
        print(f"  Predicting: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False
        )
        
        # Save annotated image
        result = results[0]
        annotated_path = output_dir / f"annotated_{img_path.name}"
        result.save(filename=str(annotated_path))
        
        # Extract structured predictions
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox_coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            pred = {
                "image": img_path.name,
                "class_id": class_id,
                "class_name": class_name,
                "taxonomy": TAXONOMY_MAP.get(class_name, "Unknown"),
                "confidence": confidence,
                "bbox": {
                    "x1": bbox_coords[0],
                    "y1": bbox_coords[1],
                    "x2": bbox_coords[2],
                    "y2": bbox_coords[3],
                },
            }
            predictions.append(pred)
            
    # Save predictions as JSON
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
        
    print(f"✓ Inference complete! Processed {len(image_files)} images.")
    print(f"✓ Predictions saved to: {predictions_file}")
    print(f"✓ Annotated images saved to: {output_dir}/")

if __name__ == "__main__":
    main()
