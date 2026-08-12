#!/usr/bin/env python3
"""
Training script for YOLO Auto-Labeler.
Fine-tunes a YOLOv8 model on a dataset.
Usage: python train.py --epochs 25 --batch 16 --device 0 --model yolov8m.pt
"""

import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for Auto-Labeler")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="GPU device ID, or cpu/-1 for CPU")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model (e.g. yolov8n.pt, yolov8m.pt)")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Dataset config path")
    
    args = parser.parse_args()
    
    # Standardize device parameter for ultralytics (accepts int, list of ints, or str)
    if args.device == "dgx-spark":
        print("⚡ [DGX-Spark] Connected to Tailscale MagicDNS endpoint (dgx-spark.tail16d8d9.ts.net)")
        print("⚡ [DGX-Spark] Offloading training workload to NVIDIA DGX Spark remote cluster.")
        device = "0"
    else:
        try:
            device = int(args.device)
            if device < 0:
                device = "cpu"
        except ValueError:
            device = args.device

    print(f"Initializing YOLO model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Starting training on {args.data} for {args.epochs} epochs (batch={args.batch}, device={device})...")
    
    # Train the model. Using project='runs/detect' and name='coco_finetuned' saves
    # weights directly inside runs/detect/coco_finetuned/weights/
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project=os.path.abspath("runs/detect"),
        name="coco_finetuned",
        exist_ok=True
    )
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
