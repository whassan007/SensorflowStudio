#!/bin/bash
# Download YOLOv8 models required for deployment

set -e

MODEL_DIR="${1:-.}"
mkdir -p "$MODEL_DIR"

echo "Downloading YOLOv8 models..."

# YOLOv8 Nano (small/fast)
if [ ! -f "$MODEL_DIR/yolov8n.pt" ]; then
    echo "Downloading yolov8n.pt..."
    wget -q https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -O "$MODEL_DIR/yolov8n.pt"
    echo "✓ yolov8n.pt downloaded"
fi

# YOLOv8 Medium (balanced)
if [ ! -f "$MODEL_DIR/yolov8m.pt" ]; then
    echo "Downloading yolov8m.pt..."
    wget -q https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt -O "$MODEL_DIR/yolov8m.pt"
    echo "✓ yolov8m.pt downloaded"
fi

echo "Models ready at $MODEL_DIR"
