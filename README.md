# Sensorflow Studio & Auto-Labeler

A production-oriented visual framework for automated object detection, training, and quality assurance in autonomous driving and computer vision pipelines. This repository contains both a command-line interface (CLI) toolset and a beautiful, self-hosted web-based Training Studio built with FastAPI, HTML5, and vanilla JavaScript.

---

## Key Features

1. **3D Perception Pipeline** (`sensorflow/`): Automated 5-stage pipeline for exabyte-scale sensor data:
   - **Ingest & Fusion**: Alpamayo + Waymo unified schema with six-axis taxonomy stratification
   - **3D Perception**: Local SAM masks lifted to LiDAR 3D bounding box proposals
   - **Temporal Tracking**: Kalman + Hungarian association for ID-smooth multi-frame tracks
   - **Quality Gate**: mAP, orientation error, ID swap rate, `process_units` benchmarking
   - **Launch Gate**: Safety threshold validation blocking export until passed
2. **Visual Pipeline (FastAPI Web UI)**: A 15-stage interactive environment including legacy 2D YOLO workflow:
   - Configure datasets and verify directories.
   - Choose base weights (`yolov8n`, `yolov8s`, `yolov8m`).
   - Run training locally with real-time logs and dynamic loss curves.
   - Run inference, explore predictions, and overlay bounding boxes on images.
   - Evaluate annotations using automated quality metrics.
   - Export models to high-performance ONNX formats.
3. **Robust CLI Tools**:
   - `train.py`: Fine-tune models with custom epochs, batch sizes, and hardware parameters.
   - `infer.py`: Batch prediction script generating structured JSON output.
   - `autograder.py`: Algorithmic review auditing prediction boxes for overlap, anomalies, and low confidence.
   - `sensorflow/cli/`: 3D pipeline CLIs (`ingest`, `auto_label`, `track`, `benchmark`).

---

## Project Structure

```
DrivingRepo/
├── app_backend.py       # FastAPI backend serving UI and pipeline APIs
├── sensorflow/          # 3D perception pipeline package
│   ├── dataset_fusion_engine.py
│   ├── perception_automator.py
│   ├── temporal_tracker.py
│   ├── quality_gate.py
│   ├── launch_gate_evaluator.py
│   ├── mitl_copilot.py
│   ├── adapters/        # Alpamayo + Waymo normalization
│   ├── metrics/         # 3D/temporal/resource metrics
│   └── cli/             # Pipeline CLI entrypoints
├── train.py             # CLI training entrypoint
├── infer.py             # CLI batch inference entrypoint
├── autograder.py        # Quality assurance diagnostics script
├── static/              # Visual Studio frontend folder
├── tests/test_pipeline/ # 3D pipeline integration tests
├── requirements.txt     # Base dependencies
├── requirements-3d.txt  # 3D pipeline dependencies (SAM, scipy, filterpy)
└── runs/pipeline/       # 3D pipeline artifacts (manifests, tracks, benchmarks)
```

---

## Quick Start (Web UI)

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
pip install -r requirements-3d.txt   # For 3D perception pipeline
bash deploy/download-models.sh .       # Downloads YOLO + SAM ViT-B weights
```

### 2. Start the Studio Server
Launch the local server:
```bash
python app_backend.py
```

### 3. Open in Browser
Navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 3D Perception Pipeline

### CLI Usage

```bash
# Stage 1: Ingest & fuse Alpamayo + Waymo data
python -m sensorflow.cli.ingest --vendors alpamayo waymo --sequence-id seq_001

# Stage 2: SAM auto-labeling (use --no-sam for dev without checkpoint)
python -m sensorflow.cli.auto_label --manifest runs/pipeline/seq_001/manifest.json --no-sam

# Stage 3: Temporal tracking
python -m sensorflow.cli.track --proposals-dir runs/pipeline/seq_001/proposals

# Stage 4: Quality gate benchmark
python -m sensorflow.cli.benchmark --manifest runs/pipeline/seq_001/manifest.json
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/dataset/ingest` | Fuse multi-vendor sensor data |
| `POST /api/perception/auto-label` | SAM + LiDAR 3D proposals |
| `POST /api/perception/track` | Kalman temporal tracking |
| `POST /api/gates/quality` | Benchmark against vendor GT |
| `POST /api/gates/launch` | Validate export thresholds |
| `GET /api/pipeline/status` | Stage completion flags |

Export is blocked (`HTTP 403`) until the launch gate passes.

---

### Stage 1: Dataset Configuration
Configure your training config (e.g. `coco8.yaml`) and inference image folder. Click **Run Pre-Check** to verify script path locations on the filesystem.

### Stage 2: Model Setup
Select your base weights size (Nano, Small, or Medium) and choose your execution device (CPU or GPU index).

### Stage 3: Training Execution
Set your epoch limit and batch size. Press **Start Training** to launch the training run. Watch real-time console outputs, check the progress bar, and view the live loss curves generated directly on the canvas.

### Stage 4: Auto-Labeler Inference
Specify weight paths (defaults to your newly trained `best.pt`), set confidence and IoU thresholds, and trigger batch predictions. Once complete, select any processed image from the dropdown list to preview the visual annotations.

### Stage 5: Auto-Grader Quality
Trigger the diagnostics engine to scan predictions. Shows a unified **Quality Score** (0-100) and prints a breakdown of issues, including:
- **Low Confidence**: Box probability below threshold.
- **Overlapping Classes**: Conflicting classifications sharing high IoU.
- **Geometric Anomalies**: Overly small areas.
- **Imbalance Indicators**: Single-class dominate bias.

### Stage 6: Export & Deploy
Compile your PyTorch checkpoint to optimized ONNX or TensorRT models directly for production deployment.

---

## CLI Usage

### 1. Training CLI
```bash
python train.py --epochs 25 --batch 16 --device 0 --model yolov8m.pt
```

### 2. Inference CLI
```bash
python infer.py --source data/test.jpg --weights runs/detect/coco_finetuned/weights/best.pt --conf 0.25
```

### 3. Grading CLI
```bash
python autograder.py --predictions runs/infer/predictions.json
```

---

## Automated Verification

Run tests to verify server endpoints:
```bash
pytest
```

---

## 🚦 FHWA Surrogate Safety Assessment Model (SSAM)

The SSAM module processes trajectory prediction files to calculate vehicle-to-vehicle conflicts. It leverages key safety surrogates to classify risk profiles:
*   **Time-to-Collision (TTC):** The time required for two vehicles to collide if they maintain their current speed and path. Values under **1.5 seconds** indicate high collision hazards.
*   **Post-Encroachment Time (PET):** The time lapse between the first vehicle leaving a conflict zone and the second vehicle entering it. Values under **5.0 seconds** indicate significant lane encroachment conflicts.
*   **Conflict Angle:** Used to classify conflict types:
    *   **Crossing:** Angle $> 85^\circ$
    *   **Lane Change:** Angle between $30^\circ$ and $85^\circ$
    *   **Rear-end:** Angle $< 30^\circ$

Severity Index calculation is derived as follows:
$$severity = 1.0 - (\frac{\min(TTC, 1.5)}{1.5}) \times 0.7 - (\frac{\min(PET, 5.0)}{5.0}) \times 0.3$$

---

## 📂 6-Layer Accident Analysis Platform

For large-scale telemetry audits, the platform employs a modular architecture:
1.  **Ingestion Layer (`accident_importer.py`):** Automatically extracts and validates datasets from local `.csv`, `.json`, `.parquet`, `.xlsx`, REST endpoints, or SQLite connection strings.
2.  **Validation Layer (`accident_validator.py`):** Assures column configuration, geospatial coordinates boundaries (US bounding box limits), and valid historical dates.
3.  **Cleaning Layer (`accident_cleaner.py`):** Normalizes header schemas to `snake_case`, handles coordinate voids, and standardizes severity classifications.
4.  **Analysis Engine (`accident_analysis.py`):** Aggregates hourly temporal curves and geospatial conflict clusters.
5.  **Insights & Reporting (`accident_insights.py` & `accident_report.py`):** Computes overall safety recommendations and exports clean HTML summaries.
6.  **API Routing Gateway (`main.py`):** Exposes FastAPI REST interfaces and a streaming WebSocket channel to push processing alerts.

---

## ⚡ NVIDIA DGX Spark Remote Cluster Configuration

To execute intensive model training or inference workloads on the **NVIDIA DGX Spark** cluster:
1.  Verify Tailscale connectivity to `dgx-spark.tail16d8d9.ts.net` (`100.113.62.112`).
2.  Run the remote listener server environment with the following environment variables:
    ```bash
    export OLLAMA_HOST=0.0.0.0
    ollama run gemma4:26b
    ```
3.  Execute training by selecting `dgx-spark` from the **Compute Device** dropdown in the visual panel. The backend automatically forwards instructions and streams Tailscale connection handshakes.

