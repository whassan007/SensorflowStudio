"""Execution integrity: ledger, dataset load, failure modes, deterministic fixture."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure repo root imports resolve
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    """Minimal valid PNG."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * width) for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Copy minimal scripts into tmp workspace so precheck/verify can find them
    for name in ("train.py", "infer.py", "autograder.py"):
        src = ROOT / name
        if src.exists():
            (tmp_path / name).write_text(src.read_text())
    # Point ledger at tmp
    from sensorflow import execution_ledger as ledger

    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "runs" / "executions")
    monkeypatch.setattr(ledger, "INDEX_PATH", tmp_path / "runs" / "executions" / "index.json")
    monkeypatch.setattr(ledger, "STRICT_MODE_PATH", tmp_path / "runs" / "studio_strict_mode.json")

    # Import app after chdir so CONFIG_PATH etc. resolve under tmp
    import importlib
    import app_backend

    importlib.reload(app_backend)
    return TestClient(app_backend.app)


def test_invalid_dataset_path_fails_with_zero_loaded(client, tmp_path):
    res = client.post(
        "/api/dataset/load",
        json={"source_path": str(tmp_path / "does_not_exist")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "FAILED"
    assert data["metrics"]["images_discovered"] == 0
    assert data["metrics"]["images_readable"] == 0
    assert data["execution_id"]
    # Must not claim 100% loaded
    assert data["metrics"].get("loaded_pct_of_discovered") in (None, 0, 0.0)


def test_corrupt_image_partial_success(client, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    good = d / "ok.png"
    good.write_bytes(_png_bytes())
    bad = d / "bad.png"
    bad.write_bytes(b"not-a-png-file!!!!!!!!!!!")

    res = client.post("/api/dataset/load", json={"source_path": str(d)})
    data = res.json()
    assert data["status"] == "PARTIAL_SUCCESS"
    assert data["metrics"]["images_discovered"] == 2
    assert data["metrics"]["images_readable"] == 1
    assert data["metrics"]["images_corrupt"] == 1
    assert data["execution_id"]


def test_yaml_class_out_of_range_validation_failed(client, tmp_path):
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    img = root / "images" / "train" / "a.png"
    img.write_bytes(_png_bytes())
    (root / "labels" / "train" / "a.txt").write_text("99 0.5 0.5 0.1 0.1\n")  # class 99 invalid

    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/train\nnames:\n  0: car\n  1: truck\n  2: person\n"
    )

    res = client.post("/api/yaml/validate", json={"path": str(yaml_path)})
    data = res.json()
    assert data["status"] == "VALIDATION_FAILED"
    assert data["class_id_out_of_range"]
    assert data["execution_id"]


def test_missing_checkpoint_inference_failed(client, tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.png").write_bytes(_png_bytes())

    res = client.post(
        "/api/infer/run",
        json={
            "weights": str(tmp_path / "missing_weights.pt"),
            "source": str(imgs),
            "conf": 0.25,
            "iou": 0.45,
        },
    )
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "execution_id" in detail
    assert "checkpoint" in detail["message"].lower() or "missing" in detail["message"].lower()


def test_deterministic_tiny_dataset_counts(client, tmp_path):
    """10 images, 10 annos, 25 objects, 3 classes — exact counts."""
    root = tmp_path / "tiny"
    img_dir = root / "images"
    lab_dir = root / "labels"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)

    # 25 objects across 10 label files, class ids 0/1/2
    object_plan = [
        [0, 0, 1],       # 3
        [1, 2],          # 2
        [0],             # 1
        [2, 2, 1, 0],    # 4
        [1, 1],          # 2
        [0, 1, 2],       # 3
        [2],             # 1
        [0, 0],          # 2
        [1, 2],          # 2
        [0, 1, 2, 0, 1], # 5  → total 25
    ]
    assert sum(len(x) for x in object_plan) == 25

    for i, classes in enumerate(object_plan):
        (img_dir / f"img_{i:02d}.png").write_bytes(_png_bytes())
        lines = [f"{c} 0.5 0.5 0.2 0.2" for c in classes]
        (lab_dir / f"img_{i:02d}.txt").write_text("\n".join(lines) + "\n")

    res = client.post("/api/dataset/load", json={"source_path": str(root)})
    data = res.json()
    assert data["status"] == "SUCCEEDED"
    assert data["metrics"]["images_discovered"] == 10
    assert data["metrics"]["images_readable"] == 10
    assert data["discovery"]["annotations_discovered"] == 10
    assert data["discovery"]["annotation_objects"] == 25
    assert set(data["discovery"]["class_ids"]) == {0, 1, 2}
    assert data["execution_id"]

    # Ledger list contains the record
    listed = client.get("/api/executions").json()["executions"]
    assert any(e["execution_id"] == data["execution_id"] for e in listed)
    full = client.get(f"/api/executions/{data['execution_id']}").json()
    assert full["status"] == "SUCCEEDED"
    assert full["records_succeeded"] == 10


def test_catalog_preprocess_is_not_executed(client):
    res = client.post("/api/dataset/preprocess", json={"dataset_type": "local"})
    data = res.json()
    assert data["status"] == "NOT_EXECUTED"
    assert data["catalog_only"] is True
    assert data["execution_id"]


def test_strict_mode_toggle(client):
    assert client.get("/api/strict-mode").json()["enabled"] is False
    client.post("/api/strict-mode", json={"enabled": True})
    assert client.get("/api/strict-mode").json()["enabled"] is True


def test_health_endpoint(client):
    data = client.get("/api/health").json()
    assert data["backend_connected"] is True
    assert "filesystem_writable" in data
    assert "scripts" in data


def test_grader_not_executed_without_predictions(client):
    data = client.get("/api/grade").json()
    assert data["status"] == "NOT_EXECUTED"
    assert data["execution_id"]
    assert data["quality_score"] is None
