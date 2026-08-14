"""Tests for local sequence ingest and full-manifest auto-label."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_tiny_jpeg(path: Path) -> None:
    # Minimal valid JPEG (1x1 pixel)
    path.write_bytes(
        bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
            "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
            "1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d"
            "0d1832211c2132323232323232323232323232323232323232323232323232323232"
            "323232323232323232323232323232323232323232ffc00011080001000103011100"
            "021101031101ffc40014000100000000000000000000000000000000ffc400141001"
            "00000000000000000000000000000000ffda000c0301000210031000003f00bf80ffd9"
        )
    )


def test_local_adapter_loads_all_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames_dir = tmp_path / "seq_frames"
    frames_dir.mkdir()
    for i in range(12):
        _write_tiny_jpeg(frames_dir / f"frame_{i:03d}.jpg")

    from sensorflow.adapters.local_adapter import LocalSequenceAdapter

    seq = LocalSequenceAdapter().load({"source_path": str(frames_dir)}, "local_seq")
    assert seq.vendor == "local"
    assert len(seq.frames) == 12
    assert seq.taxonomy_manifest.get("demo_stub") is False


def test_local_adapter_respects_max_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames_dir = tmp_path / "seq_frames"
    frames_dir.mkdir()
    for i in range(20):
        _write_tiny_jpeg(frames_dir / f"frame_{i:03d}.jpg")

    from sensorflow.adapters.local_adapter import LocalSequenceAdapter

    seq = LocalSequenceAdapter().load(
        {"source_path": str(frames_dir), "max_frames": 7},
        "capped",
    )
    assert len(seq.frames) == 7


def test_fusion_engine_local_not_capped_at_six(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames_dir = tmp_path / "drive"
    frames_dir.mkdir()
    for i in range(15):
        _write_tiny_jpeg(frames_dir / f"{i:04d}.png")

    from sensorflow.dataset_fusion_engine import DatasetFusionEngine

    engine = DatasetFusionEngine()
    seq = engine.ingest(
        ["local"],
        "real_seq",
        source_path=str(frames_dir),
        max_frames=10000,
    )
    assert len(seq.frames) == 15
    assert seq.taxonomy_manifest.get("demo_stub") is False
    manifest = engine.save_manifest(seq)
    assert manifest.exists()


def test_stub_ingest_is_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from sensorflow.dataset_fusion_engine import DatasetFusionEngine

    engine = DatasetFusionEngine()
    seq = engine.ingest(["alpamayo", "waymo"], "stub_seq")
    assert len(seq.frames) == 6
    assert seq.taxonomy_manifest.get("demo_stub") is True


def test_auto_label_consumes_all_manifest_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames_dir = tmp_path / "imgs"
    frames_dir.mkdir()
    for i in range(9):
        _write_tiny_jpeg(frames_dir / f"f{i}.jpg")

    from app_backend import app

    client = TestClient(app)
    ingest = client.post(
        "/api/dataset/ingest",
        json={
            "vendors": ["local"],
            "sequence_id": "api_local",
            "source_path": str(frames_dir),
            "max_frames": 10000,
        },
    )
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["frames"] == 9
    assert body["demo_stub"] is False

    label = client.post(
        "/api/perception/auto-label",
        json={"sequence_id": "api_local", "no_sam": True},
    )
    assert label.status_code == 200
    labeled = label.json()
    assert labeled["frames_processed"] == 9
    assert labeled["frames_expected"] == 9
    assert labeled["demo_stub"] is False

    proposals = Path("runs/pipeline/api_local/proposals")
    assert len(list(proposals.glob("*.json"))) == 9


def test_pipeline_status_pending_not_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app_backend import app

    client = TestClient(app)
    res = client.get("/api/pipeline/status?sequence_id=fresh_seq")
    assert res.status_code == 200
    data = res.json()
    assert data["tracking_complete"] is None
    assert data["benchmark_complete"] is None
    assert data["launch_gate_passed"] is None
