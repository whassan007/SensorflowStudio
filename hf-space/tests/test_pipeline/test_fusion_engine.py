"""Test dataset fusion engine."""

from pathlib import Path

from sensorflow.dataset_fusion_engine import DatasetFusionEngine


def test_ingest_mixed_vendors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = DatasetFusionEngine()
    seq = engine.ingest(["alpamayo", "waymo"], "fusion_test")
    assert seq.vendor == "mixed"
    assert len(seq.frames) >= 2
    assert "stratification" in seq.taxonomy_manifest

    manifest_path = engine.save_manifest(seq)
    assert manifest_path.exists()


def test_ingest_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = DatasetFusionEngine()
    engine.ingest(["alpamayo"], "status_test")
    engine.save_manifest(engine.ingest(["alpamayo"], "status_test"))
    status = engine.get_status("status_test")
    assert status.get("ingest_complete") is True
