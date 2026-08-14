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


def test_ingest_three_official_vendors(tmp_path, monkeypatch):
    """Dataset Configuration's three AV datasets are all accepted by ingest."""
    monkeypatch.chdir(tmp_path)
    engine = DatasetFusionEngine()
    seq = engine.ingest(["alpamayo", "waymo", "a2d2"], "three_vendors")
    assert seq.vendor == "mixed"
    assert len(seq.frames) >= 3
    assert set(seq.taxonomy_manifest["merged_vendors"]) == {"alpamayo", "waymo", "a2d2"}

    solo = engine.ingest(["a2d2"], "a2d2_only")
    assert solo.vendor == "a2d2"
    assert len(solo.frames) >= 1
    assert solo.taxonomy_manifest.get("demo_stub") is True


def test_ingest_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = DatasetFusionEngine()
    engine.ingest(["alpamayo"], "status_test")
    engine.save_manifest(engine.ingest(["alpamayo"], "status_test"))
    status = engine.get_status("status_test")
    assert status.get("ingest_complete") is True
