"""Tests for AV driving-media browse filtering."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )


def _large_png_bytes(min_bytes: int = 11 * 1024) -> bytes:
    """Valid PNG header with zero padding so size heuristics accept it in tests."""
    base = _png_bytes()
    if len(base) >= min_bytes:
        return base
    return base + b"\x00" * (min_bytes - len(base))


def _write_driving_frame(root: Path, rel: str = "images/train/frame.png") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_large_png_bytes())
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import app_backend

    importlib.reload(app_backend)
    return TestClient(app_backend.app)


def test_browse_excludes_junk_test_file_for_alpamayo(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jpg").write_bytes(_large_png_bytes())  # wrong ext still image-like size
    _write_driving_frame(data_dir, "camera/front/scene_001.jpg").write_bytes(_large_png_bytes())

    res = client.get(
        "/api/dataset/browse?source_path=data&dataset_type=alpamayo"
    )
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    assert body["browsable"] is True
    assert body["excluded_count"] >= 1
    names = {img["name"] for img in body["images"]}
    assert "scene_001.jpg" in names
    assert "test.jpg" not in names
    reasons = {e["reason"] for e in body["excluded"]}
    assert "junk_filename" in reasons or "non_av_layout" in reasons


def test_browse_excludes_docs_and_logo_paths(client, tmp_path):
    root = tmp_path / "data"
    _write_driving_frame(root, "frames/road.png")
    logo = root / "docs" / "logo.png"
    logo.parent.mkdir(parents=True)
    logo.write_bytes(_large_png_bytes())
    icon = root / "assets" / "icon.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(_large_png_bytes())

    res = client.get("/api/dataset/browse?source_path=data&dataset_type=local")
    body = res.json()
    assert body["count"] == 1
    assert body["excluded_count"] == 2
    assert all(e["reason"] == "excluded_path" for e in body["excluded"])


def test_browse_excludes_tiny_icons(client, tmp_path):
    root = tmp_path / "data"
    _write_driving_frame(root, "images/train/big.png")
    tiny = root / "images" / "train" / "favicon.png"
    tiny.write_bytes(_png_bytes())  # < 10KB and junk name

    res = client.get("/api/dataset/browse?source_path=data&dataset_type=local")
    body = res.json()
    assert body["count"] == 1
    assert body["excluded_count"] == 1
    assert body["excluded"][0]["reason"] in {"too_small", "junk_filename"}


def test_browse_lists_valid_driving_media(client, tmp_path):
    root = tmp_path / "data"
    _write_driving_frame(root, "camera/front/a.png")

    res = client.get("/api/dataset/browse?source_path=data&dataset_type=local")
    assert res.status_code == 200
    body = res.json()
    assert body["browsable"] is True
    assert body["count"] == 1
    assert body["images"][0]["preview_url"].startswith("/api/pipeline/file?")


def test_load_reports_images_excluded(client, tmp_path):
    root = tmp_path / "data"
    _write_driving_frame(root, "images/train/good.png")
    (root / "sample.jpg").write_bytes(_large_png_bytes())

    res = client.post(
        "/api/dataset/load",
        json={"source_path": str(root), "dataset_type": "local"},
    )
    data = res.json()
    assert data["metrics"]["images_discovered"] == 1
    assert data["metrics"]["images_excluded"] == 1
    assert data["discovery"]["excluded_by_reason"]


def test_repo_data_test_jpg_excluded_when_alpamayo_selected(client, tmp_path, monkeypatch):
    """Regression: loose data/test.jpg must not appear for official AV dataset types."""
    repo_data = Path(__file__).resolve().parents[2] / "data"
    if not (repo_data / "test.jpg").exists():
        pytest.skip("repo data/test.jpg not present")

    monkeypatch.chdir(tmp_path)
    shutil.copytree(repo_data, tmp_path / "data")

    import app_backend

    importlib.reload(app_backend)
    c = TestClient(app_backend.app)
    res = c.get("/api/dataset/browse?source_path=data&dataset_type=alpamayo")
    body = res.json()
    assert body["count"] == 0
    assert body["browsable"] is False
    assert body["excluded_count"] >= 1
    assert not body["images"]
