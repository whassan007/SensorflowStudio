"""Tests for pipeline output browse / artifact APIs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Import after chdir so runs/ paths land in tmp_path.
    import importlib
    import app_backend
    importlib.reload(app_backend)
    return TestClient(app_backend.app)


def test_pipeline_browse_after_ingest(client, tmp_path):
    seq_id = "browse_seq"
    ingest = client.post(
        "/api/dataset/ingest",
        json={"vendors": ["alpamayo"], "sequence_id": seq_id},
    )
    assert ingest.status_code == 200
    assert ingest.json()["frames"] >= 1

    seqs = client.get("/api/pipeline/sequences").json()
    assert any(s["sequence_id"] == seq_id for s in seqs["sequences"])

    arts = client.get(f"/api/pipeline/artifacts?sequence_id={seq_id}").json()
    assert arts["browsable"] is True
    assert arts["frames"] >= 1
    assert arts["manifest_path"]

    frames = client.get(f"/api/pipeline/frames?sequence_id={seq_id}").json()
    assert frames["browsable"] is True
    assert len(frames["frames"]) >= 1
    frame_id = frames["frames"][0]["frame_id"]

    detail = client.get(
        f"/api/pipeline/frame?sequence_id={seq_id}&frame_id={frame_id}"
    ).json()
    assert detail["frame_id"] == frame_id
    assert "cameras" in detail


def test_pipeline_file_serves_local_image(client, tmp_path):
    img = tmp_path / "data" / "frame.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    # Minimal JPEG (1x1)
    img.write_bytes(
        bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
            "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
            "1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d"
            "0d1832211c2132323232323232323232323232323232323232323232323232323232"
            "323232323232323232323232323232323232323232ffc00011080001000103011100"
            "0211031101ffc40014000100000000000000000000000000000000ffc40014100100"
            "00000000000000000000000000000000ffda000c0301000210031000003f00bf80ffd9"
        )
    )

    res = client.get(f"/api/pipeline/file?path={img}")
    assert res.status_code == 200
    assert res.content[:2] == b"\xff\xd8"


def test_pipeline_file_blocks_path_traversal(client, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    # Escape runs/pipeline via .. — must 403
    res = client.get("/api/pipeline/file?path=runs/pipeline/../../secret.txt")
    assert res.status_code in (403, 404)


def test_dataset_browse_lists_images(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.png").write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    res = client.get("/api/dataset/browse?source_path=data")
    assert res.status_code == 200
    body = res.json()
    assert body["browsable"] is True
    assert body["count"] >= 1
    assert body["images"][0]["preview_url"].startswith("/api/pipeline/file?")


def test_dataset_details_marked_catalog_only(client):
    res = client.get("/api/dataset/details?type=local")
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["catalog_only"] is True
    assert meta["browsable"] is False
    assert "browse_hint" in meta


def test_empty_sequence_not_browsable(client):
    frames = client.get("/api/pipeline/frames?sequence_id=does_not_exist").json()
    assert frames["browsable"] is False
    assert frames["empty_reason"]
