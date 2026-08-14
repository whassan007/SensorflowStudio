"""Tests for pipeline output browse / artifact APIs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _large_png_bytes(min_bytes: int = 11 * 1024) -> bytes:
    base = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )
    return base if len(base) >= min_bytes else base + b"\x00" * (min_bytes - len(base))


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
    assert ingest.json()["vendor"] == "alpamayo"

    seqs = client.get("/api/pipeline/sequences").json()
    assert any(s["sequence_id"] == seq_id for s in seqs["sequences"])

    arts = client.get(f"/api/pipeline/artifacts?sequence_id={seq_id}").json()
    assert arts["browsable"] is True
    assert arts["frames"] >= 1
    assert arts["manifest_path"]

    frames = client.get(f"/api/pipeline/frames?sequence_id={seq_id}").json()
    assert frames["browsable"] is True
    assert frames["vendor"] == "alpamayo"
    assert len(frames["frames"]) >= 1
    # Homogeneous Alpamayo IDs
    assert all(f["frame_id"].startswith("alpamayo_") for f in frames["frames"])
    frame_id = frames["frames"][0]["frame_id"]
    assert frames["frames"][0].get("preview_url")

    detail = client.get(
        f"/api/pipeline/frame?sequence_id={seq_id}&frame_id={frame_id}"
    ).json()
    assert detail["frame_id"] == frame_id
    assert "cameras" in detail
    assert detail.get("preview_url") or any(c.get("preview_url") for c in detail["cameras"])
    # Empty proposals must not imply missing camera
    assert detail["proposal_count"] == 0

    img_url = detail.get("preview_url") or detail["cameras"][0]["preview_url"]
    assert img_url.startswith("/api/pipeline/file?")
    img_res = client.get(img_url)
    assert img_res.status_code == 200
    assert img_res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_single_vendor_waymo_homogeneous(client):
    ingest = client.post(
        "/api/dataset/ingest",
        json={"vendors": ["waymo"], "sequence_id": "waymo_only"},
    )
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["vendor"] == "waymo"
    frames = client.get("/api/pipeline/frames?sequence_id=waymo_only").json()
    assert frames["vendor"] == "waymo"
    assert all(f["frame_id"].startswith("waymo_") for f in frames["frames"])
    assert all(f.get("preview_url") for f in frames["frames"])


def test_load_all_datasets_separate_sequences(client, tmp_path):
    # AV vendors stub when allow_stub
    res = client.post(
        "/api/dataset/load-all",
        json={
            "sequence_prefix": "all_demo",
            "vendors": ["alpamayo", "waymo", "a2d2"],
            "allow_stub": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["loaded"] == 3
    vendors = {r["vendor"] for r in data["results"]}
    assert vendors == {"alpamayo", "waymo", "a2d2"}
    for r in data["results"]:
        assert r["status"] == "ok"
        assert r["demo_stub"] is True
        frames = client.get(f"/api/pipeline/frames?sequence_id={r['sequence_id']}").json()
        assert frames["vendor"] == r["vendor"]
        assert frames["vendor"] != "mixed"
        prefix = f"{r['vendor']}_"
        assert all(f["frame_id"].startswith(prefix) for f in frames["frames"])


def test_load_all_not_executed_without_stub(client, tmp_path):
    res = client.post(
        "/api/dataset/load-all",
        json={
            "sequence_prefix": "strict",
            "vendors": ["waymo", "a2d2"],
            "allow_stub": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NOT_EXECUTED"
    assert data["loaded"] == 0
    assert data["not_executed"] == 2
    for r in data["results"]:
        assert r["status"] == "NOT_EXECUTED"
        assert "message" in r


def test_load_all_local_real_when_path_has_images(client, tmp_path):
    data_dir = tmp_path / "data" / "images" / "train"
    data_dir.mkdir(parents=True)
    png = _large_png_bytes()
    (data_dir / "a.png").write_bytes(png)
    res = client.post(
        "/api/dataset/load-all",
        json={
            "sequence_prefix": "with_local",
            "vendors": ["local", "waymo"],
            "source_path": "data",
            "allow_stub": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    by_v = {r["vendor"]: r for r in data["results"]}
    assert by_v["local"]["status"] == "ok"
    assert by_v["local"]["demo_stub"] is False
    assert by_v["local"]["frames"] >= 1
    assert by_v["waymo"]["demo_stub"] is True


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
    data_dir = tmp_path / "data" / "camera" / "front"
    data_dir.mkdir(parents=True)
    (data_dir / "a.png").write_bytes(_large_png_bytes())
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
