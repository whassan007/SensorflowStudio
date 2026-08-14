"""Auto-label pre-checks, status semantics, and UI-facing error payloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_tiny_jpeg(path: Path) -> None:
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


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from sensorflow import execution_ledger as ledger

    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "runs" / "executions")
    monkeypatch.setattr(ledger, "INDEX_PATH", tmp_path / "runs" / "executions" / "index.json")
    monkeypatch.setattr(ledger, "STRICT_MODE_PATH", tmp_path / "runs" / "studio_strict_mode.json")

    import importlib
    import app_backend

    importlib.reload(app_backend)
    return TestClient(app_backend.app)


def _ingest_local(client: TestClient, tmp_path: Path, seq_id: str = "seq_local", n: int = 5):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(n):
        _write_tiny_jpeg(frames_dir / f"f{i}.jpg")
    res = client.post(
        "/api/dataset/ingest",
        json={
            "vendors": ["local"],
            "sequence_id": seq_id,
            "source_path": str(frames_dir),
            "max_frames": 10000,
        },
    )
    assert res.status_code == 200
    assert res.json()["frames"] == n
    return res


def test_missing_checkpoint_returns_failed_not_partial(client, tmp_path):
    _ingest_local(client, tmp_path, "seq_ckpt", 3)
    res = client.post(
        "/api/perception/auto-label",
        json={
            "sequence_id": "seq_ckpt",
            "sam_checkpoint": str(tmp_path / "missing_sam.pth"),
            "no_sam": False,
        },
    )
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["status"] == "FAILED"
    assert detail["checkpoint"]["exists"] is False
    assert detail["frames_expected"] == 3
    assert detail["frames_processed"] == 0
    assert detail.get("next_steps")
    assert "FAILED" == detail["status"]
    assert "PARTIAL_SUCCESS" != detail["status"]


def test_empty_manifest_returns_failed(client, tmp_path):
    manifest_dir = tmp_path / "runs" / "pipeline" / "seq_empty"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sequence_id": "seq_empty",
                "vendor": "local",
                "frames": [],
                "taxonomy_manifest": {"demo_stub": False},
            }
        )
    )
    res = client.post(
        "/api/perception/auto-label",
        json={"sequence_id": "seq_empty", "no_sam": True},
    )
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["status"] == "FAILED"
    assert detail["frames_expected"] == 0
    assert "ingest" in detail["message"].lower() or "0 ingested" in detail["message"].lower()


def test_ingest_then_no_sam_auto_label_has_frames(client, tmp_path):
    _ingest_local(client, tmp_path, "seq_ok", 4)
    res = client.post(
        "/api/perception/auto-label",
        json={"sequence_id": "seq_ok", "no_sam": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["frames_expected"] == 4
    assert body["frames_processed"] == 4
    assert body["status"] == "NOT_EXECUTED"
    assert body["sam_ran"] is False
    assert isinstance(body.get("message"), str)
    assert "{" not in body["message"][:20]
