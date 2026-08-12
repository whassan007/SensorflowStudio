"""Test perception automator."""

from pathlib import Path

from sensorflow.adapters.alpamayo_adapter import AlpamayoAdapter, DEFAULT_ALPAMAYO_SAMPLES
from sensorflow.dataset_fusion_engine import DatasetFusionEngine
from sensorflow.perception_automator import PerceptionAutomator


def test_perception_automator_no_sam(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = AlpamayoAdapter()
    seq = adapter.load(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"], "perc_test")
    DatasetFusionEngine().save_manifest(seq)

    automator = PerceptionAutomator(use_sam=False)
    proposals = automator.run_sequence(seq, tmp_path / "proposals")
    assert len(proposals) >= 1
    for frame_id, props in proposals.items():
        assert len(props) >= 1
        assert len(props[0].bbox_3d) == 7


def test_perception_frame_proposals():
    adapter = AlpamayoAdapter()
    seq = adapter.load(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"], "frame_test")
    automator = PerceptionAutomator(use_sam=False)
    proposals = automator.run_frame(seq.frames[0])
    assert len(proposals) >= 1
    assert proposals[0].bbox_3d[3] > 0  # length > 0
