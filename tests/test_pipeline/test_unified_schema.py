"""Test unified schema and adapters."""

import json
from pathlib import Path

import pytest

from sensorflow.adapters.a2d2_adapter import A2D2Adapter
from sensorflow.adapters.alpamayo_adapter import AlpamayoAdapter, DEFAULT_ALPAMAYO_SAMPLES
from sensorflow.adapters.waymo_adapter import WaymoAdapter
from sensorflow.schemas.taxonomy_axes import assign_taxonomy_axes
from sensorflow.schemas.unified_frame import UnifiedSequence


def test_taxonomy_axes_pedestrian():
    axes = assign_taxonomy_axes("pedestrian", speed_kmh=30)
    assert axes.vulnerable is True
    assert axes.actor == "pedestrian"


def test_alpamayo_adapter_load():
    adapter = AlpamayoAdapter()
    seq = adapter.load(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"], "test_seq")
    assert seq.vendor == "alpamayo"
    assert len(seq.frames) >= 1
    assert seq.frames[0].lidar.num_points > 0


def test_waymo_adapter_load():
    adapter = WaymoAdapter()
    seq = adapter.load({}, "test_waymo")
    assert seq.vendor == "waymo"
    assert len(seq.frames) >= 1
    assert len(seq.frames[0].ground_truth) >= 1


def test_a2d2_adapter_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = A2D2Adapter()
    seq = adapter.load({}, "test_a2d2")
    assert seq.vendor == "a2d2"
    assert len(seq.frames) >= 1
    assert len(seq.frames[0].ground_truth) >= 1
    assert seq.taxonomy_manifest.get("demo_stub") is True


def test_unified_sequence_roundtrip(tmp_path):
    adapter = AlpamayoAdapter()
    seq = adapter.load(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"], "roundtrip")
    path = tmp_path / "manifest.json"
    seq.save(path)
    loaded = UnifiedSequence.load(path)
    assert loaded.sequence_id == "roundtrip"
    assert len(loaded.frames) == len(seq.frames)
