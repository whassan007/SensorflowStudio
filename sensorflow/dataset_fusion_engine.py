"""Dataset fusion engine: temporal sync, stratification, manifest output."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensorflow.adapters.alpamayo_adapter import AlpamayoAdapter, DEFAULT_ALPAMAYO_SAMPLES
from sensorflow.adapters.waymo_adapter import WaymoAdapter
from sensorflow.schemas.taxonomy_axes import assign_taxonomy_axes
from sensorflow.schemas.unified_frame import FusedFrame, UnifiedSequence


class DatasetFusionEngine:
    """Ingest, fuse, and stratify multi-vendor sensor sequences."""

    def __init__(self, sync_tolerance_us: int = 50_000):
        self.sync_tolerance_us = sync_tolerance_us
        self.alpamayo = AlpamayoAdapter()
        self.waymo = WaymoAdapter()

    def ingest(
        self,
        vendors: List[str],
        sequence_id: str,
        alpamayo_source: Optional[Dict[str, Any]] = None,
        waymo_source: Optional[Dict[str, Any]] = None,
    ) -> UnifiedSequence:
        sequences: List[UnifiedSequence] = []

        if "alpamayo" in vendors:
            src = alpamayo_source or DEFAULT_ALPAMAYO_SAMPLES.get("physical_ai", {})
            sequences.append(self.alpamayo.load(src, f"{sequence_id}_alp"))

        if "waymo" in vendors:
            sequences.append(self.waymo.load(waymo_source or {}, f"{sequence_id}_way"))

        if not sequences:
            sequences.append(self.alpamayo.load(DEFAULT_ALPAMAYO_SAMPLES["physical_ai"], sequence_id))

        fused = self._merge_sequences(sequences, sequence_id)
        fused = self._stratify(fused)
        return fused

    def _merge_sequences(self, sequences: List[UnifiedSequence], sequence_id: str) -> UnifiedSequence:
        if len(sequences) == 1:
            seq = sequences[0]
            seq.sequence_id = sequence_id
            return seq

        all_frames: List[FusedFrame] = []
        for seq in sequences:
            all_frames.extend(seq.frames)
        all_frames.sort(key=lambda f: f.timestamp_us)

        vendor = "mixed" if len(sequences) > 1 else sequences[0].vendor
        return UnifiedSequence(
            sequence_id=sequence_id,
            vendor=vendor,
            frames=all_frames,
            calibration=sequences[0].calibration,
            taxonomy_manifest={"merged_vendors": [s.vendor for s in sequences]},
        )

    def _stratify(self, sequence: UnifiedSequence) -> UnifiedSequence:
        """Tag frames/objects with six-axis taxonomy and build stratification manifest."""
        strata: Counter = Counter()
        for frame in sequence.frames:
            speed = frame.ego_pose.speed_kmh
            for gt in frame.ground_truth:
                gt.taxonomy_axes = assign_taxonomy_axes(gt.class_name, speed_kmh=speed)
                key = f"{gt.taxonomy_axes.mode}|{gt.taxonomy_axes.actor}|{gt.taxonomy_axes.infra}"
                strata[key] += 1
        sequence.taxonomy_manifest["stratification"] = dict(strata)
        sequence.taxonomy_manifest["total_frames"] = len(sequence.frames)
        sequence.taxonomy_manifest["total_objects"] = sum(len(f.ground_truth) for f in sequence.frames)
        return sequence

    def save_manifest(self, sequence: UnifiedSequence, output_dir: Optional[Path] = None) -> Path:
        output_dir = output_dir or Path("runs/pipeline") / sequence.sequence_id
        manifest_path = output_dir / "manifest.json"
        sequence.save(manifest_path)
        state_path = Path("runs/pipeline/state.json")
        state = self._load_state(state_path)
        state[sequence.sequence_id] = {
            "ingest_complete": True,
            "frames": len(sequence.frames),
            "vendor": sequence.vendor,
            "manifest": str(manifest_path),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        return manifest_path

    @staticmethod
    def _load_state(path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def get_status(self, sequence_id: str) -> Dict[str, Any]:
        state = self._load_state(Path("runs/pipeline/state.json"))
        return state.get(sequence_id, {"ingest_complete": False})
