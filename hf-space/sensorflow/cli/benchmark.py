#!/usr/bin/env python3
"""Run quality gate benchmarking."""

import argparse
import json
from pathlib import Path

from sensorflow.quality_gate import QualityGate
from sensorflow.schemas.unified_frame import UnifiedSequence
from sensorflow.mitl_copilot import MitlCopilot


def main():
    parser = argparse.ArgumentParser(description="Quality gate benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tracks", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    sequence = UnifiedSequence.load(manifest_path)
    tracks_path = Path(args.tracks) if args.tracks else manifest_path.parent / "tracks.json"

    with open(tracks_path) as f:
        pred_tracks = json.load(f)

    gate = QualityGate()
    results = gate.evaluate(sequence, pred_tracks)
    out_dir = gate.save_results(sequence.sequence_id, results)

    copilot = MitlCopilot()
    if not results["passed"]:
        copilot.route_edge_cases(
            sequence.sequence_id,
            results["metric_card"],
            pred_tracks,
        )

    print(f"Quality gate {'PASSED' if results['passed'] else 'FAILED'} -> {out_dir}")
    print(json.dumps(results["metric_card"], indent=2))


if __name__ == "__main__":
    main()
