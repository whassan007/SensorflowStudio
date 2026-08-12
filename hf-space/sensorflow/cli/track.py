#!/usr/bin/env python3
"""Run temporal tracking on 3D proposals."""

import argparse
from pathlib import Path

from sensorflow.temporal_tracker import TemporalTracker


def main():
    parser = argparse.ArgumentParser(description="Temporal 3D tracking")
    parser.add_argument("--proposals-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    proposals_dir = Path(args.proposals_dir)
    output_path = Path(args.output) if args.output else proposals_dir.parent / "tracks.json"

    tracker = TemporalTracker()
    proposals = TemporalTracker.load_proposals(proposals_dir)
    tracks = tracker.run_sequence(proposals, output_path)
    print(f"Tracked {len(tracks)} objects -> {output_path}")


if __name__ == "__main__":
    main()
