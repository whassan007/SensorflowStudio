#!/usr/bin/env python3
"""Ingest and fuse multi-vendor sensor data."""

import argparse
import json
from pathlib import Path

from sensorflow.dataset_fusion_engine import DatasetFusionEngine


def main():
    parser = argparse.ArgumentParser(description="Ingest and fuse sensor data")
    parser.add_argument("--vendors", nargs="+", default=["alpamayo", "waymo"])
    parser.add_argument("--sequence-id", default="seq_001")
    parser.add_argument("--alpamayo-source", default=None)
    parser.add_argument("--waymo-source", default=None)
    args = parser.parse_args()

    engine = DatasetFusionEngine()
    alp_src = json.loads(args.alpamayo_source) if args.alpamayo_source else None
    way_src = json.loads(args.waymo_source) if args.waymo_source else None

    sequence = engine.ingest(args.vendors, args.sequence_id, alp_src, way_src)
    manifest_path = engine.save_manifest(sequence)
    print(f"Ingested {len(sequence.frames)} frames -> {manifest_path}")


if __name__ == "__main__":
    main()
