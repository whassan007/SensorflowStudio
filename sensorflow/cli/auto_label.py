#!/usr/bin/env python3
"""Run SAM-based 3D auto-labeling on a fused sequence."""

import argparse
from pathlib import Path

from sensorflow.perception_automator import PerceptionAutomator
from sensorflow.schemas.unified_frame import UnifiedSequence


def main():
    parser = argparse.ArgumentParser(description="3D auto-labeling with SAM")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sam-checkpoint", default="models/sam_vit_b.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-sam", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    sequence = UnifiedSequence.load(manifest_path)
    output_dir = manifest_path.parent / "proposals"

    automator = PerceptionAutomator(
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        use_sam=not args.no_sam,
    )
    proposals = automator.run_sequence(sequence, output_dir)
    print(f"Generated proposals for {len(proposals)} frames -> {output_dir}")


if __name__ == "__main__":
    main()
