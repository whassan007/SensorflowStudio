"""Write viewable stub camera PNGs without requiring Pillow."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Tuple


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _rgb_for_seed(seed: int) -> Tuple[int, int, int]:
    """Deterministic muted road-scene palette from seed."""
    r = 40 + (seed * 37) % 80
    g = 48 + (seed * 53) % 70
    b = 56 + (seed * 29) % 90
    return r, g, b


def write_stub_camera_png(
    path: Path,
    *,
    width: int = 640,
    height: int = 360,
    seed: int = 0,
    label: str = "",
) -> str:
    """
    Write a simple RGB PNG (horizon + road bands) and return its path string.

    Used by demo-stub adapters so Pipeline Outputs can serve a real local image
    even when remote AV lakes / Unsplash URLs are unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sky_r, sky_g, sky_b = _rgb_for_seed(seed)
    road_r, road_g, road_b = max(20, sky_r - 18), max(20, sky_g - 12), max(24, sky_b - 8)
    accent = ((seed * 17) % 200) + 40

    rows = []
    horizon = height * 5 // 12
    for y in range(height):
        row = bytearray([0])  # filter None
        for x in range(width):
            if y < horizon:
                # Sky gradient
                t = y / max(horizon, 1)
                r = int(sky_r + (220 - sky_r) * (1 - t) * 0.35)
                g = int(sky_g + (230 - sky_g) * (1 - t) * 0.35)
                b = int(sky_b + (245 - sky_b) * (1 - t) * 0.45)
            else:
                # Road + lane marker
                t = (y - horizon) / max(height - horizon, 1)
                r = int(road_r + t * 30)
                g = int(road_g + t * 28)
                b = int(road_b + t * 22)
                mid = width // 2
                if abs(x - mid) < 3 and (y // 12) % 2 == 0:
                    r = g = b = min(255, accent + 80)
            # Soft vignette border so the frame reads as a camera plate
            if x < 4 or x >= width - 4 or y < 3 or y >= height - 3:
                r = g = b = 12
            row.extend((r & 0xFF, g & 0xFF, b & 0xFF))
        rows.append(bytes(row))

    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw, 9))
    if label:
        # tEXt chunk is optional metadata for debugging
        text = b"Label\x00" + label.encode("latin-1", errors="replace")[:80]
        png += _png_chunk(b"tEXt", text)
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)
    return str(path)


def stub_camera_path(sequence_id: str, frame_id: str, camera: str = "front") -> Path:
    return Path("runs/pipeline") / sequence_id / "cameras" / f"{frame_id}_{camera}.png"
