"""Minimal dependency-free PNG encoder (8-bit grayscale or RGB)."""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(img: np.ndarray) -> bytes:
    """Encode a float [0,1] or uint8 array (H,W) or (H,W,3) as PNG bytes."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if img.ndim == 2:
        color_type, arr = 0, img[:, :, None]
    else:
        color_type, arr = 2, img[:, :, :3]
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) +
            _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))


def png_data_uri(img: np.ndarray) -> str:
    return ("data:image/png;base64," +
            base64.b64encode(encode_png(img)).decode("ascii"))
