"""File persistence under runs/rotr/ (platform convention: request-scoped
FastAPI over JSON files; no services). `set_rotr_root` is the test hook,
mirroring set_safety_root / set_nextgen_root etc."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

_ROOT = os.path.join("runs", "rotr")
_ROOT_LOCK = threading.Lock()


def rotr_root() -> str:
    return _ROOT


def set_rotr_root(path: str) -> None:
    global _ROOT
    with _ROOT_LOCK:
        _ROOT = str(path)


def _path(*parts: str) -> str:
    return os.path.join(rotr_root(), *parts)


def write_json(obj: Any, *parts: str) -> str:
    path = _path(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    return path


def read_json(*parts: str) -> Optional[Dict]:
    path = _path(*parts)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_dir(*parts: str) -> List[str]:
    path = _path(*parts)
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def exists(*parts: str) -> bool:
    return os.path.exists(_path(*parts))
