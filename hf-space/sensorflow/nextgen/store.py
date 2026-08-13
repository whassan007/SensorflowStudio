"""JSON persistence for nextgen artifacts under runs/nextgen/.

Mirrors the conventions of sensorflow.safety.store (reused pattern rather
than importing it: that store hard-codes runs/safety/ and offers no root
override, which the nextgen tests need for isolation).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = "runs/nextgen"
_LOCK = threading.Lock()


def set_nextgen_root(path: str) -> None:
    global _ROOT
    _ROOT = path


def nextgen_root() -> str:
    return _ROOT


def _path(*parts: str) -> Path:
    return Path(_ROOT, *parts)


def write_json(obj: Dict, *parts: str) -> str:
    p = _path(*parts)
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, default=str)
        os.replace(tmp, p)
    return str(p)


def read_json(*parts: str) -> Optional[Dict]:
    p = _path(*parts)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def list_json(subdir: str) -> List[str]:
    d = _path(subdir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
