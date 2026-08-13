"""Filesystem layout + JSON helpers for the safety layer.

All safety artifacts persist under runs/safety/ (relocatable for tests via
set_safety_root, mirroring megaeval's set_mega_root pattern):

    runs/safety/
        gate_policy.json           configurable release-gate thresholds
        odd/{run_id}.json          synthetic gap-fill supplements per evaluation run
        ssam/{run_id}.json         cached surrogate-safety (CSI) summaries per run
        calibration/status.json    latest multi-sensor calibration validation
        discrepancy/{dataset}.json online-vs-offline mining results
        evidence/{run_id}.json/.md safety evidence packages
        scenario_db.json           curated scenario database
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

SAFETY_ROOT = os.path.join("runs", "safety")

_LOCK = threading.RLock()


def set_safety_root(path: str) -> None:
    """Test hook: relocate all safety-layer storage."""
    global SAFETY_ROOT
    SAFETY_ROOT = str(path)


def safety_path(*parts: str) -> str:
    """Absolute-ish path under the safety root; parent dirs are created."""
    path = os.path.join(SAFETY_ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def read_json(*parts: str) -> Optional[Any]:
    path = os.path.join(SAFETY_ROOT, *parts)
    if not os.path.exists(path):
        return None
    with _LOCK:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None


def write_json(payload: Any, *parts: str) -> str:
    path = safety_path(*parts)
    with _LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    return path
