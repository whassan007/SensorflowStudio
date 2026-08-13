"""JSON persistence for vitis runs under runs/vitis/ (root settable for tests)."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path("runs/vitis")


def set_vitis_root(path: str) -> None:
    global _ROOT
    _ROOT = Path(path)


def vitis_root() -> Path:
    return _ROOT


def is_default_root() -> bool:
    """True when persisting to the real runs/vitis dir (not a test tmpdir)."""
    return _ROOT == Path("runs/vitis")


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(5)}"


def save_run(kind: str, run_id: str, payload: Dict) -> Path:
    d = _ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=1, default=_json_default))
    (d / "latest.json").write_text(
        json.dumps(payload, indent=1, default=_json_default))
    return path


def load_run(kind: str, run_id: str) -> Optional[Dict]:
    path = _ROOT / kind / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_runs(kind: str) -> List[Dict]:
    d = _ROOT / kind
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob(f"{kind[:3]}*-*.json")):
        try:
            data = json.loads(p.read_text())
            out.append({"run_id": data.get("run_id", p.stem),
                        "created_at": data.get("created_at"),
                        "summary": data.get("summary", {})})
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
