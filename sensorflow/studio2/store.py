"""Filesystem layout for the studio2 control plane.

Everything persists under runs/studio2/ (relocatable for tests via
set_studio2_root, mirroring safety.store / megaeval patterns):

    runs/studio2/
        registry/<kind>/<entity_id>.json   control-plane entities
        decisions/<decision_id>.json       release decisions (append-only)
        demo/<demo_id>.json + latest.json  closed-loop demo runs
        audit.jsonl                        append-only audit trail
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

STUDIO2_ROOT = os.path.join("runs", "studio2")

_LOCK = threading.RLock()


def set_studio2_root(path: str) -> None:
    """Test hook: relocate all studio2 storage."""
    global STUDIO2_ROOT
    STUDIO2_ROOT = str(path)


def studio2_path(*parts: str) -> str:
    path = os.path.join(STUDIO2_ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def read_json(*parts: str) -> Optional[Any]:
    path = os.path.join(STUDIO2_ROOT, *parts)
    if not os.path.exists(path):
        return None
    with _LOCK:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None


def write_json(payload: Any, *parts: str) -> str:
    path = studio2_path(*parts)
    with _LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
    return path


def list_dir(*parts: str) -> List[str]:
    path = os.path.join(STUDIO2_ROOT, *parts)
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit(action: str, entity_kind: Optional[str], entity_id: Optional[str],
          actor: str = "studio2", detail: str = "",
          payload: Optional[Dict] = None) -> Dict:
    rec = {"timestamp": now_iso(), "action": action, "entity_kind": entity_kind,
           "entity_id": entity_id, "actor": actor, "detail": detail,
           "payload": payload or {}}
    path = studio2_path("audit.jsonl")
    with _LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    return rec


def read_audit(limit: int = 200) -> List[Dict]:
    path = os.path.join(STUDIO2_ROOT, "audit.jsonl")
    if not os.path.exists(path):
        return []
    out: List[Dict] = []
    with _LOCK:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out[-limit:]
