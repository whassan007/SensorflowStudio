"""Persistence + immutable audit log for the agentic subsystem.

Layout (relocatable for tests via set_agentic_root):

    runs/agentic/
        failures/{failure_id}.json      FailureEvent
        pipeline/{failure_id}.json      PipelineState
        snippets/{failure_id}/{snippet_id}.json
        evidence/{failure_id}.json      EvidenceGraph
        scorecards/{scorecard_id}.json
        reviews/{failure_id}.jsonl      HumanReviewDecision (append-only)
        suites/{suite_id}.json          EvaluationSuite
        policies/{policy_hash}.json     versioned policy documents
        audit/{failure_id}.jsonl        append-only audit trail
        audit/_global.jsonl             cross-failure audit trail

The audit log is append-only by construction: records are only ever appended
to the JSONL files and each record carries a monotonically increasing
`seq` per failure plus a hash chain (`prev_hash` -> `hash`) so tampering or
reordering is detectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional

from sensorflow.agentic.models import now_iso

_ROOT = os.path.join("runs", "agentic")
_ROOT_LOCK = threading.Lock()
_AUDIT_LOCK = threading.Lock()


def agentic_root() -> str:
    return _ROOT


def set_agentic_root(path: str) -> None:
    """Test hook: relocate all agentic storage."""
    global _ROOT
    with _ROOT_LOCK:
        _ROOT = str(path)


def _path(*parts: str) -> str:
    p = os.path.join(agentic_root(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_json(obj: Any, *parts: str) -> str:
    p = _path(*parts)
    payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    os.replace(tmp, p)
    return p


def read_json(*parts: str) -> Optional[Any]:
    p = os.path.join(agentic_root(), *parts)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def list_dir(*parts: str) -> List[str]:
    d = os.path.join(agentic_root(), *parts)
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))


def append_jsonl(record: Dict, *parts: str) -> None:
    p = _path(*parts)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(*parts: str) -> List[Dict]:
    p = os.path.join(agentic_root(), *parts)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------ audit


def _audit_file(failure_id: Optional[str]) -> str:
    name = f"{failure_id}.jsonl" if failure_id else "_global.jsonl"
    return os.path.join("audit", name)


def audit(event_type: str, failure_id: Optional[str], actor: str,
          detail: str = "", payload: Optional[Dict] = None) -> Dict:
    """Append one immutable audit record (hash-chained, per-failure seq)."""
    with _AUDIT_LOCK:
        existing = read_jsonl(_audit_file(failure_id))
        prev_hash = existing[-1]["hash"] if existing else "genesis"
        record = {
            "seq": len(existing),
            "event_type": event_type,
            "failure_id": failure_id,
            "actor": actor,
            "detail": detail,
            "payload": payload or {},
            "timestamp": now_iso(),
            "prev_hash": prev_hash,
        }
        blob = json.dumps({k: v for k, v in record.items() if k != "hash"},
                          sort_keys=True, default=str)
        record["hash"] = hashlib.sha256((prev_hash + blob).encode()).hexdigest()[:16]
        append_jsonl(record, _audit_file(failure_id))
    # mirror to the global trail for cross-failure timelines
    if failure_id is not None:
        with _AUDIT_LOCK:
            append_jsonl({**record, "mirrored_from": failure_id},
                         _audit_file(None))
    return record


def audit_trail(failure_id: str) -> List[Dict]:
    return read_jsonl(_audit_file(failure_id))


def verify_audit_chain(failure_id: str) -> Dict:
    """Recompute the hash chain; any edit/deletion/reorder breaks it."""
    records = audit_trail(failure_id)
    prev = "genesis"
    for i, rec in enumerate(records):
        blob = json.dumps({k: v for k, v in rec.items() if k != "hash"},
                          sort_keys=True, default=str)
        expected = hashlib.sha256((prev + blob).encode()).hexdigest()[:16]
        if rec.get("seq") != i or rec.get("prev_hash") != prev or rec.get("hash") != expected:
            return {"valid": False, "broken_at_seq": i, "records": len(records)}
        prev = rec["hash"]
    return {"valid": True, "records": len(records)}
