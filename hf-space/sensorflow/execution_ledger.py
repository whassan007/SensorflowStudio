"""Persistent execution ledger — verifiable evidence for Studio operations.

Statuses use QUEUED|RUNNING|SUCCEEDED|PARTIAL_SUCCESS|FAILED|CANCELLED|NOT_EXECUTED.
Never use "done" as a status substitute.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LEDGER_ROOT = Path("runs/executions")
INDEX_PATH = LEDGER_ROOT / "index.json"
STRICT_MODE_PATH = Path("runs/studio_strict_mode.json")

VALID_STATUSES = frozenset(
    {
        "QUEUED",
        "RUNNING",
        "SUCCEEDED",
        "PARTIAL_SUCCESS",
        "FAILED",
        "CANCELLED",
        "NOT_EXECUTED",
        "UNVERIFIED",
        "VALIDATION_FAILED",
    }
)

_lock = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    LEDGER_ROOT.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"executions": []}, indent=2))


def new_execution_id() -> str:
    return f"exec_{uuid.uuid4().hex[:12]}"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(snapshot: Any) -> str:
    raw = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_strict_mode() -> bool:
    if not STRICT_MODE_PATH.exists():
        return False
    try:
        return bool(json.loads(STRICT_MODE_PATH.read_text()).get("enabled", False))
    except Exception:
        return False


def set_strict_mode(enabled: bool) -> Dict[str, Any]:
    STRICT_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": bool(enabled), "updated_at": _utc_now()}
    STRICT_MODE_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def create_execution(
    operation: str,
    *,
    requested_by: str = "studio",
    configuration_snapshot: Optional[Dict[str, Any]] = None,
    input_artifacts: Optional[List[Any]] = None,
    command: Optional[List[str]] = None,
    status: str = "QUEUED",
) -> Dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    _ensure_dirs()
    execution_id = new_execution_id()
    record: Dict[str, Any] = {
        "execution_id": execution_id,
        "operation": operation,
        "status": status,
        "started_at": _utc_now(),
        "completed_at": None,
        "duration_ms": None,
        "requested_by": requested_by,
        "configuration_snapshot": configuration_snapshot or {},
        "configuration_hash": config_hash(configuration_snapshot or {}),
        "input_artifacts": input_artifacts or [],
        "output_artifacts": [],
        "records_discovered": 0,
        "records_processed": 0,
        "records_succeeded": 0,
        "records_failed": 0,
        "errors": [],
        "warnings": [],
        "logs": [],
        "events": [],
        "metrics": {},
        "process_id": None,
        "command": command,
        "exit_code": None,
        "verified": False,
        "strict_mode": get_strict_mode(),
        "unverified_reasons": [],
    }
    _append_event(record, "created", f"Operation {operation} queued")
    _write_record(record)
    _index_upsert(record)
    return record


def mark_running(execution_id: str, *, process_id: Optional[int] = None) -> Dict[str, Any]:
    record = load_execution(execution_id)
    record["status"] = "RUNNING"
    if process_id is not None:
        record["process_id"] = process_id
    _append_event(record, "running", f"Process started pid={process_id}")
    _write_record(record)
    _index_upsert(record)
    return record


def append_log(execution_id: str, line: str) -> None:
    record = load_execution(execution_id)
    record.setdefault("logs", []).append({"ts": _utc_now(), "line": line})
    if len(record["logs"]) > 2000:
        record["logs"] = record["logs"][-2000:]
    _write_record(record)


def append_event(execution_id: str, kind: str, message: str, **extra: Any) -> Dict[str, Any]:
    record = load_execution(execution_id)
    _append_event(record, kind, message, **extra)
    _write_record(record)
    _index_upsert(record)
    return record


def _append_event(record: Dict[str, Any], kind: str, message: str, **extra: Any) -> None:
    event = {"ts": _utc_now(), "kind": kind, "message": message}
    event.update(extra)
    record.setdefault("events", []).append(event)


def finalize(
    execution_id: str,
    status: str,
    *,
    exit_code: Optional[int] = None,
    output_artifacts: Optional[List[Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Any]] = None,
    warnings: Optional[List[Any]] = None,
    records_discovered: Optional[int] = None,
    records_processed: Optional[int] = None,
    records_succeeded: Optional[int] = None,
    records_failed: Optional[int] = None,
    process_invoked: bool = False,
    outputs_valid: Optional[bool] = None,
) -> Dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    record = load_execution(execution_id)
    record["status"] = status
    record["completed_at"] = _utc_now()
    try:
        start = datetime.fromisoformat(record["started_at"])
        end = datetime.fromisoformat(record["completed_at"])
        record["duration_ms"] = int((end - start).total_seconds() * 1000)
    except Exception:
        record["duration_ms"] = None
    if exit_code is not None:
        record["exit_code"] = exit_code
    if output_artifacts is not None:
        record["output_artifacts"] = output_artifacts
    if metrics is not None:
        record["metrics"] = {**record.get("metrics", {}), **metrics}
    if errors:
        record["errors"] = list(record.get("errors") or []) + list(errors)
    if warnings:
        record["warnings"] = list(record.get("warnings") or []) + list(warnings)
    if records_discovered is not None:
        record["records_discovered"] = records_discovered
    if records_processed is not None:
        record["records_processed"] = records_processed
    if records_succeeded is not None:
        record["records_succeeded"] = records_succeeded
    if records_failed is not None:
        record["records_failed"] = records_failed

    verified, reasons = verify_success(record, process_invoked=process_invoked, outputs_valid=outputs_valid)
    record["verified"] = verified
    record["unverified_reasons"] = reasons

    if get_strict_mode() and status == "SUCCEEDED" and not verified:
        record["status"] = "UNVERIFIED"
        record["warnings"] = list(record.get("warnings") or []) + [
            "Strict mode: SUCCEEDED refused — evidence incomplete"
        ]
        status = "UNVERIFIED"

    _append_event(record, "finalized", f"Status={record['status']} verified={verified}")
    _write_record(record)
    _index_upsert(record)
    return record


def verify_success(
    record: Dict[str, Any],
    *,
    process_invoked: bool = False,
    outputs_valid: Optional[bool] = None,
) -> tuple[bool, List[str]]:
    """Strict evidence checks for SUCCEEDED claims."""
    reasons: List[str] = []
    if not record.get("execution_id"):
        reasons.append("missing_execution_id")
    if not record.get("configuration_snapshot") and not record.get("configuration_hash"):
        reasons.append("missing_configuration")
    if process_invoked:
        if record.get("process_id") is None and not record.get("command"):
            reasons.append("process_not_recorded")
        if record.get("exit_code") not in (0, None) and record.get("exit_code") != 0:
            # Allow None when process_invoked False path; when invoked require 0
            pass
        if process_invoked and record.get("exit_code") not in (0,):
            if record.get("exit_code") is None:
                reasons.append("missing_exit_code")
            elif record.get("exit_code") != 0:
                reasons.append(f"nonzero_exit_code:{record.get('exit_code')}")
    outs = record.get("output_artifacts") or []
    if outputs_valid is False:
        reasons.append("outputs_invalid")
    elif outputs_valid is None and not outs:
        # Soft: only require outputs when claiming SUCCEEDED
        if record.get("status") in ("SUCCEEDED", "RUNNING"):
            reasons.append("missing_output_artifacts")
    metrics = record.get("metrics") or {}
    if record.get("status") == "SUCCEEDED" and not metrics and not outs:
        reasons.append("missing_metrics_and_outputs")
    return (len(reasons) == 0), reasons


def record_path(execution_id: str) -> Path:
    return LEDGER_ROOT / f"{execution_id}.json"


def load_execution(execution_id: str) -> Dict[str, Any]:
    path = record_path(execution_id)
    if not path.exists():
        raise FileNotFoundError(f"Execution not found: {execution_id}")
    with _lock:
        return json.loads(path.read_text())


def _write_record(record: Dict[str, Any]) -> None:
    _ensure_dirs()
    path = record_path(record["execution_id"])
    with _lock:
        path.write_text(json.dumps(record, indent=2, default=str))


def _index_upsert(record: Dict[str, Any]) -> None:
    _ensure_dirs()
    with _lock:
        try:
            index = json.loads(INDEX_PATH.read_text())
        except Exception:
            index = {"executions": []}
        entries = index.get("executions") or []
        summary = {
            "execution_id": record["execution_id"],
            "operation": record.get("operation"),
            "status": record.get("status"),
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
            "duration_ms": record.get("duration_ms"),
            "verified": record.get("verified"),
            "records_discovered": record.get("records_discovered"),
            "records_processed": record.get("records_processed"),
            "records_succeeded": record.get("records_succeeded"),
            "records_failed": record.get("records_failed"),
        }
        entries = [e for e in entries if e.get("execution_id") != record["execution_id"]]
        entries.insert(0, summary)
        index["executions"] = entries[:500]
        INDEX_PATH.write_text(json.dumps(index, indent=2))


def list_executions(
    *,
    operation: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    _ensure_dirs()
    try:
        index = json.loads(INDEX_PATH.read_text())
    except Exception:
        return []
    entries = index.get("executions") or []
    if operation:
        entries = [e for e in entries if e.get("operation") == operation]
    if status:
        entries = [e for e in entries if e.get("status") == status]
    return entries[:limit]


def get_log_text(execution_id: str) -> str:
    record = load_execution(execution_id)
    lines = []
    for entry in record.get("logs") or []:
        if isinstance(entry, dict):
            lines.append(f"{entry.get('ts', '')} {entry.get('line', '')}")
        else:
            lines.append(str(entry))
    for err in record.get("errors") or []:
        lines.append(f"ERROR: {err}")
    return "\n".join(lines)
