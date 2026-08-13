"""Persistence for retro artifacts under runs/retro/.

    runs/retro/analyses/<evaluation_id>.json   scorecard + metadata
    runs/retro/audit/<analysis_id>.jsonl       tool-call audit trail
    runs/retro/eval_cases/<case_id>.json       cases created by the write tool
    runs/retro/uploads/<name>.json             uploaded logs (allowlisted reads)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

def root() -> Path:
    """Resolved per call so tests can isolate via RETRO_RUNS_DIR."""
    return Path(os.environ.get("RETRO_RUNS_DIR", "runs/retro"))


def _dir(name: str) -> Path:
    d = root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def analyses_dir() -> Path:
    return _dir("analyses")


def audit_dir() -> Path:
    return _dir("audit")


def eval_cases_dir() -> Path:
    return _dir("eval_cases")


def uploads_dir() -> Path:
    return _dir("uploads")


def save_analysis(evaluation_id: str, payload: Dict) -> Path:
    path = analyses_dir() / f"{evaluation_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_analysis(evaluation_id: str) -> Optional[Dict]:
    path = analyses_dir() / f"{evaluation_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_analyses() -> List[Dict]:
    out = []
    for p in sorted(analyses_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text())
            sc = data.get("scorecard", {})
            out.append({
                "evaluation_id": sc.get("evaluation_id", p.stem),
                "created_at": sc.get("created_at"),
                "failure_type": sc.get("failure_type"),
                "severity": sc.get("severity"),
                "launch_recommendation": sc.get("launch_recommendation"),
                "backend_used": sc.get("backend_used"),
                "human_review_required": sc.get("human_review_required"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return out


def append_audit(analysis_id: str, record: Dict) -> None:
    path = audit_dir() / f"{analysis_id}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_audit(analysis_id: str) -> List[Dict]:
    path = audit_dir() / f"{analysis_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
