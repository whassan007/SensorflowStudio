"""Persistence for RCA investigations under runs/rca/.

Layout:
    runs/rca/<investigation_id>/investigation.json
    runs/rca/<investigation_id>/offline.csv
    runs/rca/<investigation_id>/shadow.csv
    runs/rca/<investigation_id>/meta.json
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from sensorflow.rca.models import Investigation
from sensorflow.rca.scenario import ScenarioBundle

_DEFAULT_ROOT = "runs/rca"
_root = Path(_DEFAULT_ROOT)
_lock = threading.Lock()
_cache: Dict[str, Investigation] = {}
_data_cache: Dict[str, ScenarioBundle] = {}


def set_rca_root(path: str) -> None:
    """Point the store somewhere else (tests use a tmp dir)."""
    global _root
    _root = Path(path)
    reset_rca_store()


def reset_rca_store() -> None:
    _cache.clear()
    _data_cache.clear()


def _inv_dir(inv_id: str) -> Path:
    return _root / inv_id


def save_investigation(inv: Investigation,
                       bundle: Optional[ScenarioBundle] = None) -> None:
    with _lock:
        d = _inv_dir(inv.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "investigation.json").write_text(
            json.dumps(inv.to_json_dict(), indent=1))
        if bundle is not None:
            bundle.offline.to_csv(d / "offline.csv", index=False)
            bundle.shadow.to_csv(d / "shadow.csv", index=False)
            (d / "meta.json").write_text(json.dumps(bundle.meta, indent=1))
            _data_cache[inv.id] = bundle
        _cache[inv.id] = inv


def load_investigation(inv_id: str) -> Optional[Investigation]:
    if inv_id in _cache:
        return _cache[inv_id]
    f = _inv_dir(inv_id) / "investigation.json"
    if not f.exists():
        return None
    inv = Investigation.from_json_dict(json.loads(f.read_text()))
    _cache[inv_id] = inv
    return inv


def load_bundle(inv_id: str) -> Optional[ScenarioBundle]:
    if inv_id in _data_cache:
        return _data_cache[inv_id]
    d = _inv_dir(inv_id)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    offline = pd.read_csv(d / "offline.csv")
    shadow = pd.read_csv(d / "shadow.csv")
    bundle = ScenarioBundle(cause=meta["cause"], seed=meta["seed"],
                            offline=offline, shadow=shadow, meta=meta)
    _data_cache[inv_id] = bundle
    return bundle


def list_investigations() -> List[Dict]:
    out: List[Dict] = []
    if not _root.exists():
        return out
    for d in sorted(_root.iterdir()):
        f = d / "investigation.json"
        if not f.is_file():
            continue
        inv = load_investigation(d.name)
        if inv is None:
            continue
        done = sum(1 for s in inv.stages
                   if s.status in ("complete", "complete_with_unknowns"))
        summary = {
            "id": inv.id,
            "name": inv.name,
            "baseline_model": inv.baseline_model,
            "candidate_model": inv.candidate_model,
            "training_mode": inv.training_mode,
            "revealed": inv.revealed,
            "created_at": inv.created_at,
            "claims": inv.claims,
            "stages_complete": done,
            "stages_total": len(inv.stages),
        }
        if not inv.training_mode or inv.revealed:
            summary["scenario_cause"] = inv.scenario_cause
        out.append(summary)
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out
