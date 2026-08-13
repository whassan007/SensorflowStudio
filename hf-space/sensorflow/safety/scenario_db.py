"""Curated scenario database (Safety Pool-inspired, local).

Industry concept: shared scenario databases (Safety Pool, StreetWise, ASAM
OpenSCENARIO catalogs) curate reusable test scenarios with taxonomy tags, ODD
attributes, severity and provenance. This module keeps a local, JSON-persisted
equivalent under runs/safety/scenario_db.json:

- record: id, scenario type taxonomy, ODD tags, source
  (mined | synthetic | discrepancy | rare_event), severity, provenance /
  lineage, evidence refs
- auto-population from the existing rare-event store, discrepancy mining
  reports and ODD gap-filling
- search/filter + JSON bundle export

Record ids are deterministic hashes of their provenance so repeated
auto-population is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from sensorflow.safety.store import read_json, write_json

SourceType = Literal["mined", "synthetic", "discrepancy", "rare_event"]
SEVERITIES = ["low", "medium", "high", "critical"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _det_id(*parts: str) -> str:
    return "scn-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


class ScenarioRecord(BaseModel):
    scenario_id: str
    scenario_type: str
    odd_tags: Dict[str, str] = Field(default_factory=dict)
    source: SourceType = "mined"
    severity: str = "medium"
    description: str = ""
    provenance: Dict[str, str] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class ScenarioDB:
    """JSON-persisted scenario catalog with idempotent auto-population."""

    def __init__(self):
        self._lock = threading.RLock()
        self._records: Dict[str, ScenarioRecord] = {}
        self._load()

    def _load(self) -> None:
        raw = read_json("scenario_db.json") or {}
        for sid, payload in raw.get("records", {}).items():
            try:
                self._records[sid] = ScenarioRecord.model_validate(payload)
            except Exception:
                continue

    def _save(self) -> None:
        write_json({"records": {sid: r.model_dump() for sid, r in self._records.items()},
                    "updated_at": _now()}, "scenario_db.json")

    # ---- ingestion

    def add(self, record: ScenarioRecord) -> ScenarioRecord:
        with self._lock:
            self._records[record.scenario_id] = record
            self._save()
        return record

    def add_from_rare_events(self, store) -> int:
        """Idempotently import every rare event as a mined/rare_event scenario."""
        added = 0
        with self._lock:
            for ev in store.all("rare_events"):
                sid = _det_id("rare_event", ev.event_id)
                if sid in self._records:
                    continue
                frame = store.get("frames", ev.evidence_frames[0]) if ev.evidence_frames else None
                odd_tags = {}
                if frame is not None:
                    odd_tags = {"weather": frame.weather, "lighting": frame.time_of_day}
                self._records[sid] = ScenarioRecord(
                    scenario_id=sid,
                    scenario_type=ev.scenario_type,
                    odd_tags=odd_tags,
                    source="rare_event",
                    severity=ev.severity,
                    description=ev.description,
                    provenance={"event_id": ev.event_id, "dataset_id": ev.dataset_id},
                    evidence_refs=[f"rare_event:{ev.event_id}"]
                                  + [f"frame:{f}" for f in ev.evidence_frames[:5]],
                )
                added += 1
            if added:
                self._save()
        return added

    def add_from_discrepancies(self, report: Dict) -> int:
        """Import discrepancy-mining findings (high/critical severity only)."""
        added = 0
        with self._lock:
            for d in report.get("discrepancies", []):
                if d["severity"] not in ("high", "critical"):
                    continue
                sid = _det_id("discrepancy", d["dataset_id"], d["gt_id"], d["type"])
                if sid in self._records:
                    continue
                self._records[sid] = ScenarioRecord(
                    scenario_id=sid,
                    scenario_type=f"discrepancy_{d['type'].lower()}",
                    odd_tags={"weather": d["weather"], "lighting": d["time_of_day"],
                              "actor_class": d["class_name"]},
                    source="discrepancy",
                    severity=d["severity"],
                    description=f"{d['type']} on {d['class_name']} "
                                f"({d['weather']}/{d['time_of_day']}) — online vs "
                                f"offline auto-label diff",
                    provenance={"dataset_id": d["dataset_id"], "frame_id": d["frame_id"],
                                "gt_id": d["gt_id"], "discrepancy_id": d["discrepancy_id"]},
                    evidence_refs=[f"frame:{d['frame_id']}", f"gt:{d['gt_id']}"],
                )
                added += 1
            if added:
                self._save()
        return added

    def add_gap_fill(self, run_id: str, cell: Dict[str, str], dataset_id: str,
                     n_objects: int) -> ScenarioRecord:
        sid = _det_id("gap_fill", run_id, dataset_id)
        rec = ScenarioRecord(
            scenario_id=sid,
            scenario_type="odd_gap_fill",
            odd_tags={k: str(v) for k, v in cell.items()},
            source="synthetic",
            severity="medium",
            description=f"Synthetic gap-filling scenario for ODD cell "
                        f"{'|'.join(f'{k}={v}' for k, v in cell.items())} "
                        f"({n_objects} objects)",
            provenance={"run_id": run_id, "dataset_id": dataset_id},
            evidence_refs=[f"dataset:{dataset_id}", f"odd_supplement:{run_id}"],
        )
        return self.add(rec)

    # ---- queries

    def search(self, scenario_type: Optional[str] = None, source: Optional[str] = None,
               severity: Optional[str] = None, odd_tags: Optional[Dict[str, str]] = None,
               text: Optional[str] = None, limit: int = 100) -> List[ScenarioRecord]:
        with self._lock:
            records = list(self._records.values())
        out = []
        for r in records:
            if scenario_type and scenario_type.lower() not in r.scenario_type.lower():
                continue
            if source and r.source != source:
                continue
            if severity and r.severity != severity:
                continue
            if odd_tags and any(r.odd_tags.get(k) != v for k, v in odd_tags.items()):
                continue
            if text and text.lower() not in json.dumps(r.model_dump()).lower():
                continue
            out.append(r)
        sev_rank = {s: i for i, s in enumerate(SEVERITIES)}
        out.sort(key=lambda r: (-sev_rank.get(r.severity, 0), r.created_at))
        return out[:max(1, min(limit, 1000))]

    def counts(self) -> Dict:
        with self._lock:
            records = list(self._records.values())
        by_source: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for r in records:
            by_source[r.source] = by_source.get(r.source, 0) + 1
            by_severity[r.severity] = by_severity.get(r.severity, 0) + 1
            by_type[r.scenario_type] = by_type.get(r.scenario_type, 0) + 1
        return {"total": len(records), "by_source": by_source,
                "by_severity": by_severity, "by_type": by_type}

    def export_bundle(self, **search_kwargs) -> Dict:
        records = self.search(limit=search_kwargs.pop("limit", 1000), **search_kwargs)
        return {
            "bundle_format": "sensorflow-scenario-bundle/v1",
            "exported_at": _now(),
            "filters": {k: v for k, v in search_kwargs.items() if v},
            "count": len(records),
            "scenarios": [r.model_dump() for r in records],
            "note": "local Safety Pool-inspired scenario catalog; scenario records "
                    "reference platform evidence (frames, rare events, datasets)",
        }


_DB: Optional[ScenarioDB] = None
_DB_LOCK = threading.Lock()


def get_db() -> ScenarioDB:
    global _DB
    with _DB_LOCK:
        if _DB is None:
            _DB = ScenarioDB()
        return _DB


def reset_db() -> None:
    """Test hook (also needed after set_safety_root)."""
    global _DB
    with _DB_LOCK:
        _DB = None
