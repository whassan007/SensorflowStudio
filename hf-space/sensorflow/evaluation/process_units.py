"""Process-unit accounting: deterministic cost metric (not wall-clock).

Extends the philosophy of sensorflow/metrics/resource_profile.py to the full
evaluation platform: every stage records units; unit economics are derived
from actual state (cost per verified event, per million frames, per dataset).
"""

from __future__ import annotations

from typing import Dict, Optional

from sensorflow.evaluation.records import EvalStore, ProcessUsage, new_id

# Deterministic per-item complexity factors (units per processed item).
STAGE_FACTORS: Dict[str, float] = {
    "ingestion": 0.2,
    "label_generation": 1.0,
    "anomaly_detection": 0.6,
    "rare_event_detection": 0.4,
    "rag_retrieval": 0.3,
    "vlm": 2.5,
    "validation": 0.5,
    "grading": 0.8,
    "regression_tracking": 0.3,
    "hitl": 5.0,
    "training": 12.0,
}


class ProcessMeter:
    """Records process-unit usage into the store, grouped by stage and run."""

    def __init__(self, store: EvalStore, run_id: str = ""):
        self.store = store
        self.run_id = run_id

    def record(self, stage: str, items: int, factor: Optional[float] = None) -> int:
        f = STAGE_FACTORS.get(stage, 1.0) if factor is None else factor
        units = max(1, int(round(items * f))) if items > 0 else 0
        if units > 0:
            self.store.put(
                "process_usage",
                ProcessUsage(usage_id=new_id("pu"), stage=stage, units=units, run_id=self.run_id),
            )
        return units


def usage_summary(store: EvalStore) -> Dict[str, object]:
    by_stage: Dict[str, int] = {}
    total = 0
    for u in store.all("process_usage"):
        by_stage[u.stage] = by_stage.get(u.stage, 0) + u.units
        total += u.units

    verified = len([a for a in store.all("annotations") if a.status == "VERIFIED"])
    rare_verified = len([e for e in store.all("rare_events") if e.verified])
    verified_events = verified + rare_verified
    frames = len(store.all("frames"))
    training_datasets = len(store.all("training_datasets"))

    return {
        "total": total,
        "by_stage": by_stage,
        "unit_economics": {
            "per_verified_event": round(total / verified_events, 2) if verified_events else None,
            "per_million_frames": round(total / frames * 1_000_000, 0) if frames else None,
            "per_training_dataset": round(total / training_datasets, 2) if training_datasets else None,
        },
    }
