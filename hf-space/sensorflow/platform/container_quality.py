"""Container quality profiles — wraps MegaEval container table + metric engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIMENSIONS
from sensorflow.megaeval.runs import CONTAINER_STATUS, EvaluationRun, MegaStore, get_mega_store
from sensorflow.platform.metrics_engine import container_quality_metrics, aggregate_container_rows


def _enrich_row(r: Dict[str, Any]) -> Dict[str, Any]:
    tp, fp, fn = int(r["tp"]), int(r["fp"]), int(r["fn"])
    sum_iou = float(r.get("sum_iou") or 0.0)
    n = int(r.get("n_objects") or 0)
    verified = int(r.get("verified") or 0)
    reviewed = int(r.get("reviewed") or 0)
    # MegaEval does not yet track disputed/auto_accept/hitl separately;
    # approximate: reviewed-verified ≈ HITL queue residue; rest unverified.
    hitl = max(0, reviewed - verified)
    auto_accepted = 0  # TODO(Phase 2): wire triage auto-accept
    disputed = 0  # TODO(Phase 2): grader disagreement → disputed
    metrics = container_quality_metrics(
        tp=tp, fp=fp, fn=fn, sum_iou=sum_iou, n_objects=n,
        verified=verified, reviewed=reviewed, disputed=disputed,
        auto_accepted=auto_accepted, hitl=hitl,
        anomalies=int(r.get("anomalies") or 0),
    )
    dims = {dim: DIMENSIONS[dim][int(r[dim])] for dim in pop_mod.CONTAINER_DIMS if dim in r}
    return {
        "container_id": int(r["container_id"]),
        "status": CONTAINER_STATUS[int(r["status"])] if "status" in r else None,
        "risk_score": round(float(r.get("risk_score") or 0), 4),
        "dims": dims,
        "drill_down": {
            "object_endpoint": f"/api/megaeval/runs/{{run_id}}/containers/{int(r['container_id'])}/objects",
            "annotation_ids": [],  # forensic load on demand
        },
        **metrics,
    }


def build_container_quality_profile(
    run_id: str,
    *,
    container_id: Optional[int] = None,
    sort: str = "highest_risk",
    limit: int = 50,
    offset: int = 0,
    store: Optional[MegaStore] = None,
) -> Dict[str, Any]:
    store = store or get_mega_store()
    run = store.runs.get(run_id)
    if run is None:
        raise KeyError(f"Unknown evaluation run {run_id}")
    if run.status != "published":
        raise RuntimeError(f"Run {run_id} is {run.status}; publish before quality profile")

    art = store.artifacts(run_id)
    df = art["containers"].copy()
    tp, fp, fn = df["tp"], df["fp"], df["fn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["recall"] = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        df["precision"] = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        df["mean_iou"] = np.where(tp > 0, df["sum_iou"] / tp, np.nan)

    if container_id is not None:
        df = df[df["container_id"] == int(container_id)]
        if df.empty:
            raise KeyError(f"Unknown container {container_id} in run {run_id}")

    sort_map = {
        "highest_risk": ("risk_score", False),
        "worst_recall": ("recall", True),
        "worst_precision": ("precision", True),
        "worst_iou": ("mean_iou", True),
        "most_anomalies": ("anomalies", False),
        "least_verified": ("verified", True),
    }
    metric, ascending = sort_map.get(sort, sort_map["highest_risk"])
    df = df.sort_values(metric, ascending=ascending, na_position="last")
    total = len(df)
    page = df.iloc[offset: offset + max(1, min(limit, 500))]

    rows: List[Dict[str, Any]] = []
    for _, series in page.iterrows():
        rows.append(_enrich_row(series.to_dict()))

    summary = aggregate_container_rows(rows) if rows else container_quality_metrics(
        tp=0, fp=0, fn=0
    )

    return {
        "run_id": run_id,
        "model_version": run.model_version,
        "population_id": run.population_id,
        "container_id": container_id,
        "total": total,
        "sort": sort,
        "offset": offset,
        "limit": limit,
        "summary": summary,
        "containers": rows,
        "verification_headline": summary.get("verification"),
    }


def profile_from_counts(
    *,
    container_id: str,
    tp: int,
    fp: int,
    fn: int,
    sum_iou: float = 0.0,
    n_objects: int = 0,
    verified: int = 0,
    reviewed: int = 0,
    disputed: int = 0,
    auto_accepted: int = 0,
    hitl: int = 0,
    anomalies: int = 0,
) -> Dict[str, Any]:
    """Unit-testable path without MegaStore."""
    metrics = container_quality_metrics(
        tp=tp, fp=fp, fn=fn, sum_iou=sum_iou, n_objects=n_objects or (tp + fp),
        verified=verified, reviewed=reviewed, disputed=disputed,
        auto_accepted=auto_accepted, hitl=hitl, anomalies=anomalies,
    )
    return {"container_id": container_id, **metrics}
