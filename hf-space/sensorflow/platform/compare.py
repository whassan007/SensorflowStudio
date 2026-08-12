"""Multi-model comparison (A vs B vs C) wrapping MegaEval comparative analytics.

Does not invent a second compare engine — delegates pairwise compare_runs and
folds results into a multi-way delta table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sensorflow.megaeval import analysis
from sensorflow.megaeval.runs import MegaStore, get_mega_store
from sensorflow.platform.metrics_engine import delta_metrics, round4


def _headline_map(run) -> Dict[str, Optional[float]]:
    h = run.headline or {}
    return {k: h.get(k) for k in (
        "precision", "recall", "f1", "mean_iou", "safety_recall", "anomaly_rate"
    )}


def compare_models(
    run_ids: List[str],
    *,
    baseline_run_id: Optional[str] = None,
    policy: Optional[Dict] = None,
    store: Optional[MegaStore] = None,
) -> Dict[str, Any]:
    """Compare 2+ published MegaEval runs (model versions A/B/C…).

    If baseline_run_id is omitted, the first id is the baseline.
    """
    if len(run_ids) < 2:
        raise ValueError("compare_models requires at least two run_ids")
    store = store or get_mega_store()
    baseline_id = baseline_run_id or run_ids[0]
    if baseline_id not in run_ids:
        run_ids = [baseline_id, *run_ids]

    runs = []
    for rid in run_ids:
        run = store.runs.get(rid)
        if run is None:
            raise KeyError(f"Unknown evaluation run {rid}")
        if run.status != "published":
            raise RuntimeError(f"Run {rid} is {run.status}; results require published")
        runs.append(run)

    baseline = next(r for r in runs if r.run_id == baseline_id)
    candidates = [r for r in runs if r.run_id != baseline_id]

    models = [{
        "run_id": r.run_id,
        "model_version": r.model_version,
        "role": "baseline" if r.run_id == baseline_id else "candidate",
        "headline": _headline_map(r),
    } for r in runs]

    # Multi-way metric table: one row per metric, columns per model
    metric_keys = ["precision", "recall", "f1", "mean_iou", "safety_recall", "anomaly_rate"]
    matrix = []
    for key in metric_keys:
        row: Dict[str, Any] = {"metric": key}
        for r in runs:
            row[r.model_version] = round4(r.headline.get(key)) if r.headline.get(key) is not None else None
        base_v = baseline.headline.get(key)
        deltas = {}
        for c in candidates:
            cv = c.headline.get(key)
            if base_v is not None and cv is not None:
                deltas[c.model_version] = round4(float(cv) - float(base_v))
        row["deltas_vs_baseline"] = deltas
        matrix.append(row)

    pairwise = []
    recommendations = []
    for cand in candidates:
        cmp = analysis.compare_runs(store, cand, baseline, policy)
        pairwise.append({
            "baseline_run_id": baseline.run_id,
            "candidate_run_id": cand.run_id,
            "baseline_model": baseline.model_version,
            "candidate_model": cand.model_version,
            "recommendation": cmp.get("recommendation"),
            "blockers": cmp.get("blockers") or [],
            "headline": cmp.get("headline_deltas") or delta_metrics(
                _headline_map(baseline), _headline_map(cand)
            ),
            "per_class": cmp.get("per_class") or [],
            "regressions": (cmp.get("regressions") or cmp.get("worst_cohorts") or [])[:10],
        })
        recommendations.append({
            "candidate": cand.model_version,
            "recommendation": cmp.get("recommendation"),
            "blocker_count": len(cmp.get("blockers") or []),
        })

    # Also surface LabelEval registry models when present (non-fatal)
    registry = []
    try:
        from sensorflow.evaluation.records import get_store
        for m in get_store().all("models"):
            registry.append({
                "model_id": m.model_id,
                "model_version": m.model_version,
                "metrics": m.metrics.model_dump() if hasattr(m.metrics, "model_dump") else {},
                "regression_status": m.regression_status,
            })
    except Exception:
        pass

    return {
        "baseline_run_id": baseline.run_id,
        "run_ids": run_ids,
        "models": models,
        "metric_matrix": matrix,
        "pairwise": pairwise,
        "recommendations": recommendations,
        "label_eval_registry": registry,
        "policy": {**analysis.DEFAULT_PROMOTION_POLICY, **(policy or {})},
    }
