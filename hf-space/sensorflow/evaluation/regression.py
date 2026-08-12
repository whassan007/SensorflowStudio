"""Performance regression tracking across model/dataset versions and runs.

Flags (spec §18):
- performance regression: precision/recall/safety-recall below baseline-tolerance
- tracking regression:    IDF1 down, ID switches up, fragmentation up
- annotation regression:  3D IoU down, position/orientation error up
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.evaluation.records import (
    EvalStore,
    RegressionMetricDelta,
    RegressionResult,
    new_id,
)

# metric -> (tolerance, direction). "up" means higher is better.
METRIC_SPECS: Dict[str, Dict] = {
    "precision": {"tolerance": 0.03, "better": "up", "kind": "performance"},
    "recall": {"tolerance": 0.03, "better": "up", "kind": "performance"},
    "safety_critical_recall": {"tolerance": 0.02, "better": "up", "kind": "performance"},
    "idf1": {"tolerance": 0.05, "better": "up", "kind": "tracking"},
    "id_swap_rate": {"tolerance": 0.02, "better": "down", "kind": "tracking"},
    "fragmentation_rate": {"tolerance": 0.03, "better": "down", "kind": "tracking"},
    "mean_iou_3d": {"tolerance": 0.03, "better": "up", "kind": "annotation"},
    "mean_position_error": {"tolerance": 0.10, "better": "down", "kind": "annotation"},
    "mean_orientation_error_deg": {"tolerance": 3.0, "better": "down", "kind": "annotation"},
}


def compare_runs(
    store: EvalStore,
    current_metrics: Dict[str, Optional[float]],
    model_version: str,
    dataset_version: str,
    run_id: str,
    baseline_metrics: Optional[Dict[str, Optional[float]]] = None,
    baseline_version: Optional[str] = None,
    affected_classes: Optional[List[str]] = None,
    affected_scenarios: Optional[List[str]] = None,
) -> RegressionResult:
    """Compare current run metrics vs a baseline run; persist RegressionResult."""
    deltas: List[RegressionMetricDelta] = []
    kinds: List[str] = []
    regressed_any = False

    if baseline_metrics:
        for metric, spec in METRIC_SPECS.items():
            base = baseline_metrics.get(metric)
            cur = current_metrics.get(metric)
            if base is None or cur is None:
                continue
            delta = cur - base
            if spec["better"] == "up":
                regressed = delta < -spec["tolerance"]
            else:
                regressed = delta > spec["tolerance"]
            deltas.append(RegressionMetricDelta(
                metric=metric,
                baseline=round(float(base), 4),
                current=round(float(cur), 4),
                delta=round(float(delta), 4),
                tolerance=spec["tolerance"],
                regressed=regressed,
            ))
            if regressed:
                regressed_any = True
                if spec["kind"] not in kinds:
                    kinds.append(spec["kind"])

    result = RegressionResult(
        result_id=new_id("reg"),
        model_version=model_version,
        baseline_version=baseline_version,
        dataset_version=dataset_version,
        run_id=run_id,
        regression_detected=regressed_any,
        affected_classes=affected_classes or [],
        affected_scenarios=affected_scenarios or [],
        deltas=deltas,
        kinds=kinds,
    )
    store.put("regressions", result)
    if regressed_any:
        store.audit("regression_detected", "RegressionResult", result.result_id,
                    f"{model_version} vs {baseline_version}: kinds={kinds} classes={result.affected_classes}")
    return result


def find_affected_classes(
    current_per_class: Dict[str, Dict[str, Optional[float]]],
    baseline_per_class: Dict[str, Dict[str, Optional[float]]],
) -> List[str]:
    """Classes whose precision or recall regressed beyond tolerance."""
    affected = []
    for cls, cur in current_per_class.items():
        base = baseline_per_class.get(cls)
        if not base:
            continue
        for metric in ("precision", "recall"):
            b, c = base.get(metric), cur.get(metric)
            if b is not None and c is not None and (c - b) < -METRIC_SPECS[metric]["tolerance"]:
                affected.append(cls)
                break
    return affected
