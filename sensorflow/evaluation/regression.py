"""Performance regression tracking across model/dataset versions and runs.

Flags (spec §18):
- performance regression: precision/recall/safety-recall below baseline-tolerance
- tracking regression:    IDF1 down, ID switches up, fragmentation up
- annotation regression:  3D IoU down, position/orientation error up
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sensorflow.evaluation.records import (
    EvalStore,
    RegressionMetricDelta,
    RegressionResult,
    new_id,
)

# metric -> (tolerance, direction). "up" means higher is better.
# "proportion" marks metrics that are rates in [0, 1] backed by trial counts,
# eligible for the confidence-interval decision path in compare_runs.
METRIC_SPECS: Dict[str, Dict] = {
    "precision": {"tolerance": 0.03, "better": "up", "kind": "performance", "proportion": True},
    "recall": {"tolerance": 0.03, "better": "up", "kind": "performance", "proportion": True},
    "safety_critical_recall": {"tolerance": 0.02, "better": "up", "kind": "performance", "proportion": True},
    "idf1": {"tolerance": 0.05, "better": "up", "kind": "tracking", "proportion": True},
    "id_swap_rate": {"tolerance": 0.02, "better": "down", "kind": "tracking", "proportion": True},
    "fragmentation_rate": {"tolerance": 0.03, "better": "down", "kind": "tracking", "proportion": True},
    "mean_iou_3d": {"tolerance": 0.03, "better": "up", "kind": "annotation", "proportion": False},
    "mean_position_error": {"tolerance": 0.10, "better": "down", "kind": "annotation", "proportion": False},
    "mean_orientation_error_deg": {"tolerance": 3.0, "better": "down", "kind": "annotation", "proportion": False},
}

_Z_95 = 1.959963984540054  # two-sided 95%


def wilson_interval(p: float, n: int, z: float = _Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (never degenerate at 0/1)."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_delta_ci(p_base: float, n_base: int,
                      p_cur: float, n_cur: int,
                      z: float = _Z_95) -> Tuple[float, float]:
    """Newcombe hybrid-score CI for the difference (p_cur - p_base).

    Combines the two Wilson intervals; well-behaved for small n and
    boundary proportions, unlike the Wald interval.
    """
    lb, ub = wilson_interval(p_base, n_base, z)
    lc, uc = wilson_interval(p_cur, n_cur, z)
    return (lc - ub, uc - lb)


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
    sample_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> RegressionResult:
    """Compare current run metrics vs a baseline run; persist RegressionResult.

    sample_sizes (optional, additive): metric -> (n_baseline, n_current) trial
    counts. When provided for a proportion metric, the regression decision uses
    a Newcombe 95% CI on the delta with the tolerance acting as the
    practical-significance margin: a regression is flagged only when the
    ENTIRE interval lies beyond the tolerance in the bad direction. This
    prevents noise-flagging on small runs and adds sensitivity on large runs.
    Without sample_sizes the legacy point-delta rule is preserved.

    For repeated/sequential monitoring across many runs, delegate to
    sensorflow.seqeval (anytime-valid e-processes, e-BH multiplicity) instead
    of calling this per run: fixed-level CIs are not valid under continuous
    monitoring.
    """
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
            ns = (sample_sizes or {}).get(metric)
            use_ci = (spec.get("proportion") and ns is not None
                      and ns[0] > 0 and ns[1] > 0
                      and 0.0 <= base <= 1.0 and 0.0 <= cur <= 1.0)
            if use_ci:
                lo, hi = newcombe_delta_ci(base, ns[0], cur, ns[1])
                if spec["better"] == "up":
                    regressed = hi < -spec["tolerance"]
                else:
                    regressed = lo > spec["tolerance"]
            elif spec["better"] == "up":
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
