"""Evaluation-run lineage: reproducibility records and launch validity.

Policy (deterministic): a run missing ANY lineage field is marked INVALID for
launch purposes. It may still be inspected, but LaunchRecommendation for such
a run is forced to INVALID by regression.py / scheduler.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sensorflow.nextgen.models import EvaluationRun, LineageRecord

# Current component versions of this package. Bumping any of these changes
# cache keys (cache.py) and appears in every lineage record.
COMPONENT_VERSIONS: Dict[str, str] = {
    "scenario_set_version": "bevfusion-scenes-1.0",
    "sensor_sim_version": "bevfusion-sensors-1.0",
    "simulation_version": "nextgen-closedloop-1.0",
    "feature_pipeline_version": "nextgen-features-1.0",
    "metric_version": "nextgen-metrics-1.0",
    "policy_version": "nextgen-launch-policy-1.0",
}

REQUIRED_FIELDS = [
    "model_version", "dataset_version", "scenario_set_version",
    "sensor_sim_version", "simulation_version", "feature_pipeline_version",
    "metric_version", "policy_version",
]


def build_lineage(model_version: str, dataset_version: str,
                  seeds: Dict[str, int],
                  baseline_version: str | None = None,
                  overrides: Dict[str, str] | None = None) -> LineageRecord:
    fields = {**COMPONENT_VERSIONS, **(overrides or {})}
    return LineageRecord(model_version=model_version,
                         baseline_version=baseline_version,
                         dataset_version=dataset_version,
                         seeds=dict(seeds), **fields)


def validate_lineage(lineage: LineageRecord | None) -> Tuple[bool, List[str]]:
    """Deterministic completeness check. Missing seeds are also a violation:
    without them the run is not reproducible."""
    if lineage is None:
        return False, ["lineage record entirely missing"]
    missing = [f for f in REQUIRED_FIELDS if not getattr(lineage, f)]
    if not lineage.seeds:
        missing.append("seeds")
    return (len(missing) == 0,
            [f"missing lineage field: {m}" for m in missing])


def stamp_run(run: EvaluationRun) -> EvaluationRun:
    """Attach validity verdict to a run based on its lineage."""
    ok, reasons = validate_lineage(run.lineage)
    run.lineage_valid = ok
    if not ok:
        run.valid_for_launch = False
        run.invalid_reasons = list(dict.fromkeys(run.invalid_reasons + reasons))
    return run
