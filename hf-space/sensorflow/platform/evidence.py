"""Evidence package export — JSON stub with real fields when available."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensorflow.platform.entities import EvidencePackage, Provenance, _id

EVIDENCE_DIR = Path("runs/platform/evidence")


def build_evidence_package(
    *,
    evaluation_id: Optional[str] = None,
    run_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    sequence_id: Optional[str] = None,
    candidate_run_id: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    include_gates: bool = True,
) -> EvidencePackage:
    placeholders: List[str] = []
    metrics: Dict[str, Any] = {}
    model_versions: List[str] = []
    population_id = None
    label_version = None
    dataset_version = None
    compute_usage: List[Dict[str, Any]] = []
    drill: Dict[str, List[str]] = {}
    thresholds: Dict[str, Any] = {}
    gate_results: List[Dict[str, Any]] = []

    eval_id = evaluation_id or run_id or _id("eval")

    # MegaEval run headline
    if run_id or candidate_run_id:
        rid = run_id or candidate_run_id
        try:
            from sensorflow.megaeval.runs import get_mega_store
            store = get_mega_store()
            run = store.runs.get(rid)  # type: ignore[arg-type]
            if run:
                metrics["headline"] = dict(run.headline or {})
                model_versions.append(run.model_version)
                population_id = run.population_id
                label_version = getattr(run, "label_version", None)
                drill["run_ids"] = [run.run_id]
            else:
                placeholders.append(f"mega_run_missing:{rid}")
        except Exception as exc:
            placeholders.append(f"mega_run_error:{exc}")
    else:
        placeholders.append("run_id")

    if baseline_run_id:
        try:
            from sensorflow.megaeval.runs import get_mega_store
            br = get_mega_store().runs.get(baseline_run_id)
            if br:
                model_versions.append(br.model_version)
                drill.setdefault("run_ids", []).append(br.run_id)
        except Exception:
            placeholders.append("baseline_run")

    # LabelEval dataset + process units
    if dataset_id:
        try:
            from sensorflow.evaluation.records import get_store
            from sensorflow.evaluation.process_units import usage_summary
            store = get_store()
            ds = store.get("datasets", dataset_id)
            if ds:
                dataset_version = ds.version
                drill["dataset_ids"] = [dataset_id]
            else:
                placeholders.append(f"dataset_missing:{dataset_id}")
            summary = usage_summary(store)
            compute_usage = [summary] if isinstance(summary, dict) else list(summary or [])
        except Exception as exc:
            placeholders.append(f"labeleval_error:{exc}")
            compute_usage = []
    else:
        placeholders.append("dataset_id")

    if include_gates:
        try:
            from sensorflow.platform.gates import evaluate_multi_gates, load_gate_config
            cfg = load_gate_config()
            thresholds = {k: v.get("thresholds", {}) for k, v in cfg.get("gates", {}).items()}
            gate_payload = evaluate_multi_gates(
                sequence_id=sequence_id,
                candidate_run_id=candidate_run_id or run_id,
                baseline_run_id=baseline_run_id,
            )
            gate_results = gate_payload.get("gates") or []
        except Exception as exc:
            placeholders.append(f"gates_error:{exc}")

    if not metrics:
        placeholders.append("metrics")
    if not model_versions:
        placeholders.append("model_versions")
        model_versions = ["placeholder-model"]
    if not compute_usage:
        compute_usage = [{"stage": "platform_evidence", "units": 0, "note": "placeholder"}]
        placeholders.append("compute_usage_detail")

    return EvidencePackage(
        package_id=_id("evidence"),
        evaluation_id=eval_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        population_id=population_id,
        model_versions=model_versions,
        label_version=label_version,
        metrics=metrics,
        thresholds=thresholds,
        gate_results=gate_results,
        compute_usage=compute_usage,
        drill_down_ids=drill,
        placeholders=sorted(set(placeholders)),
        provenance=Provenance(
            source_system="platform",
            notes="Phase 1 evidence stub — fill placeholders in later phases",
            parent_ids=[x for x in [run_id, dataset_id, sequence_id] if x],
        ),
    )


def export_evidence_package(package: EvidencePackage, path: Optional[Path] = None) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (EVIDENCE_DIR / f"{package.package_id}.json")
    # Append-only: never overwrite an existing package id silently
    if path.exists():
        path = EVIDENCE_DIR / f"{package.package_id}-{_id('rev')}.json"
    with open(path, "w") as f:
        json.dump(package.model_dump(), f, indent=2)
    return path
