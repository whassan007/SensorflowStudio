"""Platform Phase 1 FastAPI routes — mount into existing app_backend.

Routes:
  /api/evaluations/...
  /api/containers/...
  /api/models/compare
  /api/gates/...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.platform import levels
from sensorflow.platform.container_quality import build_container_quality_profile
from sensorflow.platform.compare import compare_models
from sensorflow.platform.evidence import build_evidence_package, export_evidence_package
from sensorflow.platform.gates import evaluate_multi_gates, list_gate_defs, load_gate_config, save_gate_config
from sensorflow.platform.levels import AggregateLevel, EvaluationScope

router = APIRouter(prefix="/api")


# ------------------------------------------------------------------ requests


class EvaluationCreateRequest(BaseModel):
    level: AggregateLevel = AggregateLevel.POPULATION
    dataset_id: Optional[str] = None
    population_id: Optional[str] = None
    run_id: Optional[str] = None
    container_id: Optional[str] = None
    mega_container_id: Optional[int] = None
    drive_id: Optional[str] = None
    scene_id: Optional[str] = None
    sequence_id: Optional[str] = None
    frame_id: Optional[str] = None
    cohort_filters: Dict[str, List[str]] = Field(default_factory=dict)
    model_version: Optional[str] = None
    label_version: Optional[str] = None


class ModelCompareRequest(BaseModel):
    run_ids: List[str] = Field(..., min_length=2)
    baseline_run_id: Optional[str] = None
    policy: Dict[str, Any] = Field(default_factory=dict)


class GateEvaluateRequest(BaseModel):
    sequence_id: Optional[str] = None
    candidate_run_id: Optional[str] = None
    baseline_run_id: Optional[str] = None


class GateConfigUpdate(BaseModel):
    gates: Dict[str, Any] = Field(default_factory=dict)


class EvidenceExportRequest(BaseModel):
    evaluation_id: Optional[str] = None
    run_id: Optional[str] = None
    dataset_id: Optional[str] = None
    sequence_id: Optional[str] = None
    candidate_run_id: Optional[str] = None
    baseline_run_id: Optional[str] = None
    persist: bool = True


# ------------------------------------------------------------------ evaluations


@router.get("/evaluations/levels")
def evaluation_levels():
    return {
        "levels": [lvl.value for lvl in AggregateLevel.ladder()],
        "backends": {lvl.value: levels.LEVEL_BACKEND[lvl] for lvl in AggregateLevel},
    }


@router.post("/evaluations/scope")
def create_evaluation_scope(req: EvaluationCreateRequest):
    scope = EvaluationScope(**req.model_dump())
    return {
        "scope": levels.summarize_scope(scope),
        "message": "Phase 1 scope descriptor — queries still served by MegaEval/LabelEval backends",
    }


@router.get("/evaluations/{run_id}")
def get_evaluation_summary(run_id: str):
    """Unified evaluation summary for a MegaEval run (async job status preserved)."""
    try:
        from sensorflow.megaeval.runs import get_mega_store
        run = get_mega_store().runs.get(run_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    if run is None:
        raise HTTPException(404, f"Unknown evaluation run {run_id}")
    scope = EvaluationScope(
        level=AggregateLevel.POPULATION,
        evaluation_id=run.run_id,
        run_id=run.run_id,
        population_id=run.population_id,
        model_version=run.model_version,
        label_version=getattr(run, "label_version", None),
    )
    return {
        "evaluation_id": run.run_id,
        "status": run.status,
        "scope": levels.summarize_scope(scope),
        "headline": run.headline,
        "model_version": run.model_version,
        "population_id": run.population_id,
        "lineage": getattr(run, "lineage", None),
        "progress": getattr(run, "progress", None),
    }


@router.post("/evaluations/evidence")
def export_evidence(req: EvidenceExportRequest):
    package = build_evidence_package(
        evaluation_id=req.evaluation_id,
        run_id=req.run_id,
        dataset_id=req.dataset_id,
        sequence_id=req.sequence_id,
        candidate_run_id=req.candidate_run_id,
        baseline_run_id=req.baseline_run_id,
    )
    path = None
    if req.persist:
        path = str(export_evidence_package(package))
    return {"package": package.model_dump(), "path": path}


# ------------------------------------------------------------------ containers


@router.get("/containers/quality")
def containers_quality(
    run_id: str,
    sort: str = "highest_risk",
    limit: int = 50,
    offset: int = 0,
):
    try:
        return build_container_quality_profile(
            run_id, sort=sort, limit=limit, offset=offset
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/containers/{container_id}/quality")
def container_quality(container_id: int, run_id: str):
    try:
        return build_container_quality_profile(run_id, container_id=container_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


# ------------------------------------------------------------------ models compare


@router.post("/models/compare")
def models_compare(req: ModelCompareRequest):
    try:
        return compare_models(
            req.run_ids,
            baseline_run_id=req.baseline_run_id,
            policy=req.policy or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


# ------------------------------------------------------------------ gates


@router.get("/gates")
def gates_list():
    cfg = load_gate_config()
    return {
        "config_path": str(cfg.get("schema_version")),
        "gates": [g.model_dump() for g in list_gate_defs(cfg)],
        "order": ["scenario", "coverage", "regression", "safety", "quality", "launch", "release"],
    }


@router.get("/gates/config")
def gates_config():
    return load_gate_config()


@router.post("/gates/config")
def gates_config_update(req: GateConfigUpdate):
    cfg = load_gate_config()
    for name, patch in req.gates.items():
        if name not in cfg["gates"]:
            cfg["gates"][name] = patch
        else:
            cfg["gates"][name].update(patch)
    save_gate_config(cfg)
    return cfg


@router.post("/gates/evaluate")
def gates_evaluate(req: GateEvaluateRequest):
    return evaluate_multi_gates(
        sequence_id=req.sequence_id,
        candidate_run_id=req.candidate_run_id,
        baseline_run_id=req.baseline_run_id,
    )


@router.get("/gates/status")
def gates_status(
    sequence_id: Optional[str] = None,
    candidate_run_id: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
):
    """Multi-gate readiness skeleton for UI cards."""
    return evaluate_multi_gates(
        sequence_id=sequence_id,
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
    )
