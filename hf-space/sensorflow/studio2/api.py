"""REST surface for the Studio 2.0 control plane under /api/studio2."""

from __future__ import annotations

import importlib
import os
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.studio2 import demo as demo_mod
from sensorflow.studio2 import funnel as funnel_mod
from sensorflow.studio2 import hardware as hw_mod
from sensorflow.studio2 import store
from sensorflow.studio2.registry import (
    DATASET_ROLES,
    KINDS,
    RoleTransitionError,
    get_registry,
    ingest_existing_stores,
)
from sensorflow.studio2.release_gate import ReleaseGate

router = APIRouter(prefix="/api/studio2", tags=["studio2"])

_SOFT_DEPS = ["sensorflow.agentic", "sensorflow.nextgen", "sensorflow.hardening",
              "sensorflow.retro"]
_HARD_DEPS = ["sensorflow.safety", "sensorflow.seqeval", "sensorflow.megaeval",
              "sensorflow.rca", "sensorflow.raremine", "sensorflow.bevfusion",
              "sensorflow.vitis", "sensorflow.evaluation"]


@router.get("/status")
def status():
    deps = {}
    for mod in _HARD_DEPS + _SOFT_DEPS:
        try:
            importlib.import_module(mod)
            deps[mod] = "importable"
        except Exception as e:
            deps[mod] = f"unavailable ({type(e).__name__})"
    return {"registry_counts": get_registry().counts(),
            "dependencies": deps,
            "soft_dependencies": _SOFT_DEPS}


# ------------------------------------------------------------------ registry


@router.get("/registry/summary")
def registry_summary():
    return {"counts": get_registry().counts(), "kinds": list(KINDS),
            "dataset_roles": list(DATASET_ROLES)}


@router.post("/registry/ingest")
def registry_ingest():
    return ingest_existing_stores(get_registry())


@router.get("/registry/{kind}")
def registry_list(kind: str):
    try:
        return {"kind": kind, "entities": get_registry().list(kind)}
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.get("/registry/{kind}/{entity_id}")
def registry_get(kind: str, entity_id: str):
    if kind not in KINDS:
        raise HTTPException(404, f"unknown kind {kind}")
    doc = get_registry().get(kind, entity_id)
    if doc is None:
        raise HTTPException(404, f"no {kind} entity {entity_id}")
    return doc


class DatasetCreate(BaseModel):
    name: str
    role: str
    lineage_parents: list = Field(default_factory=list)
    meta: Dict = Field(default_factory=dict)
    actor: str = "api"


@router.post("/registry/datasets")
def dataset_create(req: DatasetCreate):
    try:
        return get_registry().register_dataset(
            req.name, req.role, lineage_parents=req.lineage_parents,
            meta=req.meta, actor=req.actor,
            provenance={"source_package": "studio2-api"})
    except ValueError as e:
        raise HTTPException(400, str(e))


class RoleTransition(BaseModel):
    new_role: str
    actor: str
    override_reason: Optional[str] = None


@router.post("/registry/datasets/{dataset_id}/role")
def dataset_role(dataset_id: str, req: RoleTransition):
    try:
        return get_registry().transition_role(
            dataset_id, req.new_role, req.actor,
            override_reason=req.override_reason)
    except RoleTransitionError as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/audit")
def audit_log(limit: int = 100):
    return {"events": store.read_audit(limit=limit)}


# ------------------------------------------------------------------ release


class ReleaseEvaluate(BaseModel):
    """Inputs may be omitted; the gate degrades to REVIEW naming the gap.
    When use_live_sources=true the server gathers the real latest inputs."""
    safety_metrics: Optional[Dict] = None
    regression_results: Optional[Dict] = None
    distribution_shift: Optional[Dict] = None
    agentic_outcome: Optional[Dict] = None
    closed_loop: Optional[Dict] = None
    hardware_matrix: Optional[Dict] = None
    policy: Optional[Dict] = None
    context: Optional[Dict] = None
    use_live_sources: bool = False


def _gather_live_inputs(req: ReleaseEvaluate) -> ReleaseEvaluate:
    data = req.model_dump()
    try:
        from sensorflow.megaeval.runs import get_mega_store
        from sensorflow.megaeval import analysis as mega_analysis
        from sensorflow.safety import gates as safety_gates
        mega = get_mega_store()
        published = sorted(
            [r for r in mega.runs.values() if getattr(r, "status", "") == "published"],
            key=lambda r: getattr(r, "published_at", "") or "")
        if len(published) >= 2 and data["safety_metrics"] is None:
            cand = published[-1]
            data["safety_metrics"] = (safety_gates.latest_gate_result(cand.run_id)
                                      or safety_gates.evaluate_gates(
                                          mega, cand, published[-2]))
        if published and data["distribution_shift"] is None:
            data["distribution_shift"] = mega_analysis.distribution_shift(
                mega, published[-1])
    except Exception:
        pass
    if data["regression_results"] is None:
        try:
            from sensorflow.seqeval.controller import get_seqeval_store
            states = get_seqeval_store().list_states()
            done = [s for s in states if s.get("status") == "done"]
            if done:
                s = done[0]
                data["regression_results"] = {
                    "run_id": s["run_id"], "decision": s["decision"],
                    "stopping_reason": s.get("stopping_reason"),
                    "samples_used": s.get("budget", {}).get("samples_used"),
                    "affected_strata": [], "regression_map": []}
        except Exception:
            pass
    if data["hardware_matrix"] is None:
        try:
            data["hardware_matrix"] = (hw_mod.latest_matrix()
                                       or hw_mod.gate_matrix())
        except Exception:
            pass
    return ReleaseEvaluate(**data)


@router.post("/release/evaluate")
def release_evaluate(req: ReleaseEvaluate):
    if req.use_live_sources:
        req = _gather_live_inputs(req)
    gate = ReleaseGate(get_registry(), policy=req.policy)
    return gate.evaluate(
        safety_metrics=req.safety_metrics,
        regression_results=req.regression_results,
        distribution_shift=req.distribution_shift,
        agentic_outcome=req.agentic_outcome,
        closed_loop=req.closed_loop,
        hardware_matrix=req.hardware_matrix,
        context=req.context)


@router.get("/release/decisions")
def release_decisions():
    return {"decisions": get_registry().list("decisions")}


@router.get("/release/decisions/{decision_id}")
def release_decision(decision_id: str):
    doc = get_registry().get("decisions", decision_id)
    if doc is None:
        raise HTTPException(404, f"no decision {decision_id}")
    return doc


class Approval(BaseModel):
    approver: str
    rationale: str


@router.post("/release/decisions/{decision_id}/approve")
def release_approve(decision_id: str, req: Approval):
    gate = ReleaseGate(get_registry())
    try:
        return gate.approve(decision_id, req.approver, req.rationale)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


# ------------------------------------------------------------------ hardware


@router.get("/hardware/matrix")
def hardware_matrix(refresh: bool = False):
    if not refresh:
        cached = hw_mod.latest_matrix()
        if cached:
            return cached
    try:
        return hw_mod.gate_matrix()
    except Exception as e:
        raise HTTPException(503, f"hardware matrix unavailable: {e}")


# ------------------------------------------------------------------ funnel


@router.get("/funnel")
def funnel():
    return funnel_mod.build_funnel(registry=get_registry())


# ------------------------------------------------------------------ demo


class DemoRequest(BaseModel):
    seed: int = demo_mod.DEMO_SEED


@router.post("/demo/run")
def demo_run(req: DemoRequest):
    return demo_mod.run_demo(seed=req.seed, registry=get_registry())


@router.get("/demo/latest")
def demo_latest():
    doc = demo_mod.latest_demo()
    if doc is None:
        raise HTTPException(404, "no demo run yet; POST /api/studio2/demo/run")
    return doc


# ------------------------------------------------------------------ docs


_DOCS_DIR = os.path.join("docs", "architecture")


@router.get("/docs")
def docs_list():
    if not os.path.isdir(_DOCS_DIR):
        return {"docs": []}
    return {"docs": [n for n in sorted(os.listdir(_DOCS_DIR))
                     if n.startswith("studio2-") and n.endswith(".md")]}


@router.get("/docs/{name}")
def docs_get(name: str):
    if not (name.startswith("studio2-") and name.endswith(".md")
            and "/" not in name and ".." not in name):
        raise HTTPException(404, "unknown document")
    path = os.path.join(_DOCS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"no document {name}")
    with open(path) as f:
        return {"name": name, "content": f.read()}
