"""FastAPI router for the RCA investigation workbench (prefix /api/rca)."""

from __future__ import annotations

import random
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.rca import diagnostics as dg, report as report_mod, scoring, store
from sensorflow.rca.models import (FINDING_STATUS, HYPOTHESIS_LABELS,
                                   Investigation, ROOT_CAUSES, SEVERITIES,
                                   STAGES, StageOrderError,
                                   UnknownsNotAcknowledgedError, make_finding)
from sensorflow.rca.scenario import CAUSE_EXPLANATIONS, generate_scenario

router = APIRouter(prefix="/api/rca")

# The full battery is deterministic per investigation (data is immutable), so
# it is computed once and cached.
_battery_cache: Dict[str, Dict] = {}


def _inv_or_404(inv_id: str) -> Investigation:
    inv = store.load_investigation(inv_id)
    if inv is None:
        raise HTTPException(404, f"Unknown investigation {inv_id}")
    return inv


def _bundle_or_404(inv_id: str):
    bundle = store.load_bundle(inv_id)
    if bundle is None:
        raise HTTPException(404, f"No scenario data for investigation {inv_id}")
    return bundle


def _battery(inv_id: str) -> Dict:
    if inv_id not in _battery_cache:
        _battery_cache[inv_id] = dg.run_all(_bundle_or_404(inv_id))
    return _battery_cache[inv_id]


# ------------------------------------------------------------------- requests


class CreateInvestigationRequest(BaseModel):
    name: str = ""
    baseline_model: str = "detr3d-a-v41"
    candidate_model: str = "detr3d-b-v42"
    # None + training_mode -> random hidden cause ("load demo investigation").
    cause: Optional[str] = None
    seed: Optional[int] = None
    training_mode: bool = False


class CompleteStageRequest(BaseModel):
    acknowledge_unknowns: bool = False
    note: str = ""


class RecordFindingRequest(BaseModel):
    stage: str
    title: str
    status: str = "MISMATCH"
    severity: str = "WARN"
    detail: str = ""
    code: str = ""


class AssessHypothesisRequest(BaseModel):
    hypothesis: str
    confidence: str = Field(pattern="^(HIGH|MEDIUM|LOW|UNKNOWN)$")
    note: str = ""


# ---------------------------------------------------------------- meta routes


@router.get("/causes")
def list_causes():
    return {"causes": [{"id": c, "label": HYPOTHESIS_LABELS[c]}
                       for c in ROOT_CAUSES]}


@router.get("/stages")
def list_stages():
    return {"stages": STAGES}


# -------------------------------------------------------------- investigations


@router.post("/investigations")
def create_investigation(req: CreateInvestigationRequest):
    seed = req.seed if req.seed is not None else random.randint(1, 10_000_000)
    if req.cause is not None:
        if req.cause not in ROOT_CAUSES:
            raise HTTPException(400, f"Unknown cause {req.cause!r}")
        cause = req.cause
    else:
        cause = random.Random(seed).choice(ROOT_CAUSES)

    bundle = generate_scenario(cause, seed=seed)
    measured = bundle.meta["measured"]
    claims = {
        "metric": "object-level accuracy",
        "offline_delta_pp": round(measured["offline_delta"] * 100, 2),
        "shadow_delta_pp": round(measured["shadow_delta"] * 100, 2),
        "offline_n": measured["offline_n"],
        "shadow_scored_n": measured["shadow_scored_n"],
    }
    name = req.name or (f"{'Training drill' if req.training_mode else 'RCA'}: "
                        f"{req.candidate_model} vs {req.baseline_model}")
    inv = Investigation.new(name=name, baseline_model=req.baseline_model,
                            candidate_model=req.candidate_model,
                            scenario_cause=cause, seed=seed,
                            training_mode=req.training_mode, claims=claims)
    store.save_investigation(inv, bundle)
    return inv.to_dict()


@router.get("/investigations")
def list_investigations():
    return {"investigations": store.list_investigations()}


@router.get("/investigations/{inv_id}")
def get_investigation(inv_id: str):
    return _inv_or_404(inv_id).to_dict()


@router.post("/investigations/{inv_id}/reveal")
def reveal(inv_id: str):
    inv = _inv_or_404(inv_id)
    inv.revealed = True
    inv.log("revealed", "Planted root cause revealed")
    store.save_investigation(inv)
    return {"cause": inv.scenario_cause,
            "explanation": CAUSE_EXPLANATIONS[inv.scenario_cause]}


# --------------------------------------------------------------------- stages


@router.post("/investigations/{inv_id}/stages/{index}/complete")
def complete_stage(inv_id: str, index: int, req: CompleteStageRequest):
    inv = _inv_or_404(inv_id)
    try:
        stage = inv.complete_stage(index,
                                   acknowledge_unknowns=req.acknowledge_unknowns,
                                   ack_note=req.note)
    except StageOrderError as e:
        raise HTTPException(409, str(e))
    except UnknownsNotAcknowledgedError as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    store.save_investigation(inv)
    return {"stage": stage.to_dict(), "investigation": inv.to_dict()}


@router.post("/investigations/{inv_id}/stages/{index}/reopen")
def reopen_stage(inv_id: str, index: int):
    inv = _inv_or_404(inv_id)
    if not 0 <= index < len(inv.stages):
        raise HTTPException(404, f"No stage {index}")
    stage = inv.reopen_stage(index)
    store.save_investigation(inv)
    return {"stage": stage.to_dict(), "investigation": inv.to_dict()}


# ------------------------------------------------------------------- findings


@router.post("/investigations/{inv_id}/findings")
def record_finding(inv_id: str, req: RecordFindingRequest):
    inv = _inv_or_404(inv_id)
    stage_keys = [s["key"] for s in STAGES]
    if req.stage not in stage_keys:
        raise HTTPException(400, f"Unknown stage {req.stage!r}")
    if req.status not in FINDING_STATUS:
        raise HTTPException(400, f"status must be one of {FINDING_STATUS}")
    if req.severity not in SEVERITIES:
        raise HTTPException(400, f"severity must be one of {SEVERITIES}")
    finding = make_finding(req.stage, req.code or f"HUMAN:{req.stage}",
                           req.title, req.status, req.severity, req.detail,
                           source="human")
    inv.add_human_finding(finding)
    store.save_investigation(inv)
    return {"finding": finding.to_dict()}


# ---------------------------------------------------------------- diagnostics


@router.get("/investigations/{inv_id}/diagnostics/{stage_key}")
def stage_diagnostics(inv_id: str, stage_key: str):
    inv = _inv_or_404(inv_id)
    if stage_key not in dg.STAGE_DIAGNOSTICS:
        raise HTTPException(404, f"No diagnostic for stage {stage_key!r}")
    data, findings = _battery(inv_id)[stage_key]
    inv.upsert_auto_findings(stage_key, findings)
    store.save_investigation(inv)
    return {"stage": stage_key, "data": data,
            "findings": [f.to_dict() for f in findings]}


# ------------------------------------------------------- scoring and decision


@router.get("/investigations/{inv_id}/scoreboard")
def scoreboard(inv_id: str, recorded_only: bool = False):
    inv = _inv_or_404(inv_id)
    bundle = _bundle_or_404(inv_id)
    if recorded_only:
        return scoring.build_scoreboard(bundle, inv, recorded_only=True)
    return scoring.build_scoreboard(bundle, inv, battery=_battery(inv_id))


@router.post("/investigations/{inv_id}/scoreboard/assess")
def assess_hypothesis(inv_id: str, req: AssessHypothesisRequest):
    inv = _inv_or_404(inv_id)
    if req.hypothesis not in ROOT_CAUSES:
        raise HTTPException(400, f"Unknown hypothesis {req.hypothesis!r}")
    inv.human_assessments[req.hypothesis] = {"confidence": req.confidence,
                                             "note": req.note}
    inv.log("human_assessment",
            f"Human set {req.hypothesis} confidence to {req.confidence}",
            {"note": req.note})
    store.save_investigation(inv)
    bundle = _bundle_or_404(inv_id)
    return scoring.build_scoreboard(bundle, inv, battery=_battery(inv_id))


@router.get("/investigations/{inv_id}/decision-tree")
def decision_tree(inv_id: str):
    bundle = _bundle_or_404(inv_id)
    return scoring.evaluate_decision_tree(bundle, battery=_battery(inv_id))


@router.get("/investigations/{inv_id}/experiments")
def experiments(inv_id: str):
    bundle = _bundle_or_404(inv_id)
    return scoring.recommend_experiments(bundle, battery=_battery(inv_id))


@router.get("/investigations/{inv_id}/report")
def report(inv_id: str):
    inv = _inv_or_404(inv_id)
    bundle = _bundle_or_404(inv_id)
    return report_mod.build_report(bundle, inv, battery=_battery(inv_id))
