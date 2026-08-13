"""FastAPI router for the sequential regression-detection engine.

Endpoints (prefix /api/seqeval):
    POST /runs                start a sequential regression evaluation
    GET  /runs                list runs
    GET  /runs/{id}           state: stage, samples vs budget, per-node
                              confidence sequences, decisions, and the
                              sequential-evidence trajectories (points +
                              stopping boundaries) for the dashboard chart
    GET  /runs/{id}/evidence  machine-readable evidence ledger + lineage
    GET  /runs/{id}/attribution  the regression map
    GET  /policy              default statistical policy (documented)
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.seqeval import hierarchy as hierarchy_mod
from sensorflow.seqeval import ledger as ledger_mod
from sensorflow.seqeval.controller import (DEFAULT_POLICY, get_seqeval_store,
                                           merged_policy)

router = APIRouter(prefix="/api/seqeval")


class ModelSpec(BaseModel):
    model_version: str
    effects: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-stratum detection deltas, e.g. {'pedestrian|night': -0.02};"
                    " '__global__' applies everywhere (synthetic model harness).")


class StartRunRequest(BaseModel):
    population_id: str
    baseline: ModelSpec
    candidate: ModelSpec
    policy: Dict = Field(default_factory=dict)
    seed: Optional[int] = None
    sync: bool = Field(True, description="Run to completion before returning "
                                         "(seconds); false = background thread.")


@router.post("/runs")
def start_run(req: StartRunRequest):
    store = get_seqeval_store()
    try:
        run = store.create_run(req.population_id, req.baseline.model_dump(),
                               req.candidate.model_dump(),
                               policy=req.policy or None, seed=req.seed)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if req.sync:
        store.execute_sync(run)
    else:
        store.start_async(run)
    return run.to_dict(include_trajectories=False)


@router.get("/runs")
def list_runs():
    return {"runs": get_seqeval_store().list_states()}


def _state_or_404(run_id: str, include_trajectories: bool = True) -> Dict:
    state = get_seqeval_store().get_state(run_id, include_trajectories=include_trajectories)
    if state is None:
        raise HTTPException(404, f"Unknown seqeval run {run_id}")
    return state


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    return _state_or_404(run_id)


@router.get("/runs/{run_id}/evidence")
def get_evidence(run_id: str):
    _state_or_404(run_id, include_trajectories=False)
    ledger = ledger_mod.EvidenceLedger(run_id)
    return {"run_id": run_id, "records": ledger.records(),
            "lineage": ledger.lineage(),
            "required_fields": ledger_mod.REQUIRED_FIELDS}


@router.get("/runs/{run_id}/attribution")
def get_attribution(run_id: str):
    state = _state_or_404(run_id, include_trajectories=False)
    store = get_seqeval_store()
    run = store.runs.get(run_id)
    if run is not None and run.attribution is not None:
        att = run.attribution
    else:
        att = state.get("attribution")
    if att is None:
        raise HTTPException(409, f"Run {run_id} has no attribution yet "
                                 f"(status={state.get('status')})")
    return {"run_id": run_id, "decision": state.get("decision"),
            "gate": state.get("gate"), **att}


@router.get("/policy")
def get_policy():
    return {
        "default_policy": merged_policy(),
        "decision_semantics": {
            "REGRESSION": "block: anytime-valid evidence of a drop beyond the "
                          "practical margin in at least one tested node",
            "PASS": "allow: overall + all pre-registered safety primaries proven "
                    "within the margin (equivalence-style claim)",
            "INSUFFICIENT_EVIDENCE": "expand budget or report: NOT proven "
                                     "equivalent; never treated as a pass",
        },
        "test_method": ledger_mod.TEST_METHOD,
        "multiple_testing_method": hierarchy_mod.MULTIPLE_TESTING_METHOD,
        "notes": [
            "sampling plans are frozen (hashed) before candidate outcomes are seen",
            "inference units are container clusters, not frames/objects",
            "baseline predictions are cached by dataset+model fingerprints",
        ],
    }
