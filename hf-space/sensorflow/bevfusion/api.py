"""FastAPI router for the BEV-Fusion perception engine (included by app_backend).

Endpoints:
    POST /api/bevfusion/run     generate scenes, run both engines, evaluate, persist
    GET  /api/bevfusion/report  latest comparison report (per-cohort deltas + explanations)
    GET  /api/bevfusion/status  run inventory + engine identifiers
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.bevfusion import evaluate as evaluate_mod
from sensorflow.bevfusion.engines import BASELINE_ENGINE, FUSED_ENGINE

router = APIRouter(prefix="/api/bevfusion")


class RunRequest(BaseModel):
    n_sequences: int = Field(6, ge=1, le=24)
    frames_per_sequence: int = Field(24, ge=8, le=60)
    seed: int = 7


@router.post("/run")
def run(req: RunRequest):
    report = evaluate_mod.run_comparison(
        n_sequences=req.n_sequences,
        frames_per_sequence=req.frames_per_sequence,
        seed=req.seed,
    )
    return report


@router.get("/report")
def report():
    rep = evaluate_mod.latest_report()
    if rep is None:
        raise HTTPException(404, "No BEV-fusion comparison has been run yet; "
                                 "POST /api/bevfusion/run first")
    return rep


@router.get("/status")
def status():
    runs_dir = evaluate_mod.RUNS_DIR
    runs = []
    if runs_dir.exists():
        for p in sorted(runs_dir.glob("bevrun-*.json")):
            try:
                with open(p) as f:
                    r = json.load(f)
                runs.append({"run_id": r["run_id"], "created_at": r["created_at"],
                             "params": r["params"],
                             "recommendation": r.get("recommendation")})
            except Exception:
                continue
    latest: Optional[dict] = runs[-1] if runs else None
    return {
        "engines": {"baseline": BASELINE_ENGINE, "candidate": FUSED_ENGINE},
        "n_runs": len(runs),
        "runs": runs,
        "latest": latest,
        "ready": latest is not None,
    }
