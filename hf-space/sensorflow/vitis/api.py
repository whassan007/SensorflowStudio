"""FastAPI router for the Vitis acceleration layer (prefix /api/vitis)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.vitis.backend import KNOWN_DEVICES, backend_status
from sensorflow.vitis.store import list_runs, load_run

router = APIRouter(prefix="/api/vitis")

PRD_DIR = Path(__file__).resolve().parents[2] / "docs" / "prd"
PRD_FILES = {
    "vitis-hil-regression": "vitis-hil-regression.md",
    "vitis-isp-preprocessing": "vitis-isp-preprocessing.md",
    "vitis-temporal-stability": "vitis-temporal-stability.md",
}


# --------------------------------------------------------------------------
# Backends / capability
# --------------------------------------------------------------------------

@router.get("/backends/status")
def get_backend_status() -> Dict:
    return backend_status()


@router.get("/backends/devices")
def get_devices() -> Dict:
    return {"devices": [{"name": k, **v} for k, v in KNOWN_DEVICES.items()]}


# --------------------------------------------------------------------------
# Feature 1: HIL quantization-gap regression
# --------------------------------------------------------------------------

class HilRunRequest(BaseModel):
    n_sequences: int = Field(4, ge=1, le=12)
    frames_per_sequence: int = Field(14, ge=4, le=40)
    seed: int = 7
    width_bits: int = Field(10, ge=4, le=32)
    int_bits: int = Field(4, ge=2, le=16)
    max_line_buffer_depth: int = Field(2048, ge=16)
    use_lut_approx: bool = True
    lut_bits: int = Field(8, ge=4, le=12)
    device: str = "versal-ai-edge"
    regression_delta: float = Field(0.02, gt=0.0, lt=0.5)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    run_ablation: bool = True


class HilSweepRequest(BaseModel):
    n_sequences: int = Field(4, ge=1, le=12)
    frames_per_sequence: int = Field(14, ge=4, le=40)
    seed: int = 7
    widths: Optional[List[int]] = None
    int_bits: int = Field(4, ge=2, le=16)
    max_line_buffer_depth: int = Field(2048, ge=16)
    use_lut_approx: bool = True
    lut_bits: int = Field(8, ge=4, le=12)
    device: str = "versal-ai-edge"
    regression_delta: float = Field(0.02, gt=0.0, lt=0.5)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)


def _check_device(name: str) -> None:
    if name not in KNOWN_DEVICES:
        raise HTTPException(400, f"Unknown device {name!r}; "
                                 f"known: {sorted(KNOWN_DEVICES)}")


@router.post("/hil/run")
def hil_run(req: HilRunRequest) -> Dict:
    _check_device(req.device)
    from sensorflow.vitis.hil import run_hil
    if req.width_bits <= req.int_bits:
        raise HTTPException(400, "width_bits must exceed int_bits")
    return run_hil(**req.model_dump())


@router.post("/hil/sweep")
def hil_sweep(req: HilSweepRequest) -> Dict:
    _check_device(req.device)
    from sensorflow.vitis.hil import run_bitwidth_sweep
    if req.widths and any(w <= req.int_bits for w in req.widths):
        raise HTTPException(400, "every sweep width must exceed int_bits")
    return run_bitwidth_sweep(**req.model_dump())


@router.get("/hil/runs")
def hil_runs() -> Dict:
    return {"runs": list_runs("hil")}


@router.get("/hil/runs/{run_id}")
def hil_run_report(run_id: str) -> Dict:
    run = load_run("hil", run_id)
    if run is None:
        raise HTTPException(404, f"Unknown HIL run {run_id}")
    return run


# --------------------------------------------------------------------------
# Feature 2: ISP + synthetic edge-case generation
# --------------------------------------------------------------------------

class IspRunRequest(BaseModel):
    n_frames: int = Field(4, ge=1, le=24)
    seed: int = 11
    stages: Optional[List[str]] = None
    stage_params: Optional[Dict] = None
    width_bits: int = Field(12, ge=4, le=32)
    int_bits: int = Field(4, ge=2, le=16)
    max_line_buffer_depth: int = Field(2048, ge=16)
    use_lut_approx: bool = True
    lut_bits: int = Field(8, ge=4, le=12)
    device: str = "versal-ai-edge"
    include_previews: bool = True


class AugmentRequest(BaseModel):
    recipes: Optional[List[Dict]] = None
    n_variants: int = Field(12, ge=1, le=64)
    seed: int = 23
    backend: str = "vitis_emulated"
    width_bits: int = Field(12, ge=4, le=32)
    int_bits: int = Field(4, ge=2, le=16)
    device: str = "versal-ai-edge"
    include_thumbnails: bool = True


@router.post("/isp/run")
def isp_run(req: IspRunRequest) -> Dict:
    _check_device(req.device)
    from sensorflow.vitis.isp import DEFAULT_STAGES, run_isp
    if req.stages:
        bad = [s for s in req.stages if s not in DEFAULT_STAGES]
        if bad:
            raise HTTPException(400, f"Unknown ISP stages {bad}; "
                                     f"known: {DEFAULT_STAGES}")
    try:
        return run_isp(**req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/isp/stages")
def isp_stages() -> Dict:
    from sensorflow.vitis.isp import DEFAULT_STAGES
    return {"stages": DEFAULT_STAGES}


@router.get("/isp/runs")
def isp_runs() -> Dict:
    return {"runs": list_runs("isp")}


@router.get("/isp/runs/{run_id}")
def isp_run_report(run_id: str) -> Dict:
    run = load_run("isp", run_id)
    if run is None:
        raise HTTPException(404, f"Unknown ISP run {run_id}")
    return run


@router.get("/augment/recipes")
def augment_recipes() -> Dict:
    from sensorflow.vitis.augment import list_augmentations
    return {"augmentations": list_augmentations()}


@router.post("/augment/generate")
def augment_generate(req: AugmentRequest) -> Dict:
    _check_device(req.device)
    from sensorflow.vitis.augment import generate_batch
    if req.backend not in ("reference", "vitis_emulated"):
        raise HTTPException(400, f"Backend {req.backend!r} not runnable here; "
                                 "use reference or vitis_emulated")
    try:
        return generate_batch(
            recipes=req.recipes, n_variants=req.n_variants, seed=req.seed,
            backend_name=req.backend, width_bits=req.width_bits,
            int_bits=req.int_bits, device=req.device,
            include_thumbnails=req.include_thumbnails)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/augment/batches")
def augment_batches() -> Dict:
    return {"batches": list_runs("augment")}


@router.get("/augment/batches/{batch_id}")
def augment_batch(batch_id: str) -> Dict:
    run = load_run("augment", batch_id)
    if run is None:
        raise HTTPException(404, f"Unknown augmentation batch {batch_id}")
    return run


@router.get("/augment/variants")
def augment_variants() -> Dict:
    from sensorflow.vitis.augment import list_variants
    return {"variants": list_variants()}


# --------------------------------------------------------------------------
# Feature 3: temporal & stereo stability
# --------------------------------------------------------------------------

class TemporalRunRequest(BaseModel):
    engines: Optional[List[str]] = None
    n_sequences: int = Field(3, ge=1, le=8)
    frames_per_sequence: int = Field(18, ge=8, le=40)
    seed: int = 7
    width_bits: int = Field(12, ge=4, le=32)
    int_bits: int = Field(6, ge=2, le=16)
    device: str = "versal-ai-edge"


@router.get("/temporal/engines")
def temporal_engines() -> Dict:
    from sensorflow.vitis.temporal import KNOWN_ENGINES
    return {"engines": list(KNOWN_ENGINES)}


@router.post("/temporal/run")
def temporal_run(req: TemporalRunRequest) -> Dict:
    _check_device(req.device)
    from sensorflow.vitis.temporal import KNOWN_ENGINES, run_temporal_profile
    if req.engines:
        bad = [e for e in req.engines if e not in KNOWN_ENGINES]
        if bad:
            raise HTTPException(400, f"Unknown engines {bad}; "
                                     f"known: {list(KNOWN_ENGINES)}")
    return run_temporal_profile(**req.model_dump())


@router.get("/temporal/runs")
def temporal_runs() -> Dict:
    return {"runs": list_runs("temporal")}


@router.get("/temporal/runs/{run_id}")
def temporal_run_report(run_id: str) -> Dict:
    run = load_run("temporal", run_id)
    if run is None:
        raise HTTPException(404, f"Unknown temporal run {run_id}")
    return run


# --------------------------------------------------------------------------
# PRDs
# --------------------------------------------------------------------------

@router.get("/prd")
def prd_list() -> Dict:
    docs = []
    for key, fname in PRD_FILES.items():
        path = PRD_DIR / fname
        docs.append({"id": key, "file": fname, "available": path.exists()})
    return {"prds": docs}


@router.get("/prd/{prd_id}")
def prd_get(prd_id: str) -> Dict:
    fname = PRD_FILES.get(prd_id)
    if fname is None:
        raise HTTPException(404, f"Unknown PRD {prd_id!r}; "
                                 f"known: {sorted(PRD_FILES)}")
    path = PRD_DIR / fname
    if not path.exists():
        raise HTTPException(404, f"PRD file {fname} missing on disk")
    return {"id": prd_id, "file": fname, "markdown": path.read_text()}
