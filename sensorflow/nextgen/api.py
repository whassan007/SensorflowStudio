"""FastAPI router for the Next-Generation AV Perception Evaluation platform.

Included by app_backend with prefix /api/nextgen. All artifacts persist
under runs/nextgen/ (store.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.nextgen import causal as causal_mod
from sensorflow.nextgen import compute as compute_mod
from sensorflow.nextgen import counterfactual as cf_mod
from sensorflow.nextgen import lineage as lineage_mod
from sensorflow.nextgen import safety_metrics as sm_mod
from sensorflow.nextgen import scheduler as sched_mod
from sensorflow.nextgen import store
from sensorflow.nextgen import validity as validity_mod
from sensorflow.nextgen.closedloop import DEFAULT_ENGINE, ENGINE_PROFILES, run_closed_loop
from sensorflow.nextgen.models import (
    AGENTIC_SCORECARD_SOURCE, DataLabel, DistributionShiftAssessment,
    TransformationStep,
)

router = APIRouter(prefix="/api/nextgen")

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "architecture"
_DOC_FILES = {
    "comparison": "nextgen-architecture-comparison.md",
    "adr": "nextgen-adr.md",
    "rollout": "nextgen-rollout.md",
    "worldmodel": "nextgen-worldmodel-generative-comparison.md",
}


# ------------------------------------------------------------ status


@router.get("/status")
def status():
    return {
        "package": "sensorflow.nextgen",
        "engines": sorted(ENGINE_PROFILES),
        "transformations": sorted(cf_mod.TRANSFORMATIONS),
        "counterfactuals": len(store.list_json("counterfactuals")),
        "gauntlets": sched_mod.list_gauntlets(),
        "compute_report_available": compute_mod.latest_report() is not None,
        "agentic_scorecard_source": AGENTIC_SCORECARD_SOURCE,
        "data_labels": [d.value for d in DataLabel],
        "component_versions": lineage_mod.COMPONENT_VERSIONS,
    }


# ------------------------------------------------------------ counterfactuals


class GenerateRequest(BaseModel):
    recipe: List[TransformationStep]
    seed: int = 7
    n_scenarios: int = Field(3, ge=1, le=12)
    frames_per_sequence: int = Field(40, ge=10, le=80)


@router.get("/counterfactuals/catalogue")
def catalogue():
    return {"transformations": cf_mod.transformation_catalogue()}


@router.post("/counterfactuals/generate")
def generate(req: GenerateRequest):
    if not req.recipe:
        raise HTTPException(422, "recipe must contain at least one transformation")
    try:
        scenarios = cf_mod.generate_counterfactuals(
            req.recipe, seed=req.seed, n_scenarios=req.n_scenarios,
            frames_per_sequence=req.frames_per_sequence)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"scenarios": [s.model_dump(mode="json") for s in scenarios]}


@router.get("/counterfactuals")
def list_counterfactuals():
    return {"scenarios": [s.model_dump(mode="json") for s in cf_mod.list_scenarios()]}


@router.post("/counterfactuals/{scenario_id}/validate")
def validate(scenario_id: str):
    bundle = cf_mod.load_bundle(scenario_id)
    if bundle is None:
        raise HTTPException(404, f"unknown scenario {scenario_id}")
    scenario = bundle["scenario"]
    report = validity_mod.validate_scenario(
        scenario_id, bundle["sequence"], bundle["actors"],
        scenario.environment, scenario.provenance.seed,
        source_overlap_pairs=bundle.get("source_overlap_pairs"),
        source_features=bundle.get("source_features"))
    scenario.validity = report
    cf_mod.save_scenario(scenario)
    return report.model_dump(mode="json")


@router.get("/counterfactuals/suite-weights")
def suite_weights():
    """Suite-level weight policy over every validated scenario."""
    reports = [s.validity for s in cf_mod.list_scenarios() if s.validity]
    return validity_mod.apply_suite_weight_policy(reports)


# ------------------------------------------------------------ closed loop


class ReplayRequest(BaseModel):
    scenario_id: str
    engine: str = DEFAULT_ENGINE
    seed: int = 0
    corrected: bool = False
    faults: List[Dict[str, Any]] = Field(default_factory=list)


def _resolve_scenario(scenario_id: str, faults: List[Dict[str, Any]],
                      default_demo_fault: bool = False):
    """Returns (actors, environment, scenario_id, data_label, faults)."""
    if scenario_id in ("demo", "demo-occluded-emergence"):
        from sensorflow.nextgen.closedloop import demo_emergence_scenario
        demo = demo_emergence_scenario()
        if not faults and default_demo_fault:
            # Out-of-the-box causal demo: the emergent pedestrian is missed
            # until far too late.
            faults = [{"type": "miss",
                       "instance_id": demo["critical_instance_id"],
                       "from_s": 0.0, "until_s": 4.0}]
        return (demo["actors"], demo["environment"], demo["scenario_id"],
                DataLabel.COUNTERFACTUAL, faults)
    bundle = cf_mod.load_bundle(scenario_id)
    if bundle is None:
        raise HTTPException(404, f"unknown scenario {scenario_id}; generate "
                                 f"counterfactuals first or use scenario_id='demo'")
    scenario = bundle["scenario"]
    return (bundle["actors"], scenario.environment, scenario.scenario_id,
            scenario.provenance.data_label, faults)


@router.post("/simulation/replay")
def simulation_replay(req: ReplayRequest):
    if req.engine not in ENGINE_PROFILES:
        raise HTTPException(422, f"unknown engine {req.engine}")
    actors, env, sid, label, faults = _resolve_scenario(req.scenario_id, req.faults)
    assessment = run_closed_loop(actors, env, sid, label, engine=req.engine,
                                 seed=req.seed, corrected=req.corrected,
                                 faults=faults)
    result = assessment.model_dump(mode="json")
    store.write_json(result, "closedloop", f"{sid}-{req.engine}.json")
    return result


@router.post("/causal/replay")
def causal_replay(req: ReplayRequest):
    if req.engine not in ENGINE_PROFILES:
        raise HTTPException(422, f"unknown engine {req.engine}")
    actors, env, sid, label, faults = _resolve_scenario(
        req.scenario_id, req.faults, default_demo_fault=True)
    result = causal_mod.causal_replay(actors, env, sid, label,
                                      engine=req.engine, seed=req.seed,
                                      faults=faults)
    payload = result.model_dump(mode="json")
    store.write_json(payload, "causal", f"{sid}.json")
    return payload


# ------------------------------------------------------------ safety metrics


class SafetyReportRequest(BaseModel):
    objects: Optional[List[Dict[str, Any]]] = None
    ego_speed_mps: float = 10.0
    reaction_time_s: float = 0.9
    brake_capability_mps2: float = 7.0
    friction_mu: float = 0.9


@router.post("/metrics/safety-report")
def safety_report(req: SafetyReportRequest):
    params = sm_mod.SafetyRegionParams(
        reaction_time_s=req.reaction_time_s,
        brake_capability_mps2=req.brake_capability_mps2,
        friction_mu=req.friction_mu)
    if req.objects:
        return sm_mod.safety_report(req.objects, req.ego_speed_mps, params)
    return sm_mod.divergence_demo(req.ego_speed_mps)


@router.get("/metrics/divergence-demo")
def divergence_demo():
    return sm_mod.divergence_demo()


# ------------------------------------------------------------ gauntlet


class GauntletRequest(BaseModel):
    candidate_version: str = "candidate-v4"
    baseline_version: str = "baseline-v3"
    effects: Dict[str, float] = Field(default_factory=dict)
    budget_units: int = Field(80_000, ge=1_000, le=500_000)
    batch_units: int = Field(2_000, ge=100, le=20_000)
    seed: int = 11


@router.post("/gauntlet/run")
def gauntlet_run(req: GauntletRequest):
    return sched_mod.run_gauntlet(
        candidate_version=req.candidate_version,
        baseline_version=req.baseline_version,
        effects=req.effects,
        config={"budget_units": req.budget_units, "batch_units": req.batch_units},
        seed=req.seed)


@router.get("/gauntlet/{run_id}/status")
def gauntlet_status(run_id: str):
    g = sched_mod.get_gauntlet(run_id)
    if g is None:
        raise HTTPException(404, f"unknown gauntlet {run_id}")
    return {"run_id": g["run_id"], "status": g["status"],
            "scale": g["scale"], "timing": g["timing"],
            "events": g["events"], "halted": g["halted"],
            "recommendation": g["recommendation"]["recommendation"]}


@router.get("/gauntlet/{run_id}/results")
def gauntlet_results(run_id: str):
    g = sched_mod.get_gauntlet(run_id)
    if g is None:
        raise HTTPException(404, f"unknown gauntlet {run_id}")
    return g


@router.get("/gauntlets")
def gauntlets():
    return {"gauntlets": sched_mod.list_gauntlets()}


# ------------------------------------------------------------ compute


@router.get("/compute/report")
def compute_report():
    rep = compute_mod.latest_report()
    if rep is None:
        raise HTTPException(404, "no compute benchmark yet; POST "
                                 "/api/nextgen/compute/benchmark first")
    return rep


class BenchmarkRequest(BaseModel):
    n_scenarios: int = Field(6, ge=2, le=16)
    frames_per_sequence: int = Field(20, ge=8, le=40)
    seed: int = 7


@router.post("/compute/benchmark")
def compute_benchmark(req: BenchmarkRequest):
    result = compute_mod.benchmark(n_scenarios=req.n_scenarios,
                                   frames_per_sequence=req.frames_per_sequence,
                                   seed=req.seed)
    return result.model_dump(mode="json")


# ------------------------------------------------------------ distribution


class DistributionRequest(BaseModel):
    megaeval_run_id: Optional[str] = None
    scenario_id: Optional[str] = None


@router.post("/distribution/analyze")
def distribution_analyze(req: DistributionRequest):
    """Distribution-shift analysis. Preferred path DELEGATES to
    sensorflow.megaeval.analysis.distribution_shift for a megaeval run; the
    scenario path compares a counterfactual's feature distributions to the
    real-scene reference (validity.check_distribution)."""
    if req.megaeval_run_id:
        from sensorflow.megaeval.analysis import distribution_shift
        from sensorflow.megaeval.runs import get_mega_store
        mstore = get_mega_store()
        run = mstore.runs.get(req.megaeval_run_id)
        if run is None:
            raise HTTPException(404, f"unknown megaeval run {req.megaeval_run_id}")
        shift = distribution_shift(mstore, run)
        return DistributionShiftAssessment(
            run_id=req.megaeval_run_id, method=shift["method"],
            shifts=shift["shifts"],
            data_labels=[DataLabel.SIMULATED]).model_dump(mode="json")
    if req.scenario_id:
        bundle = cf_mod.load_bundle(req.scenario_id)
        if bundle is None:
            raise HTTPException(404, f"unknown scenario {req.scenario_id}")
        check, _reasons, realism = validity_mod.check_distribution(bundle["sequence"])
        # parse psi/js out of the structured check
        return DistributionShiftAssessment(
            run_id=req.scenario_id, method=check["detail"],
            shifts=[check],
            psi=None, js_divergence=None, magnitude=None,
            data_labels=[bundle["scenario"].provenance.data_label]
        ).model_dump(mode="json")
    raise HTTPException(422, "provide megaeval_run_id or scenario_id")


# ------------------------------------------------------------ architecture


@router.get("/architecture/docs")
def architecture_docs():
    docs = {}
    for key, fname in _DOC_FILES.items():
        p = _DOCS_DIR / fname
        docs[key] = {"file": f"docs/architecture/{fname}",
                     "content": p.read_text() if p.exists() else None}
    missing = [k for k, v in docs.items() if v["content"] is None]
    if len(missing) == len(docs):
        raise HTTPException(404, "architecture docs not found")
    return {"docs": docs, "missing": missing}
