"""FastAPI router for the safety & compliance layer (/api/safety/*).

Endpoint map (frontend integration pass consumes this):

ODD coverage (ISO 34503 / ASAM OpenODD-inspired)
  GET  /api/safety/odd/taxonomy
  GET  /api/safety/odd/coverage?run=...&dims=weather,lighting&include_cells=...
  POST /api/safety/odd/fill-gap            {run, cell, num_sequences?, ...}

Release gating + Safety Evidence Package (ISO 26262 / SOTIF / UL 4600-inspired)
  POST /api/safety/gates/evaluate          {candidate_run, baseline_run, policy_overrides?}
  GET  /api/safety/gates/policy
  POST /api/safety/gates/policy            {overrides} (persisted)
  GET  /api/safety/gates/result/{run_id}
  GET  /api/safety/evidence/{run_id}?format=json|markdown

Extended SSAM surrogate safety (FHWA SSAM)
  POST /api/safety/ssam/analyze            {trajectories?|scenario?, params?, ...}
  GET  /api/safety/ssam/summary?run=...

Multi-sensor calibration validation (Deepen-style)
  POST /api/safety/calibration/validate    {mode, rotation_offset_deg?, ...}
  GET  /api/safety/calibration/status

Auto-label discrepancy mining (continuous-learning loop)
  POST /api/safety/discrepancy/mine        {dataset_id?, profile?, seed?}
  GET  /api/safety/discrepancy/summary?dataset_id=...

Scenario database (Safety Pool-inspired, local)
  GET  /api/safety/scenarios?source=&severity=&scenario_type=&text=&limit=
  POST /api/safety/scenarios/populate      (auto-import rare events)
  POST /api/safety/scenarios/export        {filters?}

Neuro-symbolic semantic mining
  POST /api/safety/semantic-search         {concept, filters?, run?, target?, k?, use_llm?}

Consensus extensions (additive evidence over the grader subsystem)
  GET  /api/safety/consensus/summary?dataset_id=...
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sensorflow.safety import calibration as calib_mod
from sensorflow.safety import discrepancy as disc_mod
from sensorflow.safety import gates as gates_mod
from sensorflow.safety import odd as odd_mod
from sensorflow.safety import scenario_db as sdb_mod
from sensorflow.safety import semantic as sem_mod
from sensorflow.safety import ssam_ext

router = APIRouter(prefix="/api/safety", tags=["safety"])


def _mega_store():
    from sensorflow.megaeval.runs import get_mega_store
    return get_mega_store()


def _eval_store():
    from sensorflow.evaluation.records import get_store
    return get_store()


def _published_run(run_id: str):
    run = _mega_store().runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"Unknown evaluation run {run_id}")
    if run.status != "published":
        raise HTTPException(409, f"Run {run_id} is {run.status}; safety analyses "
                                 "need a published run")
    return run


# ------------------------------------------------------------------ ODD


@router.get("/odd/taxonomy")
def odd_taxonomy():
    return odd_mod.taxonomy()


@router.get("/odd/coverage")
def odd_coverage(run: str,
                 dims: Optional[str] = Query(None, description="comma-separated"),
                 min_samples: int = Query(odd_mod.DEFAULT_MIN_SAMPLES, ge=1, le=100_000),
                 max_ci_width: float = Query(odd_mod.DEFAULT_MAX_CI_WIDTH, gt=0, le=1),
                 target_recall: Optional[float] = Query(None, gt=0, le=1),
                 include_cells: bool = False,
                 max_gaps: int = Query(50, ge=1, le=500)):
    r = _published_run(run)
    dim_list = [d.strip() for d in dims.split(",") if d.strip()] if dims else None
    try:
        return odd_mod.coverage_for_run(
            _mega_store(), r, dims=dim_list, min_samples=min_samples,
            max_ci_width=max_ci_width, target_recall=target_recall,
            include_cells=include_cells, max_gaps=max_gaps)
    except ValueError as e:
        raise HTTPException(422, str(e))


class FillGapRequest(BaseModel):
    run: str
    cell: Dict[str, str]
    num_sequences: int = Field(2, ge=1, le=10)
    frames_per_sequence: int = Field(20, ge=5, le=60)
    seed: Optional[int] = None


@router.post("/odd/fill-gap")
def odd_fill_gap(req: FillGapRequest):
    r = _published_run(req.run)
    try:
        return odd_mod.fill_gap(_mega_store(), r, req.cell,
                                num_sequences=req.num_sequences,
                                frames_per_sequence=req.frames_per_sequence,
                                seed=req.seed)
    except ValueError as e:
        raise HTTPException(422, str(e))


# ------------------------------------------------------------------ gates


class GateEvaluateRequest(BaseModel):
    candidate_run: str
    baseline_run: str
    policy_overrides: Optional[Dict] = None


@router.post("/gates/evaluate")
def gates_evaluate(req: GateEvaluateRequest):
    cand = _published_run(req.candidate_run)
    base = _published_run(req.baseline_run)
    return gates_mod.evaluate_gates(_mega_store(), cand, base,
                                    policy_overrides=req.policy_overrides)


@router.get("/gates/policy")
def gates_policy():
    return {"policy": gates_mod.get_policy(),
            "defaults": gates_mod.DEFAULT_POLICY}


class PolicyUpdateRequest(BaseModel):
    overrides: Dict


@router.post("/gates/policy")
def gates_policy_update(req: PolicyUpdateRequest):
    return {"policy": gates_mod.set_policy(req.overrides)}


@router.get("/gates/result/{run_id}")
def gates_result(run_id: str):
    result = gates_mod.latest_gate_result(run_id)
    if result is None:
        raise HTTPException(404, f"No gate evaluation recorded for run {run_id}")
    return result


@router.get("/evidence/{run_id}")
def evidence(run_id: str, format: str = Query("json", pattern="^(json|markdown)$")):
    pkg = gates_mod.load_evidence(run_id)
    if pkg is None:
        raise HTTPException(404, f"No Safety Evidence Package for run {run_id}; "
                                 "run POST /api/safety/gates/evaluate first")
    if format == "markdown":
        return {"run_id": run_id, "format": "markdown",
                "markdown": gates_mod.render_markdown(pkg)}
    return pkg


# ------------------------------------------------------------------ SSAM


class TrajectoryState(BaseModel):
    t: float
    x: float
    y: float
    speed: float
    heading: float


class Trajectory(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"
    length: float = Field(4.5, gt=0)
    width: float = Field(1.9, gt=0)
    states: List[TrajectoryState]


class SsamAnalyzeRequest(BaseModel):
    trajectories: Optional[List[Trajectory]] = None
    scenario: Optional[str] = Field(
        None, description="crossing | rear_end | lane_change | mixed — generates "
                          "synthetic TRJ-like trajectories when none are supplied")
    seed: int = 0
    reaction_delay_s: float = Field(0.6, ge=0, le=5)
    params: Optional[Dict] = None


@router.post("/ssam/analyze")
def ssam_analyze(req: SsamAnalyzeRequest):
    if req.trajectories:
        trajs = [t.model_dump() for t in req.trajectories]
        generated = None
    else:
        scenario = req.scenario or "mixed"
        try:
            trajs = ssam_ext.generate_trajectories(
                seed=req.seed, scenario=scenario, reaction_delay_s=req.reaction_delay_s)
        except ValueError as e:
            raise HTTPException(422, str(e))
        generated = {"scenario": scenario, "seed": req.seed,
                     "reaction_delay_s": req.reaction_delay_s, "simulated": True}
    result = ssam_ext.analyze_trajectories(trajs, params=req.params)
    if generated:
        result["generated"] = generated
    return result


@router.get("/ssam/summary")
def ssam_summary(run: str, force: bool = False):
    r = _published_run(run)
    return ssam_ext.csi_for_run(r, force=force)


# ------------------------------------------------------------------ calibration


class CalibrationValidateRequest(BaseModel):
    mode: str = Field("clean", pattern="^(clean|miscalibrated|perception_failure)$")
    rotation_offset_deg: float = Field(2.0, ge=0, le=30)
    translation_offset_m: float = Field(0.0, ge=0, le=5)
    tamper_fraction: float = Field(0.25, ge=0, le=1)
    num_objects: int = Field(14, ge=3, le=60)
    seed: int = 3


@router.post("/calibration/validate")
def calibration_validate(req: CalibrationValidateRequest):
    return calib_mod.run_validation(
        mode=req.mode, rotation_offset_deg=req.rotation_offset_deg,
        translation_offset_m=req.translation_offset_m,
        tamper_fraction=req.tamper_fraction, num_objects=req.num_objects,
        seed=req.seed)


@router.get("/calibration/status")
def calibration_status():
    status = calib_mod.latest_status()
    if status is None:
        return {"status": "NEVER_RUN",
                "note": "POST /api/safety/calibration/validate to run a validation"}
    return status


# ------------------------------------------------------------------ discrepancy mining


class DiscrepancyMineRequest(BaseModel):
    dataset_id: Optional[str] = Field(
        None, description="defaults to the newest labeleval dataset")
    profile: Optional[Dict] = None
    seed: int = 101


@router.post("/discrepancy/mine")
def discrepancy_mine(req: DiscrepancyMineRequest):
    store = _eval_store()
    dataset_id = req.dataset_id
    if dataset_id is None:
        datasets = store.all("datasets")
        if not datasets:
            raise HTTPException(404, "No labeleval datasets exist; generate one first "
                                     "(POST /api/labeleval/datasets/generate)")
        dataset_id = max(datasets, key=lambda d: d.created_at).dataset_id
    try:
        report = disc_mod.mine(store, dataset_id, profile=req.profile, seed=req.seed)
    except KeyError as e:
        raise HTTPException(404, str(e))
    # full per-discrepancy list can be large; cap the response (persisted intact)
    slim = dict(report)
    slim["discrepancies"] = report["discrepancies"][:100]
    slim["discrepancies_truncated"] = len(report["discrepancies"]) > 100
    return slim


@router.get("/discrepancy/summary")
def discrepancy_summary(dataset_id: Optional[str] = None):
    if dataset_id:
        summary = disc_mod.latest_summary(dataset_id)
        if summary is None:
            raise HTTPException(404, f"No discrepancy mining report for {dataset_id}")
        return summary
    store = _eval_store()
    summaries = []
    for ds in store.all("datasets"):
        s = disc_mod.latest_summary(ds.dataset_id)
        if s is not None:
            summaries.append(s)
    return {"datasets": summaries}


# ------------------------------------------------------------------ scenario database


@router.get("/scenarios")
def scenarios(scenario_type: Optional[str] = None, source: Optional[str] = None,
              severity: Optional[str] = None, weather: Optional[str] = None,
              lighting: Optional[str] = None, text: Optional[str] = None,
              limit: int = Query(100, ge=1, le=1000)):
    odd_tags = {}
    if weather:
        odd_tags["weather"] = weather
    if lighting:
        odd_tags["lighting"] = lighting
    db = sdb_mod.get_db()
    records = db.search(scenario_type=scenario_type, source=source, severity=severity,
                        odd_tags=odd_tags or None, text=text, limit=limit)
    return {"counts": db.counts(),
            "scenarios": [r.model_dump() for r in records]}


@router.post("/scenarios/populate")
def scenarios_populate():
    added = sdb_mod.get_db().add_from_rare_events(_eval_store())
    return {"imported_rare_events": added, "counts": sdb_mod.get_db().counts()}


class ScenarioExportRequest(BaseModel):
    scenario_type: Optional[str] = None
    source: Optional[str] = None
    severity: Optional[str] = None
    text: Optional[str] = None
    limit: int = Field(1000, ge=1, le=5000)


@router.post("/scenarios/export")
def scenarios_export(req: ScenarioExportRequest):
    return sdb_mod.get_db().export_bundle(
        scenario_type=req.scenario_type, source=req.source,
        severity=req.severity, text=req.text, limit=req.limit)


# ------------------------------------------------------------------ semantic search


class SemanticSearchRequest(BaseModel):
    concept: str = Field(..., min_length=2, max_length=500)
    filters: Optional[Dict] = None
    target: str = Field("containers", pattern="^(containers|scenarios)$")
    run: Optional[str] = Field(None, description="megaeval run for target=containers; "
                                                 "defaults to the newest published run")
    k: int = Field(12, ge=1, le=100)
    use_llm: Optional[bool] = Field(
        None, description="None=auto (short-timeout Ollama attempt), true/false to force")


@router.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    if req.target == "scenarios":
        return sem_mod.search_scenarios(req.concept, filters=req.filters, k=req.k)
    store = _mega_store()
    if req.run:
        run = _published_run(req.run)
    else:
        published = [r for r in store.runs.values() if r.status == "published"]
        if not published:
            raise HTTPException(404, "No published megaeval runs; start one via "
                                     "/api/megaeval or use target=scenarios")
        run = max(published, key=lambda r: r.created_at)
    return sem_mod.search_containers(store, run, req.concept, filters=req.filters,
                                     k=req.k, use_llm=req.use_llm)


# ------------------------------------------------------------------ consensus evidence


@router.get("/consensus/summary")
def consensus_summary(dataset_id: str, examples: int = Query(5, ge=0, le=50)):
    from sensorflow.evaluation.graders import dataset_grader_statistics
    store = _eval_store()
    if store.get("datasets", dataset_id) is None:
        raise HTTPException(404, f"Unknown dataset {dataset_id}")
    stats = dataset_grader_statistics(store, dataset_id)
    if not stats:
        raise HTTPException(404, f"No grader comparisons for {dataset_id}; run the "
                                 "evaluation pipeline first")
    anns = store.where("annotations", dataset_id=dataset_id)
    sample = []
    for a in sorted(anns, key=lambda x: x.annotation_id):
        c = store.get("grader_comparisons", a.annotation_id)
        if c is None:
            continue
        vector = {k[4:]: v for k, v in c.kappa_stats.items() if k.startswith("csv_")}
        mbr = {k[4:]: v for k, v in c.kappa_stats.items() if k.startswith("mbr_")}
        if not vector:
            continue
        sample.append({"annotation_id": a.annotation_id,
                       "consensus_scalar": c.consensus,
                       "consensus_score_vector": vector,
                       "mbr": mbr})
        if len(sample) >= examples:
            break
    return {"dataset_id": dataset_id, "statistics": stats,
            "examples": sample,
            "note": "grader panel is simulated (see graders.py); kappa/tau/MBR "
                    "math is exact"}
