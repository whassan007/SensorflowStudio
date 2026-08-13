"""FastAPI router for the aggregate-first mega-scale evaluation layer."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.megaeval import analysis, errors as errors_mod, population as pop_mod
from sensorflow.megaeval import sampling
from sensorflow.megaeval.cube import SKETCH_METRICS
from sensorflow.megaeval.population import DIMENSIONS
from sensorflow.megaeval.runs import CONTAINER_STATUS, EvaluationRun, get_mega_store

router = APIRouter(prefix="/api")


def _run_or_404(run_id: str) -> EvaluationRun:
    run = get_mega_store().runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"Unknown evaluation run {run_id}")
    return run


def _published_or_409(run_id: str) -> EvaluationRun:
    run = _run_or_404(run_id)
    if run.status != "published":
        raise HTTPException(409, f"Run {run_id} is {run.status}; results are available "
                                 "once it is published")
    return run


# ------------------------------------------------------------------ requests


class GeneratePopulationRequest(BaseModel):
    name: str = "mega-perception"
    num_objects: int = Field(320_000, ge=1_000, le=1_200_000)
    seed: int = 42


class CreateRunRequest(BaseModel):
    population_id: str
    model_version: str = "model-v42"
    overrides: Dict = Field(default_factory=dict)
    worker_delay_s: float = Field(0.5, ge=0.0, le=5.0)
    workers: int = Field(4, ge=1, le=8)
    seed: Optional[int] = None
    label_version: str = "labels-v1"


class EvaluationQueryRequest(BaseModel):
    evaluation_id: str
    filters: Dict[str, List[str]] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    limit: int = Field(200, ge=1, le=2000)


class ErrorSearchRequest(BaseModel):
    run_id: str
    error_types: List[str] = Field(default_factory=list)
    filters: Dict[str, List[str]] = Field(default_factory=dict)
    confidence_max: Optional[float] = None
    confidence_min: Optional[float] = None
    risk_min: Optional[float] = None
    severity_min: Optional[float] = None
    safety_only: bool = False
    limit_containers: int = Field(25, ge=1, le=100)


class CompareRequest(BaseModel):
    candidate_run_id: str
    baseline_run_id: str
    policy: Dict = Field(default_factory=dict)


class WhyRequest(BaseModel):
    run_id: str
    filters: Dict[str, List[str]] = Field(default_factory=dict)
    metric: str = "recall"


class SimilarityRequest(BaseModel):
    run_id: str
    container_id: int
    filters: Dict[str, List[str]] = Field(default_factory=dict)
    k: int = Field(12, ge=1, le=50)


class ReviewPlanRequest(BaseModel):
    target_n: Optional[int] = Field(None, ge=100, le=20_000)


# ------------------------------------------------------------------ populations


@router.post("/megaeval/population/generate")
def generate_population(req: GeneratePopulationRequest):
    meta = pop_mod.generate_population(name=req.name, num_objects=req.num_objects,
                                       seed=req.seed)
    return meta


@router.get("/megaeval/populations")
def list_populations():
    return {"populations": pop_mod.list_populations()}


@router.get("/megaeval/dimensions")
def dimensions():
    return {"dimensions": DIMENSIONS}


# ------------------------------------------------------------------ runs (async jobs)


@router.post("/megaeval/runs")
def create_run(req: CreateRunRequest):
    store = get_mega_store()
    try:
        run = store.create_run(population_id=req.population_id,
                               model_version=req.model_version,
                               overrides=req.overrides,
                               seed=req.seed,
                               worker_delay_s=req.worker_delay_s,
                               workers=req.workers,
                               label_version=req.label_version)
    except KeyError as e:
        raise HTTPException(404, str(e))
    store.start_async(run)
    return run.to_dict()


@router.get("/megaeval/runs")
def list_runs():
    store = get_mega_store()
    runs = sorted(store.runs.values(), key=lambda r: r.created_at, reverse=True)
    return {"runs": [r.to_dict() for r in runs]}


@router.get("/megaeval/runs/{run_id}")
def get_run(run_id: str):
    return _run_or_404(run_id).to_dict()


@router.get("/megaeval/status")
def megaeval_status():
    store = get_mega_store()
    active = [r.progress_dict() for r in store.runs.values()
              if r.status in ("queued", "running", "reducing", "materializing")]
    return {"active_runs": active, "cache": store.router.cache.stats(),
            "total_runs": len(store.runs)}


# ------------------------------------------------------------------ THE query API


@router.post("/evaluations/query")
def evaluations_query(req: EvaluationQueryRequest):
    """Aggregate-first query endpoint. Routed: cache -> cube -> record scan."""
    store = get_mega_store()
    run = _published_or_409(req.evaluation_id)
    art = store.artifacts(run.run_id)

    def scan_fn(filters, group_by, metrics, limit):
        return store.scan_records(run, filters, group_by, metrics, limit)

    return store.router.query(run, art["cube"], art.get("sketches", {}),
                              req.filters, req.metrics, req.group_by,
                              req.limit, scan_fn=scan_fn)


@router.get("/megaeval/cache")
def cache_stats():
    return get_mega_store().router.cache.stats()


# ------------------------------------------------------------------ quality funnel


@router.get("/megaeval/runs/{run_id}/funnel")
def quality_funnel(run_id: str):
    store = get_mega_store()
    run = _published_or_409(run_id)
    art = store.artifacts(run_id)
    errs = art.get("errors")
    review = art.get("review") or {}
    n_err = int(len(errs)) if errs is not None else 0
    n_high = int((errs["risk_score"] >= 0.6).sum()) if errs is not None and len(errs) else 0
    results = (review.get("results") or {})
    prec = results.get("precision") or {}
    stages = [
        {"stage": "Objects Evaluated", "count": int(run.objects_total)},
        {"stage": "Containers", "count": int(run.headline.get("containers", 0))},
        {"stage": "Potential Errors", "count": n_err},
        {"stage": "High-Confidence Errors", "count": n_high},
        {"stage": "Review Sample", "count": int((review.get("funnel") or {}).get("statistically_selected", 0))},
        {"stage": "Human Verified", "count": int((review.get("funnel") or {}).get("reviewed", 0))},
    ]
    base = stages[0]["count"] or 1
    for s in stages:
        s["pct_of_population"] = round(s["count"] / base * 100, 3)
    return {"stages": stages,
            "estimated_precision": ({"estimate": prec.get("estimate"),
                                     "ci_low": prec.get("ci_low"),
                                     "ci_high": prec.get("ci_high"),
                                     "n_reviewed": prec.get("n_reviewed")}
                                    if prec else None)}


# ------------------------------------------------------------------ containers


_SORT_PRESETS = {
    "worst_recall": ("recall", True),
    "worst_precision": ("precision", True),
    "worst_iou": ("mean_iou", True),
    "most_anomalies": ("anomalies", False),
    "least_verified": ("reviewed", True),
    "highest_risk": ("risk_score", False),
}


@router.get("/megaeval/runs/{run_id}/containers")
def containers(run_id: str, sort: str = "highest_risk", limit: int = 50, offset: int = 0):
    store = get_mega_store()
    _published_or_409(run_id)
    art = store.artifacts(run_id)
    df = art["containers"].copy()
    tp, fp, fn = df["tp"], df["fp"], df["fn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["recall"] = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        df["precision"] = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        df["mean_iou"] = np.where(tp > 0, df["sum_iou"] / tp, np.nan)
    metric, ascending = _SORT_PRESETS.get(sort, _SORT_PRESETS["highest_risk"])
    df = df.sort_values(metric, ascending=ascending, na_position="last")
    total = len(df)
    page = df.iloc[offset:offset + max(1, min(limit, 500))]
    rows = []
    for _, r in page.iterrows():
        rows.append({
            "container_id": int(r["container_id"]),
            **{dim: DIMENSIONS[dim][int(r[dim])] for dim in pop_mod.CONTAINER_DIMS},
            "n_objects": int(r["n_objects"]),
            "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
            "anomalies": int(r["anomalies"]),
            "reviewed": int(r["reviewed"]), "verified": int(r["verified"]),
            "recall": None if np.isnan(r["recall"]) else round(float(r["recall"]), 4),
            "precision": None if np.isnan(r["precision"]) else round(float(r["precision"]), 4),
            "mean_iou": None if np.isnan(r["mean_iou"]) else round(float(r["mean_iou"]), 4),
            "risk_score": round(float(r["risk_score"]), 4),
            "status": CONTAINER_STATUS[int(r["status"])],
        })
    return {"total": total, "sort": sort, "rows": rows}


@router.get("/megaeval/runs/{run_id}/containers/{container_id}/objects")
def container_objects(run_id: str, container_id: int):
    store = get_mega_store()
    run = _published_or_409(run_id)
    try:
        rows = store.container_objects(run, container_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No forensic data for container {container_id}")
    if not rows:
        raise HTTPException(404, f"Unknown container {container_id}")
    return {"container_id": container_id, "objects": rows}


# ------------------------------------------------------------------ error index


@router.post("/megaeval/errors/search")
def error_search(req: ErrorSearchRequest):
    store = get_mega_store()
    _published_or_409(req.run_id)
    art = store.artifacts(req.run_id)
    return errors_mod.search_errors(
        art.get("errors"), art.get("containers"),
        error_types=req.error_types or None, filters=req.filters or None,
        confidence_max=req.confidence_max, confidence_min=req.confidence_min,
        risk_min=req.risk_min, severity_min=req.severity_min,
        safety_only=req.safety_only, limit_containers=req.limit_containers)


# ------------------------------------------------------------------ compare / shift / why / similarity


@router.post("/megaeval/compare")
def compare(req: CompareRequest):
    store = get_mega_store()
    cand = _published_or_409(req.candidate_run_id)
    base = _published_or_409(req.baseline_run_id)
    return analysis.compare_runs(store, cand, base, req.policy or None)


@router.get("/megaeval/runs/{run_id}/shift")
def shift(run_id: str):
    return analysis.distribution_shift(get_mega_store(), _published_or_409(run_id))


@router.post("/megaeval/why")
def why(req: WhyRequest):
    return analysis.why(get_mega_store(), _published_or_409(req.run_id),
                        req.filters or None, req.metric)


@router.post("/megaeval/similarity")
def similarity(req: SimilarityRequest):
    try:
        return analysis.similar_containers(get_mega_store(), _published_or_409(req.run_id),
                                           req.container_id, req.filters or None, req.k)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ------------------------------------------------------------------ review sampling


@router.get("/megaeval/runs/{run_id}/review")
def review_state(run_id: str):
    store = get_mega_store()
    _published_or_409(run_id)
    art = store.artifacts(run_id)
    review = art.get("review")
    if review is None:
        return {"run_id": run_id, "planned": False, "executed": False}
    slim = {k: v for k, v in review.items() if k != "plans"}
    slim["planned"] = True
    slim["strata"] = {name: plan["strata"] for name, plan in review["plans"].items()}
    return slim


@router.post("/megaeval/runs/{run_id}/review/plan")
def review_plan(run_id: str, req: ReviewPlanRequest):
    store = get_mega_store()
    run = _published_or_409(run_id)
    review = sampling.build_review_plan(store, run, target_n=req.target_n)
    slim = {k: v for k, v in review.items() if k != "plans"}
    slim["planned"] = True
    slim["strata"] = {name: plan["strata"] for name, plan in review["plans"].items()}
    return slim


@router.post("/megaeval/runs/{run_id}/review/execute")
def review_execute(run_id: str):
    store = get_mega_store()
    run = _published_or_409(run_id)
    art = store.artifacts(run_id)
    if art.get("review") is None:
        sampling.build_review_plan(store, run)
    review = sampling.execute_reviews(store, run)
    slim = {k: v for k, v in review.items() if k != "plans"}
    slim["planned"] = True
    slim["strata"] = {name: plan["strata"] for name, plan in review["plans"].items()}
    return slim


# ------------------------------------------------------------------ distributions (sketches)


@router.get("/megaeval/runs/{run_id}/distributions")
def distributions(run_id: str):
    store = get_mega_store()
    run = _published_or_409(run_id)
    art = store.artifacts(run_id)
    sk = art.get("sketches", {})
    out = {"run_id": run_id, "exact": False,
           "note": "distributions are sketch-based (fixed-bin quantile histograms); "
                   "counts in the cube are exact"}
    for name in ("confidence", "iou"):
        hist = sk.get(name)
        if hist is None:
            continue
        out[name] = {
            "bins": hist.bins, "lo": hist.lo, "hi": hist.hi,
            "counts": hist.counts.tolist(),
            "percentiles": {q: round(hist.percentile(float(q)), 4)
                            for q in ("10", "25", "50", "75", "90", "99")},
        }
    out["containers_hll_estimate"] = sk.get("container_hll_estimate")
    out["containers_exact"] = run.headline.get("containers")
    out["available_sketch_metrics"] = SKETCH_METRICS
    return out
