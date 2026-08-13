"""FastAPI router for the label evaluation platform (included by app_backend)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sensorflow.evaluation import copilot, reporting, synthetic
from sensorflow.evaluation.pipeline import get_pipeline
from sensorflow.evaluation.process_units import usage_summary
from sensorflow.evaluation.records import RelabelingAction, get_store
from sensorflow.evaluation.triage import QualityPolicy, load_policy, save_policy

router = APIRouter(prefix="/api")


# ------------------------------------------------------------------ request bodies


class GenerateDatasetRequest(BaseModel):
    name: str = "synthetic-perception"
    num_sequences: int = 6
    frames_per_sequence: int = 30
    seed: int = 7


class RunPipelineRequest(BaseModel):
    dataset_id: str
    policy_id: Optional[str] = None


class PrecheckRequest(BaseModel):
    dataset_id: Optional[str] = None


class ReviewActionRequest(BaseModel):
    action: str
    corrected_bbox_3d: Optional[List[float]] = None
    corrected_class: Optional[str] = None
    merge_with_track_id: Optional[str] = None
    note: Optional[str] = None


class TrainRequest(BaseModel):
    dataset_version: str
    model_version: Optional[str] = None
    configuration: Dict = Field(default_factory=dict)
    quality_policy: Optional[str] = None
    training_parameters: Dict = Field(default_factory=dict)


class BenchmarkTechniquesRequest(BaseModel):
    dataset_id: Optional[str] = None


class CopilotRequest(BaseModel):
    context_type: str = "general"
    annotation_id: Optional[str] = None
    event_id: Optional[str] = None
    model_version: Optional[str] = None
    extra: Dict = Field(default_factory=dict)


# ------------------------------------------------------------------ serializers


def _dataset_summary(d) -> Dict:
    store = get_store()
    anns = store.where("annotations", dataset_id=d.dataset_id)
    verified = sum(1 for a in anns if a.status == "VERIFIED")
    coverage = d.gt_coverage
    conf = "none"
    if coverage > 0:
        conf = "high" if d.gt_type in ("HUMAN_VERIFIED_GROUND_TRUTH", "GOLD_STANDARD") else (
            "medium" if d.gt_type == "VENDOR_GROUND_TRUTH" else "low")
    return {
        "dataset_id": d.dataset_id,
        "name": d.name,
        "version": d.version,
        "created_at": d.created_at,
        "num_scenes": d.num_scenes,
        "num_sequences": d.num_sequences,
        "num_frames": d.num_frames,
        "num_annotations": len(anns) or d.num_annotations,
        "gt_availability": {
            "has_reference": coverage > 0,
            "gt_type": d.gt_type if coverage > 0 else None,
            "coverage": coverage,
            "evaluation_confidence": conf,
        },
        "lineage": d.lineage.model_dump(),
        "status": d.status,
        "verified_count": verified,
    }


def _pipeline_state() -> Dict:
    pipe = get_pipeline()
    store = get_store()
    open_tasks = len([t for t in store.all("review_tasks") if t.status != "resolved"])
    return {
        "running": pipe.running,
        "stage": pipe.stage,
        "services": pipe.service_status(),
        "queue": pipe.queue.stats(),
        "counters": reporting.counters(store, pipe.active_dataset),
        "review_queue_count": open_tasks,
        "regression_alert": pipe.regression_alert,
        "last_run_id": pipe.last_run_id,
    }


def _active_training_job() -> Optional[Dict]:
    store = get_store()
    jobs = sorted(store.all("training_jobs"), key=lambda j: j.started_at)
    if not jobs:
        return None
    running = [j for j in jobs if j.status == "running"]
    job = running[-1] if running else jobs[-1]
    d = job.model_dump()
    d["logs"] = d["logs"][-50:]
    return d


# ------------------------------------------------------------------ datasets


@router.get("/datasets")
def list_datasets():
    store = get_store()
    ds = sorted(store.all("datasets"), key=lambda d: d.created_at)
    return {"datasets": [_dataset_summary(d) for d in ds]}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    store = get_store()
    d = store.get("datasets", dataset_id)
    if d is None:
        raise HTTPException(404, f"Unknown dataset {dataset_id}")
    return _dataset_summary(d)


@router.post("/dataset/precheck")
def dataset_precheck(req: PrecheckRequest):
    store = get_store()
    checks = []
    dataset = None
    if req.dataset_id:
        dataset = store.get("datasets", req.dataset_id)
        if dataset is None:
            raise HTTPException(404, f"Unknown dataset {req.dataset_id}")
    else:
        ds = sorted(store.all("datasets"), key=lambda d: d.created_at)
        dataset = ds[-1] if ds else None

    def check(gate, actual, threshold, passed, applicable=True):
        checks.append({"gate": gate, "actual": actual, "threshold": threshold,
                       "passed": passed, "applicable": applicable})

    if dataset is None:
        check("dataset_exists", False, True, False)
        return {"status": "failed", "message": "No dataset available — generate one first.", "checks": checks}

    frames = store.where("frames", dataset_id=dataset.dataset_id)
    check("dataset_exists", True, True, True)
    check("frames_present", len(frames), ">= 10", len(frames) >= 10)
    seqs = store.where("sequences", dataset_id=dataset.dataset_id)
    check("sequences_present", len(seqs), ">= 1", len(seqs) >= 1)
    with_gt = sum(1 for f in frames if f.gt_boxes)
    check("reference_gt_coverage", f"{with_gt}/{len(frames)} frames", "> 0",
          with_gt > 0)
    check("gt_type_labeled", dataset.gt_type or "none", "explicit GT type", dataset.gt_type is not None)
    check("lidar_points_expected", frames[0].num_lidar_points if frames else 0, "> 0",
          bool(frames and frames[0].num_lidar_points > 0))
    policy = load_policy()
    check("quality_policy_loaded", policy.policy_id, "loadable", True)
    passed = all(c["passed"] for c in checks if c["applicable"])
    return {
        "status": "success" if passed else "warning",
        "message": ("Dataset ready for evaluation pipeline."
                    if passed else "Precheck found issues; review the failed gates."),
        "checks": checks,
        "dataset_id": dataset.dataset_id,
    }


@router.post("/labeleval/datasets/generate")
def generate_dataset(req: GenerateDatasetRequest):
    store = get_store()
    dataset = synthetic.generate_dataset(
        store, name=req.name, num_sequences=req.num_sequences,
        frames_per_sequence=req.frames_per_sequence, seed=req.seed,
        version=f"v{len(store.all('datasets')) + 1}",
    )
    pipe = get_pipeline()
    pipe.active_dataset = dataset.dataset_id
    store.save()
    return _dataset_summary(dataset)


@router.post("/labeleval/run")
def run_pipeline(req: RunPipelineRequest):
    pipe = get_pipeline()
    store = get_store()
    if store.get("datasets", req.dataset_id) is None:
        raise HTTPException(404, f"Unknown dataset {req.dataset_id}")
    try:
        run_id = pipe.run(req.dataset_id, background=True)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"run_id": run_id, "status": "started"}


# ------------------------------------------------------------------ pipeline / queue / overview


@router.get("/labeleval/pipeline")
def pipeline_state():
    return _pipeline_state()


@router.get("/queue/status")
def queue_status():
    return get_pipeline().queue.stats()


@router.get("/labeleval/overview")
def overview():
    return reporting.overview(get_store(), get_pipeline())


@router.get("/labeleval/funnel")
def funnel():
    return reporting.funnel(get_store(), get_pipeline().active_dataset)


# ------------------------------------------------------------------ frames / evaluations


@router.get("/labeleval/frames")
def list_frames(dataset_id: Optional[str] = None):
    store = get_store()
    did = dataset_id or get_pipeline().active_dataset
    frames = sorted(store.where("frames", dataset_id=did), key=lambda f: f.index) if did else []
    return {"frame_ids": [f.frame_id for f in frames]}


@router.get("/labeleval/frames/{frame_id}")
def get_frame(frame_id: str):
    store = get_store()
    summary = reporting.frame_summary(store, frame_id)
    if summary is None:
        raise HTTPException(404, f"Unknown frame {frame_id}")
    frame = store.get("frames", frame_id)
    seq = store.get("sequences", frame.sequence_id)
    prev_id = next_id = None
    if seq:
        idx = seq.frame_ids.index(frame_id) if frame_id in seq.frame_ids else -1
        if idx > 0:
            prev_id = seq.frame_ids[idx - 1]
        if 0 <= idx < len(seq.frame_ids) - 1:
            next_id = seq.frame_ids[idx + 1]
    return {"frame": summary, "prev": prev_id, "next": next_id}


@router.get("/labeleval/evaluations/{annotation_id}")
def get_evaluation(annotation_id: str):
    rec = reporting.evaluation_record(get_store(), annotation_id)
    if rec is None:
        raise HTTPException(404, f"Unknown annotation {annotation_id}")
    return rec


@router.get("/labeleval/evaluations")
def list_evaluations(dataset_id: Optional[str] = None, limit: int = 200, offset: int = 0):
    store = get_store()
    did = dataset_id or get_pipeline().active_dataset
    anns = sorted(store.where("annotations", dataset_id=did), key=lambda a: a.annotation_id) if did else []
    total = len(anns)
    out = []
    for a in anns[offset:offset + limit]:
        rec = reporting.evaluation_record(store, a.annotation_id)
        if rec:
            out.append(rec)
    return {"total": total, "records": out}


# ------------------------------------------------------------------ haystack / rare events


@router.get("/labeleval/haystack")
def haystack(dataset_id: Optional[str] = None):
    did = dataset_id or get_pipeline().active_dataset
    return {"points": reporting.haystack(get_store(), did)}


@router.get("/labeleval/rare-events")
def rare_events():
    store = get_store()
    did = get_pipeline().active_dataset
    events = store.where("rare_events", dataset_id=did) if did else store.all("rare_events")
    events = sorted(events, key=lambda e: (-{"critical": 3, "high": 2, "medium": 1, "low": 0}[e.severity], -e.rarity_score))
    return {"events": [e.model_dump() for e in events]}


@router.get("/labeleval/rare-events/{event_id}")
def rare_event(event_id: str):
    e = get_store().get("rare_events", event_id)
    if e is None:
        raise HTTPException(404, f"Unknown rare event {event_id}")
    return e.model_dump()


# ------------------------------------------------------------------ quality


@router.get("/quality/metrics")
def quality_metrics(dataset_id: Optional[str] = None):
    did = dataset_id or get_pipeline().active_dataset
    return reporting.quality_metrics(get_store(), did)


@router.get("/quality/groups")
def quality_groups(dataset_id: Optional[str] = None):
    did = dataset_id or get_pipeline().active_dataset
    if did is None:
        return {"dataset_id": None, "total": 0, "verification_rate": 0.0, "groups": []}
    return reporting.quality_groups(get_store(), did)


@router.get("/quality/groups/{group_id}")
def quality_group_detail(group_id: str):
    detail = reporting.group_detail(get_store(), group_id)
    if detail is None:
        raise HTTPException(404, f"Unknown group {group_id}")
    return detail


@router.get("/labeleval/policy")
def get_policy():
    return load_policy().model_dump()


@router.post("/labeleval/policy")
def set_policy(policy: QualityPolicy):
    save_policy(policy)
    get_store().audit("policy_updated", "QualityPolicy", policy.policy_id, "thresholds updated", "operator")
    return policy.model_dump()


# ------------------------------------------------------------------ regression


@router.get("/regression")
def regression():
    store = get_store()
    entries = sorted(store.all("regressions"), key=lambda r: r.date, reverse=True)
    return {
        "entries": [e.model_dump() for e in entries],
        "current_alert": get_pipeline().regression_alert,
    }


# ------------------------------------------------------------------ review / HITL


@router.get("/review/tasks")
def review_tasks(status: Optional[str] = None):
    store = get_store()
    tasks = sorted(store.all("review_tasks"), key=lambda t: t.created_at, reverse=True)
    if status:
        tasks = [t for t in tasks if t.status == status]
    return {"tasks": [_task_payload(t, with_evidence=False) for t in tasks[:300]]}


def _task_payload(t, with_evidence: bool = True) -> Dict:
    d = t.model_dump()
    d["evidence"] = reporting.evaluation_record(get_store(), t.annotation_id) if with_evidence else None
    return d


@router.get("/review/tasks/{task_id}")
def review_task(task_id: str):
    t = get_store().get("review_tasks", task_id)
    if t is None:
        raise HTTPException(404, f"Unknown review task {task_id}")
    return _task_payload(t)


@router.post("/review/tasks/{task_id}")
def act_on_review_task(task_id: str, req: ReviewActionRequest):
    valid = {"approve", "reject", "correct", "merge_tracks", "split_track", "mark_ignore"}
    if req.action not in valid:
        raise HTTPException(400, f"Invalid action {req.action}; expected one of {sorted(valid)}")
    pipe = get_pipeline()
    try:
        result = pipe.apply_review(task_id, RelabelingAction(**req.model_dump()))
    except KeyError as e:
        raise HTTPException(404, str(e))
    task = result["task"]
    return {
        "task": _task_payload(task),
        "revalidation": reporting.evaluation_record(get_store(), task.annotation_id),
        "message": result["message"],
    }


# ------------------------------------------------------------------ models / training


@router.get("/models")
def list_models():
    store = get_store()
    models = sorted(store.all("models"), key=lambda m: m.created_at)
    out = [m.model_dump() for m in models]
    if not out:
        out = [{
            "model_id": "mdl-bootstrap",
            "model_version": "model-v1",
            "name": "synthlab-detector",
            "trained_on_dataset": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "metrics": {"precision": None, "recall": None, "f1": None, "map_3d": None,
                        "safety_critical_recall": None, "rare_recall": None},
            "regression_status": "baseline",
        }]
    return {"models": out}


@router.post("/train")
def start_training(req: TrainRequest):
    pipe = get_pipeline()
    try:
        job = pipe.start_training(
            dataset_version=req.dataset_version,
            model_version=req.model_version,
            configuration=req.configuration,
            quality_policy=req.quality_policy or "",
            training_parameters=req.training_parameters,
        )
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"job_id": job.job_id, "model_id": job.model_id,
            "model_version": job.model_version, "status": job.status}


@router.get("/train/jobs")
def training_jobs():
    store = get_store()
    jobs = sorted(store.all("training_jobs"), key=lambda j: j.started_at, reverse=True)
    out = []
    for j in jobs:
        d = j.model_dump()
        d["logs"] = d["logs"][-50:]
        out.append(d)
    return {"jobs": out}


@router.get("/train/jobs/{job_id}")
def training_job(job_id: str):
    j = get_store().get("training_jobs", job_id)
    if j is None:
        raise HTTPException(404, f"Unknown training job {job_id}")
    return j.model_dump()


# ------------------------------------------------------------------ anomaly config / benchmark


@router.get("/labeleval/config")
def get_anomaly_config():
    return get_pipeline().anomaly_config


@router.post("/labeleval/config")
def set_anomaly_config(config: Dict):
    pipe = get_pipeline()
    pipe.anomaly_config = config
    store = get_store()
    store.meta["anomaly_config"] = config
    store.audit("anomaly_config_updated", "AnomalyConfig", "singleton",
                f"strategy={config.get('advanced', {}).get('ensemble_strategy')}", "operator")
    store.save()
    return pipe.anomaly_config


@router.post("/benchmark/techniques")
def run_benchmark_techniques(req: BenchmarkTechniquesRequest):
    store = get_store()
    did = req.dataset_id or get_pipeline().active_dataset
    try:
        bench = reporting.benchmark_techniques(store, did, get_pipeline().anomaly_config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    store.save()
    return _bench_payload(bench)


@router.get("/benchmark/techniques")
def latest_benchmark():
    store = get_store()
    benches = sorted(store.all("benchmarks"), key=lambda b: b.created_at)
    if not benches:
        return {"benchmark_id": "", "rows": [], "highlights": {}, "created_at": ""}
    return _bench_payload(benches[-1])


def _bench_payload(bench) -> Dict:
    return {
        "benchmark_id": bench.benchmark_id,
        "rows": [r.model_dump() for r in bench.rows],
        "highlights": bench.highlights,
        "created_at": bench.created_at,
    }


# ------------------------------------------------------------------ process units / alerts / audit


@router.get("/labeleval/process-units")
def process_units():
    return usage_summary(get_store())


@router.get("/labeleval/alerts")
def alerts():
    store = get_store()
    out = sorted(store.all("alerts"), key=lambda a: a.created_at, reverse=True)
    return {"alerts": [{
        "alert_id": a.alert_id,
        "kind": a.kind,
        "severity": a.severity,
        "message": a.message,
        "evidence_link": {"page": a.evidence_page, "id": a.evidence_id},
        "created_at": a.created_at,
    } for a in out]}


@router.get("/labeleval/audit")
def audit(limit: int = 200):
    store = get_store()
    events = sorted(store.all("audit_events"), key=lambda e: e.timestamp, reverse=True)
    return {"events": [e.model_dump() for e in events[:limit]]}


# ------------------------------------------------------------------ copilot (evidence-based)


@router.post("/labeleval/copilot/explain")
def copilot_explain_evidence(req: CopilotRequest):
    return copilot.explain(get_store(), req.model_dump())


# ------------------------------------------------------------------ SSE stream


def _megaeval_status() -> Dict:
    try:
        from sensorflow.megaeval.runs import get_mega_store
        mstore = get_mega_store()
        active = [r.progress_dict() for r in mstore.runs.values()
                  if r.status in ("queued", "running", "reducing", "materializing")]
        return {"active_runs": active, "cache": mstore.router.cache.stats(),
                "total_runs": len(mstore.runs)}
    except Exception:
        return {"active_runs": [], "cache": None, "total_runs": 0}


@router.get("/events/stream")
async def events_stream(ticks: int = 600):
    async def gen():
        for _ in range(max(1, min(ticks, 3600))):  # bounded; client reconnects
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "pipeline": _pipeline_state(),
                "training": _active_training_job(),
                "alerts_count": len(get_store().all("alerts")),
                "megaeval": _megaeval_status(),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
