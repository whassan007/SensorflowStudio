"""FastAPI router for the rare-event miner (included by app_backend).

All state persists under runs/raremine/. The optional LLM narrative follows
the sensorflow.evaluation.copilot pattern: try local Ollama, always fall back
to a deterministic evidence-derived analysis so nothing requires a model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sensorflow.raremine import curator_metrics, dedup as dedup_mod, lineage as lineage_mod
from sensorflow.raremine import pipeline as pipe_mod
from sensorflow.raremine import quantval, scenes as scenes_mod
from sensorflow.raremine.lineage import LeakageError
from sensorflow.raremine.models import DESTINATIONS, get_store

router = APIRouter(prefix="/api/raremine")


class GenerateScenesRequest(BaseModel):
    n: int = Field(60, ge=10, le=2000)
    seed: int = 7


class MineRequest(BaseModel):
    bank_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    diversity_budget: int = Field(12, ge=1, le=200)


class ReviewRequest(BaseModel):
    action: str  # approve | reject
    note: str = ""
    reviewer: str = "human-reviewer"
    destination: Optional[str] = None


class OverrideRequest(BaseModel):
    track_candidate_id: str
    actor: str
    reason: str


class PromoteRequest(BaseModel):
    track_candidate_id: str
    curator: str = "human-curator"


def _bank_or_404(bank_id: Optional[str] = None):
    store = get_store()
    if bank_id:
        bank = store.get("banks", bank_id)
        if bank is None:
            raise HTTPException(404, f"unknown scene bank {bank_id}")
        return bank
    banks = store.all("banks")
    if not banks:
        raise HTTPException(404, "no scene bank generated yet")
    return sorted(banks, key=lambda b: b.created_at)[-1]


def _tc_or_404(track_candidate_id: str):
    tc = get_store().get("track_candidates", track_candidate_id)
    if tc is None:
        raise HTTPException(404, f"unknown track candidate {track_candidate_id}")
    return tc


def _tc_summary(tc) -> Dict[str, Any]:
    rep = tc.representative
    return {
        "track_candidate_id": tc.track_candidate_id,
        "sequence_id": tc.sequence_id,
        "track_id": tc.track_id,
        "frame_count": tc.frame_count,
        "duration_frames": tc.duration_frames,
        "stage": tc.stage,
        "destination": tc.destination,
        "diversity_selected": tc.diversity_selected,
        "duplicate_of": tc.duplicate_of,
        "max_difficulty": tc.max_difficulty,
        "max_visibility": tc.max_visibility,
        "representative_frames": tc.representative_frames.model_dump(),
        "auto_validation": tc.auto_validation,
        "human_validation": tc.human_validation,
        "candidate": rep.model_dump(),
    }


# ------------------------------------------------------------------ scene bank


@router.post("/scenes/generate")
def generate_scenes(req: GenerateScenesRequest):
    store = get_store()
    bank = scenes_mod.generate_scene_bank(store, n_scenes=req.n, seed=req.seed)
    return {"status": "ok", "bank": bank.model_dump()}


@router.get("/status")
def status():
    store = get_store()
    banks = sorted(store.all("banks"), key=lambda b: b.created_at)
    runs = sorted(store.all("runs"), key=lambda r: r.created_at)
    tcs = store.all("track_candidates")
    priority_hist: Dict[str, int] = {}
    for t in tcs:
        if t.duplicate_of is None and t.representative.edge_case_detected:
            p = t.representative.curation_priority
            priority_hist[p] = priority_hist.get(p, 0) + 1
    return {
        "bank": banks[-1].model_dump() if banks else None,
        "last_run": runs[-1].model_dump() if runs else None,
        "num_track_candidates": len([t for t in tcs if t.duplicate_of is None]),
        "num_detected": sum(1 for t in tcs
                            if t.duplicate_of is None and t.representative.edge_case_detected),
        "priority_histogram": priority_hist,
        "dedup_report": (store.meta.get("last_run") or {}).get("dedup_report"),
        "diversity_report": (store.meta.get("last_run") or {}).get("diversity_report"),
    }


# ------------------------------------------------------------------ mining


@router.post("/mine")
def mine(req: MineRequest):
    store = get_store()
    bank = _bank_or_404(req.bank_id)
    run = pipe_mod.run_full_pipeline(store, bank.bank_id, config=req.config or None,
                                     diversity_budget=req.diversity_budget)
    return {"status": "ok", "run": run.model_dump(),
            "dedup_report": store.meta["last_run"]["dedup_report"],
            "diversity_report": store.meta["last_run"]["diversity_report"]}


@router.get("/candidates")
def list_candidates(priority: Optional[str] = None, status: Optional[str] = None,
                    costume: Optional[str] = None, difficulty: Optional[str] = None,
                    detected_only: bool = True, bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    items = pipe_mod.review_queue(store, bank.bank_id, priority=priority,
                                  status=status, costume=costume, difficulty=difficulty)
    if detected_only:
        items = [t for t in items if t.representative.edge_case_detected]
    return {"count": len(items), "candidates": [_tc_summary(t) for t in items]}


@router.get("/candidates/{track_candidate_id}")
def get_candidate(track_candidate_id: str):
    store = get_store()
    tc = _tc_or_404(track_candidate_id)
    frames = [store.get("candidates", cid) for cid in tc.frame_candidate_ids]
    rec = lineage_mod.get_lineage(store, track_candidate_id)
    return {
        **_tc_summary(tc),
        "frame_candidates": [f.model_dump() for f in frames if f is not None],
        "lineage": rec.model_dump() if rec else None,
    }


@router.get("/candidates/{track_candidate_id}/scene")
def candidate_scene(track_candidate_id: str, scene_id: Optional[str] = None):
    """Schematic scene data for the BEV/2D canvas: object positions + bboxes,
    with the candidate object flagged. GT overlay only when GT is available."""
    store = get_store()
    tc = _tc_or_404(track_candidate_id)
    sid = scene_id or tc.representative.scene_id
    scene = store.get("scenes", sid)
    if scene is None:
        raise HTTPException(404, f"unknown scene {sid}")
    objects = []
    for obj in scene.objects:
        entry = {
            "object_id": obj.object_id,
            "is_candidate": obj.track_id == tc.track_id,
            "position": obj.position,
            "distance_m": obj.distance_m,
            "bbox_2d": obj.bbox_2d,
            "context": obj.context,
        }
        if scene.modalities.get("gt_annotations"):
            gt = next((g for g in scene.gt_boxes if g.object_id == obj.object_id), None)
            entry["gt"] = gt.model_dump() if gt else None
        objects.append(entry)
    preds = [p.model_dump(exclude={"planted_failure", "iou_with_gt"})
             for p in scene.baseline_predictions] if scene.modalities.get("baseline_predictions") else None
    return {
        "scene_id": scene.scene_id,
        "sequence_id": scene.sequence_id,
        "frame_index": scene.frame_index,
        "lighting": scene.lighting,
        "weather": scene.weather,
        "modalities": scene.modalities,
        "objects": objects,
        "baseline_predictions": preds,
    }


@router.get("/tracks")
def tracks_view(bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    tcs = [t for t in store.where("track_candidates", bank_id=bank.bank_id)
           if t.duplicate_of is None and t.frame_count > 1]
    tcs.sort(key=lambda t: -t.frame_count)
    out = []
    for t in tcs:
        frames = [store.get("candidates", cid) for cid in t.frame_candidate_ids]
        out.append({
            **_tc_summary(t),
            "frames": [{
                "candidate_id": f.candidate_id,
                "scene_id": f.scene_id,
                "frame_index": f.frame_index,
                "edge_case_detected": f.edge_case_detected,
                "confidence_rare_event": f.confidence_rare_event,
                "evidence_quality": f.evidence_quality,
                "perception_difficulty": f.perception_difficulty,
                "failure_observed": bool(f.observed_model_behavior
                                         and f.observed_model_behavior.failure_observed),
            } for f in frames if f is not None],
        })
    return {"count": len(out), "tracks": out}


@router.get("/dedup/report")
def dedup_report():
    store = get_store()
    report = (store.meta.get("last_run") or {}).get("dedup_report")
    if report is None:
        raise HTTPException(404, "no mining run yet")
    return report


@router.get("/diversity/report")
def diversity_report():
    store = get_store()
    report = (store.meta.get("last_run") or {}).get("diversity_report")
    if report is None:
        raise HTTPException(404, "no mining run yet")
    return report


# ------------------------------------------------------------------ review + governance


@router.post("/review/{track_candidate_id}")
def review(track_candidate_id: str, req: ReviewRequest):
    store = get_store()
    _tc_or_404(track_candidate_id)
    if req.destination is not None and req.destination not in DESTINATIONS:
        raise HTTPException(400, f"invalid destination {req.destination}")
    try:
        tc = pipe_mod.human_review(store, track_candidate_id, req.action,
                                   note=req.note, reviewer=req.reviewer,
                                   destination=req.destination)
    except LeakageError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    rec = lineage_mod.get_lineage(store, track_candidate_id)
    return {"status": "ok", **_tc_summary(tc),
            "lineage": rec.model_dump() if rec else None}


@router.post("/governance/promote-training")
def promote_training(req: PromoteRequest):
    """Attempt to route an example to TRAINING_CANDIDATE. Protected evaluation
    examples are refused (403) unless a governance override exists."""
    store = get_store()
    _tc_or_404(req.track_candidate_id)
    try:
        rec = lineage_mod.promote_to_training(store, req.track_candidate_id,
                                              curator=req.curator)
    except LeakageError as e:
        raise HTTPException(403, str(e))
    store.save()
    return {"status": "ok", "lineage": rec.model_dump()}


@router.post("/governance/override")
def governance_override(req: OverrideRequest):
    store = get_store()
    _tc_or_404(req.track_candidate_id)
    try:
        rec = lineage_mod.governance_override(store, req.track_candidate_id,
                                              actor=req.actor, reason=req.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    store.save()
    return {"status": "ok", "lineage": rec.model_dump()}


@router.get("/destinations")
def destinations(bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    return pipe_mod.destinations_report(store, bank.bank_id)


@router.get("/lineage/{track_candidate_id}")
def lineage(track_candidate_id: str):
    store = get_store()
    _tc_or_404(track_candidate_id)
    rec = lineage_mod.get_lineage(store, track_candidate_id)
    if rec is None:
        raise HTTPException(404, "no lineage record")
    return rec.model_dump()


@router.get("/lineage")
def lineage_all(bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    return lineage_mod.lineage_report(store, bank.bank_id)


# ------------------------------------------------------------------ reports


@router.get("/reports/quantval")
def quantval_report(bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    return quantval.quantitative_report(store, bank.bank_id)


@router.get("/reports/curator")
def curator_report(bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    return curator_metrics.curator_report(store, bank.bank_id)


@router.get("/reports/improvement")
def improvement(bank_id: Optional[str] = None):
    store = get_store()
    bank = _bank_or_404(bank_id)
    return curator_metrics.improvement_report(store, bank.bank_id)


# ------------------------------------------------------------------ optional LLM narrative


def _deterministic_narrative(summary: Dict[str, Any]) -> str:
    c = summary["candidate"]
    lines = [f"# Candidate {summary['track_candidate_id']}", ""]
    lines.append(f"Proposed event: {c['event_type']} "
                 f"({', '.join(c['costume_type']) or 'costume family undetermined'})")
    lines.append("")
    lines.append("## Separate confidences (never combined)")
    lines.append(f"- human identity: {c['confidence_human_identity']:.2f}")
    lines.append(f"- costume present: {c['confidence_costume']:.2f}")
    lines.append(f"- rare event: {c['confidence_rare_event']:.2f}")
    lines.append("")
    lines.append("## Evidence")
    for ev in c["visual_evidence"] + c["human_identity_evidence"]:
        lines.append(f"- [{ev['modality']}] {ev['description']}")
    lines.append("")
    lines.append("## Alternative hypotheses")
    for a in c["alternative_hypotheses"]:
        lines.append(f"- {a['hypothesis']} ({a['status']}, {a['confidence']:.2f}): {a['reason']}")
    lines.append("")
    lines.append(f"Priority {c['curation_priority']}: {c['priority_reason']}")
    lines.append("")
    lines.append("_This is a PROPOSAL derived from sensor evidence rules. It is not ground "
                 "truth; validation measures it, humans confirm it, statistics decide its "
                 "importance, and training usage is governed separately._")
    return "\n".join(lines)


@router.post("/candidates/{track_candidate_id}/explain")
def explain_candidate(track_candidate_id: str):
    """Copilot-pattern narrative: local Ollama when reachable, deterministic
    evidence-based analysis otherwise. Advisory only."""
    tc = _tc_or_404(track_candidate_id)
    summary = _tc_summary(tc)
    offline = _deterministic_narrative(summary)
    try:
        from sensorflow.evaluation.copilot import OLLAMA_ENDPOINTS  # read-only reuse
    except Exception:
        OLLAMA_ENDPOINTS = []
    prompt = (
        "You are an advisory perception-QA copilot. A rule-based miner proposed this "
        "rare-event candidate (costumed pedestrian). Explain the evidence and risks; do NOT "
        "change any confidences or decisions.\n\nCANDIDATE (JSON):\n"
        + json.dumps(summary["candidate"], indent=2)[:6000]
    )
    for ep in OLLAMA_ENDPOINTS:
        try:
            res = httpx.post(ep["url"], json={
                "model": ep["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }, timeout=15.0)
            if res.status_code == 200:
                text = res.json().get("message", {}).get("content", "")
                if text:
                    return {"status": "ok", "provider": ep["url"], "analysis": text}
        except Exception:
            continue
    return {"status": "ok", "provider": "offline_deterministic", "analysis": offline}
