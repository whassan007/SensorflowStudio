"""FastAPI router for the Agentic Retrospective Safety Analyzer.

Mounted at /api/retro by app_backend.py. Artifacts persist under runs/retro/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from sensorflow.retro import store
from sensorflow.retro.agent.orchestrator import analyze
from sensorflow.retro.inference.client import all_backend_statuses
from sensorflow.retro.inference.compat import check_vllm_compatibility, format_report
from sensorflow.retro.inference.env_detect import detect_environment
from sensorflow.retro.rag.evalharness import run_retrieval_eval
from sensorflow.retro.rag.retriever import get_index
from sensorflow.retro.tools.builtin import FIXTURES_DIR, build_registry

router = APIRouter(prefix="/api/retro", tags=["retro"])


@router.get("/env")
def env_report():
    env = detect_environment()
    return {"status": "ok", "environment": env.model_dump()}


@router.get("/compat")
def compat_report(model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                  quantization: str = "none"):
    env = detect_environment(probe_ollama=False)
    report = check_vllm_compatibility(env, model_name=model_name,
                                      quantization=quantization)
    return {"status": "ok", "report": report.model_dump(),
            "formatted": format_report(report)}


@router.get("/backends")
def backends_status():
    return {"status": "ok",
            "backends": [s.model_dump() for s in all_backend_statuses()]}


@router.get("/rag/search")
def rag_search(q: str = Query(min_length=3), k: int = Query(4, ge=1, le=10)):
    index = get_index()
    hits = index.search(q, k=k)
    return {"status": "ok", "store_backend": index.backend_name,
            "hits": [h.model_dump() for h in hits]}


@router.get("/rag/eval")
def rag_eval(k: int = Query(4, ge=1, le=10)):
    return {"status": "ok", "report": run_retrieval_eval(k=k).model_dump()}


@router.get("/fixtures")
def list_fixtures():
    out = []
    for p in sorted(Path(FIXTURES_DIR).glob("*.json")):
        try:
            data = json.loads(p.read_text())
            out.append({
                "fixture_id": p.stem,
                "evaluation_id": data.get("evaluation_id"),
                "description": (data.get("scenario") or {}).get("description", ""),
                "weather": (data.get("scenario") or {}).get("weather"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return {"status": "ok", "fixtures": out}


@router.post("/analyze")
def analyze_failure(fixture_id: Optional[str] = None,
                    backend: str = Query("mock", pattern="^(mock|ollama|vllm)$"),
                    log: Optional[Dict[str, Any]] = Body(None)):
    """Analyze a fixture by id, or an uploaded JSON eval log in the body."""
    if not fixture_id and not log:
        raise HTTPException(status_code=400,
                            detail="provide ?fixture_id= or a JSON log body")
    try:
        if fixture_id:
            scorecard = analyze(fixture_id=fixture_id, backend=backend)
        else:
            eval_id = log.get("evaluation_id", "uploaded")
            safe = "".join(c for c in eval_id if c.isalnum() or c in "-_")[:64]
            upload_path = store.uploads_dir() / f"{safe or 'uploaded'}.json"
            upload_path.write_text(json.dumps(log, indent=2))
            scorecard = analyze(path=str(upload_path), backend=backend)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:  # backend unavailable
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "ok",
            "scorecard": json.loads(scorecard.model_dump_json()),
            "markdown": scorecard.render_markdown()}


@router.get("/analyses")
def list_analyses():
    return {"status": "ok", "analyses": store.list_analyses()}


@router.get("/analyses/{evaluation_id}")
def get_analysis(evaluation_id: str):
    data = store.load_analysis(evaluation_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=f"no analysis for {evaluation_id}")
    return {"status": "ok", **data}


@router.get("/analyses/{evaluation_id}/audit")
def get_audit(evaluation_id: str):
    data = store.load_analysis(evaluation_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=f"no analysis for {evaluation_id}")
    audit_id = data.get("audit_analysis_id", evaluation_id)
    return {"status": "ok", "audit_analysis_id": audit_id,
            "records": store.read_audit(audit_id)}


@router.get("/tools")
def tool_registry_listing():
    reg = build_registry(persist_audit=False)
    return {"status": "ok",
            "tools": [s.model_dump() for s in reg.specs()],
            "note": "write tools require policy_authorization=true per call; "
                    "all calls are audited"}
