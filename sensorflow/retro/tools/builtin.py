"""Built-in analysis tools registered on the audited registry.

All tools are read-only except CreateEvaluationCaseTool, which requires an
explicit policy_authorization flag per call (enforced by the registry).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from sensorflow.retro import metrics, store
from sensorflow.retro.rag.embedder import HashedTfEmbedder
from sensorflow.retro.rag.retriever import get_index
from sensorflow.retro.tools.registry import ToolRegistry

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# Dotted paths the analyzer needs; absent ones become UNKNOWN, never guessed.
REQUIRED_FIELDS = [
    "ego.speed_mps",
    "ground_truth.class",
    "ground_truth.distance_m",
    "ground_truth.relative_velocity_mps",
    "planner_response",
    "counterfactual_planner_response",
    "timing.detection_latency_ms",
    "timing.planner_latency_ms",
    "scenario.weather",
    "models.baseline",
    "models.candidate",
    "collision",
    "traffic_context.following_vehicle",
]


def _dotted_get(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] is None:
            return None
        cur = cur[part]
    return cur


# ------------------------------------------------------------- LogReaderTool

class LogReaderInput(BaseModel):
    fixture_id: Optional[str] = Field(
        None, description="fixture name (stem) under sensorflow/retro/fixtures")
    path: Optional[str] = Field(
        None, description="path under the approved allowlist "
                          "(runs/retro/uploads or the fixtures dir)")


class LogReaderOutput(BaseModel):
    source_path: str
    log: Dict[str, Any]
    present_fields: List[str]
    missing_fields: List[str]


def _allowed_roots() -> List[Path]:
    return [FIXTURES_DIR.resolve(), store.uploads_dir().resolve()]


def log_reader(inp: LogReaderInput) -> LogReaderOutput:
    if not inp.fixture_id and not inp.path:
        raise ValueError("provide fixture_id or path")
    if inp.fixture_id:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", inp.fixture_id):
            raise PermissionError(f"invalid fixture id {inp.fixture_id!r}")
        target = (FIXTURES_DIR / f"{inp.fixture_id}.json").resolve()
    else:
        target = Path(inp.path).resolve()
    if not any(str(target).startswith(str(root) + "/") or target.parent == root
               for root in _allowed_roots()):
        raise PermissionError(
            f"path {target} is outside the approved artifact allowlist "
            f"({[str(r) for r in _allowed_roots()]})")
    if not target.exists():
        raise FileNotFoundError(f"artifact not found: {target}")
    log = json.loads(target.read_text())
    present, missing = [], []
    for fieldpath in REQUIRED_FIELDS:
        (present if _dotted_get(log, fieldpath) is not None else missing).append(fieldpath)
    return LogReaderOutput(source_path=str(target), log=log,
                           present_fields=present, missing_fields=missing)


# ------------------------------------------------------ SafetyStandardRAGTool

class RAGSearchInput(BaseModel):
    query: str = Field(min_length=3)
    k: int = Field(4, ge=1, le=10)


class RAGSearchOutput(BaseModel):
    store_backend: str
    hits: List[Dict[str, Any]]


def rag_search(inp: RAGSearchInput) -> RAGSearchOutput:
    index = get_index()
    hits = index.search(inp.query, k=inp.k)
    return RAGSearchOutput(store_backend=index.backend_name,
                           hits=[h.model_dump() for h in hits])


# ------------------------------------------------------- MetricCalculatorTool

class MetricCalcInput(BaseModel):
    operation: str = Field(description="stopping_distance | ttc | scr_impact | "
                                       "behavioral_impact")
    params: Dict[str, Any] = Field(default_factory=dict)


class MetricCalcOutput(BaseModel):
    operation: str
    result: Dict[str, Any]


def metric_calculator(inp: MetricCalcInput) -> MetricCalcOutput:
    ops = {
        "stopping_distance": lambda p: metrics.stopping_distance(**p),
        "ttc": lambda p: metrics.time_to_collision(**p),
        "scr_impact": lambda p: metrics.scr_impact(p.get("evaluation_context")),
        "behavioral_impact": lambda p: metrics.behavioral_impact(
            p.get("observed"), p.get("counterfactual")),
    }
    if inp.operation not in ops:
        raise ValueError(f"unknown operation {inp.operation!r}; "
                         f"expected one of {sorted(ops)}")
    result = ops[inp.operation](inp.params)
    return MetricCalcOutput(operation=inp.operation, result=result.model_dump())


# ------------------------------------------------ HistoricalFailureSearchTool

class HistoricalSearchInput(BaseModel):
    query_text: str = Field(min_length=3)
    k: int = Field(3, ge=1, le=10)
    exclude_evaluation_id: Optional[str] = None


class HistoricalSearchOutput(BaseModel):
    matches: List[Dict[str, Any]]
    corpus_size: int


def historical_search(inp: HistoricalSearchInput) -> HistoricalSearchOutput:
    """Similarity over previously analyzed failures (persisted scorecards)."""
    embedder = HashedTfEmbedder()
    docs, meta = [], []
    for entry in store.list_analyses():
        eid = entry.get("evaluation_id")
        if not eid or eid == inp.exclude_evaluation_id:
            continue
        full = store.load_analysis(eid) or {}
        sc = full.get("scorecard", {})
        text = " ".join([
            str(sc.get("failure_type", "")),
            str(sc.get("behavioral_consequence", "")),
            json.dumps(sc.get("scenario", {})),
            " ".join(h.get("hypothesis", "") for h in sc.get("root_cause_hypotheses", [])),
        ])
        docs.append(text)
        meta.append(entry)
    if not docs:
        return HistoricalSearchOutput(matches=[], corpus_size=0)
    mat = embedder.embed(docs)
    q = embedder.embed([inp.query_text])[0]
    sims = mat @ q
    order = np.argsort(-sims)[:inp.k]
    matches = [{**meta[i], "similarity": round(float(np.clip(sims[i], 0, 1)), 4)}
               for i in order]
    return HistoricalSearchOutput(matches=matches, corpus_size=len(docs))


# -------------------------------------------------- DistributionAnalysisTool

class DistributionInput(BaseModel):
    distribution_shift: Optional[Dict[str, Any]] = Field(
        None, description="slice-level shift data from the eval artifact")
    run_id: Optional[str] = Field(
        None, description="megaeval run id — when provided and resolvable, the "
                          "analysis delegates to sensorflow.megaeval.analysis")


class DistributionOutput(BaseModel):
    source: str
    findings: List[str]
    shift: Optional[Dict[str, Any]] = None


def distribution_analysis(inp: DistributionInput) -> DistributionOutput:
    if inp.run_id:
        # Delegate to the platform's megaeval statistics when a real run exists.
        try:
            from sensorflow.megaeval.analysis import distribution_shift
            from sensorflow.megaeval.runs import get_mega_store
            store = get_mega_store()
            run = store.runs.get(inp.run_id)
            if run is None:
                raise KeyError(f"unknown megaeval run {inp.run_id!r}")
            if run.status != "published":
                raise ValueError(
                    f"megaeval run {inp.run_id!r} not published "
                    f"(status={run.status}); artifacts unavailable")
            res = distribution_shift(store, run)
            shifts = res.get("shifts", [])
            findings = [f"megaeval shift analysis for run {inp.run_id}: "
                        f"{len(shifts)} shifted cohort(s) above threshold"]
            for s in shifts[:3]:
                findings.append(
                    f"cohort {s['cohort']}: train share {s['train_share']:.4f} "
                    f"-> eval share {s['eval_share']:.4f} "
                    f"(rel {s['relative_change']:+.2f}), cohort recall "
                    f"{s['cohort_recall']} vs overall {s['overall_recall']}")
            return DistributionOutput(
                source="sensorflow.megaeval.analysis.distribution_shift",
                findings=findings,
                shift=res)
        except Exception as exc:
            return DistributionOutput(
                source="megaeval-delegation-failed",
                findings=[f"could not delegate to megaeval ({exc}); no "
                          "distribution claim is made"], shift=None)
    ds = inp.distribution_shift
    if not ds:
        return DistributionOutput(
            source="none",
            findings=["no distribution data in artifact -> distribution shift "
                      "UNKNOWN"], shift=None)
    findings = []
    cand, base = ds.get("candidate_fn_rate", ds.get("candidate_fp_rate")), \
        ds.get("baseline_fn_rate", ds.get("baseline_fp_rate"))
    if cand is not None and base is not None and base > 0:
        ratio = cand / base
        findings.append(
            f"slice '{ds.get('slice', 'unknown')}': error rate {base:.3f} -> "
            f"{cand:.3f} ({ratio:.2f}x) per the evaluation artifact")
        if ratio > 1.3:
            findings.append("relative increase exceeds 1.3x: concentrated "
                            "regression in this slice rather than uniform noise")
    else:
        findings.append("artifact shift block lacks comparable rates -> UNKNOWN")
    return DistributionOutput(source="artifact-provided slice rates",
                              findings=findings, shift=ds)


# ------------------------------------------------- CreateEvaluationCaseTool

class CreateCaseInput(BaseModel):
    title: str = Field(min_length=5)
    description: str = Field(min_length=10)
    scenario_tags: List[str] = Field(default_factory=list)
    source_evaluation_id: str


class CreateCaseOutput(BaseModel):
    case_id: str
    path: str
    created_at: str


def create_evaluation_case(inp: CreateCaseInput) -> CreateCaseOutput:
    """THE write tool. The registry refuses this call without an explicit
    policy_authorization=True flag; every call is audited."""
    case_id = f"case-{uuid.uuid4().hex[:10]}"
    created = datetime.now(timezone.utc).isoformat()
    payload = {"case_id": case_id, "created_at": created, **inp.model_dump()}
    path = store.eval_cases_dir() / f"{case_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return CreateCaseOutput(case_id=case_id, path=str(path), created_at=created)


# --------------------------------------------------------------- registration

def build_registry(analysis_id: Optional[str] = None,
                   persist_audit: bool = True) -> ToolRegistry:
    reg = ToolRegistry(analysis_id=analysis_id, persist_audit=persist_audit)
    reg.register("log_reader", log_reader, LogReaderInput, LogReaderOutput,
                 "Read an approved local failure artifact (allowlisted to the "
                 "fixtures dir and runs/retro/uploads); returns the parsed log "
                 "plus present/missing required fields.",
                 read_only=True, timeout_s=5.0)
    reg.register("safety_standard_rag", rag_search, RAGSearchInput, RAGSearchOutput,
                 "Retrieve safety-case requirements from the (synthetic, "
                 "clearly-labeled) corpus with source/version/section/"
                 "relevance-score metadata.",
                 read_only=True, timeout_s=20.0)
    reg.register("metric_calculator", metric_calculator, MetricCalcInput,
                 MetricCalcOutput,
                 "Deterministic safety metrics: stopping_distance (reaction/"
                 "latency/friction/grade/decel parameterized), ttc (SSAM "
                 "rectangle projection with validity flags), scr_impact "
                 "(policy-defined criticality), behavioral_impact (observed vs "
                 "corrected-perception planner response).",
                 read_only=True, timeout_s=10.0)
    reg.register("historical_failure_search", historical_search,
                 HistoricalSearchInput, HistoricalSearchOutput,
                 "Similarity search over previously analyzed failures "
                 "(persisted retro scorecards).",
                 read_only=True, timeout_s=10.0)
    reg.register("distribution_analysis", distribution_analysis,
                 DistributionInput, DistributionOutput,
                 "Distribution-shift findings: delegates to megaeval stats for "
                 "real run ids, else analyzes artifact-provided slice rates.",
                 read_only=True, timeout_s=15.0)
    reg.register("create_evaluation_case", create_evaluation_case,
                 CreateCaseInput, CreateCaseOutput,
                 "WRITE TOOL: persist a new evaluation case candidate derived "
                 "from this retrospective. Requires explicit policy "
                 "authorization on every call; fully audited.",
                 read_only=False, timeout_s=5.0)
    return reg
