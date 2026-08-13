"""REST surface for the ROTR capability: APIRouter under /api/rotr.

Request-scoped over file persistence (platform convention); no background
services. Endpoints cover the full loop: bank generation, detection runs,
violation inspection with taxonomy queries, the attribution matrix,
consequence replay, metrics + regression, clusters, HITL actions, flywheel
artifacts + contamination guard, stop-ship policy, and docs serving."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from sensorflow.rotr import SOFTWARE_VERSION, pipeline, store
from sensorflow.rotr import flywheel as flywheel_mod
from sensorflow.rotr.flywheel import ContaminationError
from sensorflow.rotr.metrics import evaluate_candidate
from sensorflow.rotr.models import now_iso
from sensorflow.rotr.rules import CONFIG, RULES, RULESET_VERSION
from sensorflow.rotr.scenes import MODEL_PROFILES
from sensorflow.rotr.stopship import STOPSHIP_POLICY, policy_version
from sensorflow.rotr.taxonomy import ROTRQuery, matches, parse_query

router = APIRouter(prefix="/api/rotr", tags=["rotr"])


@router.get("/health")
def health() -> Dict:
    return {"status": "ok", "software_version": SOFTWARE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "stopship_policy_version": policy_version(),
            "time": now_iso()}


# ------------------------------------------------------------ banks


class BankRequest(BaseModel):
    n_scenarios: int = 28
    seed: int = 7
    model_version: str = "stack-v1"


@router.post("/banks")
def create_bank(req: BankRequest) -> Dict:
    if req.model_version not in MODEL_PROFILES:
        raise HTTPException(422, f"unknown model_version {req.model_version}; "
                                 f"known: {sorted(MODEL_PROFILES)}")
    if not (1 <= req.n_scenarios <= 500):
        raise HTTPException(422, "n_scenarios must be in [1, 500]")
    return pipeline.create_bank(req.n_scenarios, req.seed, req.model_version)


@router.get("/banks")
def list_banks() -> List[Dict]:
    return pipeline.list_banks()


@router.get("/banks/{bank_id}")
def get_bank(bank_id: str) -> Dict:
    doc = pipeline.load_bank(bank_id)
    if doc is None:
        raise HTTPException(404, f"unknown bank {bank_id}")
    return pipeline.bank_summary(doc)


@router.get("/banks/{bank_id}/scenarios/{scenario_id}")
def get_scenario(bank_id: str, scenario_id: str) -> Dict:
    sc = pipeline.load_scenario(bank_id, scenario_id)
    if sc is None:
        raise HTTPException(404, f"unknown scenario {scenario_id}")
    return sc.model_dump()


# ------------------------------------------------------------ runs


class RunRequest(BaseModel):
    bank_id: Optional[str] = None
    n_scenarios: int = 28
    seed: int = 7
    model_version: str = "stack-v1"
    forward_to_agentic: bool = False


@router.post("/runs")
def create_run(req: RunRequest) -> Dict:
    try:
        doc = pipeline.execute_run(req.bank_id, req.n_scenarios, req.seed,
                                   req.model_version,
                                   forward_to_agentic=req.forward_to_agentic)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _run_summary(doc)


@router.get("/runs")
def list_runs() -> List[Dict]:
    return pipeline.list_runs()


def _run_summary(doc: Dict) -> Dict:
    return {k: doc[k] for k in
            ("run_id", "bank_id", "model_version", "ruleset_version",
             "n_scenarios", "scenario_summaries", "metrics", "clusters",
             "gate")} | {"n_violations": len(doc["violations"])}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return _run_summary(doc)


# ------------------------------------------------------------ violations


@router.get("/runs/{run_id}/violations")
def list_violations(run_id: str, actor: Optional[str] = None,
                    vulnerability: Optional[str] = None,
                    legality: Optional[str] = None,
                    environment: Optional[str] = None,
                    interaction: Optional[str] = None,
                    behavior: Optional[str] = None,
                    road_geometry: Optional[str] = None,
                    traffic_control: Optional[str] = None,
                    visibility: Optional[str] = None,
                    lighting: Optional[str] = None,
                    weather: Optional[str] = None,
                    consequence_class: Optional[str] = None,
                    primary_layer: Optional[str] = None,
                    text: Optional[str] = None) -> Dict:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    query = parse_query(text) if text else ROTRQuery()
    explicit = {"actor": actor, "vulnerability": vulnerability,
                "legality": legality, "environment": environment,
                "interaction": interaction, "behavior": behavior,
                "road_geometry": road_geometry,
                "traffic_control": traffic_control, "visibility": visibility,
                "lighting": lighting, "weather": weather,
                "consequence_class": consequence_class,
                "primary_layer": primary_layer}
    for axis, val in explicit.items():
        if val is not None:
            setattr(query, axis, val)
    return _query_run(doc, query)


class QueryRequest(BaseModel):
    run_id: str
    text: Optional[str] = None
    filters: Optional[Dict[str, str]] = None


@router.post("/query")
def structured_query(req: QueryRequest) -> Dict:
    doc = pipeline.load_run(req.run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {req.run_id}")
    query = parse_query(req.text) if req.text else ROTRQuery()
    for k, v in (req.filters or {}).items():
        if not hasattr(query, k):
            raise HTTPException(422, f"unknown filter axis {k!r}")
        setattr(query, k, v)
    return _query_run(doc, query)


def _query_run(doc: Dict, query: ROTRQuery) -> Dict:
    env_by_scenario = {s["scenario_id"]: s for s in doc["scenario_summaries"]}
    cluster_by_vid = {}
    for c in doc["clusters"]:
        for vid in c["member_violation_ids"]:
            cluster_by_vid[vid] = c["cluster_id"]
    results = []
    for v in doc["violations"]:
        attr = doc["attributions"].get(v["violation_id"], {})
        cons = doc["consequences"].get(v["violation_id"], {})
        env = env_by_scenario.get(v["scenario_id"], {})
        if not matches(query, v.get("taxonomy", {}),
                       cons.get("consequence_class"),
                       attr.get("primary_layer"),
                       {"lighting": env.get("lighting"),
                        "weather": env.get("weather")}):
            continue
        results.append({
            "violation_id": v["violation_id"],
            "scenario_id": v["scenario_id"],
            "rule_id": v["rule_id"],
            "rule_version": v["rule_version"],
            "description": v["description"],
            "taxonomy": v.get("taxonomy", {}),
            "evidence": v["evidence"],
            "confidence": v["confidence"],
            "primary_layer": attr.get("primary_layer"),
            "consequence_class": cons.get("consequence_class"),
            "cluster_id": cluster_by_vid.get(v["violation_id"]),
            "environment": {"visibility": env.get("visibility"),
                            "lighting": env.get("lighting"),
                            "weather": env.get("weather")},
            "provenance": v["provenance"],
        })
    return {"query": query.model_dump(), "n_results": len(results),
            "results": results}


@router.get("/runs/{run_id}/violations/{violation_id}")
def get_violation(run_id: str, violation_id: str) -> Dict:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    v = next((v for v in doc["violations"]
              if v["violation_id"] == violation_id), None)
    if v is None:
        raise HTTPException(404, f"unknown violation {violation_id}")
    return {"violation": v,
            "attribution": doc["attributions"].get(violation_id),
            "consequence": doc["consequences"].get(violation_id)}


@router.get("/runs/{run_id}/attribution")
def attribution_matrix(run_id: str) -> Dict:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    rows = []
    for v in doc["violations"]:
        attr = doc["attributions"].get(v["violation_id"], {})
        rows.append({
            "violation_id": v["violation_id"],
            "scenario_id": v["scenario_id"],
            "rule_id": v["rule_id"],
            "primary_layer": attr.get("primary_layer"),
            "layers": attr.get("layers", {}),
            "note": attr.get("note", ""),
        })
    return {"run_id": run_id, "rows": rows,
            "invariant": "downstream failure never auto-implies perception; "
                         "layers are implicated only by their own positive "
                         "evidence"}


@router.get("/runs/{run_id}/violations/{violation_id}/consequence")
def get_consequence(run_id: str, violation_id: str) -> Dict:
    doc = pipeline.load_consequence(run_id, violation_id)
    if doc is None:
        raise HTTPException(404, "no consequence replay for "
                                 f"{violation_id} in {run_id}")
    # Attach the scenario geometry the UI needs for the BEV replay view.
    run = pipeline.load_run(run_id)
    if run:
        v = next((v for v in run["violations"]
                  if v["violation_id"] == violation_id), None)
        if v:
            sc = pipeline.load_scenario(run["bank_id"], v["scenario_id"])
            if sc:
                doc["scenario_geometry"] = {
                    "actual_context": sc.actual_context.model_dump(),
                    "actors": [{
                        "actor_id": a.actor_id, "class_name": a.class_name,
                        "dims": a.dims,
                        "states": [{"t": s.t, "x": s.x, "y": s.y,
                                    "yaw": s.yaw} for s in a.states],
                    } for a in sc.actors],
                }
    return doc


# ------------------------------------------------------------ metrics + regression


@router.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> Dict:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return doc["metrics"]


@router.get("/runs/{run_id}/clusters")
def get_clusters(run_id: str) -> List[Dict]:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return doc["clusters"]


class RegressionRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    seed: int = 11


@router.post("/regression")
def run_regression(req: RegressionRequest) -> Dict:
    base = pipeline.load_run(req.baseline_run_id)
    cand = pipeline.load_run(req.candidate_run_id)
    if base is None or cand is None:
        raise HTTPException(404, "unknown baseline or candidate run")
    rid = f"reg-{req.baseline_run_id}-vs-{req.candidate_run_id}"
    result = evaluate_candidate(rid, base, cand, seed=req.seed)
    store.write_json(result.model_dump(), "regressions", f"{rid}.json")
    return result.model_dump()


@router.get("/regressions")
def list_regressions() -> List[Dict]:
    out = []
    for name in store.list_dir("regressions"):
        doc = store.read_json("regressions", name)
        if doc:
            out.append(doc)
    return out


# ------------------------------------------------------------ HITL + flywheel


@router.get("/runs/{run_id}/hitl")
def hitl_queue(run_id: str) -> List[Dict]:
    if pipeline.load_run(run_id) is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return flywheel_mod.get_queue(run_id)


class HITLActionRequest(BaseModel):
    run_id: str
    review_id: str
    action: str                       # VALIDATE | REJECT
    actor: str
    notes: str = ""


@router.post("/hitl/action")
def hitl_action(req: HITLActionRequest) -> Dict:
    try:
        return flywheel_mod.act(req.run_id, req.review_id, req.action,
                                req.actor, req.notes)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/flywheel/suite")
def regression_suite() -> Dict:
    return flywheel_mod.get_suite()


@router.get("/flywheel/candidates")
def training_candidates() -> List[Dict]:
    return flywheel_mod.list_candidates()


class PromoteRequest(BaseModel):
    candidate_id: str
    actor: str


@router.post("/flywheel/promote")
def promote(req: PromoteRequest) -> Dict:
    try:
        return flywheel_mod.promote_to_training(req.candidate_id, req.actor)
    except ContaminationError as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))


class OverrideRequest(BaseModel):
    candidate_id: str
    actor: str
    reason: str


@router.post("/flywheel/override")
def override(req: OverrideRequest) -> Dict:
    try:
        return flywheel_mod.governance_override(req.candidate_id, req.actor,
                                                req.reason)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


# ------------------------------------------------------------ policy + docs


@router.get("/stopship/policy")
def stopship_policy() -> Dict:
    return {"policy": STOPSHIP_POLICY, "policy_version": policy_version(),
            "note": "deterministic conjunction; explicitly NOT LLM-driven"}


@router.get("/runs/{run_id}/gate")
def run_gate(run_id: str) -> Dict:
    doc = pipeline.load_run(run_id)
    if doc is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return doc["gate"]


@router.get("/rules")
def list_rules() -> Dict:
    return {"ruleset_version": RULESET_VERSION,
            "config": CONFIG,
            "config_note": "thresholds are ILLUSTRATIVE (synthetic substrate)",
            "rules": [{"rule_id": rid, "description": desc}
                      for rid, desc, _fn in RULES]}


@router.get("/docs/architecture")
def architecture_doc() -> Dict:
    path = Path(__file__).resolve().parents[2] / "docs" / "architecture" / \
        "rotr-architecture.md"
    if not path.exists():
        raise HTTPException(404, "architecture doc missing")
    return {"path": str(path), "markdown": path.read_text()}
