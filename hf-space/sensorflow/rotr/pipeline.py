"""Run orchestration: bank -> detection -> attribution -> consequence ->
metrics -> clusters -> gate -> HITL queue, persisted under runs/rotr/.

Deterministic end to end: run ids derive from bank ids, which derive from
(n_scenarios, seed, model_version, generator version)."""

from __future__ import annotations

from typing import Dict, List, Optional

from sensorflow.rotr import store
from sensorflow.rotr.attribution import attribute
from sensorflow.rotr.consequence import classify
from sensorflow.rotr.flywheel import build_queue
from sensorflow.rotr.metrics import compute_metrics
from sensorflow.rotr.models import ROTRScenario
from sensorflow.rotr.rules import RULESET_VERSION, detect
from sensorflow.rotr.scenes import bank_id_for, generate_bank
from sensorflow.rotr.stopship import evaluate_gate
from sensorflow.rotr.taxonomy import build_clusters, signature

KIND_VULNERABILITY = {"fail_yield_pedestrian": "VRU"}


# ------------------------------------------------------------ banks


def create_bank(n_scenarios: int = 28, seed: int = 7,
                model_version: str = "stack-v1") -> Dict:
    scenarios = generate_bank(n_scenarios, seed, model_version)
    bank_id = scenarios[0].bank_id
    doc = {
        "bank_id": bank_id, "n_scenarios": n_scenarios, "seed": seed,
        "model_version": model_version,
        "scenarios": [s.model_dump() for s in scenarios],
    }
    store.write_json(doc, "banks", f"{bank_id}.json")
    return bank_summary(doc)


def bank_summary(doc: Dict) -> Dict:
    return {
        "bank_id": doc["bank_id"], "n_scenarios": doc["n_scenarios"],
        "seed": doc["seed"], "model_version": doc["model_version"],
        "scenarios": [{
            "scenario_id": s["scenario_id"],
            "description": s["description"],
            "kind": s["planted"]["kind"],
            "committed": s["planted"]["committed"],
            "is_violation_opportunity": s["planted"]["is_violation_opportunity"],
            "environment": s["environment"],
        } for s in doc["scenarios"]],
    }


def load_bank(bank_id: str) -> Optional[Dict]:
    return store.read_json("banks", f"{bank_id}.json")


def list_banks() -> List[Dict]:
    out = []
    for name in store.list_dir("banks"):
        doc = store.read_json("banks", name)
        if doc:
            out.append({k: doc[k] for k in
                        ("bank_id", "n_scenarios", "seed", "model_version")})
    return out


def load_scenario(bank_id: str, scenario_id: str) -> Optional[ROTRScenario]:
    doc = load_bank(bank_id)
    if not doc:
        return None
    for s in doc["scenarios"]:
        if s["scenario_id"] == scenario_id:
            return ROTRScenario.model_validate(s)
    return None


# ------------------------------------------------------------ runs


def execute_run(bank_id: Optional[str] = None, n_scenarios: int = 28,
                seed: int = 7, model_version: str = "stack-v1",
                forward_to_agentic: bool = False) -> Dict:
    if bank_id is None:
        bank_id = bank_id_for(n_scenarios, seed, model_version)
        if not store.exists("banks", f"{bank_id}.json"):
            create_bank(n_scenarios, seed, model_version)
    doc = load_bank(bank_id)
    if doc is None:
        raise KeyError(f"unknown bank {bank_id}")
    scenarios = [ROTRScenario.model_validate(s) for s in doc["scenarios"]]
    run_id = f"run-{bank_id}"

    violations, attributions, consequences, gate_items = [], {}, {}, []
    scenario_summaries = []
    for sc in scenarios:
        vs = detect(sc)
        scenario_summaries.append({
            "scenario_id": sc.scenario_id, "kind": sc.planted.kind,
            "committed": sc.planted.committed,
            "is_violation_opportunity": sc.planted.is_violation_opportunity,
            "expected_rule_id": sc.planted.expected_rule_id,
            "cause_layer": sc.planted.cause_layer,
            "vulnerability": KIND_VULNERABILITY.get(sc.planted.kind, "NON_VRU"),
            "visibility": sc.environment.visibility,
            "lighting": sc.environment.lighting,
            "weather": sc.environment.weather,
            "n_violations_detected": len(vs),
        })
        for v in vs:
            v.taxonomy = signature(v, sc)
            attr = attribute(sc, v)
            cf = classify(sc, v, attr)
            violations.append(v.model_dump())
            attributions[v.violation_id] = attr.model_dump()
            consequences[v.violation_id] = {
                "consequence_class": cf.consequence_class,
                "corrected_layers": cf.corrected_layers,
                "engine": cf.planner_evaluation.engine,
                "max_position_divergence_m":
                    cf.planner_evaluation.max_position_divergence_m,
                "observed_safety": cf.observed_safety.model_dump(),
                "corrected_safety": cf.corrected_safety.model_dump(),
            }
            # Full replay (trajectories) persisted separately: large payload.
            store.write_json(cf.model_dump(), "runs", run_id, "consequences",
                             f"{v.violation_id}.json")
            gate_items.append({
                "violation_id": v.violation_id,
                "signature": v.taxonomy,
                "attribution": attributions[v.violation_id],
                "consequence": consequences[v.violation_id],
            })

    metrics = compute_metrics(scenario_summaries, violations, consequences)
    cluster_items = [{
        "violation_id": v["violation_id"], "signature": v["taxonomy"],
        "primary_layer": attributions[v["violation_id"]]["primary_layer"],
        "consequence_class":
            consequences[v["violation_id"]]["consequence_class"],
    } for v in violations]
    clusters = build_clusters(cluster_items)
    gate = evaluate_gate(run_id, gate_items,
                         forward_to_agentic=forward_to_agentic)

    run_doc = {
        "run_id": run_id, "bank_id": bank_id,
        "model_version": doc["model_version"],
        "ruleset_version": RULESET_VERSION,
        "n_scenarios": len(scenarios),
        "scenario_summaries": scenario_summaries,
        "violations": violations,
        "attributions": attributions,
        "consequences": consequences,
        "metrics": metrics,
        "clusters": clusters,
        "gate": gate.model_dump(),
    }
    store.write_json(run_doc, "runs", run_id, "run.json")
    build_queue(run_id, violations, clusters)
    return run_doc


def load_run(run_id: str) -> Optional[Dict]:
    return store.read_json("runs", run_id, "run.json")


def list_runs() -> List[Dict]:
    out = []
    for name in store.list_dir("runs"):
        doc = store.read_json("runs", name, "run.json")
        if doc:
            out.append({
                "run_id": doc["run_id"], "bank_id": doc["bank_id"],
                "model_version": doc["model_version"],
                "n_scenarios": doc["n_scenarios"],
                "n_violations": len(doc["violations"]),
                "gate_outcome": doc["gate"]["outcome"],
                "rotr_recall": doc["metrics"]["rotr_recall"],
            })
    return out


def load_consequence(run_id: str, violation_id: str) -> Optional[Dict]:
    return store.read_json("runs", run_id, "consequences",
                           f"{violation_id}.json")
