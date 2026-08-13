"""Closed-loop demo: one seeded scenario moves through the whole system.

Stitches the REAL engines end-to-end:

  1. bevfusion scenario suite (deterministic multi-sensor scenes)
  2. baseline vs candidate perception comparison (bevfusion engines)
  3. seqeval anytime-valid regression verdict (with a planted night-stratum
     regression so governance has something to catch)
  4. nextgen counterfactual + closed-loop causal replay (when importable)
  5. safety layered gates + evidence package (reusing the latest published
     megaeval candidate/baseline runs)
  6. megaeval distribution-shift report
  7. studio2 ReleaseGate composition -> ReleaseDecision with the evidence tuple
  8. flywheel step: the failure is registered as a REGRESSION-role dataset
     (protected by the contamination guard)

Every step records availability; steps whose engine/store is absent are
skipped with the reason, and the release gate degrades accordingly (that IS
the demo of graceful degradation). Deterministic for a fixed seed.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from sensorflow.studio2 import store
from sensorflow.studio2.registry import Registry, content_hash, get_registry
from sensorflow.studio2.release_gate import ReleaseGate
from sensorflow.studio2 import hardware as hw_mod

DEMO_SEED = 977


def _step(name: str, available: bool, detail: Dict) -> Dict:
    return {"step": name, "available": available, **detail}


def run_demo(seed: int = DEMO_SEED, registry: Optional[Registry] = None,
             repo_root: str = ".", persist: bool = True) -> Dict:
    registry = registry or get_registry()
    steps = []

    # ---- 1+2. scenario + perception comparison (bevfusion, landed) --------
    scenario_entity = None
    perception = None
    try:
        from sensorflow.bevfusion.evaluate import run_comparison
        report = run_comparison(n_sequences=3, frames_per_sequence=16,
                                seed=seed, persist=persist)
        params = report["params"]
        scenario_entity = registry.register_scenario(
            name=f"demo-suite-seed{seed}",
            generator="bevfusion.scenes.generate_sequences",
            seed=seed, recipe=params,
            provenance={"source_package": "bevfusion",
                        "run_id": report["run_id"]})
        dataset_entity = registry.register_dataset(
            name=f"demo-scenes-seed{seed}", role="TEST",
            provenance={"source_package": "bevfusion",
                        "generator_seed": seed})
        model_entity = registry.register_model(
            name=report["engines"]["candidate"],
            version=report["engines"]["candidate"],
            provenance={"source_package": "bevfusion"})
        baseline_entity = registry.register_model(
            name=report["engines"]["baseline"],
            version=report["engines"]["baseline"],
            provenance={"source_package": "bevfusion"})
        experiment = registry.register_experiment(
            name=f"studio2-demo-seed{seed}",
            candidate_model_id=model_entity["entity_id"],
            baseline_model_id=baseline_entity["entity_id"])
        gate_policy_entity = registry.register_policy(
            name="studio2-demo-run-config",
            doc={"n_sequences": 3, "frames_per_sequence": 16, "seed": seed})
        run_entity = registry.register_run(
            name=f"demo-bev-{report['run_id']}", engine="bevfusion",
            experiment_id=experiment["entity_id"],
            tuple_components={
                "model_version_id": model_entity["entity_id"],
                "dataset_version_id": dataset_entity["entity_id"],
                "scenario_version_id": scenario_entity["entity_id"],
                "config_hash": content_hash(params, exclude=()),
                # bevfusion sensor models are pinned in code; version the pin
                "calibration_version": "bevfusion-sensors-v1",
                "seed": seed,
                "policy_version_id": gate_policy_entity["entity_id"],
            },
            results={"recommendation": report["recommendation"],
                     "headline_deltas": report["headline_deltas"]},
            provenance={"source_package": "bevfusion",
                        "run_id": report["run_id"]})
        perception = report
        steps.append(_step("scenario_and_perception", True, {
            "bevfusion_run_id": report["run_id"],
            "scenario_version": scenario_entity["entity_id"],
            "evaluation_run": run_entity["entity_id"],
            "reproducibility": run_entity["reproducibility"],
            "recommendation": report["recommendation"]}))
    except Exception as e:
        steps.append(_step("scenario_and_perception", False,
                           {"reason": f"{type(e).__name__}: {e}"}))

    # ---- 3. seqeval verdict (landed; needs a megaeval population) ---------
    seq_verdict = None
    try:
        pop_dir = os.path.join(repo_root, "runs", "megaeval", "populations")
        pops = sorted(os.listdir(pop_dir)) if os.path.isdir(pop_dir) else []
        if not pops:
            raise RuntimeError("no megaeval population available")
        from sensorflow.seqeval import evaluate_regression
        # planted regression in the pedestrian|night stratum: the point of
        # the demo is to show governance catching a real per-stratum failure
        planted = {"pedestrian|night": -0.05}
        seq_verdict = evaluate_regression(
            pops[-1],
            baseline={"model_version": "demo-baseline"},
            candidate={"model_version": "demo-candidate", "effects": planted},
            seed=seed)
        steps.append(_step("sequential_regression", True, {
            "seqeval_run_id": seq_verdict["run_id"],
            "decision": seq_verdict["decision"],
            "affected_strata": seq_verdict["affected_strata"],
            "samples_used": seq_verdict["samples_used"],
            "planted_effect": planted}))
    except Exception as e:
        steps.append(_step("sequential_regression", False,
                           {"reason": f"{type(e).__name__}: {e}"}))

    # ---- 4. nextgen counterfactual + closed loop (in-flight, guarded) -----
    closed_loop = None
    try:
        from sensorflow.nextgen.causal import causal_replay
        from sensorflow.nextgen.closedloop import demo_emergence_scenario
        from sensorflow.nextgen.models import DataLabel
        scn = demo_emergence_scenario()
        result = causal_replay(scn["actors"], scn["environment"],
                               scn["scenario_id"], DataLabel.COUNTERFACTUAL,
                               seed=seed)
        closed_loop = {"scenario_id": result.scenario_id,
                       "verdict": result.verdict,
                       "diffs": result.diffs}
        steps.append(_step("closed_loop_replay", True, {
            "scenario_id": result.scenario_id, "verdict": result.verdict}))
    except Exception as e:
        steps.append(_step("closed_loop_replay", False,
                           {"reason": f"nextgen unavailable or failed: "
                                      f"{type(e).__name__}: {e}"}))

    # ---- 5+6. safety gates + shift on the latest published megaeval runs --
    safety_result = None
    shift_report = None
    try:
        from sensorflow.megaeval.runs import get_mega_store
        from sensorflow.megaeval import analysis as mega_analysis
        from sensorflow.safety import gates as safety_gates
        mega = get_mega_store()
        published = sorted(
            [r for r in mega.runs.values() if getattr(r, "status", "") == "published"],
            key=lambda r: getattr(r, "published_at", "") or "")
        if len(published) < 2:
            raise RuntimeError("fewer than two published megaeval runs")
        candidate, baseline = published[-1], published[-2]
        cached = safety_gates.latest_gate_result(candidate.run_id)
        safety_result = cached or safety_gates.evaluate_gates(
            mega, candidate, baseline)
        shift_report = mega_analysis.distribution_shift(mega, candidate)
        steps.append(_step("safety_gates_and_shift", True, {
            "candidate_run_id": candidate.run_id,
            "baseline_run_id": baseline.run_id,
            "safety_decision": safety_result["decision"],
            "blocking_gates": safety_result["blocking_gates"],
            "shifts_reported": len(shift_report.get("shifts") or []),
            "reused_cached_gate_result": bool(cached)}))
    except Exception as e:
        steps.append(_step("safety_gates_and_shift", False,
                           {"reason": f"{type(e).__name__}: {e}"}))

    # ---- hardware matrix (studio2, best-effort) ----------------------------
    hardware_matrix = None
    try:
        hardware_matrix = hw_mod.gate_matrix(persist=persist)
        steps.append(_step("hardware_matrix", True, {
            "status": hardware_matrix["status"],
            "n_combinations": hardware_matrix["n_combinations"],
            "n_insufficient": hardware_matrix["n_insufficient"]}))
    except Exception as e:
        steps.append(_step("hardware_matrix", False,
                           {"reason": f"{type(e).__name__}: {e}"}))

    # ---- agentic policy outcome (in-flight, guarded) -----------------------
    agentic_outcome = None
    try:
        from sensorflow.agentic.policy import PolicyInput, evaluate as agentic_eval
        night_regressed = bool(seq_verdict and any(
            "night" in str(s) for s in (seq_verdict.get("affected_strata") or [])))
        agentic_outcome = agentic_eval(PolicyInput(
            failure_id=f"demo-failure-seed{seed}",
            safety_critical_class=False,
            behavioral_evidence="uncertain",
            significant=night_regressed,
            novelty="known_regression" if night_regressed else "unknown",
            evidence_confidence="CONFIRMED" if night_regressed else "LIKELY",
            concentration_dimension="lighting",
            concentrated=night_regressed))
        steps.append(_step("agentic_policy", True, {
            "outcome": agentic_outcome["outcome"],
            "severity": agentic_outcome["severity"],
            "policy_version": agentic_outcome["policy_version"]}))
    except Exception as e:
        steps.append(_step("agentic_policy", False,
                           {"reason": f"agentic unavailable or failed: "
                                      f"{type(e).__name__}: {e}"}))

    # ---- 7. release decision ----------------------------------------------
    gate = ReleaseGate(registry)
    decision = gate.evaluate(
        safety_metrics=safety_result,
        regression_results=seq_verdict,
        distribution_shift=shift_report,
        agentic_outcome=agentic_outcome,
        closed_loop=closed_loop,
        hardware_matrix=hardware_matrix,
        context={"demo_seed": seed,
                 "scenario_version": (scenario_entity or {}).get("entity_id"),
                 "perception_run": (perception or {}).get("run_id")})
    steps.append(_step("release_decision", True, {
        "decision_id": decision["entity_id"],
        "status": decision["status"],
        "blocking_conditions": decision["blocking_conditions"],
        "degraded_inputs": decision["degraded_inputs"],
        "evidence_completeness": decision["evidence_completeness"]}))

    # ---- 8. flywheel: failure -> REGRESSION-role dataset -------------------
    regression_dataset = None
    if decision["status"] in ("NO_GO", "REVIEW"):
        parents = [p for p in [
            (scenario_entity or {}).get("entity_id"),
            (seq_verdict or {}).get("run_id"),
            decision["entity_id"]] if p]
        regression_dataset = registry.register_dataset(
            name=f"demo-regression-suite-seed{seed}", role="REGRESSION",
            lineage_parents=parents,
            provenance={"source": "studio2.demo",
                        "release_decision": decision["entity_id"],
                        "seqeval_run": (seq_verdict or {}).get("run_id")},
            meta={"affected_strata": (seq_verdict or {}).get("affected_strata"),
                  "note": "failure registered for perpetual re-evaluation; "
                          "protected from training by the contamination guard"})
        steps.append(_step("flywheel_registration", True, {
            "dataset_id": regression_dataset["entity_id"],
            "role": regression_dataset["role"],
            "protected_evaluation": regression_dataset["protected_evaluation"]}))
    else:
        steps.append(_step("flywheel_registration", False,
                           {"reason": "decision was GO; nothing to register"}))

    demo = {
        "demo_id": f"demo-{content_hash({'seed': seed}, exclude=())}",
        "seed": seed,
        "steps": steps,
        "decision": decision,
        "regression_dataset": regression_dataset,
        "generated_at": store.now_iso(),
    }
    if persist:
        store.write_json(demo, "demo", f"{demo['demo_id']}.json")
        store.write_json(demo, "demo", "latest.json")
    return demo


def latest_demo() -> Optional[Dict]:
    return store.read_json("demo", "latest.json")
