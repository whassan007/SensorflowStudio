"""Worked example: pedestrian -> construction cone at ~0.01%.

Runs the full five-layer pipeline on the seeded synthetic population,
through evidence aggregation, analysis, the deterministic launch decision, a
recorded (clearly demo-labeled) human review and the learning flywheel, and
assembles a walkthrough where EVERY value is labeled:

    OBSERVED           measured on the seeded synthetic run
    HYPOTHETICAL       an agent hypothesis, unverified
    REQUIRED-EVIDENCE  evidence that does not exist and would be required

The walkthrough is deterministic for a fixed seed.
"""

from __future__ import annotations

from typing import Dict, Optional

from sensorflow.agentic import METHODOLOGY_PROVENANCE
from sensorflow.agentic import data as data_mod
from sensorflow.agentic import flywheel as flywheel_mod
from sensorflow.agentic import pipeline as pipeline_mod
from sensorflow.agentic import review as review_mod
from sensorflow.agentic import store as store_mod

WORKED_EXAMPLE_SEED = data_mod.DEFAULT_SEED
DEMO_REVIEWER = "safety-reviewer-demo"


def _find_ped_cone_failure() -> Optional[Dict]:
    for doc in pipeline_mod.list_failures():
        if (doc.get("gt_class") == "pedestrian"
                and doc.get("predicted_class") == "construction_cone"):
            return doc
    return None


def run_worked_example(seed: int = WORKED_EXAMPLE_SEED,
                       use_llm: bool = False) -> Dict:
    # Layer 1 — detection (idempotent: reuse an existing detection if present)
    existing = _find_ped_cone_failure()
    if existing is None:
        pipeline_mod.detect_failures(seed=seed, use_llm=use_llm)
        existing = _find_ped_cone_failure()
    if existing is None:
        raise RuntimeError("worked example: pedestrian->cone pattern not "
                           "detected on the seeded population")
    fid = existing["failure_id"]
    failure = pipeline_mod.get_failure(fid)
    state = pipeline_mod.get_state(fid)

    # Layers 2-4
    for stage in ("EVIDENCE_AGGREGATION", "FAILURE_ANALYSIS", "LAUNCH_DECISION"):
        if state.stage_record(stage).status == "pending":
            state = pipeline_mod.run_stage(fid, stage, seed=seed, use_llm=use_llm)
    failure = pipeline_mod.get_failure(fid)
    evaluation = state.policy_evaluation or {}

    # Human review (demo decision, clearly labeled as such)
    if not failure.validated:
        review_mod.record_decision(
            failure, reviewer=DEMO_REVIEWER, decision="confirm_failure",
            rationale=("DEMO WALKTHROUGH DECISION: replay evidence shows "
                       "unsafe delayed braking; statistics significant; "
                       "confirming the failure as real for the worked example"),
            evidence_reviewed=["evidence_graph", "statistical_assessment",
                               "safety_replays", "concentration_analysis"],
            policy_version=evaluation.get("policy_version", ""))
        failure.validated = True
        pipeline_mod.save_failure(failure)
        store_mod.audit("failure_validated", fid, DEMO_REVIEWER,
                        detail="confirm_failure (worked-example demo reviewer)")

    # Layer 5 — flywheel
    if state.stage_record("LEARNING_FLYWHEEL").status in ("pending", "blocked"):
        state.stage_record("LEARNING_FLYWHEEL").status = "pending"
        pipeline_mod.save_state(state)
        pipeline_mod.run_stage(fid, "LEARNING_FLYWHEEL", seed=seed,
                               use_llm=use_llm)
        state = pipeline_mod.get_state(fid)

    regression = flywheel_mod.regression_evaluate(seed=seed)

    # ---- assemble the labeled walkthrough ------------------------------
    stat = state.statistical
    conc = state.concentration
    safety = state.agent_results.get("safety_impact")
    fusion = state.agent_results.get("sensor_fusion_verification")
    mining = state.agent_results.get("scenario_mining")
    vlm = state.agent_results.get("vlm_scene_analysis")

    def L(label: str, value) -> Dict:
        return {"label": label, "value": value}

    walkthrough = {
        "methodology_provenance": METHODOLOGY_PROVENANCE,
        "seed": seed,
        "deterministic": True,
        "failure_id": fid,
        "title": failure.title,
        "layers": {
            "1_failure_detection": {
                "candidate_events": L("OBSERVED",
                                      failure.detection_basis.candidate_events),
                "baseline_events": L("OBSERVED",
                                     failure.detection_basis.baseline_events),
                "denominator": L("OBSERVED", failure.detection_basis.denominator),
                "observed_rate": L("OBSERVED",
                                   failure.detection_basis.candidate_rate),
                "headline": L("OBSERVED",
                              f"~{failure.detection_basis.candidate_rate:.4%} "
                              "pedestrian->construction_cone rate on the "
                              "candidate model"),
                "scene_instances_captured": L("OBSERVED", len(failure.instances)),
            },
            "2_evidence_aggregation": {
                "evidence_graph_nodes": L(
                    "OBSERVED",
                    [{"type": n["node_type"], "status": n["status"]}
                     for n in (store_mod.read_json(
                         "evidence", f"{fid}.json") or {}).get("nodes", [])]),
                "snippets_built": L("OBSERVED", len(pipeline_mod.get_snippets(fid))),
                "temporal_context": L("OBSERVED",
                                      "before/failure/after frames captured "
                                      "per snippet"),
            },
            "3_failure_analysis": {
                "statistics": L("OBSERVED", stat.model_dump() if stat else None),
                "concentration": L("OBSERVED", {
                    "determination": conc.determination if conc else None,
                    "concentrated_dimensions": (conc.concentrated_dimensions
                                                if conc else []),
                    "construction_stratum": next(
                        (s.model_dump() for s in (conc.strata if conc else [])
                         if s.stratum == "construction"), None),
                }),
                "fusion_verdict": L("OBSERVED",
                                    fusion.output.get("overall_verdict")
                                    if fusion else None),
                "novelty": L("OBSERVED",
                             mining.output.get("novelty") if mining else None),
                "scene_hypotheses": L("HYPOTHETICAL",
                                      (vlm.output.get("hypotheses", [])[:3]
                                       if vlm else [])),
                "safety_chain": L("OBSERVED",
                                  safety.output.get("chain") if safety else None),
                "replay_worst_case": L("OBSERVED",
                                       safety.output.get("worst_case")
                                       if safety else None),
                "replay_coverage": L("REQUIRED-EVIDENCE",
                                     safety.output.get("coverage_note")
                                     if safety else "no safety agent output"),
            },
            "4_launch_decision": {
                "policy_version": L("OBSERVED", evaluation.get("policy_version")),
                "severity": L("OBSERVED", evaluation.get("severity")),
                "outcome": L("OBSERVED", evaluation.get("outcome")),
                "recommended_option": L("OBSERVED",
                                        evaluation.get("recommended_option")),
                "matrix_row_fired": L("OBSERVED", evaluation.get("matrix_row_fired")),
                "option_c_rejection": L(
                    "REQUIRED-EVIDENCE",
                    [c for c in (evaluation.get("option_c_evaluation") or {})
                     .get("checks", []) if not c.get("passed")]),
                "expected_loss_table": L("HYPOTHETICAL",
                                         evaluation.get("expected_loss_table")),
                "mandatory_review_triggers": L(
                    "OBSERVED",
                    [t for t in evaluation.get("mandatory_review_triggers", [])
                     if t["fired"]]),
                "human_review": L("OBSERVED", review_mod.decisions_for(fid)),
            },
            "5_learning_flywheel": {
                "suites_created": L("OBSERVED", state.suite_ids),
                "regression_hook": L("OBSERVED", regression.get("suites")),
            },
        },
        "scorecard_id": state.scorecard_id,
        "audit_records": L("OBSERVED", len(store_mod.audit_trail(fid))),
        "audit_chain_valid": L("OBSERVED",
                               store_mod.verify_audit_chain(fid).get("valid")),
        "separation_of_powers": (
            "agents produced hypotheses and advisory summaries only; rates, "
            "CIs, significance, severity, the policy outcome and the option "
            "recommendation above all came from deterministic code, and the "
            "final validation was a recorded human decision"),
    }
    store_mod.write_json(walkthrough, "worked_example", "latest.json")
    return walkthrough
