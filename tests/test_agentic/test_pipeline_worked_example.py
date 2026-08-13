"""Five-layer pipeline, worked-example determinism, epistemic labeling and
audit completeness (hash-chained, append-only)."""

from __future__ import annotations

import copy
import json

from sensorflow.agentic import store as store_mod
from sensorflow.agentic import worked_example as we_mod

STAGES = ["FAILURE_DETECTION", "EVIDENCE_AGGREGATION", "FAILURE_ANALYSIS",
          "LAUNCH_DECISION", "LEARNING_FLYWHEEL"]


def _strip_ids(obj):
    """Remove run-specific identifiers so two runs can be compared on
    substance."""
    if isinstance(obj, dict):
        return {k: _strip_ids(v) for k, v in obj.items()
                if not (k.endswith("_id") or k.endswith("_at")
                        or k in ("timestamp", "hash", "prev_hash",
                                 "evidence_ref", "narrative"))}
    if isinstance(obj, list):
        return [_strip_ids(v) for v in obj]
    if isinstance(obj, str):
        # ids are embedded in some prose fields; normalize them away
        for prefix in ("fail-", "snip-", "suite-", "inst-", "rev-", "card-"):
            if prefix in obj:
                return "<contains-id>"
        return obj
    return obj


def test_worked_example_completes_all_five_layers(walkthrough):
    assert set(walkthrough["layers"]) == {
        "1_failure_detection", "2_evidence_aggregation", "3_failure_analysis",
        "4_launch_decision", "5_learning_flywheel"}
    from sensorflow.agentic import pipeline as pipeline_mod
    state = pipeline_mod.get_state(walkthrough["failure_id"])
    for stage in STAGES:
        assert state.stage_record(stage).status == "complete", stage
    # the methodology provenance disclosure is part of the artifact
    assert "was not provided" in walkthrough["methodology_provenance"]


def test_worked_example_values_are_epistemically_labeled(walkthrough):
    labels = set()

    def collect(obj):
        if isinstance(obj, dict):
            if "label" in obj and "value" in obj:
                labels.add(obj["label"])
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    collect(walkthrough)
    assert labels <= {"OBSERVED", "HYPOTHETICAL", "REQUIRED-EVIDENCE",
                      "DERIVED"}
    # all three spec labels actually occur
    assert {"OBSERVED", "HYPOTHETICAL", "REQUIRED-EVIDENCE"} <= labels


def test_worked_example_headline_numbers(walkthrough):
    l1 = walkthrough["layers"]["1_failure_detection"]
    assert l1["denominator"]["value"] == 240_000
    assert l1["candidate_events"]["value"] == 25
    rate = l1["observed_rate"]["value"]
    assert abs(rate - 1.0e-4) < 3.0e-5  # the motivating ~0.01%

    l4 = walkthrough["layers"]["4_launch_decision"]
    assert l4["outcome"]["value"] == "LAUNCH_REVIEW_REQUIRED"
    assert l4["severity"]["value"] == "S4"
    assert l4["recommended_option"]["value"] == "OPTION_A_DELAY"
    # the final authorization is human, recorded, and references the policy
    decisions = l4["human_review"]["value"]
    assert decisions
    assert decisions[0]["decision"] == "confirm_failure"
    assert decisions[0]["policy_version"]
    assert decisions[0]["reviewer"] == "safety-reviewer-demo"


def test_scorecard_downstream_impact_observed_with_replay(walkthrough):
    """The safety agent escalates (unsafe replay behavior) but still produced
    OBSERVED replay evidence — the scorecard must not degrade that to
    REQUIRED_EVIDENCE just because the agent status is 'escalated'."""
    from sensorflow.agentic import scorecard as scorecard_mod
    card = scorecard_mod.load_scorecard(walkthrough["scorecard_id"])
    assert card["downstream_impact"]["tag"] == "OBSERVED"
    assert card["downstream_impact"]["value"]["behavioral_evidence"] == "observed"
    assert card["novelty"]["tag"] == "OBSERVED"


def test_worked_example_is_deterministic(tmp_path):
    """Two runs from two pristine storage roots must agree on every
    substantive value (only ids/timestamps may differ)."""
    original_root = store_mod.agentic_root()
    try:
        store_mod.set_agentic_root(str(tmp_path / "run_a"))
        run_a = we_mod.run_worked_example()
        store_mod.set_agentic_root(str(tmp_path / "run_b"))
        run_b = we_mod.run_worked_example()
    finally:
        store_mod.set_agentic_root(original_root)
    a, b = _strip_ids(run_a), _strip_ids(run_b)
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)


def test_audit_trail_is_complete_and_chained(walkthrough):
    fid = walkthrough["failure_id"]
    trail = store_mod.audit_trail(fid)
    events = {r["event_type"] for r in trail}
    # every stage transition, agent output, policy evaluation, human decision
    assert {"stage_transition", "agent_output", "policy_evaluated",
            "human_review_decision", "scorecard_generated",
            "suite_created_or_updated", "failure_validated"} <= events
    stage_details = " ".join(r["detail"] for r in trail
                             if r["event_type"] == "stage_transition")
    for stage in STAGES[1:]:  # detection transitions live in the global trail
        assert stage in stage_details
    agents_seen = {r["actor"] for r in trail if r["event_type"] == "agent_output"}
    assert {"vlm_scene_analysis", "sensor_fusion_verification",
            "scenario_mining", "statistical_regression", "safety_impact",
            "launch_decision", "eval_flywheel"} <= agents_seen
    # hash chain is intact and sequenced
    check = store_mod.verify_audit_chain(fid)
    assert check["valid"] is True
    assert check["records"] == len(trail)


def test_audit_tampering_is_detected(walkthrough):
    fid = walkthrough["failure_id"]
    trail = store_mod.audit_trail(fid)
    tampered = copy.deepcopy(trail)
    tampered[2]["detail"] = "history rewritten"
    import os
    path = os.path.join(store_mod.agentic_root(), "audit", f"{fid}.jsonl")
    with open(path) as f:
        original = f.read()
    try:
        with open(path, "w") as f:
            for rec in tampered:
                f.write(json.dumps(rec, default=str) + "\n")
        check = store_mod.verify_audit_chain(fid)
        assert check["valid"] is False
        assert check["broken_at_seq"] == 2
    finally:
        with open(path, "w") as f:
            f.write(original)
    assert store_mod.verify_audit_chain(fid)["valid"] is True
