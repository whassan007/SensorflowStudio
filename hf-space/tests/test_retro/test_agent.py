"""Agent pipeline tests: canonical fixtures, fact/inference separation,
stripped-telemetry -> UNKNOWN, scorecard completeness and tier tagging."""

from __future__ import annotations

import json

import pytest

from sensorflow.retro import store as retro_store
from sensorflow.retro.agent.orchestrator import analyze, analyze_fixture
from sensorflow.retro.scorecard import (EvidenceTier, LaunchRecommendation,
                                        RetrospectiveScorecard, Severity)


@pytest.fixture()
def phantom(retro_root) -> RetrospectiveScorecard:
    return analyze_fixture("phantom_brake_plastic_bag", backend="mock")


@pytest.fixture()
def missed_ped(retro_root) -> RetrospectiveScorecard:
    return analyze_fixture("missed_pedestrian_rain", backend="mock")


# ----------------------------------------------------- canonical fixture A

def test_phantom_brake_fixture_analysis(phantom):
    sc = phantom
    assert sc.failure_type == "FALSE_POSITIVE"
    assert sc.severity == Severity.CRITICAL          # hard brake + tailgater
    assert sc.launch_recommendation == LaunchRecommendation.FAIL
    assert sc.object_class == "plastic_bag"
    # causal hypotheses reference the actual evidence
    texts = " ".join(h.hypothesis.lower() for h in sc.root_cause_hypotheses)
    assert "deformable" in texts or "training-distribution" in texts
    assert any("velocity latch" in h.hypothesis.lower()
               for h in sc.root_cause_hypotheses)
    # phantom braking requirement was retrieved
    assert any(s.doc_id == "SFS-SAFE-001" for s in sc.retrieved_standards)


# ----------------------------------------------------- canonical fixture B

def test_missed_pedestrian_fixture_analysis(missed_ped):
    sc = missed_ped
    assert sc.failure_type == "FALSE_NEGATIVE"
    assert sc.severity == Severity.CRITICAL
    assert sc.launch_recommendation == LaunchRecommendation.FAIL
    assert sc.human_review_required
    # kinematics: pedestrian inside stopping distance
    assert sc.stopping_distance_m is not None
    assert sc.distance_to_object_m <= sc.stopping_distance_m * 1.05
    assert sc.ttc_s is not None and sc.ttc_s < 2.5
    # rain-conditioned hypothesis grounded in sensor evidence
    assert any("fusion" in h.hypothesis.lower() for h in sc.root_cause_hypotheses)
    # SCR regression computed and significant -> reflected in the gate
    assert sc.safety_critical_recall_impact == pytest.approx(-3 / 1240, abs=1e-6)
    assert sc.statistical_significance.significant is True
    assert any(s.doc_id in ("SFS-SAFE-001", "SFS-PERC-002")
               for s in sc.retrieved_standards)


# --------------------------------------------------- fact/inference separation

def test_fact_inference_separation(missed_ped):
    sc = missed_ped
    tiers = {e.tier for e in sc.evidence}
    assert {EvidenceTier.OBSERVED, EvidenceTier.DERIVED,
            EvidenceTier.RETRIEVED, EvidenceTier.AI_HYPOTHESIS} <= tiers
    # hypotheses never appear as observed/derived facts
    for e in sc.evidence:
        if e.tier == EvidenceTier.AI_HYPOTHESIS:
            assert "inference" in e.provenance
    for h in sc.root_cause_hypotheses:
        assert h.tier == EvidenceTier.AI_HYPOTHESIS
        assert h.supporting_evidence_keys, "hypotheses must cite evidence"
    # observed items carry artifact provenance
    observed = [e for e in sc.evidence if e.tier == EvidenceTier.OBSERVED]
    assert all("artifact" in e.provenance or "eval log field" in e.provenance
               for e in observed)
    # rendered scorecard keeps the five-way distinction
    md = sc.render_markdown()
    for heading in ("FACT", "DERIVED FACT", "RETRIEVED REQUIREMENT",
                    "AI HYPOTHES", "SAFETY DETERMINATION"):
        assert heading in md


# ------------------------------------------------- stripped telemetry -> UNKNOWN

def test_stripped_telemetry_produces_unknown_not_guesses(retro_root):
    sc = analyze_fixture("missing_telemetry_variant", backend="mock")
    assert sc.failure_type == "FALSE_NEGATIVE"
    # stripped fields must be UNKNOWN, never guessed
    assert sc.ego_speed_mps is None
    assert sc.stopping_distance_m is None
    assert sc.ttc_s is None
    assert "ego.speed_mps" in sc.uncertainty.missing_fields
    assert "counterfactual_planner_response" in sc.uncertainty.missing_fields
    assert "stopping_distance" in sc.uncertainty.unknown_metrics
    assert "ttc" in sc.uncertainty.unknown_metrics
    # UNKNOWN markers surface in the evidence chain
    unknown_ev = [e for e in sc.evidence if e.value == "UNKNOWN"]
    assert unknown_ev
    # and the gate refuses to make a decision
    assert sc.launch_recommendation == LaunchRecommendation.INSUFFICIENT_EVIDENCE
    assert sc.human_review_required


def test_dynamically_stripped_field_becomes_unknown(retro_root):
    """Strip a field from fixture B at runtime -> the output marks it UNKNOWN."""
    from sensorflow.retro.tools.builtin import FIXTURES_DIR
    log = json.loads((FIXTURES_DIR / "missed_pedestrian_rain.json").read_text())
    del log["ego"]["speed_mps"]
    log["evaluation_id"] = "EVAL-STRIPPED-01"
    upload = retro_store.uploads_dir() / "stripped.json"
    upload.write_text(json.dumps(log))

    sc = analyze(path=str(upload), backend="mock")
    assert sc.ego_speed_mps is None
    assert "ego.speed_mps" in sc.uncertainty.missing_fields
    assert sc.stopping_distance_m is None  # not computable without speed
    assert sc.launch_recommendation == LaunchRecommendation.INSUFFICIENT_EVIDENCE


# ------------------------------------------------- severity divergence flow

def test_llm_severity_divergence_flagged(retro_root):
    """On the missing-telemetry variant the scripted 'LLM' proposes CRITICAL
    (near-miss heuristic) while policy computes DISRUPTIVE from the sparse
    evidence -> divergence recorded, policy wins, human review required."""
    sc = analyze_fixture("missing_telemetry_variant", backend="mock")
    assert sc.ai_proposed_severity == Severity.CRITICAL
    assert sc.severity == Severity.DISRUPTIVE
    assert sc.severity_divergence
    assert any("diverged" in r for r in sc.human_review_reasons)
    assert any("severity adjudication" in n for n in sc.uncertainty.notes)


# ------------------------------------------------------ variants + persistence

def test_benign_variants_pass(retro_root):
    fp = analyze_fixture("benign_fp_static_sign", backend="mock")
    fn = analyze_fixture("noncritical_fn_parked_vehicle", backend="mock")
    assert fp.severity == Severity.BENIGN
    assert fp.launch_recommendation == LaunchRecommendation.PASS
    assert fn.severity == Severity.BENIGN
    assert fn.launch_recommendation == LaunchRecommendation.PASS
    assert not fn.human_review_required


def test_analysis_persisted_with_audit(retro_root):
    sc = analyze_fixture("phantom_brake_plastic_bag", backend="mock")
    stored = retro_store.load_analysis(sc.evaluation_id)
    assert stored and stored["scorecard"]["failure_type"] == "FALSE_POSITIVE"
    assert "# Retrospective Scorecard" in stored["markdown"]
    audit = retro_store.read_audit(stored["audit_analysis_id"])
    tools_used = {r["tool"] for r in audit}
    assert {"log_reader", "metric_calculator", "safety_standard_rag",
            "historical_failure_search", "distribution_analysis"} <= tools_used
    # no write-tool call happened during analysis
    assert "create_evaluation_case" not in tools_used


def test_scorecard_completeness_spec_fields(missed_ped):
    """Every spec field exists and the strictly-typed model round-trips."""
    sc = missed_ped
    dumped = json.loads(sc.model_dump_json())
    for field in ["evaluation_id", "failure_type", "severity",
                  "safety_critical_recall_impact", "behavioral_consequence",
                  "launch_recommendation", "baseline_model", "candidate_model",
                  "scenario", "object_class", "ground_truth", "prediction",
                  "confidence", "ego_speed_mps", "distance_to_object_m",
                  "stopping_distance_m", "ttc_s", "planner_response",
                  "disengagement_probability", "metric_delta",
                  "statistical_significance", "distribution_shift",
                  "root_cause_hypotheses", "evidence", "retrieved_standards",
                  "uncertainty", "human_review_required", "policy_version",
                  "agent_version"]:
        assert field in dumped, f"missing spec field {field}"
    assert dumped["policy_version"].startswith("retro-policy/")
    assert dumped["agent_version"].startswith("retro-agent/")
    restored = RetrospectiveScorecard(**dumped)
    assert restored.severity == sc.severity


def test_historical_search_finds_prior_analyses(retro_root):
    analyze_fixture("missed_pedestrian_rain", backend="mock")
    from sensorflow.retro.tools.builtin import build_registry
    reg = build_registry(analysis_id="hist-test")
    res = reg.call("historical_failure_search",
                   {"query_text": "pedestrian rain night missed detection", "k": 3})
    assert res.result["corpus_size"] >= 1
    assert any(m["evaluation_id"] == "EVAL-2026-0802-MP01"
               for m in res.result["matches"])
