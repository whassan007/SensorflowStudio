"""Layered release gating: policy handling, blocking behavior with the right
evidence, and the Safety Evidence Package (JSON + markdown)."""

from __future__ import annotations

import pytest

from sensorflow.safety import gates

LENIENT = {
    "scenario_quality": {"min_geometric_pass_rate": 0.05},
    "coverage": {"min_coverage_rate": 0.0, "min_production_weighted_coverage": 0.0,
                 "min_samples": 5, "max_ci_width": 0.99},
}


@pytest.fixture()
def eval_store(tmp_path):
    from sensorflow.evaluation.records import reset_store
    return reset_store(tmp_path / "eval")


def test_policy_defaults_merge_and_persist(fresh_safety_root):
    p = gates.get_policy()
    assert p == gates.DEFAULT_POLICY
    updated = gates.set_policy({"safety": {"max_csi_increase_ratio": 0.5}})
    assert updated["safety"]["max_csi_increase_ratio"] == 0.5
    # untouched sections keep defaults
    assert updated["coverage"]["min_samples"] == gates.DEFAULT_POLICY["coverage"]["min_samples"]
    assert gates.get_policy()["safety"]["max_csi_increase_ratio"] == 0.5


def test_all_gates_pass_for_good_candidate(mega_env, fresh_safety_root, eval_store):
    res = gates.evaluate_gates(mega_env["store"], mega_env["good"], mega_env["good"],
                               policy_overrides=LENIENT)
    assert res["decision"] == "RELEASE_READY"
    assert res["blocking_gates"] == []
    by_gate = {g["gate"]: g for g in res["gates"]}
    assert set(by_gate) == {"scenario_quality", "coverage", "regression",
                            "safety", "release_readiness"}
    assert all(g["status"] == "PASS" for g in res["gates"])
    # candidate == baseline -> zero CSI delta, PROMOTE recommendation
    assert by_gate["safety"]["evidence"]["csi_increase_ratio"] == pytest.approx(0.0)
    assert by_gate["regression"]["evidence"]["blockers"] == []
    # every gate carries checks with thresholds and actuals
    for g in res["gates"]:
        for c in g["checks"]:
            assert {"check", "actual", "threshold", "passed"} <= set(c)


def test_regressed_candidate_blocks_regression_and_safety(mega_env, fresh_safety_root,
                                                          eval_store):
    overrides = {**LENIENT, "safety": {"max_csi_increase_ratio": 0.05}}
    res = gates.evaluate_gates(mega_env["store"], mega_env["bad"], mega_env["good"],
                               policy_overrides=overrides)
    assert res["decision"] == "BLOCKED"
    assert "regression" in res["blocking_gates"]
    assert "safety" in res["blocking_gates"]
    by_gate = {g["gate"]: g for g in res["gates"]}
    # regression gate carries the megaeval blockers as evidence
    assert by_gate["regression"]["evidence"]["blockers"]
    # safety gate shows candidate CSI above baseline
    ev = by_gate["safety"]["evidence"]
    assert ev["candidate_csi"] > ev["baseline_csi"]
    # readiness gate blocked and names the first blocking gate
    rr = by_gate["release_readiness"]
    assert rr["status"] == "BLOCK"
    assert rr["evidence"]["first_blocking_gate"] in res["blocking_gates"]


def test_coverage_gate_blocks_on_impossible_threshold(mega_env, fresh_safety_root,
                                                      eval_store):
    overrides = {**LENIENT,
                 "coverage": {"min_coverage_rate": 1.01,
                              "min_production_weighted_coverage": 0.0}}
    res = gates.evaluate_gates(mega_env["store"], mega_env["good"], mega_env["good"],
                               policy_overrides=overrides)
    assert "coverage" in res["blocking_gates"]
    cov = next(g for g in res["gates"] if g["gate"] == "coverage")
    failed = [c for c in cov["checks"] if not c["passed"]]
    assert failed and failed[0]["check"] == "coverage_rate"
    assert cov["evidence"]["top_gaps"] is not None


def test_scenario_quality_gate_blocks_on_impossible_threshold(mega_env,
                                                              fresh_safety_root,
                                                              eval_store):
    overrides = {**LENIENT, "scenario_quality": {"min_geometric_pass_rate": 1.01}}
    res = gates.evaluate_gates(mega_env["store"], mega_env["good"], mega_env["good"],
                               policy_overrides=overrides)
    assert "scenario_quality" in res["blocking_gates"]
    sq = next(g for g in res["gates"] if g["gate"] == "scenario_quality")
    assert sq["evidence"]["annotations_sampled"] > 0


def test_evidence_package_structure_and_wording(mega_env, fresh_safety_root, eval_store):
    gates.evaluate_gates(mega_env["store"], mega_env["good"], mega_env["good"],
                         policy_overrides=LENIENT)
    run_id = mega_env["good"].run_id
    pkg = gates.load_evidence(run_id)
    assert pkg is not None
    assert pkg["package_id"] == f"sep-{run_id}"
    # precise wording: supports a safety case, does not certify
    assert "SUPPORT a safety case" in pkg["disclaimer"]
    assert "not certify" in pkg["disclaimer"]
    # standard mappings cover the three families
    standards = " ".join(m["standard"] for m in pkg["standard_mappings"])
    for token in ("ISO 26262", "21448", "UL 4600"):
        assert token in standards
    # methodology + lineage present
    assert pkg["sampling_methodology"]["config"] is not None
    assert pkg["lineage"]["candidate"]["evaluation_id"] == run_id
    assert pkg["decision"]["release_ready"] is True

    # persisted gate result retrievable
    result = gates.latest_gate_result(run_id)
    assert result is not None and result["evidence_package_id"] == pkg["package_id"]

    # markdown rendering
    md = gates.render_markdown(pkg)
    assert "# Safety Evidence Package" in md
    assert "RELEASE READY" in md
    assert "Standard mappings" in md
    assert pkg["disclaimer"][:40] in md
