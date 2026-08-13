"""Policy engine: determinism, version hashing, pre-authorized stop-ship,
INDETERMINATE fail-safe, severity taxonomy, the full option matrix and the
safety-as-hard-constraint guarantee."""

from __future__ import annotations

import copy

from sensorflow.agentic import policy as policy_mod
from sensorflow.agentic.policy import PolicyInput


def _base_input(**overrides) -> PolicyInput:
    kwargs = dict(
        failure_id="fail-test",
        safety_critical_class=True,
        behavioral_evidence="observed_unsafe",
        downstream_contained=False,
        rate=1.0e-4, rate_ci=[7.0e-5, 1.6e-4], denominator=240_000,
        significant=True, small_sample=False,
        novelty="novel", evidence_confidence="CONFIRMED",
        downstream_consequence="observed_critical",
        gt_available=True, lineage_complete=True, telemetry_available=True,
        fusion_verdict="multi_modal_supported", agent_conflict=False,
    )
    kwargs.update(overrides)
    return PolicyInput(**kwargs)


# ------------------------------------------------------------------ versioning


def test_policy_hash_versioning_and_determinism():
    doc = policy_mod.save_policy(copy.deepcopy(policy_mod.DEFAULT_POLICY))
    assert doc["policy_version"] == policy_mod.policy_hash(doc)
    # identical content -> identical hash; changed content -> new hash
    again = policy_mod.policy_hash(copy.deepcopy(policy_mod.DEFAULT_POLICY))
    assert again == doc["policy_version"]
    mutated = copy.deepcopy(policy_mod.DEFAULT_POLICY)
    mutated["expected_loss"]["delay_cost"] = 999.0
    assert policy_mod.policy_hash(mutated) != doc["policy_version"]
    # persisted + retrievable by version
    fetched = policy_mod.get_policy(doc["policy_version"])
    assert fetched["policy_name"] == doc["policy_name"]
    # example thresholds must be flagged as placeholders
    assert doc["placeholder_values"] is True


def test_policy_evaluation_is_deterministic():
    inp = _base_input()
    r1 = policy_mod.evaluate(inp)
    r2 = policy_mod.evaluate(inp)
    for key in ("outcome", "severity", "recommended_option", "policy_version"):
        assert r1[key] == r2[key]
    assert r1["expected_loss_table"] == r2["expected_loss_table"]


# ------------------------------------------------------------------ stop-ship


def test_automatic_stop_ship_fires_only_on_preauthorized_conditions():
    # ASS-1: S4 + CONFIRMED + significant -> fires
    r = policy_mod.evaluate(_base_input())
    assert r["severity"] == "S4"
    assert r["outcome"] == "AUTOMATIC_STOP_SHIP"
    assert r["automatic_stop_ship_condition"]["condition_id"] == "ASS-1"
    assert r["recommended_option"] == "STOP_SHIP"

    # same facts but only LIKELY confidence -> NOT pre-authorized
    r2 = policy_mod.evaluate(_base_input(evidence_confidence="LIKELY"))
    assert r2["outcome"] != "AUTOMATIC_STOP_SHIP"

    # not significant -> NOT pre-authorized
    r3 = policy_mod.evaluate(_base_input(significant=False))
    assert r3["outcome"] != "AUTOMATIC_STOP_SHIP"


# ------------------------------------------------------------------ INDETERMINATE


def test_indeterminate_on_missing_gt_lineage_telemetry_and_conflict():
    cases = [
        {"gt_available": False},
        {"lineage_complete": False},
        {"telemetry_available": False, "denominator": None},
        {"fusion_verdict": "verification_failed"},
        {"agent_conflict": True, "conflict_details": ["vlm vs fusion"]},
    ]
    for overrides in cases:
        r = policy_mod.evaluate(_base_input(**overrides))
        assert r["outcome"] == "INDETERMINATE", overrides
        # never a pass, always escalates to a human
        assert r["outcome"] != "NO_LAUNCH_IMPACT"
        assert r["recommended_option"] == "HUMAN_SAFETY_REVIEW"
        assert r["indeterminate_reasons"]


# ------------------------------------------------------------------ severity


def test_severity_taxonomy_criteria():
    doc = policy_mod.get_policy()
    assert set(doc["severity_taxonomy"]) == {"S0", "S1", "S2", "S3", "S4", "S5"}

    def sev(**kw):
        return policy_mod.assign_severity(doc, _base_input(**kw))["severity"]

    assert sev(collision_observed=True) == "S5"
    assert sev(behavioral_evidence="observed_unsafe") == "S4"
    assert sev(behavioral_evidence="none", downstream_contained=False) == "S3"
    assert sev(behavioral_evidence="none", downstream_contained=True) == "S2"
    assert sev(safety_critical_class=False, behavioral_evidence="none",
               functional_impact=True) == "S1"
    assert sev(safety_critical_class=False, behavioral_evidence="none",
               functional_impact=False) == "S0"


# ------------------------------------------------------------------ option matrix


def test_option_matrix_all_seven_rows():
    # row 1: gate violated -> STOP_SHIP
    r = policy_mod.evaluate(_base_input(gate_violated=True,
                                        evidence_confidence="LIKELY"))
    assert r["matrix_row_fired"]["row"] == 1
    assert r["recommended_option"] == "STOP_SHIP"

    # row 2: high-confidence uncontained critical -> OPTION_A_DELAY
    r = policy_mod.evaluate(_base_input(evidence_confidence="LIKELY"))
    assert r["matrix_row_fired"]["row"] == 2
    assert r["recommended_option"] == "OPTION_A_DELAY"

    # row 3: validated mitigation -> OPTION_B_MITIGATION
    r = policy_mod.evaluate(_base_input(evidence_confidence="LIKELY",
                                        mitigation_validated=True,
                                        mitigation_description="class-gate patch"))
    assert r["matrix_row_fired"]["row"] == 3
    assert r["recommended_option"] == "OPTION_B_MITIGATION"

    # row 4: reliably detectable ODD concentration -> OPTION_C_REDUCED_ODD
    r = policy_mod.evaluate(_base_input(
        behavioral_evidence="observed_contained", downstream_contained=True,
        downstream_consequence="observed_degraded",
        evidence_confidence="LIKELY",
        concentrated=True, concentration_dimension="lighting",
        odd_detector_recall=0.999, odd_exclusion_share=0.10,
        residual_rate_outside_odd=4.0e-6))
    assert r["matrix_row_fired"]["row"] == 4
    assert r["recommended_option"] == "OPTION_C_REDUCED_ODD"
    assert r["option_c_evaluation"]["feasible"] is True

    # row 4 must NOT fire when the dimension is not instrumented in the ODD
    # taxonomy (geography), even with perfect detector numbers
    r_geo = policy_mod.evaluate(_base_input(
        behavioral_evidence="observed_contained", downstream_contained=True,
        downstream_consequence="observed_degraded",
        evidence_confidence="LIKELY",
        concentrated=True, concentration_dimension="geo_bucket",
        odd_detector_recall=0.999, odd_exclusion_share=0.10,
        residual_rate_outside_odd=4.0e-6))
    assert r_geo["option_c_evaluation"]["feasible"] is False
    assert r_geo["recommended_option"] != "OPTION_C_REDUCED_ODD"

    # row 5: insufficient evidence -> EXPAND_EVALUATION
    r = policy_mod.evaluate(_base_input(
        behavioral_evidence="none", downstream_contained=True,
        significant=False, small_sample=True,
        evidence_confidence="INSUFFICIENT_EVIDENCE",
        downstream_consequence="uncertain"))
    assert r["matrix_row_fired"]["row"] == 5
    assert r["recommended_option"] == "EXPAND_EVALUATION"

    # row 6: modality conflict -> HUMAN_SAFETY_REVIEW
    r = policy_mod.evaluate(_base_input(
        safety_critical_class=False, behavioral_evidence="none",
        evidence_confidence="LIKELY", rate=1.0e-6,
        fusion_verdict="modality_conflict",
        downstream_consequence="uncertain"))
    assert r["matrix_row_fired"]["row"] == 6
    assert r["recommended_option"] == "HUMAN_SAFETY_REVIEW"

    # row 7: no safety impact -> PROCEED
    r = policy_mod.evaluate(_base_input(
        safety_critical_class=False, behavioral_evidence="none",
        significant=False, rate=1.0e-6, novelty="known_stable",
        evidence_confidence="LIKELY",
        downstream_consequence="none_observed"))
    assert r["matrix_row_fired"]["row"] == 7
    assert r["recommended_option"] == "PROCEED"


# ------------------------------------------------------------------ hard constraint


def test_cheaper_unsafe_option_never_selected():
    # Craft a policy where PROCEED is by far the cheapest option.
    doc = copy.deepcopy(policy_mod.DEFAULT_POLICY)
    doc["expected_loss"]["incident_cost"] = 0.0
    doc["expected_loss"]["delay_cost"] = 10_000.0
    doc = policy_mod.save_policy(doc)

    # S3 severity, POSSIBLE confidence: rows 1-6 all miss, row 7 (PROCEED)
    # matches — but PROCEED is infeasible under the hard constraint because
    # the residual rate exceeds MAX_ACCEPTABLE_RESIDUAL_RISK at S3+.
    # (odd_exclusion_share gives every alternative a nonzero business cost,
    # so PROCEED is STRICTLY cheapest.)
    inp = _base_input(behavioral_evidence="none", downstream_contained=False,
                      evidence_confidence="POSSIBLE", significant=True,
                      small_sample=False, rate=1.0e-4,
                      odd_exclusion_share=0.5,
                      downstream_consequence="uncertain")
    r = policy_mod.evaluate(inp, policy=doc)
    table = {row["option"]: row for row in r["expected_loss_table"]}
    # PROCEED is the cheapest option in the table...
    cheapest = min(r["expected_loss_table"], key=lambda x: x["expected_loss"])
    assert cheapest["option"] == "PROCEED"
    # ...but it is infeasible and therefore NOT selected.
    assert table["PROCEED"]["feasible"] is False
    assert r["recommended_option"] != "PROCEED"
    assert r["recommended_option"] == "HUMAN_SAFETY_REVIEW"  # fail-safe

    # restore the default active policy for the rest of the suite
    policy_mod.save_policy(copy.deepcopy(policy_mod.DEFAULT_POLICY))
