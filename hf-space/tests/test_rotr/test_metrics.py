"""Metric math: recall/false-accusation denominators, exposure-derived
weight calibration (monotone in measured harm), BCR/CFR."""

from __future__ import annotations

from sensorflow.rotr.metrics import compute_metrics


def _scenario(sid, kind="k", committed=True, opportunity=True,
              rule="R-X", vul="NON_VRU", vis="clear", light="day"):
    return {"scenario_id": sid, "kind": kind, "committed": committed,
            "is_violation_opportunity": opportunity,
            "expected_rule_id": rule if committed else None,
            "cause_layer": "planning" if committed else None,
            "vulnerability": vul, "visibility": vis, "lighting": light,
            "weather": "clear", "n_violations_detected": 0}


def _violation(vid, sid, rule="R-X"):
    return {"violation_id": vid, "scenario_id": sid, "rule_id": rule,
            "taxonomy": {}}


class TestRecallMath:
    def test_recall_and_false_accusations(self):
        scenarios = [
            _scenario("s1", committed=True),
            _scenario("s2", committed=True),
            _scenario("s3", committed=False, opportunity=False),
            _scenario("s4", committed=False, opportunity=False),
        ]
        violations = [_violation("v1", "s1"),       # true positive
                      _violation("v3", "s3")]       # false accusation
        m = compute_metrics(scenarios, violations, {})
        assert m["rotr_recall"] == 0.5              # 1 of 2 committed found
        assert m["false_accusation_rate"] == 0.5    # 1 of 2 negatives accused
        assert m["n_committed_violations"] == 2
        assert m["n_planted_non_violations"] == 2

    def test_behavior_rates_use_opportunity_denominator(self):
        scenarios = [
            _scenario("s1", committed=True, vul="VRU"),
            _scenario("s2", committed=False, opportunity=True, vul="VRU"),
            _scenario("s3", committed=False, opportunity=False),
        ]
        m = compute_metrics(scenarios, [], {})
        assert m["behavior_rates"]["VRU"] == {
            "opportunities": 2, "committed": 1, "violation_rate": 0.5}
        assert m["behavior_rates"]["ALL"]["opportunities"] == 2


class TestWeightCalibration:
    def test_weights_monotone_in_measured_harm(self):
        # Stratum A: 2/2 safety-critical; stratum B: 0/2.
        scenarios = [
            _scenario("a1", kind="A", vul="VRU"),
            _scenario("a2", kind="A", vul="VRU"),
            _scenario("b1", kind="B"),
            _scenario("b2", kind="B"),
        ]
        violations = [_violation(f"v-{s}", s) for s in
                      ("a1", "a2", "b1", "b2")]
        consequences = {
            "v-a1": {"consequence_class": "SAFETY_CRITICAL"},
            "v-a2": {"consequence_class": "SAFETY_CRITICAL"},
            "v-b1": {"consequence_class": "NO_MATERIAL_CONSEQUENCE"},
            "v-b2": {"consequence_class": "DEGRADED_COMFORT"},
        }
        m = compute_metrics(scenarios, violations, consequences)
        strata = m["weight_calibration"]["strata"]
        wa = strata["A|VRU"]["weight"]
        wb = strata["B|NON_VRU"]["weight"]
        assert strata["A|VRU"]["harm_fraction"] > strata["B|NON_VRU"]["harm_fraction"]
        assert wa > wb, "higher measured harm must never get a lower weight"
        assert m["weight_calibration"]["method"].startswith("stratum weight")
        assert "ILLUSTRATIVE" in m["weight_calibration"]["method"]

    def test_zero_harm_strata_get_floor_not_zero(self):
        scenarios = [_scenario("a1", kind="A"), _scenario("b1", kind="B")]
        violations = [_violation("v-a1", "a1"), _violation("v-b1", "b1")]
        consequences = {
            "v-a1": {"consequence_class": "SAFETY_CRITICAL"},
            "v-b1": {"consequence_class": "NO_MATERIAL_CONSEQUENCE"}}
        m = compute_metrics(scenarios, violations, consequences)
        assert m["weight_calibration"]["strata"]["B|NON_VRU"]["weight"] == 0.1


class TestRates:
    def test_bcr_and_cfr(self):
        scenarios = [_scenario(f"s{i}") for i in range(4)]
        violations = [_violation("v0", "s0"), _violation("v1", "s1")]
        consequences = {
            "v0": {"consequence_class": "SAFETY_CRITICAL"},
            "v1": {"consequence_class": "NO_MATERIAL_CONSEQUENCE"}}
        m = compute_metrics(scenarios, violations, consequences)
        assert m["bcr"] == 0.5          # 1 of 2 detected is consequential
        assert m["cfr"] == 0.25         # 1 SC over 4 scenarios
        lo, hi = m["cfr_wilson_95"]
        assert 0.0 <= lo < 0.25 < hi <= 1.0
