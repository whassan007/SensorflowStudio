"""Counterfactual consequence: cosmetic violations classify as
NO_MATERIAL_CONSEQUENCE; a missed crossing pedestrian is SAFETY_CRITICAL;
every verdict carries measured evidence from both branches."""

from __future__ import annotations

import pytest

from sensorflow.rotr.attribution import attribute
from sensorflow.rotr.consequence import classify


@pytest.fixture(scope="module")
def classified(bank_v1, detections_v1):
    out = {}
    for sc in bank_v1:
        for v in detections_v1[sc.scenario_id]:
            attr = attribute(sc, v)
            out[v.violation_id] = (sc, v, attr, classify(sc, v, attr))
    return out


def _by_kind_cause(classified, kind, cause):
    return [t for t in classified.values()
            if t[0].planted.kind == kind and t[0].planted.cause_layer == cause]


class TestClasses:
    def test_missed_crossing_pedestrian_is_safety_critical(self, classified):
        cases = _by_kind_cause(classified, "fail_yield_pedestrian", "perception")
        assert cases
        for sc, v, attr, cf in cases:
            assert cf.consequence_class == "SAFETY_CRITICAL"
            assert cf.observed_safety.min_clearance_m is not None
            assert cf.observed_safety.min_clearance_m < 1.5

    def test_cosmetic_wrong_lane_is_no_material_consequence(self, classified):
        cases = _by_kind_cause(classified, "wrong_lane_association",
                               "localization")
        assert cases
        for sc, v, attr, cf in cases:
            assert cf.consequence_class == "NO_MATERIAL_CONSEQUENCE"
            assert cf.planner_evaluation.max_position_divergence_m < 2.0

    def test_control_fault_replays_the_plan(self, classified):
        cases = _by_kind_cause(classified, "stop_overshoot", "control")
        assert cases
        for sc, v, attr, cf in cases:
            assert cf.planner_evaluation.engine == "planned-trajectory"
            assert cf.consequence_class != "NO_MATERIAL_CONSEQUENCE"


class TestEvidence:
    def test_both_branches_measured(self, classified):
        for sc, v, attr, cf in classified.values():
            pe = cf.planner_evaluation
            assert pe.observed_trajectory and pe.corrected_trajectory
            assert pe.engine
            for sa in (cf.observed_safety, cf.corrected_safety):
                assert sa.max_braking_mps2 >= 0.0
                assert "surrogate" in sa.surrogate_caveat.lower() or \
                    "TTC" in sa.surrogate_caveat

    def test_counterfactual_provenance_labeled(self, classified):
        for sc, v, attr, cf in classified.values():
            assert cf.provenance.source == "COUNTERFACTUAL"
            assert cf.corrected_layers
