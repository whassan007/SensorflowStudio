"""Causal-layer attribution: planted-cause correctness per layer, and the
package's core invariant — a violation is NEVER auto-attributed to
perception."""

from __future__ import annotations

import pytest

from sensorflow.rotr.attribution import attribute


@pytest.fixture(scope="module")
def attributions(bank_v1, detections_v1):
    out = []
    for sc in bank_v1:
        for v in detections_v1[sc.scenario_id]:
            out.append((sc, v, attribute(sc, v)))
    return out


def _cases(attributions, cause):
    return [(sc, v, a) for sc, v, a in attributions
            if sc.planted.cause_layer == cause]


class TestPlantedCauses:
    def test_every_planted_cause_matches_primary_layer(self, attributions):
        for sc, v, a in attributions:
            assert a.primary_layer == sc.planted.cause_layer, \
                (f"{v.violation_id}: planted {sc.planted.cause_layer}, "
                 f"attributed {a.primary_layer}")

    def test_planning_error_with_perfect_perception(self, attributions):
        """THE keystone case: a planted planning fault where perception was
        perfect must attribute to planning with perception RULED_OUT."""
        cases = _cases(attributions, "planning")
        assert cases, "bank must contain planning-cause violations"
        for sc, v, a in cases:
            assert a.primary_layer == "planning"
            assert a.layers["perception"].status == "RULED_OUT"
            assert a.layers["planning"].status == "SUPPORTED"
            assert "OWN world view" in a.layers["planning"].evidence

    def test_map_error_case(self, attributions):
        cases = _cases(attributions, "map")
        assert cases
        for sc, v, a in cases:
            assert a.primary_layer == "map"
            assert a.layers["map"].status == "SUPPORTED"
            # The plan was compliant with the (wrong) map: planning ruled out.
            assert a.layers["planning"].status == "RULED_OUT"

    def test_localization_error_case(self, attributions):
        cases = _cases(attributions, "localization")
        assert cases
        for sc, v, a in cases:
            assert a.primary_layer == "localization"
            assert a.layers["localization"].status == "SUPPORTED"
            assert a.layers["perception"].status == "RULED_OUT"

    def test_control_error_case(self, attributions):
        cases = _cases(attributions, "control")
        assert cases
        for sc, v, a in cases:
            assert a.primary_layer == "control"
            assert a.layers["control"].status == "SUPPORTED"
            assert a.layers["planning"].status == "RULED_OUT"

    def test_prediction_error_case(self, attributions):
        cases = _cases(attributions, "prediction")
        assert cases
        for sc, v, a in cases:
            assert a.primary_layer == "prediction"
            assert a.layers["perception"].status == "RULED_OUT"

    def test_data_label_case_keeps_unassessable_layers_unknown(self, attributions):
        cases = _cases(attributions, "data_label")
        assert cases
        for sc, v, a in cases:
            assert a.primary_layer == "data_label"
            assert a.layers["perception"].status == "UNKNOWN"
            assert a.layers["prediction"].status == "UNKNOWN"


class TestNoAutoPerception:
    def test_perception_primary_requires_positive_perception_evidence(
            self, attributions):
        for sc, v, a in attributions:
            if a.primary_layer == "perception":
                assert a.layers["perception"].status == "SUPPORTED"

    def test_tri_state_everywhere(self, attributions):
        for _, _, a in attributions:
            assert set(a.layers) == {"perception", "prediction", "planning",
                                     "localization", "map", "control",
                                     "policy_rule", "data_label"}
            for le in a.layers.values():
                assert le.status in ("SUPPORTED", "RULED_OUT", "UNKNOWN")
                assert le.evidence
