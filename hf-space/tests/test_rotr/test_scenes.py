"""Scenario substrate: determinism, planted truth, environment taxonomy."""

from __future__ import annotations

import pytest

from sensorflow.rotr.scenes import MODEL_PROFILES, bank_id_for, generate_bank


class TestDeterminism:
    def test_same_inputs_identical_banks(self):
        a = generate_bank(14, seed=3, model_version="stack-v1")
        b = generate_bank(14, seed=3, model_version="stack-v1")
        # Wall-clock provenance timestamps are the only permitted difference.
        da = [s.model_dump(exclude={"provenance": {"timestamp"}}) for s in a]
        db = [s.model_dump(exclude={"provenance": {"timestamp"}}) for s in b]
        assert da == db

    def test_seed_changes_bank_id_and_content(self):
        assert bank_id_for(14, 3, "stack-v1") != bank_id_for(14, 4, "stack-v1")

    def test_model_version_changes_committed_pattern_not_geometry(self):
        v1 = generate_bank(28, seed=7, model_version="stack-v1")
        v2 = generate_bank(28, seed=7, model_version="stack-v2-improved")
        # Same opportunity mix (same seed) ...
        assert [s.planted.kind for s in v1] == [s.planted.kind for s in v2]
        # ... but a better perception stack commits fewer perception-caused.
        def committed_perception(bank):
            return sum(1 for s in bank if s.planted.committed
                       and s.planted.cause_layer == "perception")
        assert committed_perception(v2) < committed_perception(v1)

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError):
            generate_bank(4, seed=1, model_version="stack-vX")


class TestPlantedTruth:
    def test_bank_contains_violations_and_lookalike_non_violations(self, bank_v1):
        kinds = {s.planted.kind for s in bank_v1}
        assert {"fail_yield_pedestrian", "restricted_path_entry",
                "wrong_lane_association", "intersection_conflict",
                "unsafe_merge", "stop_overshoot"} <= kinds
        non_viol = {s.planted.kind for s in bank_v1
                    if not s.planted.is_violation_opportunity}
        assert {"legal_assertive_merge", "yield_with_right_of_way",
                "green_proceed"} <= non_viol

    def test_non_opportunities_never_committed(self, bank_v1):
        for s in bank_v1:
            if not s.planted.is_violation_opportunity:
                assert not s.planted.committed
                assert s.planted.expected_rule_id is None

    def test_environment_attributes_cover_low_visibility_yield_cases(self, bank_v1):
        # The spec's taxonomy query needs this cohort to exist.
        assert any(s.planted.kind == "fail_yield_pedestrian"
                   and s.planted.committed
                   and s.environment.visibility == "low" for s in bank_v1)

    def test_provenance_complete(self, bank_v1):
        for s in bank_v1:
            p = s.provenance
            assert p.scenario_id == s.scenario_id
            assert p.dataset_version and p.model_version and p.software_version
            assert p.source == "SYNTHETIC"


class TestProfiles:
    def test_profiles_cover_all_planted_causes(self):
        causes = {"perception", "prediction", "planning", "localization",
                  "map", "control", "data_label"}
        for profile in MODEL_PROFILES.values():
            assert causes <= set(profile)
