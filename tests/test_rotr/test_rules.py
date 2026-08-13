"""Rule engine: every planted violation found, every planted NON-violation
rejected (no false accusations), deterministic output."""

from __future__ import annotations

from sensorflow.rotr.rules import RULESET_VERSION, detect


class TestPlantedViolations:
    def test_every_committed_violation_detected_with_expected_rule(
            self, bank_v1, detections_v1):
        for sc in bank_v1:
            if not sc.planted.committed:
                continue
            hits = detections_v1[sc.scenario_id]
            assert any(v.rule_id == sc.planted.expected_rule_id
                       for v in hits), \
                f"{sc.scenario_id} ({sc.planted.kind}) missed"

    def test_violations_carry_structured_evidence_and_version(
            self, bank_v1, detections_v1):
        for sc in bank_v1:
            for v in detections_v1[sc.scenario_id]:
                assert v.rule_version == RULESET_VERSION
                assert v.evidence, "evidence fields required"
                assert "ruleset_fingerprint" in v.evidence
                assert 0.0 < v.confidence <= 1.0
                assert v.provenance.model_version == "stack-v1"


class TestPlantedNonViolations:
    def test_no_false_accusations(self, bank_v1, detections_v1):
        """Lookalike scenarios (legal assertive merge, yield-with-ROW,
        green proceed) and non-committed variants must produce ZERO
        violations."""
        for sc in bank_v1:
            if sc.planted.committed:
                continue
            assert detections_v1[sc.scenario_id] == [], \
                f"false accusation on {sc.scenario_id} ({sc.planted.kind})"


class TestDeterminism:
    def test_detect_is_pure(self, bank_v1):
        sc = next(s for s in bank_v1 if s.planted.committed)
        a = [v.model_dump(exclude={"provenance"}) for v in detect(sc)]
        b = [v.model_dump(exclude={"provenance"}) for v in detect(sc)]
        assert a == b
