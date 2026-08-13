"""Stop-ship trigger: fires ONLY on the full deterministic conjunction."""

from __future__ import annotations

from sensorflow.rotr.stopship import (
    STOPSHIP_POLICY, evaluate_gate, policy_version,
)


def _item(vru=True, missed=True, sc=True, vid="v-1"):
    return {
        "violation_id": vid,
        "signature": {"vulnerability": "VRU" if vru else "NON_VRU"},
        "attribution": {"layers": {"perception": {
            "status": "SUPPORTED" if missed else "RULED_OUT",
            "evidence": "actor: missed detection — GT visible" if missed
            else "detected fine"}}},
        "consequence": {"consequence_class":
                        "SAFETY_CRITICAL" if sc else "DEGRADED_COMFORT"},
    }


class TestConjunction:
    def test_full_conjunction_fires_no_go(self):
        gate = evaluate_gate("run-x", [_item()], forward_to_agentic=False)
        assert gate.outcome == "NO_GO"
        assert any(e["fired"] for e in gate.events)

    def test_any_two_of_three_do_not_fire(self):
        for kwargs in ({"vru": False}, {"missed": False}, {"sc": False}):
            gate = evaluate_gate("run-x", [_item(**kwargs)],
                                 forward_to_agentic=False)
            assert gate.outcome == "GO", f"fired on partial match {kwargs}"

    def test_perception_supported_but_not_missed_does_not_fire(self):
        item = _item()
        item["attribution"]["layers"]["perception"]["evidence"] = \
            "detected but mislocalized (mean |err| 1.4 m)"
        gate = evaluate_gate("run-x", [item], forward_to_agentic=False)
        assert gate.outcome == "GO"

    def test_empty_run_is_go(self):
        gate = evaluate_gate("run-x", [], forward_to_agentic=False)
        assert gate.outcome == "GO"
        assert gate.events == []


class TestPolicy:
    def test_policy_version_is_content_hash(self):
        v = policy_version()
        assert v == policy_version()
        assert v != policy_version({**STOPSHIP_POLICY, "policy_semver": "9.9.9"})
        assert len(v) == 16

    def test_not_llm_driven_is_explicit_policy_text(self):
        assert "NOT LLM-driven" in STOPSHIP_POLICY["notes"]


class TestAgenticForwarding:
    def test_no_go_forwards_advisory_when_agentic_importable(self):
        gate = evaluate_gate("run-x", [_item()], forward_to_agentic=True)
        assert gate.outcome == "NO_GO"
        # agentic is a concurrent workstream: either a real advisory or an
        # honest 'unavailable' record — never a crash, never a changed gate.
        assert gate.agentic_advisory is not None
        assert "engine" in gate.agentic_advisory

    def test_gate_outcome_never_depends_on_forwarding(self):
        a = evaluate_gate("run-x", [_item()], forward_to_agentic=False)
        b = evaluate_gate("run-x", [_item()], forward_to_agentic=True)
        assert a.outcome == b.outcome == "NO_GO"
