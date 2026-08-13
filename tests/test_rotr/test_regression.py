"""Baseline-vs-candidate regression: seqeval delegation verdicts and the
six-outcome distinction (slowest module: two 140-scenario runs + a
sequential test over a dedicated megaeval population)."""

from __future__ import annotations

import pytest

from sensorflow.rotr import pipeline
from sensorflow.rotr import store as rotr_store
from sensorflow.rotr.metrics import evaluate_candidate


@pytest.fixture(scope="module")
def reg_env(tmp_path_factory):
    from sensorflow.megaeval import population as pop_mod
    from sensorflow.seqeval import ledger as ledger_mod
    from sensorflow.seqeval.controller import reset_seqeval_store

    root = tmp_path_factory.mktemp("rotr_reg")
    pop_mod.set_mega_root(str(root / "mega"))
    ledger_mod.set_seqeval_root(str(root / "seq"))
    reset_seqeval_store()
    rotr_store.set_rotr_root(str(root / "rotr"))
    runs = {
        "v2": pipeline.execute_run(n_scenarios=140, seed=7,
                                   model_version="stack-v2-improved"),
        "v3": pipeline.execute_run(n_scenarios=140, seed=7,
                                   model_version="stack-v3-planning-regression"),
    }
    yield runs
    pop_mod.set_mega_root("runs/megaeval")
    ledger_mod.set_seqeval_root("runs/seqeval")
    reset_seqeval_store()


class TestSixOutcomes:
    def test_planted_planning_regression_is_caught_and_safety_critical(self, reg_env):
        r = evaluate_candidate("t-reg", reg_env["v2"], reg_env["v3"], seed=11)
        assert r.six_outcomes["observed_difference"]
        assert r.six_outcomes["statistically_significant"], \
            f"seqeval said {r.seqeval and r.seqeval.get('decision')}"
        assert r.six_outcomes["practically_significant"]
        assert r.six_outcomes["safety_critical"]
        assert not r.six_outcomes["distribution_driven"], \
            "same-seed banks share the opportunity mix"
        assert r.primary_outcome == "SAFETY_CRITICAL_REGRESSION"
        assert r.seqeval["decision"] == "REGRESSION"
        assert r.metric_deltas["violation_rate_VRU"] > 0

    def test_self_comparison_shows_no_difference(self, reg_env):
        r = evaluate_candidate("t-self", reg_env["v2"], reg_env["v2"], seed=11)
        assert not r.six_outcomes["observed_difference"]
        assert not r.six_outcomes["statistically_significant"]
        assert not r.six_outcomes["safety_critical"]
        assert r.primary_outcome in ("NO_DIFFERENCE", "INSUFFICIENT_EVIDENCE")

    def test_delegation_payload_is_attached_with_translated_effects(self, reg_env):
        r = evaluate_candidate("t-payload", reg_env["v2"], reg_env["v3"], seed=11)
        assert r.seqeval and "run_id" in r.seqeval
        eff = r.seqeval["translated_effects"]
        # rate increases translate to NEGATIVE performance effects
        assert eff["pedestrian|night"] < 0
        assert eff["__global__"] < 0
