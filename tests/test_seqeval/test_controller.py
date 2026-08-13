"""End-to-end controller behavior: type-I control, power on the masking case,
early-stopping efficiency, three-outcome logic, ledger completeness and
reproducibility."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sensorflow.seqeval import ledger as ledger_mod
from sensorflow.seqeval import paired
from sensorflow.seqeval.controller import (INSUFFICIENT_LANGUAGE,
                                           evaluate_regression,
                                           get_seqeval_store)
from sensorflow.seqeval.sequential import (DECISION_INSUFFICIENT, DECISION_PASS,
                                           DECISION_REGRESSION)
from tests.test_seqeval.conftest import FAST_POLICY

BASE = {"model_version": "ctl-base-v1"}


def _run(seq_env, candidate, policy=None, seed=None):
    merged = {**FAST_POLICY, **(policy or {})}
    return evaluate_regression(seq_env["meta"]["population_id"], BASE,
                               candidate, policy=merged, seed=seed)


class TestTypeIControl:
    def test_false_regression_rate_at_most_alpha(self, seq_env):
        """Monte Carlo over no-regression candidates: the whole pipeline
        (screening + sequential looks + hierarchy) must not cry wolf."""
        reps = 24
        false_regressions = 0
        for i in range(reps):
            res = _run(seq_env, {"model_version": f"null-cand-{i}"})
            if res["decision"] == DECISION_REGRESSION:
                false_regressions += 1
        # alpha=0.05: even 2/24 (~0.083) would signal a broken guarantee given
        # the margin buffer; allow exactly the alpha level rounded up.
        assert false_regressions <= max(1, int(np.ceil(0.05 * reps)))


class TestPowerAndMasking:
    def test_2pp_pedestrian_night_regression_detected_despite_overall_gain(self, seq_env):
        meta = seq_env["meta"]
        candidate = {"model_version": "masked-reg-v1",
                     "effects": {"pedestrian|night": -0.02, "__global__": 0.004}}
        res = _run(seq_env, candidate,
                   policy={"target_n": 16000, "safety_floor": 2500,
                           "escalation": {"enabled": True,
                                          "max_extra_per_stratum": 4000,
                                          "batch_objects": 500}})
        assert res["decision"] == DECISION_REGRESSION
        assert "stratum:pedestrian|night" in res["affected_strata"]
        assert res["samples_used"] < 0.5 * res["full_population"]

        # the masking case: the OVERALL population metric actually improved
        cache = paired.get_prediction_cache()
        b = cache.get_or_compute(meta, paired.SimulatedModel(**BASE))
        c = cache.get_or_compute(meta, paired.SimulatedModel(
            candidate["model_version"], candidate["effects"]))
        assert float(c.mean() - b.mean()) > 0  # naive headline says "better"

        # attribution carries magnitude + CI + decision for the regressed stratum
        row = next(r for r in res["regression_map"]
                   if r["node"] == "stratum:pedestrian|night")
        assert row["abs_delta"] < -0.01
        # the e-process crossed its anytime-valid boundary 1/alpha_allocated
        assert row["e_regression"] >= 1.0 / row["alpha_allocated"]
        # and the CS brackets the true planted effect (~ -2pp)
        assert row["delta_ci"][0] < -0.02 < row["delta_ci"][1]
        assert row["safety_primary"] is True
        assert row["baseline_value"] is not None and row["candidate_value"] is not None


class TestEarlyStopping:
    def test_large_regression_stops_far_before_budget(self, seq_env):
        res = _run(seq_env, {"model_version": "huge-reg-v1",
                             "effects": {"pedestrian|night": -0.10}},
                   policy={"target_n": 16000, "safety_floor": 2500})
        assert res["decision"] == DECISION_REGRESSION
        assert res["stopping_reason"] == "regression_confirmed"
        assert res["samples_used"] < 0.4 * res["planned_total"]
        assert res["samples_used"] < 0.06 * res["full_population"]


class TestThreeOutcomeGate:
    def test_tiny_budget_yields_insufficient_never_pass(self, seq_env):
        res = _run(seq_env, {"model_version": "small-budget-cand"},
                   policy={"target_n": 700, "min_per_stratum": 20,
                           "safety_floor": 40,
                           "stage_fractions": [1.0], "batches_per_stage": 1,
                           "escalation": {"enabled": False}})
        assert res["decision"] == DECISION_INSUFFICIENT
        assert res["gate"] == "expand_or_report"
        assert "NOT PROVEN EQUIVALENT" in res["message"]

    def test_equivalent_candidate_passes_with_realistic_margin(self, seq_env):
        res = _run(seq_env, {"model_version": "equiv-cand-v1"},
                   policy={"delta_margin": 0.02, "target_n": 9000,
                           "safety_floor": 1200})
        assert res["decision"] == DECISION_PASS
        assert res["gate"] == "allow"


class TestEvidenceLedger:
    def test_ledger_complete_and_reproducible(self, seq_env):
        candidate = {"model_version": "ledger-cand-v1",
                     "effects": {"pedestrian|night": -0.02}}
        res1 = _run(seq_env, candidate, seed=777)
        res2 = _run(seq_env, candidate, seed=777)

        # reproducibility: same seed/models/policy -> same decisions everywhere
        assert res1["decision"] == res2["decision"]
        assert res1["samples_used"] == res2["samples_used"]
        map1 = {r["node"]: r["decision"] for r in res1["regression_map"]}
        map2 = {r["node"]: r["decision"] for r in res2["regression_map"]}
        assert map1 == map2

        # completeness: every record carries every required field, non-null core
        ledger = ledger_mod.EvidenceLedger(res1["run_id"])
        records = ledger.records()
        assert len(records) >= 20  # overall + classes + strata + difficulty
        for rec in records:
            for field in ledger_mod.REQUIRED_FIELDS:
                assert field in rec, f"missing {field}"
            assert rec["n"] > 0
            assert rec["n_effective"] > 0
            assert rec["decision"] in (DECISION_REGRESSION, DECISION_PASS,
                                       DECISION_INSUFFICIENT)
            assert rec["test_method"]
            assert rec["multiple_testing_method"]
        lineage = ledger.lineage()
        assert lineage["evaluator_version"].startswith("seqeval-")
        assert lineage["plan_hash"]
        assert lineage["statistical_config"]["delta_margin"] == 0.005

    def test_plan_hash_stable_across_candidates(self, seq_env):
        """Sampling-bias prevention at the pipeline level: two candidates with
        opposite outcomes see the exact same frozen plan."""
        r1 = _run(seq_env, {"model_version": "hash-cand-good"})
        r2 = _run(seq_env, {"model_version": "hash-cand-bad",
                            "effects": {"pedestrian|night": -0.10}})
        store = get_seqeval_store()
        s1 = store.get_state(r1["run_id"], include_trajectories=False)
        s2 = store.get_state(r2["run_id"], include_trajectories=False)
        assert s1["plan"]["plan_hash"] == s2["plan"]["plan_hash"]

    def test_run_state_persisted_to_disk(self, seq_env):
        res = _run(seq_env, {"model_version": "persist-cand"})
        path = os.path.join(ledger_mod.run_dir(res["run_id"]), "run.json")
        assert os.path.exists(path)
        with open(path) as f:
            state = json.load(f)
        assert state["decision"] == res["decision"]
        assert state["budget"]["samples_used"] == res["samples_used"]
