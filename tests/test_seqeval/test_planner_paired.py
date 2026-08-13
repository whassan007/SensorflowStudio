"""Frozen-plan bias prevention, allocation properties, paired harness and the
prediction cache."""

from __future__ import annotations

import numpy as np
import pytest

from sensorflow.seqeval import paired, planner


SAFETY = ["pedestrian|night", "cyclist|night"]


def _plan(meta, **kw):
    args = dict(target_n=6000, seed=99, safety_primaries=SAFETY,
                min_per_stratum=120, safety_floor=800, persist=False)
    args.update(kw)
    return planner.build_plan(meta, **args)


class TestFrozenPlan:
    def test_plan_hash_reproducible_and_candidate_independent(self, seq_env):
        """The planner signature admits no candidate/outcome inputs; identical
        (population, config, seed) must give identical plans."""
        meta = seq_env["meta"]
        p1 = _plan(meta)
        p2 = _plan(meta)
        assert p1["plan_hash"] == p2["plan_hash"]
        assert p1["frozen_before_candidate_outcomes"] is True

    def test_plan_hash_changes_with_seed_and_config(self, seq_env):
        meta = seq_env["meta"]
        assert _plan(meta)["plan_hash"] != _plan(meta, seed=100)["plan_hash"]
        assert _plan(meta)["plan_hash"] != _plan(meta, target_n=7000)["plan_hash"]

    def test_allocation_floors_and_weights(self, seq_env):
        meta = seq_env["meta"]
        plan = _plan(meta)
        for s, v in plan["strata"].items():
            assert v["allocated"] <= v["N"]
            assert v["allocated"] >= min(120, v["N"])
            # HT weight consistent with allocation
            assert v["weight"] == pytest.approx(v["N"] / v["allocated"], rel=0.01)
            if v["key"] in SAFETY:
                assert v["safety_primary"]
                assert v["allocated"] >= min(800, int(0.85 * v["N"]))

    def test_whole_clusters_selected(self, seq_env):
        """Every selected container contributes ALL of its stratum objects, so
        the cluster inference unit is intact."""
        meta = seq_env["meta"]
        plan = _plan(meta)
        frame = paired.load_frame(meta)
        s = next(int(k) for k, v in plan["strata"].items()
                 if v["key"] == "pedestrian|night")
        clusters = planner.stratum_clusters(plan, s)
        assert clusters, "expected non-empty stratum"
        for arr in clusters[:50]:
            cid = frame["container_id"][arr]
            assert np.unique(cid).size == 1  # one container per cluster unit
            members = np.where((frame["container_id"] == cid[0])
                               & (frame["class"] == 1) & (frame["lighting"] == 1))[0]
            assert set(arr.tolist()) == set(members.tolist())

    def test_persisted_plan_roundtrip(self, seq_env):
        meta = seq_env["meta"]
        plan = _plan(meta, persist=True)
        loaded = planner.load_plan(plan["plan_id"])
        assert loaded is not None
        assert loaded["plan_hash"] == plan["plan_hash"]
        s = next(iter(loaded["strata"]))
        np.testing.assert_array_equal(loaded["_arrays"][f"s{s}_ids"],
                                      plan["_arrays"][f"s{s}_ids"])


class TestPairedHarness:
    def test_planted_effect_shows_up_where_planted(self, seq_env):
        meta = seq_env["meta"]
        frame = paired.load_frame(meta)
        base = paired.SimulatedModel("harness-base")
        cand = paired.SimulatedModel("harness-cand",
                                     {"pedestrian|night": -0.05, "__global__": 0.0})
        ids = np.arange(meta["num_objects"], dtype=np.int64)
        oc = paired.paired_outcomes(meta, base, cand, ids)
        ped_night = (frame["class"] == 1) & (frame["lighting"] == 1)
        delta_in = oc["d"][ped_night].mean()
        delta_out = oc["d"][~ped_night].mean()
        assert delta_in < -0.03
        assert abs(delta_out) < 0.005

    def test_pairing_reduces_variance(self, seq_env):
        """Paired discordance must be far below what independent runs of two
        models with these accuracies would produce."""
        meta = seq_env["meta"]
        base = paired.SimulatedModel("harness-base")
        cand = paired.SimulatedModel("harness-cand2")
        ids = np.arange(meta["num_objects"], dtype=np.int64)
        oc = paired.paired_outcomes(meta, base, cand, ids)
        p = oc["baseline"].mean()
        independent_disagreement = 2 * p * (1 - p)
        paired_disagreement = (oc["baseline"] != oc["candidate"]).mean()
        assert paired_disagreement < 0.4 * independent_disagreement

    def test_deterministic_given_fingerprints(self, seq_env):
        meta = seq_env["meta"]
        m = paired.SimulatedModel("det-check", {"pedestrian|night": -0.02})
        ids = np.arange(0, 5000, dtype=np.int64)
        a = paired.paired_outcomes(meta, paired.SimulatedModel("det-base"), m, ids)
        paired.get_prediction_cache().clear_memory()
        b = paired.paired_outcomes(meta, paired.SimulatedModel("det-base"), m, ids)
        np.testing.assert_array_equal(a["d"], b["d"])


class TestPredictionCache:
    def test_baseline_not_recomputed_across_candidates(self, seq_env):
        meta = seq_env["meta"]
        paired.reset_prediction_cache()
        base = paired.SimulatedModel("cache-base-v1")
        key = f"{paired.dataset_fingerprint(meta)}-{base.fingerprint()}"
        ids = np.arange(0, 20_000, dtype=np.int64)
        for i in range(3):  # three successive candidate updates
            cand = paired.SimulatedModel(f"cache-cand-v{i}")
            paired.paired_outcomes(meta, base, cand, ids)
        assert paired.COMPUTE_COUNTS[key] == 1

    def test_cache_key_tracks_model_effects(self, seq_env):
        meta = seq_env["meta"]
        m1 = paired.SimulatedModel("same-version", {"pedestrian|night": -0.02})
        m2 = paired.SimulatedModel("same-version", {"pedestrian|night": -0.03})
        assert m1.fingerprint() != m2.fingerprint()
