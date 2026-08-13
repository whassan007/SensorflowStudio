"""Cube correctness: cube aggregates must EXACTLY match brute-force computation
over the raw per-object records, and re-running with the same lineage must be
bit-identical (deterministic seeds)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from sensorflow.megaeval import cube as cube_mod
from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIM_NAMES, DIMENSIONS
from sensorflow.megaeval.runs import run_dir


def _brute_force(env, group_dims):
    """Recompute tp/fp/fn/n per group directly from raw per-object outputs."""
    run = env["good"]
    frames = []
    d = run_dir(run.run_id)
    for p in range(run.partitions_total):
        pop_cols = pop_mod.load_partition(run.population_id, p)
        with np.load(os.path.join(d, f"objects-part-{p:04d}.npz")) as z:
            df = pd.DataFrame({
                **{k: pop_cols[k] for k in DIM_NAMES},
                "tp": z["tp"].astype(np.int64),
                "detected": z["detected"].astype(bool),
            })
        df["fn"] = 1 - df["tp"]
        df["fp"] = (df["detected"] & (df["tp"] == 0)).astype(np.int64)
        df["n"] = 1
        frames.append(df[group_dims + ["n", "tp", "fp", "fn"]])
        with np.load(os.path.join(d, f"fp-part-{p:04d}.npz")) as z:
            if z["container_id"].size:
                fpdf = pd.DataFrame({k: z[k] for k in DIM_NAMES})
                fpdf["n"] = 0
                fpdf["tp"] = 0
                fpdf["fp"] = 1
                fpdf["fn"] = 0
                frames.append(fpdf[group_dims + ["n", "tp", "fp", "fn"]])
    full = pd.concat(frames, ignore_index=True)
    return full.groupby(group_dims, as_index=False).sum()


@pytest.mark.parametrize("group_dims", [["class"], ["class", "lighting"], ["scenario"]])
def test_cube_matches_brute_force_exactly(mega_env, group_dims):
    run = mega_env["good"]
    cube = mega_env["store"].artifacts(run.run_id)["cube"]
    rows, _ = cube_mod.aggregate(cube, None, group_dims,
                                 ["n", "tp", "fp", "fn", "precision", "recall"], 2000)
    brute = _brute_force(mega_env, group_dims)
    assert len(rows) == len(brute)
    for _, b in brute.iterrows():
        key = {d: DIMENSIONS[d][int(b[d])] for d in group_dims}
        match = [r for r in rows if all(r[d] == key[d] for d in group_dims)]
        assert len(match) == 1, f"missing cube row for {key}"
        r = match[0]
        assert r["n"] == int(b["n"])
        assert r["tp"] == int(b["tp"])
        assert r["fp"] == int(b["fp"])
        assert r["fn"] == int(b["fn"])
        # derived ratios match brute force (API rounds ratios to 6 decimals)
        if b["tp"] + b["fp"] > 0:
            assert r["precision"] == pytest.approx(b["tp"] / (b["tp"] + b["fp"]), abs=1e-6)
        if b["tp"] + b["fn"] > 0:
            assert r["recall"] == pytest.approx(b["tp"] / (b["tp"] + b["fn"]), abs=1e-6)


def test_rollup_consistency(mega_env):
    """Any grouped rollup must sum to the global totals (no double counting)."""
    run = mega_env["good"]
    cube = mega_env["store"].artifacts(run.run_id)["cube"]
    total_rows, _ = cube_mod.aggregate(cube, None, None, ["n", "tp", "fp", "fn"], 1)
    total = total_rows[0]
    for dim in ("class", "weather", "occlusion", "distance_band"):
        rows, _ = cube_mod.aggregate(cube, None, [dim], ["n", "tp", "fp", "fn"], 2000)
        for stat in ("n", "tp", "fp", "fn"):
            assert sum(r[stat] for r in rows) == total[stat], f"{dim}/{stat} rollup mismatch"


def test_filtered_aggregate_equals_manual_subset(mega_env):
    run = mega_env["good"]
    cube = mega_env["store"].artifacts(run.run_id)["cube"]
    filt = {"class": ["pedestrian"], "lighting": ["night"]}
    rows, _ = cube_mod.aggregate(cube, filt, None, ["n", "tp", "fn", "recall"], 1)
    sub = cube[(cube["class"] == DIMENSIONS["class"].index("pedestrian"))
               & (cube["lighting"] == DIMENSIONS["lighting"].index("night"))]
    assert rows[0]["n"] == int(sub["n"].sum())
    assert rows[0]["tp"] == int(sub["tp"].sum())
    assert rows[0]["recall"] == pytest.approx(sub["tp"].sum() / max(sub["tp"].sum() + sub["fn"].sum(), 1), abs=1e-6)


def test_reproducible_run_same_lineage(mega_env):
    """Same population + model_version (same derived seed) -> identical cube."""
    store, meta = mega_env["store"], mega_env["meta"]
    r1 = store.create_run(population_id=meta["population_id"],
                          model_version="model-repro", worker_delay_s=0.0)
    store.execute_sync(r1)
    r2 = store.create_run(population_id=meta["population_id"],
                          model_version="model-repro", worker_delay_s=0.0)
    store.execute_sync(r2)
    assert r1.seed == r2.seed
    c1 = mega_env["store"].artifacts(r1.run_id)["cube"].sort_values(DIM_NAMES).reset_index(drop=True)
    c2 = mega_env["store"].artifacts(r2.run_id)["cube"].sort_values(DIM_NAMES).reset_index(drop=True)
    pd.testing.assert_frame_equal(c1, c2)
    assert r1.headline == {**r2.headline}
