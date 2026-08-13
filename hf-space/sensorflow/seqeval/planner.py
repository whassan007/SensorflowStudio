"""Frozen stratified sampling planner (hybrid Neyman + risk floors + safety minimums).

The plan is built from the POPULATION ONLY (dimension codes, counts, and a
candidate-independent prior on baseline variance) and is persisted with its
seed, config and a content hash BEFORE any candidate outcome is observed.
This is the sampling-bias firewall: no outcome-dependent selection is possible
because the selection provably cannot depend on outcomes (the planner never
sees them), and the hash gives auditors a cheap way to verify the plan did not
change after results started arriving.

Allocation (hybrid, in order):
  1. Neyman baseline: n_h proportional to N_h * sigma_h, where sigma_h =
     sqrt(p_h (1 - p_h)) uses a PRIOR per-stratum baseline success rate
     (deterministic profile; never candidate data).
  2. Risk weighting: VRU-class and night/adverse-condition strata get a
     multiplicative boost (they are where regressions hurt).
  3. Floors: every stratum gets a mandatory minimum; pre-registered
     safety-critical strata get a much larger minimum sized for the target MDE.
  4. Truncation at stratum population size; whole CLUSTERS (container groups)
     are selected so the inference unit stays intact.

Because allocation is non-proportional, every sampled object carries the
Horvitz-Thompson style weight w_h = N_h / n_h; population-level estimates are
weight-corrected and therefore remain unbiased despite oversampling.

The plan also stores a per-stratum escalation RESERVE (the remaining cluster
units in shuffled order) so the controller can expand the sample without
breaking the frozen-selection property: the reserve order was fixed at plan
time too.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

from sensorflow.megaeval import population as pop_mod
from sensorflow.seqeval import ledger as ledger_mod
from sensorflow.seqeval import paired as paired_mod

DIFFICULTY_BANDS = [0.30, 0.55]  # easy < 0.30 <= medium < 0.55 <= hard
BAND_NAMES = ["easy", "medium", "hard"]

# Candidate-independent prior baseline success rates used for Neyman sigma_h.
_PRIOR_BASE = 0.90
_PRIOR_NIGHT_DROP = 0.04
_PRIOR_VRU_DROP = 0.03


def stratum_key(cls_name: str, cond_name: str) -> str:
    return f"{cls_name}|{cond_name}"


def _prior_rate(cls_code: int, cond_code: int, condition_dim: str) -> float:
    p = _PRIOR_BASE
    if condition_dim == "lighting" and cond_code == 1:
        p -= _PRIOR_NIGHT_DROP
    if cls_code in paired_mod.VRU_CLASSES:
        p -= _PRIOR_VRU_DROP
    return p


def _risk_weight(cls_code: int, cond_code: int, condition_dim: str) -> float:
    w = 1.0
    if cls_code in paired_mod.VRU_CLASSES:
        w *= 1.5
    if condition_dim == "lighting" and cond_code == 1:
        w *= 1.3
    return w


def plan_path(plan_id: str) -> str:
    return os.path.join(ledger_mod.seq_root(), "plans", plan_id)


def build_plan(meta: Dict, target_n: int, seed: int,
               condition_dim: str = "lighting",
               safety_primaries: Optional[List[str]] = None,
               min_per_stratum: int = 200,
               safety_floor: int = 2500,
               persist: bool = True) -> Dict:
    """Build (and persist) a frozen stratified cluster-sampling plan.

    Takes NO candidate/model arguments by design — see module docstring.
    """
    safety_primaries = list(safety_primaries or [])
    frame = paired_mod.load_frame(meta)
    cls_vocab = pop_mod.DIMENSIONS["class"]
    cond_vocab = pop_mod.DIMENSIONS[condition_dim]
    n_cond = len(cond_vocab)
    strata_ids = frame["class"].astype(np.int64) * n_cond + frame[condition_dim].astype(np.int64)
    n_strata = len(cls_vocab) * n_cond

    counts = np.bincount(strata_ids, minlength=n_strata)
    keys = [stratum_key(cls_vocab[s // n_cond], cond_vocab[s % n_cond])
            for s in range(n_strata)]

    # --- 1-2. Neyman shares with risk multipliers (prior variance, no outcomes)
    shares = np.zeros(n_strata)
    for s in range(n_strata):
        if counts[s] == 0:
            continue
        p = _prior_rate(s // n_cond, s % n_cond, condition_dim)
        sigma = np.sqrt(p * (1 - p))
        shares[s] = counts[s] * sigma * _risk_weight(s // n_cond, s % n_cond, condition_dim)
    shares = shares / shares.sum() if shares.sum() > 0 else shares

    # --- 3. floors and safety minimums, truncated at N_h
    alloc = np.zeros(n_strata, dtype=np.int64)
    for s in range(n_strata):
        if counts[s] == 0:
            continue
        n_h = int(round(target_n * shares[s]))
        n_h = max(n_h, min(min_per_stratum, int(counts[s])))
        if keys[s] in safety_primaries:
            n_h = max(n_h, min(safety_floor, int(0.85 * counts[s])))
        alloc[s] = min(n_h, int(counts[s]))

    # --- 4. whole-cluster selection in seeded shuffled order + reserve
    rng = np.random.default_rng([int(seed), int(meta["seed"]), 20260812])
    strata = {}
    arrays = {}
    for s in range(n_strata):
        if counts[s] == 0:
            continue
        rows = np.where(strata_ids == s)[0]
        containers = frame["container_id"][rows]
        order = np.argsort(containers, kind="stable")
        rows, containers = rows[order], containers[order]
        uniq, start = np.unique(containers, return_index=True)
        bounds = np.append(start, rows.size)
        cluster_order = rng.permutation(uniq.size)

        sel_ids, sel_bounds = [], [0]
        res_ids, res_bounds = [], [0]
        taken = 0
        for ci in cluster_order:
            members = frame["object_id"][rows[bounds[ci]:bounds[ci + 1]]]
            if taken < alloc[s]:
                sel_ids.append(members)
                sel_bounds.append(sel_bounds[-1] + members.size)
                taken += members.size
            else:
                res_ids.append(members)
                res_bounds.append(res_bounds[-1] + members.size)
        sel = np.concatenate(sel_ids) if sel_ids else np.empty(0, dtype=np.int64)
        res = np.concatenate(res_ids) if res_ids else np.empty(0, dtype=np.int64)
        arrays[f"s{s}_ids"] = sel
        arrays[f"s{s}_bounds"] = np.asarray(sel_bounds, dtype=np.int64)
        arrays[f"s{s}_res_ids"] = res
        arrays[f"s{s}_res_bounds"] = np.asarray(res_bounds, dtype=np.int64)
        strata[str(s)] = {
            "key": keys[s],
            "class": cls_vocab[s // n_cond],
            "condition": cond_vocab[s % n_cond],
            "N": int(counts[s]),
            "allocated": int(sel.size),
            "clusters": len(sel_bounds) - 1,
            "reserve": int(res.size),
            "weight": round(float(counts[s]) / max(sel.size, 1), 4),
            "safety_primary": keys[s] in safety_primaries,
        }

    config = {
        "target_n": int(target_n),
        "condition_dim": condition_dim,
        "safety_primaries": safety_primaries,
        "min_per_stratum": int(min_per_stratum),
        "safety_floor": int(safety_floor),
        "difficulty_bands": DIFFICULTY_BANDS,
        "allocation": "neyman(prior sigma) * risk_weights, floors, whole clusters",
    }
    h = hashlib.sha256()
    h.update(json.dumps({"dataset": paired_mod.dataset_fingerprint(meta),
                         "seed": int(seed), "config": config}, sort_keys=True).encode())
    for s in sorted(int(k) for k in strata):
        h.update(arrays[f"s{s}_ids"].tobytes())
        h.update(arrays[f"s{s}_res_ids"].tobytes())
    plan_hash = h.hexdigest()[:24]

    plan = {
        "plan_id": f"plan-{plan_hash[:10]}",
        "plan_hash": plan_hash,
        "population_id": meta["population_id"],
        "dataset_fingerprint": paired_mod.dataset_fingerprint(meta),
        "seed": int(seed),
        "created_at": ledger_mod.now_iso(),
        "config": config,
        "population_objects": int(meta["num_objects"]),
        "total_allocated": int(sum(v["allocated"] for v in strata.values())),
        "strata": strata,
        "frozen_before_candidate_outcomes": True,
    }
    if persist:
        d = plan_path(plan["plan_id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "plan.json"), "w") as f:
            json.dump(plan, f)
        np.savez_compressed(os.path.join(d, "units.npz"), **arrays)
    plan["_arrays"] = arrays  # in-memory only; not serialized
    return plan


def load_plan(plan_id: str) -> Optional[Dict]:
    d = plan_path(plan_id)
    meta_path = os.path.join(d, "plan.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        plan = json.load(f)
    with np.load(os.path.join(d, "units.npz")) as z:
        plan["_arrays"] = {k: z[k] for k in z.files}
    return plan


def stratum_clusters(plan: Dict, s: int, reserve: bool = False) -> List[np.ndarray]:
    """Ordered list of cluster-unit object_id arrays for stratum s."""
    suffix = "_res" if reserve else ""
    ids = plan["_arrays"].get(f"s{s}{suffix}_ids")
    bounds = plan["_arrays"].get(f"s{s}{suffix}_bounds")
    if ids is None or bounds is None or ids.size == 0:
        return []
    return [ids[bounds[i]:bounds[i + 1]] for i in range(bounds.size - 1)]
