"""Human review as statistical sampling.

Instead of a flat "review N labels" queue, review effort is a designed sample:

  population -> containers -> suspicious containers -> candidate pool
             -> stratified risk-weighted selection -> reviews -> CIs

Two estimands, each with its own sampling frame:
  - precision: frame = predicted positives, strata = confidence tier x safety
  - recall:    frame = ground-truth objects,  strata = model outcome x safety

Risk weighting happens through *allocation* (risky strata get more samples);
estimates stay honest because strata are combined with their true population
weights (classic stratified estimator):

    p_hat = sum_h W_h * p_hat_h
    SE    = sqrt( sum_h W_h^2 * p_hat_h (1 - p_hat_h) / n_h )     (normal approx)

Per-stratum intervals use Wilson. Simulated reviewers agree with ground truth
with 98.5% fidelity (the synthetic stand-in for human accuracy).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIM_NAMES
from sensorflow.megaeval.runs import EvaluationRun, MegaStore, run_dir

HUMAN_FIDELITY = 0.985
Z95 = 1.959964


def wilson_interval(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _load_unit_frames(store: MegaStore, run: EvaluationRun) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One batch scan (NOT on the query path) to build the two sampling frames."""
    d = run_dir(run.run_id)
    gt_frames, pp_frames = [], []
    for p in range(run.partitions_total):
        pop_cols = pop_mod.load_partition(run.population_id, p)
        with np.load(os.path.join(d, f"objects-part-{p:04d}.npz")) as z:
            detected = z["detected"].astype(bool)
            tp = z["tp"].astype(bool)
            conf = z["confidence"]
        base = pd.DataFrame({
            "unit_id": pop_cols["object_id"],
            "container_id": pop_cols["container_id"],
            **{k: pop_cols[k] for k in DIM_NAMES},
            "safety": pop_cols["safety_critical"].astype(bool),
            "detected": detected, "tp": tp, "confidence": conf,
        })
        gt_frames.append(base)
        pp = base[detected].copy()
        pp["correct"] = pp["tp"]
        pp_frames.append(pp)
        with np.load(os.path.join(d, f"fp-part-{p:04d}.npz")) as z:
            if z["container_id"].size:
                fpdf = pd.DataFrame({
                    "unit_id": -np.arange(1, z["container_id"].size + 1) - p * 10_000_000,
                    "container_id": z["container_id"],
                    **{k: z[k] for k in DIM_NAMES},
                    "safety": False, "detected": True, "tp": False,
                    "confidence": z["confidence"], "correct": False,
                })
                pp_frames.append(fpdf)
    return pd.concat(gt_frames, ignore_index=True), pd.concat(pp_frames, ignore_index=True)


def _stratify_precision(pp: pd.DataFrame) -> pd.Series:
    tier = np.select([pp["confidence"] < 0.45, pp["confidence"] < 0.75], [0, 1], default=2)
    return pd.Series(tier * 2 + pp["safety"].astype(int).to_numpy(), index=pp.index)


def _stratify_recall(gt: pd.DataFrame) -> pd.Series:
    return pd.Series((~gt["detected"]).astype(int).to_numpy() * 2
                     + gt["safety"].astype(int).to_numpy(), index=gt.index)


_PRECISION_STRATA = {0: "low_conf", 1: "low_conf_safety", 2: "mid_conf", 3: "mid_conf_safety",
                     4: "high_conf", 5: "high_conf_safety"}
_RECALL_STRATA = {0: "detected", 1: "detected_safety", 2: "missed", 3: "missed_safety"}
# risk-weighted allocation multipliers (riskier strata get proportionally more reviews)
_PRECISION_RISK_W = {0: 3.0, 1: 4.0, 2: 1.5, 3: 2.5, 4: 1.0, 5: 2.0}
_RECALL_RISK_W = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}


def _allocate(counts: Dict[int, int], risk_w: Dict[int, float], total_n: int) -> Dict[int, int]:
    weights = {h: counts[h] * risk_w.get(h, 1.0) for h in counts}
    wsum = sum(weights.values()) or 1.0
    alloc = {}
    for h, c in counts.items():
        n_h = int(round(total_n * weights[h] / wsum))
        alloc[h] = max(1, min(n_h, c)) if c > 0 else 0
    return alloc


def build_review_plan(store: MegaStore, run: EvaluationRun,
                      target_n: Optional[int] = None, seed: int = 99) -> Dict:
    """Build the sampling funnel + stratified selection. Persists review.json."""
    target_n = target_n or run.lineage["sampling_config"].get("target_n", 1500)
    rng = np.random.default_rng(seed + run.seed)
    gt, pp = _load_unit_frames(store, run)
    art = store.artifacts(run.run_id)
    containers = art["containers"]
    errors = art.get("errors", pd.DataFrame())

    suspicious = int((containers["risk_score"] >= 0.2).sum())
    candidates = int(len(errors)) if len(errors) else 0

    plans = {}
    for name, frame, strat_fn, labels, risk_w in (
            ("precision", pp, _stratify_precision, _PRECISION_STRATA, _PRECISION_RISK_W),
            ("recall", gt, _stratify_recall, _RECALL_STRATA, _RECALL_RISK_W)):
        strata = strat_fn(frame)
        counts = strata.value_counts().to_dict()
        alloc = _allocate(counts, risk_w, target_n // 2)
        selected = {}
        for h, n_h in alloc.items():
            if n_h <= 0:
                continue
            idx = frame.index[strata == h].to_numpy()
            chosen = rng.choice(idx, size=min(n_h, idx.size), replace=False)
            selected[str(h)] = frame.loc[chosen, "unit_id"].astype(np.int64).tolist()
        plans[name] = {
            "frame_size": int(len(frame)),
            "strata": {str(h): {"label": labels[h], "N": int(counts.get(h, 0)),
                                "allocated": int(alloc.get(h, 0))}
                       for h in labels},
            "selected_ids": selected,
        }

    selected_total = sum(len(v) for p in plans.values() for v in p["selected_ids"].values())
    review = {
        "run_id": run.run_id,
        "method": "stratified risk-weighted sampling; Wilson per-stratum, "
                  "stratified normal-approx combined CI",
        "target_n": target_n,
        "funnel": {
            "population_objects": int(run.objects_total),
            "containers": int(len(containers)),
            "suspicious_containers": suspicious,
            "candidate_pool": candidates,
            "statistically_selected": int(selected_total),
            "reviewed": 0,
        },
        "plans": plans,
        "executed": False,
        "results": None,
    }
    with open(os.path.join(run_dir(run.run_id), "review.json"), "w") as f:
        json.dump(review, f)
    store.invalidate_artifacts(run.run_id)
    return review


def execute_reviews(store: MegaStore, run: EvaluationRun, seed: int = 7) -> Dict:
    """Simulate human review of the selected sample; compute CIs; update the cube."""
    d = run_dir(run.run_id)
    with open(os.path.join(d, "review.json")) as f:
        review = json.load(f)
    rng = np.random.default_rng(seed + run.seed)
    gt, pp = _load_unit_frames(store, run)
    gt_idx = gt.set_index("unit_id")
    pp_idx = pp.set_index("unit_id")

    results = {}
    reviewed_units = []  # (dims..., verified) for cube update — recall-frame units only
    total_reviewed = 0
    for name, frame_idx, truth_col in (("precision", pp_idx, "correct"),
                                       ("recall", gt_idx, "tp")):
        plan = review["plans"][name]
        strata_out = []
        p_hat_comb, var_comb = 0.0, 0.0
        frame_N = plan["frame_size"]
        n_total = 0
        for h, sinfo in plan["strata"].items():
            ids = plan["selected_ids"].get(h, [])
            if not ids:
                continue
            units = frame_idx.loc[[i for i in ids if i in frame_idx.index]]
            if isinstance(units, pd.Series):
                units = units.to_frame().T
            truth = units[truth_col].astype(bool).to_numpy()
            agrees = rng.random(truth.size) < HUMAN_FIDELITY
            verdict = np.where(agrees, truth, ~truth)
            k, n_h = int(verdict.sum()), int(verdict.size)
            W_h = sinfo["N"] / frame_N
            p_h = k / n_h if n_h else 0.0
            lo, hi = wilson_interval(k, n_h)
            strata_out.append({"stratum": sinfo["label"], "N": sinfo["N"], "n": n_h,
                               "correct": k, "p": round(p_h, 4),
                               "wilson_ci": [round(lo, 4), round(hi, 4)],
                               "weight": round(W_h, 4)})
            p_hat_comb += W_h * p_h
            var_comb += (W_h**2) * p_h * (1 - p_h) / max(n_h, 1)
            n_total += n_h
            total_reviewed += n_h
            if name == "recall":
                sub = units.copy()
                sub["verified"] = verdict
                reviewed_units.append(sub)
        se = math.sqrt(var_comb)
        results[name] = {
            "estimate": round(p_hat_comb, 4),
            "ci_low": round(max(0.0, p_hat_comb - Z95 * se), 4),
            "ci_high": round(min(1.0, p_hat_comb + Z95 * se), 4),
            "n_reviewed": n_total,
            "frame_size": frame_N,
            "method": review["method"],
            "strata": strata_out,
        }

    # ---- fold reviews back into the cube + container table (reviewed/verified counters)
    if reviewed_units:
        ru = pd.concat(reviewed_units)
        ru = ru.reset_index(drop=True)
        upd = ru.groupby(DIM_NAMES, as_index=False, observed=True).agg(
            add_reviewed=("verified", "size"), add_verified=("verified", "sum"))
        from sensorflow.megaeval import cube as cube_mod
        cube_path = os.path.join(d, "cube.npz")
        cube_df = cube_mod.load_cube(cube_path)
        merged = cube_df.merge(upd, on=DIM_NAMES, how="left")
        merged["reviewed"] = merged["reviewed"] + merged["add_reviewed"].fillna(0).astype(np.int64)
        merged["verified"] = merged["verified"] + merged["add_verified"].fillna(0).astype(np.int64)
        cube_mod.save_cube(cube_path, merged.drop(columns=["add_reviewed", "add_verified"]))

        cupd = ru.groupby("container_id", as_index=False).agg(
            add_reviewed=("verified", "size"), add_verified=("verified", "sum"))
        cont_path = os.path.join(d, "containers.npz")
        with np.load(cont_path) as z:
            cont = pd.DataFrame({k: z[k] for k in z.files})
        cont = cont.merge(cupd, on="container_id", how="left")
        cont["reviewed"] = cont["reviewed"] + cont["add_reviewed"].fillna(0).astype(np.int64)
        cont["verified"] = cont["verified"] + cont["add_verified"].fillna(0).astype(np.int64)
        cont = cont.drop(columns=["add_reviewed", "add_verified"])
        np.savez_compressed(cont_path, **{c: cont[c].to_numpy() for c in cont.columns})

    review["executed"] = True
    review["funnel"]["reviewed"] = total_reviewed
    review["results"] = results
    with open(os.path.join(d, "review.json"), "w") as f:
        json.dump(review, f)
    store.invalidate_artifacts(run.run_id)
    return review
