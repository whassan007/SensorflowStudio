"""Intelligent launch-evaluation scheduler: the priority-ordered gauntlet.

Priority order (deterministic policy):
    1. safety_critical        2. historical_regressions   3. new_odd
    4. distribution_shift     5. rare_events              6. nominal

Budget-aware batched execution with sequential early stopping DELEGATED to
sensorflow.seqeval (reuse, not reimplementation): every stopping decision is
an anytime-valid e-process decision from seqeval.PairedSequentialTest.

STATISTICAL VALIDITY CONDITIONS FOR EARLY STOPPING (documented + enforced):
* Only ANYTIME-VALID statistics may stop a stratum: the e-process guarantees
  (Ville's inequality) hold at every data-dependent stopping time, so
  peeking after every batch costs nothing. Enforced structurally — the loop
  has no other stopping test; the descriptive Wilson-style CIs in reports
  are never consulted for control flow.
* Observations fed to the tests are CLUSTER means (seqeval.units), so
  within-scene correlation cannot make stopping anti-conservative.
* Decisions are sticky (seqeval semantics): once REGRESSION/PASS is
  declared at level alpha it is never revisited.

Early-stop policy:
* catastrophic regression (safety_critical stratum, REGRESSION decision,
  delta estimate below -catastrophic_delta) -> HALT the entire candidate;
* high-confidence regression in any stratum -> related strata (RELATED map)
  are promoted to the front of the remaining queue;
* critical-suite PASS (safety_critical passes) -> EXPAND: the nominal
  stratum's unit allowance is multiplied (cheap volume evidence only after
  the safety-critical evidence is in).

Scenario units are lightweight paired-outcome units (the 100k-scale
demonstration measures scheduling + caching math with real timings; each
unit stands for one scenario evaluation). Unit outcome vectors are cached
content-addressed per (stratum, model pair, versions) via cache.py, so a
re-run of the same gauntlet hits the cache instead of recomputing — the
same dedup math that makes a real 100k-scenario gauntlet sublinear.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from sensorflow.seqeval.sequential import (
    DECISION_INSUFFICIENT, DECISION_PASS, DECISION_REGRESSION,
    PairedSequentialTest,
)
from sensorflow.seqeval.units import cluster_units

from sensorflow.nextgen import store
from sensorflow.nextgen.cache import CacheKeyVersions, get_feature_cache
from sensorflow.nextgen.lineage import build_lineage
from sensorflow.nextgen.models import DataLabel, EvaluationRun
from sensorflow.nextgen.regression import launch_recommendation
from sensorflow.nextgen import lineage as lineage_mod

PRIORITY_ORDER = ["safety_critical", "historical_regressions", "new_odd",
                  "distribution_shift", "rare_events", "nominal"]

RELATED = {
    "safety_critical": ["historical_regressions", "rare_events"],
    "historical_regressions": ["safety_critical"],
    "new_odd": ["distribution_shift"],
    "distribution_shift": ["new_odd", "rare_events"],
    "rare_events": ["safety_critical"],
    "nominal": [],
}

# Default synthetic gauntlet: >= 100k scenario units total.
DEFAULT_STRATA: Dict[str, Dict] = {
    "safety_critical":        {"units": 12_000, "base_rate": 0.90, "cluster_size": 20,
                               "data_label": "COUNTERFACTUAL"},
    "historical_regressions": {"units": 10_000, "base_rate": 0.88, "cluster_size": 20,
                               "data_label": "REPLAYED"},
    "new_odd":                {"units": 14_000, "base_rate": 0.86, "cluster_size": 25,
                               "data_label": "SIMULATED"},
    "distribution_shift":     {"units": 14_000, "base_rate": 0.90, "cluster_size": 25,
                               "data_label": "REPLAYED"},
    "rare_events":            {"units": 10_000, "base_rate": 0.82, "cluster_size": 10,
                               "data_label": "COUNTERFACTUAL"},
    "nominal":                {"units": 40_000, "base_rate": 0.95, "cluster_size": 40,
                               "data_label": "REPLAYED"},
}

DEFAULT_CONFIG = {
    "batch_units": 2_000,
    "budget_units": 80_000,
    "alpha": 0.05,
    "margin_delta": 0.005,
    "catastrophic_delta": 0.03,
    "nominal_expand_factor": 2.0,
    "nominal_base_allowance": 8_000,
}


# ------------------------------------------------------- unit simulation


def _outcome_vectors(stratum: str, cfg: Dict, effect: float,
                     baseline_version: str, candidate_version: str,
                     seed: int, versions: CacheKeyVersions) -> Dict:
    """Paired unit outcomes for one stratum, content-addressed cached.

    The pairing construction mirrors seqeval.paired (shared latent + per-
    model flip noise) but over abstract scenario units; the cache key covers
    stratum, both model versions, seed and the component versions, so a
    version bump is a miss by construction.
    """
    cache = get_feature_cache()
    key = versions.key_for(
        f"gauntlet|{stratum}|{baseline_version}|{candidate_version}|{seed}"
        f"|{cfg['units']}|{cfg['base_rate']}|{effect}", kind="gauntlet-units")

    def _compute():
        n = int(cfg["units"])
        rng = np.random.default_rng([seed, hash(stratum) % (2 ** 31)])
        u = rng.random(n)
        cluster_size = int(cfg["cluster_size"])
        clusters = np.arange(n) // cluster_size
        scene = rng.normal(0.0, 0.02, size=int(np.ceil(n / cluster_size)))
        p_base = np.clip(cfg["base_rate"] + scene[clusters], 0.02, 0.995)
        p_cand = np.clip(p_base + effect, 0.02, 0.995)
        flip = rng.random(n) < 0.04
        fresh = rng.random(n)
        u_base = u
        u_cand = np.where(flip, fresh, u)
        b = (u_base < p_base)
        c = (u_cand < p_cand)
        return {"baseline": b.astype(int).tolist(),
                "candidate": c.astype(int).tolist(),
                "clusters": clusters.astype(int).tolist()}

    return cache.get_or_compute(key, _compute, kind="gauntlet-units")


# ------------------------------------------------------- gauntlet runner


def run_gauntlet(candidate_version: str = "candidate-v4",
                 baseline_version: str = "baseline-v3",
                 effects: Optional[Dict[str, float]] = None,
                 strata: Optional[Dict[str, Dict]] = None,
                 config: Optional[Dict] = None,
                 seed: int = 11,
                 versions: Optional[CacheKeyVersions] = None,
                 persist: bool = True) -> Dict:
    """Run the full priority-ordered gauntlet. `effects` plants per-stratum
    candidate deltas (absolute points), e.g. {"safety_critical": -0.06} for a
    catastrophic safety regression."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    strata_cfg = {k: dict(v) for k, v in (strata or DEFAULT_STRATA).items()}
    effects = effects or {}
    versions = versions or CacheKeyVersions()
    cache = get_feature_cache()
    hits0, misses0 = cache.stats.hits, cache.stats.misses

    run_id = f"gauntlet-{uuid4().hex[:10]}"
    t_start = time.perf_counter()

    queue: List[str] = [s for s in PRIORITY_ORDER if s in strata_cfg]
    nominal_allowance = int(cfg["nominal_base_allowance"])
    budget_left = int(cfg["budget_units"])
    total_units = sum(int(s["units"]) for s in strata_cfg.values())

    events: List[Dict] = []
    stratum_results: List[Dict] = []
    halted = False
    processed_order: List[str] = []
    units_evaluated = 0

    while queue and budget_left > 0 and not halted:
        name = queue.pop(0)
        scfg = strata_cfg[name]
        processed_order.append(name)
        effect = float(effects.get(name, 0.0))
        vectors = _outcome_vectors(name, scfg, effect, baseline_version,
                                   candidate_version, seed, versions)
        b = np.asarray(vectors["baseline"], dtype=np.float64)
        c = np.asarray(vectors["candidate"], dtype=np.float64)
        clusters = np.asarray(vectors["clusters"])

        limit = int(scfg["units"])
        if name == "nominal":
            limit = min(limit, nominal_allowance)
        limit = min(limit, budget_left)

        test = PairedSequentialTest(delta=cfg["margin_delta"], alpha=cfg["alpha"])
        used = 0
        decision = DECISION_INSUFFICIENT
        batch = int(cfg["batch_units"])
        while used < limit:
            end = min(used + batch, limit)
            db = c[used:end] - b[used:end]
            means, _ = cluster_units(db, clusters[used:end])
            test.update_clusters(means)
            test.record_objects(b[used:end].astype(bool), c[used:end].astype(bool))
            used = end
            test.record_trajectory_point()
            # ONLY anytime-valid stopping: seqeval e-process decision.
            decision = test.evaluate()
            if decision != DECISION_INSUFFICIENT:
                break

        budget_left -= used
        units_evaluated += used
        est = test.delta_estimate() or 0.0
        snapshot = test.snapshot()
        stratum_results.append({
            "stratum": name, "priority": PRIORITY_ORDER.index(name) + 1,
            "data_label": scfg.get("data_label", "SIMULATED"),
            "planted_effect": effect,
            "units_available": int(scfg["units"]),
            "units_evaluated": used,
            "units_saved_by_early_stop": max(0, limit - used),
            **snapshot,
            "trajectory": test.trajectory[-20:],
        })

        if decision == DECISION_REGRESSION:
            if name == "safety_critical" and est < -cfg["catastrophic_delta"]:
                halted = True
                events.append({
                    "event": "CATASTROPHIC_HALT", "stratum": name,
                    "delta_estimate": round(est, 5),
                    "detail": f"safety-critical regression {est:+.4f} below "
                              f"-{cfg['catastrophic_delta']}: candidate halted, "
                              f"remaining strata not evaluated"})
            else:
                promoted = [r for r in RELATED.get(name, []) if r in queue]
                for r in reversed(promoted):
                    queue.remove(r)
                    queue.insert(0, r)
                events.append({
                    "event": "REGRESSION_PRIORITIZE_RELATED", "stratum": name,
                    "delta_estimate": round(est, 5),
                    "promoted": promoted,
                    "detail": f"high-confidence regression in {name}; related "
                              f"strata promoted: {promoted or 'none pending'}"})
        elif decision == DECISION_PASS and name == "safety_critical":
            nominal_allowance = int(nominal_allowance * cfg["nominal_expand_factor"])
            events.append({
                "event": "CRITICAL_PASS_EXPAND", "stratum": name,
                "detail": f"safety-critical suite passed; nominal allowance "
                          f"expanded to {nominal_allowance} units"})

    elapsed = time.perf_counter() - t_start
    hits = cache.stats.hits - hits0
    misses = cache.stats.misses - misses0

    lineage = build_lineage(candidate_version, f"gauntlet-strata-{seed}",
                            seeds={"gauntlet": seed},
                            baseline_version=baseline_version)
    data_labels = list(dict.fromkeys(
        s.get("data_label", "SIMULATED") for s in strata_cfg.values()))

    # Strata records for the launch recommendation (statistical verdicts come
    # from the seqeval snapshots; safety significance from the point deltas).
    from sensorflow.nextgen.regression import SAFETY_MARGINS
    strata_records = []
    for r in stratum_results:
        kind = "safety" if r["stratum"] in ("safety_critical", "rare_events") else "global"
        strata_records.append({
            "stratum": r["stratum"], "kind": kind,
            "delta_abs": r["delta_estimate"],
            "statistical": {"decision": r["decision"]},
            "safety": {"significant": (r["delta_estimate"] is not None
                                       and r["delta_estimate"] < -SAFETY_MARGINS[kind]),
                       "margin": SAFETY_MARGINS[kind]},
        })
    recommendation = launch_recommendation(run_id, strata_records, lineage,
                                           data_labels)
    if halted:
        recommendation.recommendation = "DO_NOT_LAUNCH"
        recommendation.blockers.insert(
            0, "gauntlet halted: catastrophic safety-critical regression")

    result = {
        "run_id": run_id,
        "candidate_version": candidate_version,
        "baseline_version": baseline_version,
        "config": cfg,
        "planted_effects": effects,
        "scale": {
            "total_units_defined": total_units,
            "units_evaluated": units_evaluated,
            "units_saved": total_units - units_evaluated,
            "budget_units": cfg["budget_units"],
            "budget_remaining": budget_left,
        },
        "timing": {"wall_s": round(elapsed, 4),
                   "units_per_second": (round(units_evaluated / elapsed)
                                        if elapsed > 0 else None)},
        "cache": {"hits": hits, "misses": misses,
                  "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None},
        "priority_order": PRIORITY_ORDER,
        "processed_order": processed_order,
        "halted": halted,
        "events": events,
        "strata": stratum_results,
        "recommendation": recommendation.model_dump(mode="json"),
        "statistical_validity": (
            "all early stopping via seqeval anytime-valid e-processes on "
            "cluster means; sticky decisions; no fixed-n test is ever used "
            "for control flow"),
        "status": "halted" if halted else "completed",
    }

    run = EvaluationRun(run_id=run_id, kind="gauntlet",
                        status=result["status"],
                        data_labels=[DataLabel(d) for d in data_labels],
                        lineage=lineage,
                        params={"effects": effects, "config": cfg, "seed": seed},
                        results={k: result[k] for k in
                                 ("scale", "timing", "cache", "events")})
    lineage_mod.stamp_run(run)
    run.valid_for_launch = run.lineage_valid and not halted
    result["lineage"] = lineage.model_dump(mode="json")
    result["lineage_valid"] = run.lineage_valid

    if persist:
        store.write_json(result, "gauntlets", f"{run_id}.json")
        store.write_json({"latest": run_id}, "gauntlets", "latest.json")
    return result


def get_gauntlet(run_id: str) -> Optional[Dict]:
    if run_id == "latest":
        pointer = store.read_json("gauntlets", "latest.json")
        if not pointer:
            return None
        run_id = pointer["latest"]
    return store.read_json("gauntlets", f"{run_id}.json")


def list_gauntlets() -> List[Dict]:
    out = []
    for rid in store.list_json("gauntlets"):
        if rid == "latest":
            continue
        raw = store.read_json("gauntlets", f"{rid}.json")
        if raw:
            out.append({"run_id": raw["run_id"], "status": raw["status"],
                        "recommendation": raw["recommendation"]["recommendation"],
                        "units_evaluated": raw["scale"]["units_evaluated"],
                        "wall_s": raw["timing"]["wall_s"]})
    return out
