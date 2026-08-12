"""Cross-run analysis: model-vs-model comparison, distribution shift, the "why"
layer, and hybrid similarity search.

All computations run on cubes and the error index — never on raw records.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sensorflow.megaeval import cube as cube_mod
from sensorflow.megaeval import population as pop_mod
from sensorflow.megaeval.population import DIMENSIONS, DIM_NAMES
from sensorflow.megaeval.runs import ERROR_TYPES, EvaluationRun, MegaStore, run_dir

# ------------------------------------------------------------------ model vs model

DEFAULT_PROMOTION_POLICY = {
    "max_recall_drop": 0.010,
    "max_precision_drop": 0.015,
    "max_safety_recall_drop": 0.005,
    "max_cohort_recall_drop": 0.050,
    "min_cohort_support": 200,
    "cohort_dims": ["road_type", "lighting", "class"],
}

_HEADLINE_METRICS = ["precision", "recall", "f1", "mean_iou", "safety_recall", "anomaly_rate"]


def compare_runs(store: MegaStore, candidate: EvaluationRun, baseline: EvaluationRun,
                 policy: Optional[Dict] = None) -> Dict:
    policy = {**DEFAULT_PROMOTION_POLICY, **(policy or {})}
    cand_art, base_art = store.artifacts(candidate.run_id), store.artifacts(baseline.run_id)
    cand_cube, base_cube = cand_art["cube"], base_art["cube"]

    headline = []
    for m in _HEADLINE_METRICS:
        c, b = candidate.headline.get(m), baseline.headline.get(m)
        if c is None or b is None:
            continue
        headline.append({"metric": m, "baseline": round(b, 4), "candidate": round(c, 4),
                         "delta": round(c - b, 4)})

    per_class = []
    for cls in DIMENSIONS["class"]:
        c, b = candidate.per_class.get(cls), baseline.per_class.get(cls)
        if not c or not b:
            continue
        per_class.append({
            "class": cls, "n": c.get("n"),
            "recall_baseline": b.get("recall"), "recall_candidate": c.get("recall"),
            "recall_delta": round((c.get("recall") or 0) - (b.get("recall") or 0), 4),
            "precision_baseline": b.get("precision"), "precision_candidate": c.get("precision"),
            "precision_delta": round((c.get("precision") or 0) - (b.get("precision") or 0), 4),
        })

    dims = policy["cohort_dims"]
    cand_rows, _ = cube_mod.aggregate(cand_cube, None, dims, ["n", "recall", "precision"], 2000)
    base_rows, _ = cube_mod.aggregate(base_cube, None, dims, ["n", "recall", "precision"], 2000)
    base_map = {tuple(r[d] for d in dims): r for r in base_rows}
    cohorts = []
    for r in cand_rows:
        key = tuple(r[d] for d in dims)
        b = base_map.get(key)
        if not b or r["n"] < policy["min_cohort_support"]:
            continue
        if r.get("recall") is None or b.get("recall") is None:
            continue
        cohorts.append({
            "cohort": "/".join(key), **{d: r[d] for d in dims}, "n": r["n"],
            "recall_baseline": b["recall"], "recall_candidate": r["recall"],
            "recall_delta": round(r["recall"] - b["recall"], 4),
        })
    cohorts.sort(key=lambda x: x["recall_delta"])
    regressions = [c for c in cohorts if c["recall_delta"] < -policy["max_cohort_recall_drop"]]

    blockers: List[str] = []
    hd = {h["metric"]: h["delta"] for h in headline}
    if hd.get("recall", 0) < -policy["max_recall_drop"]:
        blockers.append(f"headline recall dropped {hd['recall']:+.4f} "
                        f"(policy allows -{policy['max_recall_drop']})")
    if hd.get("precision", 0) < -policy["max_precision_drop"]:
        blockers.append(f"headline precision dropped {hd['precision']:+.4f}")
    if hd.get("safety_recall", 0) < -policy["max_safety_recall_drop"]:
        blockers.append(f"safety recall dropped {hd['safety_recall']:+.4f} "
                        f"(policy allows -{policy['max_safety_recall_drop']})")
    for c in regressions[:5]:
        blockers.append(
            f"REGRESSION: {c['cohort']} recall "
            f"{c['recall_baseline']*100:.1f}% -> {c['recall_candidate']*100:.1f}% "
            f"(Δ {c['recall_delta']*100:+.1f})")

    # integrate with the existing regression tracker so the Regression page shows it
    try:
        from sensorflow.evaluation import regression as legacy_regression
        from sensorflow.evaluation.records import get_store as get_legacy_store
        legacy_regression.compare_runs(
            get_legacy_store(),
            current_metrics={"precision": candidate.headline.get("precision"),
                             "recall": candidate.headline.get("recall"),
                             "safety_critical_recall": candidate.headline.get("safety_recall"),
                             "mean_iou_3d": candidate.headline.get("mean_iou")},
            model_version=candidate.model_version,
            dataset_version=candidate.population_id,
            run_id=candidate.run_id,
            baseline_metrics={"precision": baseline.headline.get("precision"),
                              "recall": baseline.headline.get("recall"),
                              "safety_critical_recall": baseline.headline.get("safety_recall"),
                              "mean_iou_3d": baseline.headline.get("mean_iou")},
            baseline_version=baseline.model_version,
            affected_classes=[c["class"] for c in per_class if c["recall_delta"] < -0.03],
            affected_scenarios=sorted({c.get("road_type", "") for c in regressions[:5] if c}),
        )
    except Exception:
        pass

    return {
        "candidate": {"run_id": candidate.run_id, "model_version": candidate.model_version},
        "baseline": {"run_id": baseline.run_id, "model_version": baseline.model_version},
        "headline_deltas": headline,
        "per_class": per_class,
        "worst_cohorts": cohorts[:15],
        "regressions": regressions[:10],
        "policy": policy,
        "recommendation": "DO_NOT_PROMOTE" if blockers else "PROMOTE",
        "blockers": blockers,
    }


# ------------------------------------------------------------------ distribution shift


def distribution_shift(store: MegaStore, run: EvaluationRun,
                       min_eval_count: int = 300, rel_threshold: float = 0.35) -> Dict:
    cache_path = os.path.join(run_dir(run.run_id), "shift.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    meta = pop_mod.load_meta(run.population_id)
    cube_df = store.artifacts(run.run_id)["cube"]
    overall_rows, _ = cube_mod.aggregate(cube_df, None, None, ["recall"], 1)
    overall_recall = overall_rows[0].get("recall")

    shifts = []
    for row in meta.get("train_mix", []):
        train, evals = row["train_share"], row["eval_share"]
        if train <= 0 or row["eval_count"] < min_eval_count:
            continue
        rel = (evals - train) / train
        if abs(rel) < rel_threshold:
            continue
        filt = {"class": [row["class"]], "weather": [row["weather"]],
                "lighting": [row["lighting"]]}
        crows, _ = cube_mod.aggregate(cube_df, filt, None, ["n", "recall", "precision"], 1)
        c = crows[0] if crows else {}
        shifts.append({
            "cohort": f"{row['class']}/{row['weather']}/{row['lighting']}",
            "class": row["class"], "weather": row["weather"], "lighting": row["lighting"],
            "train_share": round(train, 5), "eval_share": round(evals, 5),
            "relative_change": round(rel, 4),
            "eval_count": row["eval_count"],
            "cohort_recall": c.get("recall"),
            "overall_recall": overall_recall,
            "recall_gap": (round(c["recall"] - overall_recall, 4)
                           if c.get("recall") is not None and overall_recall is not None else None),
        })
    shifts.sort(key=lambda s: abs(s["relative_change"]) * (1 + abs(s.get("recall_gap") or 0) * 5),
                reverse=True)

    # surface the worst shifts as alerts in the existing alerting system (once per run)
    try:
        from sensorflow.evaluation.records import Alert, get_store as get_legacy_store, new_id
        legacy = get_legacy_store()
        for s in shifts[:3]:
            if s.get("recall_gap") is not None and s["recall_gap"] < -0.03:
                legacy.put("alerts", Alert(
                    alert_id=new_id("alert"), kind="distribution_shift",
                    severity="critical" if s["recall_gap"] < -0.08 else "warning",
                    message=(f"Distribution shift {s['cohort']}: train {s['train_share']*100:.1f}% "
                             f"-> eval {s['eval_share']*100:.1f}% "
                             f"(rel {s['relative_change']*100:+.0f}%), cohort recall "
                             f"{(s['cohort_recall'] or 0)*100:.1f}% vs overall "
                             f"{(s['overall_recall'] or 0)*100:.1f}%"),
                    evidence_page="command", evidence_id=run.run_id))
        legacy.save()
    except Exception:
        pass

    result = {"run_id": run.run_id, "shifts": shifts[:20],
              "method": "train mix vs eval mix over (class, weather, lighting); "
                        "relative share change with cube recall impact",
              "thresholds": {"min_eval_count": min_eval_count, "rel_threshold": rel_threshold}}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


# ------------------------------------------------------------------ the "why" layer


_FACTOR_ORDER = ["occlusion", "low_illumination", "long_range", "sensor_disagreement", "other"]


def why(store: MegaStore, run: EvaluationRun, filters: Optional[Dict] = None,
        metric: str = "recall") -> Dict:
    """Decompose the failure population of a filtered slice by primary factor."""
    art = store.artifacts(run.run_id)
    errors = art.get("errors")
    cube_df = art["cube"]
    if errors is None or not len(errors):
        return {"factors": [], "top_cohorts": [], "failure_count": 0}

    mask = np.ones(len(errors), dtype=bool)
    fail_types = [ERROR_TYPES.index(t) for t in ("FN", "FP", "LOCALIZATION")]
    if metric == "recall":
        fail_types = [ERROR_TYPES.index("FN"), ERROR_TYPES.index("LOCALIZATION")]
    elif metric == "precision":
        fail_types = [ERROR_TYPES.index("FP"), ERROR_TYPES.index("LOCALIZATION")]
    mask &= errors["error_type"].isin(fail_types).to_numpy()
    for dim, values in (filters or {}).items():
        if dim not in DIM_NAMES:
            continue
        vocab = DIMENSIONS[dim]
        codes = [vocab.index(v) for v in (values if isinstance(values, list) else [values])
                 if v in vocab]
        mask &= errors[dim].isin(codes).to_numpy()
    sub = errors[mask]
    total = len(sub)
    if total == 0:
        return {"factors": [], "top_cohorts": [], "failure_count": 0}

    occl = sub["occlusion"].to_numpy() >= 1
    night = sub["lighting"].to_numpy() == 1
    far = sub["distance_band"].to_numpy() >= 2
    sensor = sub["sensor_disagree"].to_numpy().astype(bool)
    factor = np.full(total, 4, dtype=np.int8)  # other
    factor[sensor] = 3
    factor[far] = 2
    factor[night] = 1
    factor[occl] = 0  # priority: occlusion > low light > range > sensor > other
    counts = np.bincount(factor, minlength=5)
    factors = [{"factor": _FACTOR_ORDER[i], "count": int(counts[i]),
                "share": round(float(counts[i]) / total, 4)}
               for i in range(5) if counts[i] > 0]
    factors.sort(key=lambda x: -x["count"])

    rows, _ = cube_mod.aggregate(cube_df, filters, ["class", "lighting", "weather"],
                                 ["n", metric, "fn", "fp"], 2000)
    rows = [r for r in rows if r["n"] >= 150 and r.get(metric) is not None]
    rows.sort(key=lambda r: r[metric])
    top = [{"cohort": f"{r['class']}/{r['lighting']}/{r['weather']}",
            "n": r["n"], metric: r[metric], "fn": r.get("fn"), "fp": r.get("fp")}
           for r in rows[:8]]

    return {"metric": metric, "filters": filters or {}, "failure_count": int(total),
            "factors": factors, "top_cohorts": top,
            "method": "primary-factor attribution over the error index "
                      "(priority: occlusion > low illumination > long range > "
                      "sensor disagreement > other)"}


# ------------------------------------------------------------------ hybrid similarity


def similar_containers(store: MegaStore, run: EvaluationRun, container_id: int,
                       filters: Optional[Dict] = None, k: int = 12) -> Dict:
    art = store.artifacts(run.run_id)
    ids, emb = art["emb_ids"], art["emb"]
    containers = art["containers"].set_index("container_id")
    pos = np.where(ids == container_id)[0]
    if pos.size == 0:
        raise KeyError(f"Unknown container {container_id}")
    q = emb[pos[0]]
    sims = emb @ q

    mask = np.ones(ids.shape[0], dtype=bool)
    for dim, values in (filters or {}).items():
        if dim not in pop_mod.CONTAINER_DIMS:
            continue
        vocab = DIMENSIONS[dim]
        codes = [vocab.index(v) for v in (values if isinstance(values, list) else [values])
                 if v in vocab]
        col = containers.loc[ids, dim].to_numpy()
        mask &= np.isin(col, codes)
    mask[pos[0]] = False
    sims = np.where(mask, sims, -np.inf)

    kk = min(k, int(mask.sum()))
    if kk <= 0:
        return {"query_container": container_id, "results": []}
    top = np.argpartition(-sims, kk - 1)[:kk]
    top = top[np.argsort(-sims[top])]
    results = []
    for i in top.tolist():
        cid = int(ids[i])
        row = containers.loc[cid]
        results.append({
            "container_id": cid,
            "similarity": round(float(sims[i]), 4),
            **{dim: DIMENSIONS[dim][int(row[dim])] for dim in pop_mod.CONTAINER_DIMS},
            "n_objects": int(row["n_objects"]),
            "fn": int(row["fn"]), "fp": int(row["fp"]),
            "anomalies": int(row["anomalies"]),
            "risk_score": round(float(row["risk_score"]), 4),
        })
    return {"query_container": container_id,
            "retrieval": "cosine over 32-d structural embeddings + structured filters",
            "results": results}
