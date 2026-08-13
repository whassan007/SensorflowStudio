"""Metric cube: partial sufficient statistics -> reduce -> aggregate queries.

The cube is the heart of aggregate-first evaluation. During a run, each worker
computes *partial* additive statistics per partition, grouped by the 9 cube
dimensions. Reducing = concatenating partials and summing per cell (map-side
combine + shuffle reduce in Spark terms). Queries NEVER scan raw records unless
they ask for something outside the cube (then the router honestly reports
source="scan").

Cell key: class x weather x lighting x road_type x scenario x sensor x
          distance_band x speed_band x occlusion         (per evaluation run)
Cell stats (all additive): n, tp, fp, fn, loc_err, anomalies, sum_iou,
          sum_conf, sum_conf2, safety_n, safety_tp, reviewed, verified
Derived on demand: precision, recall, f1, mean_iou, anomaly_rate,
          conf_mean, conf_std, safety_recall, review_coverage, verified_rate
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sensorflow.megaeval.population import DIMENSIONS, DIM_NAMES

STAT_COLS = ["n", "tp", "fp", "fn", "loc_err", "anomalies", "sum_iou",
             "sum_conf", "sum_conf2", "safety_n", "safety_tp", "reviewed", "verified"]

DERIVED_METRICS = ["precision", "recall", "f1", "mean_iou", "anomaly_rate",
                   "conf_mean", "conf_std", "safety_recall", "review_coverage",
                   "verified_rate", "error_rate"]
COUNT_METRICS = ["n", "tp", "fp", "fn", "loc_err", "anomalies", "reviewed", "verified",
                 "safety_n", "safety_tp"]
SKETCH_METRICS = ["confidence_p50", "confidence_p10", "confidence_p90",
                  "iou_p50", "iou_p10", "iou_p90"]
ALL_METRICS = COUNT_METRICS + DERIVED_METRICS + SKETCH_METRICS


# ------------------------------------------------------------------ partial stats


def partial_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Group one partition's contribution rows by the 9 dims and sum stat columns.

    `rows` must contain DIM_NAMES code columns plus STAT_COLS contribution columns
    (e.g. a TP object contributes n=1, tp=1, sum_iou=iou; an FP row contributes
    n=0, fp=1, sum_conf=conf).
    """
    return rows.groupby(DIM_NAMES, as_index=False, observed=True, sort=False)[STAT_COLS].sum()


def reduce_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    """Reduce partial stats into the final cube (concat + groupby-sum)."""
    if not partials:
        return pd.DataFrame(columns=DIM_NAMES + STAT_COLS)
    return (pd.concat(partials, ignore_index=True)
            .groupby(DIM_NAMES, as_index=False, observed=True, sort=False)[STAT_COLS]
            .sum())


def derive_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    """Attach derived ratio metrics to an aggregated (already summed) frame."""
    out = agg.copy()
    tp, fp, fn = out["tp"], out["fp"], out["fn"]
    pp = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        out["precision"] = np.where(pp > 0, tp / pp, np.nan)
        out["recall"] = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        p, r = out["precision"], out["recall"]
        out["f1"] = np.where((p + r) > 0, 2 * p * r / (p + r), np.nan)
        out["mean_iou"] = np.where(tp > 0, out["sum_iou"] / tp, np.nan)
        out["anomaly_rate"] = np.where(out["n"] > 0, out["anomalies"] / out["n"], np.nan)
        out["conf_mean"] = np.where(pp > 0, out["sum_conf"] / pp, np.nan)
        var = np.where(pp > 1,
                       (out["sum_conf2"] - out["sum_conf"] ** 2 / pp.replace(0, 1)) / (pp - 1).replace(0, 1),
                       np.nan)
        out["conf_std"] = np.sqrt(np.clip(var, 0, None))
        out["safety_recall"] = np.where(out["safety_n"] > 0, out["safety_tp"] / out["safety_n"], np.nan)
        out["review_coverage"] = np.where(out["n"] > 0, out["reviewed"] / out["n"], np.nan)
        out["verified_rate"] = np.where(out["reviewed"] > 0, out["verified"] / out["reviewed"], np.nan)
        out["error_rate"] = np.where(out["n"] > 0, (fn + fp + out["loc_err"]) / out["n"], np.nan)
    return out


# ------------------------------------------------------------------ persistence


def save_cube(path: str, cube: pd.DataFrame) -> None:
    np.savez_compressed(path, **{c: cube[c].to_numpy() for c in cube.columns})


def load_cube(path: str) -> pd.DataFrame:
    with np.load(path) as z:
        return pd.DataFrame({k: z[k] for k in z.files})


# ------------------------------------------------------------------ query router


class QueryCache:
    """Evaluation cache: key = hash(dataset+model+filters+group_by+metrics)."""

    def __init__(self, max_entries: int = 512):
        self._data: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(dataset_version: str, model_version: str, filters: Dict,
            group_by: List[str], metrics: List[str]) -> str:
        blob = json.dumps({
            "d": dataset_version, "m": model_version,
            "f": {k: sorted(v) for k, v in sorted((filters or {}).items())},
            "g": list(group_by or []), "x": sorted(metrics or []),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            hit = self._data.get(key)
            if hit is not None:
                self.hits += 1
            else:
                self.misses += 1
            return hit

    def put(self, key: str, value: Dict) -> None:
        with self._lock:
            if len(self._data) >= self.max_entries:
                self._data.pop(next(iter(self._data)))
            self._data[key] = value

    def invalidate(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> Dict:
        with self._lock:
            total = self.hits + self.misses
            return {"entries": len(self._data), "hits": self.hits, "misses": self.misses,
                    "hit_rate": round(self.hits / total, 4) if total else None}


def _codes_for(dim: str, values: List[str]) -> List[int]:
    vocab = DIMENSIONS[dim]
    out = []
    for v in values:
        if v in vocab:
            out.append(vocab.index(v))
    return out


def apply_filters(cube: pd.DataFrame, filters: Optional[Dict[str, List[str]]]) -> pd.DataFrame:
    if not filters:
        return cube
    mask = np.ones(len(cube), dtype=bool)
    for dim, values in filters.items():
        if dim not in DIM_NAMES:
            continue
        codes = _codes_for(dim, values if isinstance(values, list) else [values])
        mask &= cube[dim].isin(codes).to_numpy()
    return cube[mask]


def aggregate(cube: pd.DataFrame, filters: Optional[Dict], group_by: Optional[List[str]],
              metrics: Optional[List[str]], limit: int = 200) -> Tuple[List[Dict], int]:
    """Aggregate-lookup path: filter cells, group, sum, derive. Never touches records."""
    sub = apply_filters(cube, filters)
    group_by = [g for g in (group_by or []) if g in DIM_NAMES]
    metrics = [m for m in (metrics or []) if m in ALL_METRICS and m not in SKETCH_METRICS] \
        or ["n", "precision", "recall", "f1", "mean_iou", "anomaly_rate"]

    if group_by:
        agg = sub.groupby(group_by, as_index=False, observed=True, sort=False)[STAT_COLS].sum()
    else:
        sums = sub[STAT_COLS].sum()
        agg = pd.DataFrame([sums])
    agg = derive_metrics(agg)

    cell_count = len(sub)
    sort_metric = next((m for m in metrics if m in agg.columns), None)
    if group_by and sort_metric:
        agg = agg.sort_values("n", ascending=False)
    agg = agg.head(max(1, min(limit, 2000)))

    rows: List[Dict] = []
    for _, r in agg.iterrows():
        row: Dict = {}
        for g in group_by:
            row[g] = DIMENSIONS[g][int(r[g])]
        row["n"] = int(r["n"])
        for m in metrics:
            if m == "n":
                continue
            v = r.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row[m] = None
            elif m in COUNT_METRICS:
                row[m] = int(v)
            else:
                row[m] = round(float(v), 6)
        rows.append(row)
    return rows, cell_count


class QueryRouter:
    """Routes a query: cache -> cube aggregate -> (only if unavoidable) record scan.

    This is the OLAP-engine seam: callers never know how data is stored.
    """

    def __init__(self):
        self.cache = QueryCache()

    def query(self, run: "object", cube: pd.DataFrame, sketches: Dict,
              filters: Optional[Dict], metrics: Optional[List[str]],
              group_by: Optional[List[str]], limit: int = 200,
              scan_fn=None) -> Dict:
        t0 = time.perf_counter()
        metrics = metrics or []
        key = QueryCache.key(run.population_id, run.model_version,
                             filters or {}, group_by or [], metrics)
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached,
                    "meta": {**cached["meta"], "cache_hit": True, "source": "cache",
                             "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}}

        sketch_wanted = [m for m in metrics if m in SKETCH_METRICS]
        needs_scan = any(g not in DIM_NAMES for g in (group_by or []))

        if needs_scan and scan_fn is not None:
            rows, cell_count = scan_fn(filters, group_by, metrics, limit)
            source, exact = "scan", True
        else:
            rows, cell_count = aggregate(cube, filters, group_by, metrics, limit)
            source, exact = "cube", True
            if sketch_wanted and not group_by and not filters:
                # population-level percentile questions answered from sketches
                for m in sketch_wanted:
                    metric_name, q = m.rsplit("_p", 1)
                    hist = sketches.get(metric_name)
                    if hist is not None and rows:
                        rows[0][m] = round(float(hist.percentile(float(q))), 6)
                exact = False  # percentiles are approximate
            elif sketch_wanted:
                exact = True  # sketch metrics ignored on grouped/filtered queries

        latency = round((time.perf_counter() - t0) * 1000, 3)
        result = {
            "rows": rows,
            "meta": {
                "source": source,
                "cache_hit": False,
                "latency_ms": latency,
                "cells_touched": cell_count,
                "exact": exact,
                "approximate_fields": sketch_wanted if not exact else [],
            },
        }
        self.cache.put(key, result)
        return result
