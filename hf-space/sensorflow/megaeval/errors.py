"""Error index: fast multi-criteria search over evaluation errors.

The index is a columnar table (one row per error) built during the run:
    error_id, object_id, container_id, error_type, severity, confidence,
    risk_score, safety_critical, sensor_disagree, + the 9 cube dimensions.

Searches return the *worst-N containers* (aggregated), never raw record dumps —
this backs the Investigation UI. Individual exemplar errors are capped.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sensorflow.megaeval.population import DIMENSIONS, DIM_NAMES
from sensorflow.megaeval.runs import ERROR_TYPES


def search_errors(
    errors: pd.DataFrame,
    containers: pd.DataFrame,
    error_types: Optional[List[str]] = None,
    filters: Optional[Dict[str, List[str]]] = None,
    confidence_max: Optional[float] = None,
    confidence_min: Optional[float] = None,
    risk_min: Optional[float] = None,
    severity_min: Optional[float] = None,
    safety_only: bool = False,
    limit_containers: int = 25,
    limit_examples: int = 50,
) -> Dict:
    if errors is None or not len(errors):
        return {"matched_errors": 0, "worst_containers": [], "examples": [],
                "by_type": {}}
    mask = np.ones(len(errors), dtype=bool)
    if error_types:
        codes = [ERROR_TYPES.index(t) for t in error_types if t in ERROR_TYPES]
        mask &= errors["error_type"].isin(codes).to_numpy()
    for dim, values in (filters or {}).items():
        if dim not in DIM_NAMES:
            continue
        vocab = DIMENSIONS[dim]
        codes = [vocab.index(v) for v in (values if isinstance(values, list) else [values])
                 if v in vocab]
        mask &= errors[dim].isin(codes).to_numpy()
    if confidence_max is not None:
        mask &= errors["confidence"].to_numpy() <= confidence_max
    if confidence_min is not None:
        mask &= errors["confidence"].to_numpy() >= confidence_min
    if risk_min is not None:
        mask &= errors["risk_score"].to_numpy() >= risk_min
    if severity_min is not None:
        mask &= errors["severity"].to_numpy() >= severity_min
    if safety_only:
        mask &= errors["safety_critical"].to_numpy().astype(bool)

    sub = errors[mask]
    by_type = {ERROR_TYPES[int(c)]: int(n)
               for c, n in sub["error_type"].value_counts().items()}

    # aggregate to worst containers: rank by count x mean risk
    grp = sub.groupby("container_id", as_index=False).agg(
        error_count=("error_id", "size"), mean_risk=("risk_score", "mean"),
        max_severity=("severity", "max"), safety_hits=("safety_critical", "sum"))
    grp["rank_score"] = grp["error_count"] * grp["mean_risk"]
    grp = grp.sort_values("rank_score", ascending=False).head(limit_containers)
    cmeta = containers.set_index("container_id") if containers is not None else None
    worst = []
    for _, r in grp.iterrows():
        cid = int(r["container_id"])
        item = {"container_id": cid, "error_count": int(r["error_count"]),
                "mean_risk": round(float(r["mean_risk"]), 4),
                "max_severity": round(float(r["max_severity"]), 4),
                "safety_hits": int(r["safety_hits"])}
        if cmeta is not None and cid in cmeta.index:
            row = cmeta.loc[cid]
            for dim in ("weather", "lighting", "road_type", "scenario"):
                item[dim] = DIMENSIONS[dim][int(row[dim])]
            item["n_objects"] = int(row["n_objects"])
            item["risk_score"] = round(float(row["risk_score"]), 4)
        worst.append(item)

    ex = sub.sort_values("risk_score", ascending=False).head(limit_examples)
    examples = []
    for _, r in ex.iterrows():
        examples.append({
            "error_id": int(r["error_id"]),
            "annotation_id": (f"obj-{int(r['object_id'])}" if r["object_id"] >= 0
                              else f"fp-{int(r['error_id'])}"),
            "container_id": int(r["container_id"]),
            "error_type": ERROR_TYPES[int(r["error_type"])],
            "class": DIMENSIONS["class"][int(r["class"])],
            "severity": round(float(r["severity"]), 4),
            "confidence": round(float(r["confidence"]), 4),
            "risk_score": round(float(r["risk_score"]), 4),
            "safety_critical": bool(r["safety_critical"]),
            "scenario": DIMENSIONS["scenario"][int(r["scenario"])],
            "lighting": DIMENSIONS["lighting"][int(r["lighting"])],
            "weather": DIMENSIONS["weather"][int(r["weather"])],
        })

    return {"matched_errors": int(len(sub)), "by_type": by_type,
            "worst_containers": worst, "examples": examples}
