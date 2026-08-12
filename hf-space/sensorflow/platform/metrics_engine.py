"""Metric engine: precision/recall/F1/IoU + verification rates.

Pure functions — no I/O. Callers feed counts from MegaEval containers or
LabelEval match stats.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def round4(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 4)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round4(precision),
        "recall": round4(recall),
        "f1": round4(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def mean_iou(sum_iou: float, tp: int) -> Optional[float]:
    return round4(safe_div(sum_iou, tp))


def verification_rates(
    *,
    n_objects: int,
    verified: int = 0,
    reviewed: int = 0,
    disputed: int = 0,
    auto_accepted: int = 0,
    hitl: int = 0,
    unverified: Optional[int] = None,
) -> Dict[str, Any]:
    """Verification / HITL funnel rates at container (or dataset) scope.

    unverified defaults to n_objects - verified when not provided.
    """
    n = max(0, int(n_objects))
    verified = max(0, int(verified))
    reviewed = max(0, int(reviewed))
    disputed = max(0, int(disputed))
    auto_accepted = max(0, int(auto_accepted))
    hitl = max(0, int(hitl))
    if unverified is None:
        unverified = max(0, n - verified)

    def rate(count: int) -> Optional[float]:
        return round4(safe_div(count, n)) if n > 0 else None

    return {
        "n_objects": n,
        "verified": verified,
        "unverified": unverified,
        "disputed": disputed,
        "auto_accepted": auto_accepted,
        "hitl": hitl,
        "reviewed": reviewed,
        "verified_rate": rate(verified),
        "unverified_rate": rate(unverified),
        "disputed_rate": rate(disputed),
        "auto_accept_rate": rate(auto_accepted),
        "hitl_rate": rate(hitl),
        "review_coverage": rate(reviewed),
    }


def container_quality_metrics(
    *,
    tp: int,
    fp: int,
    fn: int,
    sum_iou: float = 0.0,
    n_objects: int = 0,
    verified: int = 0,
    reviewed: int = 0,
    disputed: int = 0,
    auto_accepted: int = 0,
    hitl: int = 0,
    anomalies: int = 0,
) -> Dict[str, Any]:
    prf = precision_recall_f1(tp, fp, fn)
    n = n_objects or (tp + fp)
    ver = verification_rates(
        n_objects=n,
        verified=verified,
        reviewed=reviewed,
        disputed=disputed,
        auto_accepted=auto_accepted,
        hitl=hitl,
    )
    return {
        **prf,
        "mean_iou": mean_iou(sum_iou, tp),
        "anomalies": anomalies,
        "anomaly_rate": round4(safe_div(anomalies, n)) if n else None,
        "verification": ver,
    }


def aggregate_container_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up a page of container metric dicts into one profile summary."""
    tp = fp = fn = verified = reviewed = disputed = auto_acc = hitl = anomalies = n = 0
    sum_iou = 0.0
    for r in rows:
        tp += int(r.get("tp") or 0)
        fp += int(r.get("fp") or 0)
        fn += int(r.get("fn") or 0)
        verified += int(r.get("verified") or 0)
        reviewed += int(r.get("reviewed") or 0)
        disputed += int(r.get("disputed") or 0)
        auto_acc += int(r.get("auto_accepted") or 0)
        hitl += int(r.get("hitl") or 0)
        anomalies += int(r.get("anomalies") or 0)
        n += int(r.get("n_objects") or 0)
        mi = r.get("mean_iou")
        tpi = int(r.get("tp") or 0)
        if mi is not None and tpi > 0:
            sum_iou += float(mi) * tpi
        elif r.get("sum_iou") is not None:
            sum_iou += float(r["sum_iou"])
    return container_quality_metrics(
        tp=tp, fp=fp, fn=fn, sum_iou=sum_iou, n_objects=n,
        verified=verified, reviewed=reviewed, disputed=disputed,
        auto_accepted=auto_acc, hitl=hitl, anomalies=anomalies,
    )


def delta_metrics(
    baseline: Dict[str, Optional[float]],
    candidate: Dict[str, Optional[float]],
    keys: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    keys = list(keys or ("precision", "recall", "f1", "mean_iou", "safety_recall", "anomaly_rate"))
    out = []
    for k in keys:
        b, c = baseline.get(k), candidate.get(k)
        if b is None or c is None:
            continue
        out.append({
            "metric": k,
            "baseline": round4(float(b)),
            "candidate": round4(float(c)),
            "delta": round4(float(c) - float(b)),
        })
    return out
