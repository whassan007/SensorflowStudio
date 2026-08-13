"""Grader disagreement subsystem.

Grader sources (spec §11): auto label, vendor GT, historical model, VLM label,
human reviewer (when present), independent detector. Computes classification,
spatial, temporal and statistical agreement, and a canonical
grader_consensus_score while preserving the components.

Statistical agreement — used correctly, not interchangeably:
- Cohen's Kappa:        exactly 2 graders, categorical labels.
- Fleiss' Kappa:        >2 graders, every grader rates every (fixed) item.
- Krippendorff's Alpha: any number of graders, tolerates missing ratings.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sensorflow.evaluation.records import (
    Annotation,
    EvalStore,
    Frame,
    GraderComparison,
)
from sensorflow.metrics.perception_3d import bev_iou

GRADER_SOURCES = ["auto_label", "vendor_gt", "historical_model", "vlm_label", "independent_detector"]

CLASS_LIST = ["pedestrian", "cyclist", "vehicle", "motorcycle", "truck"]


# ------------------------------------------------------------------ statistics


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for exactly two graders over the same items."""
    assert len(a) == len(b) and len(a) > 0
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = Counter(a)
    pb = Counter(b)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    if math.isclose(pe, 1.0):
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: List[Dict[str, int]]) -> float:
    """Fleiss' kappa: ratings[i] = {category: count} with a fixed rater count
    per item (every grader rates every item)."""
    if not ratings:
        return 0.0
    n_raters = sum(ratings[0].values())
    cats = sorted({c for r in ratings for c in r})
    N = len(ratings)
    p_j = {c: sum(r.get(c, 0) for r in ratings) / (N * n_raters) for c in cats}
    P_i = []
    for r in ratings:
        s = sum(cnt * (cnt - 1) for cnt in r.values())
        P_i.append(s / (n_raters * (n_raters - 1)) if n_raters > 1 else 1.0)
    P_bar = float(np.mean(P_i))
    P_e = sum(p ** 2 for p in p_j.values())
    if math.isclose(P_e, 1.0):
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha(data: List[List[Optional[str]]]) -> float:
    """Krippendorff's alpha (nominal) tolerating missing ratings.

    data[rater][item] = category or None.
    """
    # Collect pairable values per item (unit).
    units: List[List[str]] = []
    n_items = max((len(r) for r in data), default=0)
    for i in range(n_items):
        vals = [r[i] for r in data if i < len(r) and r[i] is not None]
        if len(vals) >= 2:
            units.append(vals)
    if not units:
        return 0.0
    # Observed disagreement.
    Do_num, Do_den = 0.0, 0.0
    all_vals: List[str] = []
    for vals in units:
        m = len(vals)
        all_vals.extend(vals)
        cnt = Counter(vals)
        pairs_disagree = 0
        for c1 in cnt:
            for c2 in cnt:
                if c1 != c2:
                    pairs_disagree += cnt[c1] * cnt[c2]
        Do_num += pairs_disagree / (m - 1)
        Do_den += m
    Do = Do_num / Do_den if Do_den else 0.0
    # Expected disagreement.
    total = len(all_vals)
    cnt_all = Counter(all_vals)
    De_num = 0.0
    for c1 in cnt_all:
        for c2 in cnt_all:
            if c1 != c2:
                De_num += cnt_all[c1] * cnt_all[c2]
    De = De_num / (total * (total - 1)) if total > 1 else 0.0
    if math.isclose(De, 0.0):
        return 1.0
    return 1 - Do / De


# ------------------------------------------------------------------ simulated graders

def _simulate_grader_votes(ann: Annotation, gt_class: Optional[str], rng: np.random.Generator) -> Dict[str, str]:
    """Deterministic per-annotation grader simulation.

    Each grader source produces a class vote. Injected GRADER_DISAGREEMENT
    forces the VLM + independent detector to disagree.
    """
    base = gt_class or ann.class_name
    votes: Dict[str, str] = {"auto_label": ann.class_name}
    votes["vendor_gt"] = gt_class if gt_class else ann.class_name

    def maybe_flip(cls: str, p: float) -> str:
        if rng.random() < p:
            alts = [c for c in CLASS_LIST if c != cls]
            return alts[int(rng.integers(0, len(alts)))]
        return cls

    disagree = "GRADER_DISAGREEMENT" in ann.injected_errors
    votes["historical_model"] = maybe_flip(base, 0.02)
    votes["vlm_label"] = maybe_flip(base, 1.0 if disagree else 0.04)
    votes["independent_detector"] = maybe_flip(base, 1.0 if disagree else 0.03)
    return votes


def _spatial_agreement(ann: Annotation, gt_box: Optional[List[float]], rng: np.random.Generator) -> Tuple[float, Dict[str, float]]:
    """Spatial agreement between graders' boxes (auto vs reference vs an
    independent perturbation): IoU, center distance, dims, orientation."""
    if not ann.bbox_3d:
        return 0.0, {}
    ref = gt_box if gt_box else ann.bbox_3d
    iou = bev_iou(ann.bbox_3d, ref)
    center_dist = math.hypot(ann.bbox_3d[0] - ref[0], ann.bbox_3d[1] - ref[1])
    dim_diff = float(np.mean([abs(ann.bbox_3d[i] - ref[i]) / max(ref[i], 0.1) for i in (3, 4, 5)]))
    yaw_diff = abs(ann.bbox_3d[6] - ref[6]) % (2 * math.pi)
    yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)
    detail = {
        "iou": round(iou, 4),
        "center_distance": round(center_dist, 3),
        "dimension_diff": round(dim_diff, 4),
        "orientation_diff_deg": round(math.degrees(yaw_diff), 2),
    }
    score = float(np.clip(
        0.5 * iou + 0.25 * max(0, 1 - center_dist / 2.0) + 0.15 * max(0, 1 - dim_diff) + 0.10 * max(0, 1 - yaw_diff / math.pi),
        0, 1))
    return score, detail


def grade_annotation(
    store: EvalStore,
    ann: Annotation,
    frame: Frame,
    temporal_quality: Optional[float] = None,
    seed: int = 7,
) -> GraderComparison:
    rng = np.random.default_rng(abs(hash((ann.annotation_id, seed))) % (2 ** 32))
    gt = next((g for g in frame.gt_boxes if g.gt_id == ann.matched_gt_id), None)

    votes = _simulate_grader_votes(ann, gt.class_name if gt else None, rng)
    classes = list(votes.values())
    # Reliability-weighted class agreement: authoritative sources count more.
    weights = {"auto_label": 1.0, "vendor_gt": 2.0, "historical_model": 0.75,
               "vlm_label": 0.5, "independent_detector": 0.75}
    weight_by_class: Dict[str, float] = {}
    for source, cls in votes.items():
        weight_by_class[cls] = weight_by_class.get(cls, 0.0) + weights.get(source, 1.0)
    total_weight = sum(weights.get(s, 1.0) for s in votes)
    class_agreement = max(weight_by_class.values()) / total_weight

    spatial, spatial_detail = _spatial_agreement(ann, gt.bbox_3d if gt else None, rng)
    temporal = temporal_quality if temporal_quality is not None else (1.0 if ann.track_id else 0.5)

    # Per-item agreement indicators; proper kappa/alpha statistics are computed
    # at dataset level in dataset_grader_statistics (kappa is undefined on a
    # single item).
    kappa_stats: Dict[str, float] = {
        "auto_vs_vendor_agree": 1.0 if votes["auto_label"] == votes["vendor_gt"] else 0.0,
        "vote_agreement": round(class_agreement, 4),
    }

    disagreements: List[str] = []
    if class_agreement < 1.0:
        disagreements.append("classification")
    if spatial < 0.6:
        disagreements.append("spatial")
    if temporal < 0.6:
        disagreements.append("temporal")

    consensus = float(np.clip(0.5 * class_agreement + 0.35 * spatial + 0.15 * temporal, 0, 1))

    # --- additive consensus evidence (canonical consensus above is unchanged)
    csv = consensus_score_vector(votes, spatial_detail, float(temporal), weights)
    candidates = _grader_box_candidates(ann, gt, votes, rng)
    mbr = mbr_consensus_select(candidates)
    extra_stats: Dict[str, float] = {f"csv_{k}": v for k, v in csv.items()}
    for c in mbr["candidates"]:
        extra_stats[f"mbr_risk_{c['source']}"] = c["expected_risk"]
    extra_stats["mbr_selected_idx"] = float(mbr["selected_index"])
    extra_stats["mbr_selected_risk"] = mbr["selected"]["expected_risk"]

    cmp = GraderComparison(
        annotation_id=ann.annotation_id,
        grader_count=len(votes),
        graders=list(votes.keys()),
        class_votes=votes,
        class_agreement=round(class_agreement, 4),
        spatial_agreement=round(spatial, 4),
        temporal_agreement=round(float(temporal), 4),
        consensus=round(consensus, 4),
        disagreement_types=disagreements,
        kappa_stats={**kappa_stats,
                     **{f"spatial_{k}": v for k, v in spatial_detail.items()},
                     **extra_stats},
    )
    store.put("grader_comparisons", cmp)
    return cmp


def dataset_grader_statistics(store: EvalStore, dataset_id: str) -> Dict[str, float]:
    """Dataset-level statistical agreement across all graded items."""
    anns = store.where("annotations", dataset_id=dataset_id)
    comparisons = [store.get("grader_comparisons", a.annotation_id) for a in anns]
    comparisons = [c for c in comparisons if c is not None]
    if not comparisons:
        return {}

    # Two-grader Cohen's kappa: auto label vs vendor GT over all items.
    auto = [c.class_votes.get("auto_label", "") for c in comparisons]
    vendor = [c.class_votes.get("vendor_gt", "") for c in comparisons]
    cohen = cohens_kappa(auto, vendor)

    # Fleiss' kappa: all graders rate all items (fixed panel).
    ratings = [dict(Counter(c.class_votes.values())) for c in comparisons]
    fleiss = fleiss_kappa(ratings)

    # Krippendorff's alpha: tolerate a missing vendor rating (simulated missing
    # data on unmatched labels).
    raters: List[List[Optional[str]]] = [
        [c.class_votes.get(source) for c in comparisons] for source in GRADER_SOURCES
    ]
    # Mark vendor rating missing where the label had no matched GT.
    ann_by_id = {a.annotation_id: a for a in anns}
    for j, c in enumerate(comparisons):
        a = ann_by_id.get(c.annotation_id)
        if a is not None and a.matched_gt_id is None:
            raters[GRADER_SOURCES.index("vendor_gt")][j] = None
    kripp = krippendorff_alpha(raters)

    stats = {
        "cohen_kappa": round(cohen, 4),
        "fleiss_kappa": round(fleiss, 4),
        "krippendorff_alpha": round(kripp, 4),
        "mean_consensus": round(float(np.mean([c.consensus for c in comparisons if c.consensus is not None])), 4),
    }

    # Additive: Kendall's tau between the model-confidence ranking of items and
    # the grader-consensus ranking (do graders order item quality like the
    # model's own confidence does?). Needs >= 2 items.
    paired = [(ann_by_id[c.annotation_id].confidence, c.consensus or 0.0)
              for c in comparisons if c.annotation_id in ann_by_id]
    conf = [p[0] for p in paired]
    cons = [p[1] for p in paired]
    if len(conf) >= 2:
        tau, p = kendalls_tau(conf, cons)
        stats["kendall_tau_confidence_vs_consensus"] = tau
        stats["kendall_tau_p_value"] = p
    return stats


# ------------------------------------------------------------------ additive consensus extensions
#
# The canonical scalar consensus above stays authoritative. The functions
# below add richer evidence (spec: consensus score vector, Kendall's tau
# ranking agreement, Minimum Bayes Risk selection) surfaced through
# GraderComparison.kappa_stats with csv_* / mbr_* prefixes.


def consensus_score_vector(votes: Dict[str, str], spatial_detail: Dict[str, float],
                           temporal: float,
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Per-element consensus scores instead of one scalar.

    Elements: majority class share, reliability-weighted class agreement,
    spatial sub-scores (IoU, center, dimensions, orientation) and temporal
    consistency — each individually in [0, 1].
    """
    n = max(len(votes), 1)
    counts = Counter(votes.values())
    majority_share = max(counts.values()) / n if counts else 0.0
    weights = weights or {}
    weight_by_class: Dict[str, float] = {}
    for source, cls in votes.items():
        weight_by_class[cls] = weight_by_class.get(cls, 0.0) + weights.get(source, 1.0)
    total_w = sum(weights.get(s, 1.0) for s in votes) or 1.0
    weighted = max(weight_by_class.values()) / total_w if weight_by_class else 0.0

    center = spatial_detail.get("center_distance")
    dims = spatial_detail.get("dimension_diff")
    orient = spatial_detail.get("orientation_diff_deg")
    return {
        "class_majority_share": round(majority_share, 4),
        "class_weighted_agreement": round(weighted, 4),
        "spatial_iou": round(float(spatial_detail.get("iou", 0.0)), 4),
        "spatial_center": round(max(0.0, 1 - (center or 2.0) / 2.0), 4),
        "spatial_dims": round(max(0.0, 1 - (dims if dims is not None else 1.0)), 4),
        "spatial_orientation": round(max(0.0, 1 - (orient if orient is not None else 180.0) / 180.0), 4),
        "temporal": round(float(temporal), 4),
    }


def kendalls_tau(scores_a: Sequence[float], scores_b: Sequence[float]) -> Tuple[float, float]:
    """Kendall's tau-b ranking agreement between two score lists over the same
    items (scipy.stats.kendalltau). Returns (tau, p_value)."""
    from scipy.stats import kendalltau
    tau, p = kendalltau(list(scores_a), list(scores_b))
    if math.isnan(tau):
        return 0.0, 1.0
    return round(float(tau), 4), round(float(p), 6)


def mbr_utility(cand_a: Dict, cand_b: Dict, iou_weight: float = 0.5) -> float:
    """Pairwise utility for MBR: class match + BEV IoU (each in [0, 1])."""
    class_match = 1.0 if cand_a.get("class_name") == cand_b.get("class_name") else 0.0
    box_a, box_b = cand_a.get("bbox_3d"), cand_b.get("bbox_3d")
    iou = bev_iou(box_a, box_b) if box_a and box_b else 0.0
    return (1 - iou_weight) * class_match + iou_weight * iou


def mbr_consensus_select(candidates: List[Dict], iou_weight: float = 0.5) -> Dict:
    """Minimum Bayes Risk consensus selection.

    Among candidate labels (each {source, class_name, bbox_3d}), select the one
    minimizing expected disagreement risk against the others treated as
    pseudo-references: risk(c) = 1 - mean_utility(c, others).
    """
    if not candidates:
        raise ValueError("mbr_consensus_select requires at least one candidate")
    scored = []
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        util = (float(np.mean([mbr_utility(cand, o, iou_weight) for o in others]))
                if others else 1.0)
        scored.append({"source": cand.get("source", f"candidate_{i}"),
                       "class_name": cand.get("class_name"),
                       "expected_utility": round(util, 4),
                       "expected_risk": round(1 - util, 4)})
    best = min(range(len(scored)),
               key=lambda i: (scored[i]["expected_risk"], i))
    return {"selected_index": best, "selected": scored[best], "candidates": scored,
            "utility": f"{1 - iou_weight:.2f}*class_match + {iou_weight:.2f}*bev_iou"}


def _grader_box_candidates(ann: Annotation, gt, votes: Dict[str, str],
                           rng: np.random.Generator) -> List[Dict]:
    """Candidate labels for MBR selection. auto_label and vendor_gt use their
    real boxes; the remaining graders' boxes are SIMULATED as deterministic
    small perturbations of their vote's base box (consistent with the simulated
    grader panel above)."""
    base_box = list(gt.bbox_3d) if gt is not None else (list(ann.bbox_3d) if ann.bbox_3d else None)
    out: List[Dict] = []
    for source in GRADER_SOURCES:
        if source not in votes:
            continue
        if source == "auto_label":
            box = list(ann.bbox_3d) if ann.bbox_3d else None
        elif source == "vendor_gt":
            box = list(gt.bbox_3d) if gt is not None else (list(ann.bbox_3d) if ann.bbox_3d else None)
        elif base_box is not None:
            jitter = rng.normal(0.0, 0.12, size=3)
            box = list(base_box)
            box[0] += float(jitter[0])
            box[1] += float(jitter[1])
            box[6] += float(jitter[2] * 0.2)
        else:
            box = None
        out.append({"source": source, "class_name": votes[source], "bbox_3d": box})
    return out
