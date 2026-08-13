"""Quantitative validation: for candidates with model outputs, compute the
OBSERVED difficulty from actual baseline behavior and compare it with the
miner's PREDICTED difficulty (agreement matrix).

Observed metrics per track candidate: mean baseline confidence, per-example
correctness, localization quality (IoU), miss rate, class confusion, track
stability. Predicted difficulty comes from the miner and is never edited here
— this stage only measures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sensorflow.raremine.models import RareMineStore, TrackCandidate

DIFF_ORDER = ["EASY", "MODERATE", "HARD", "EXTREME"]


def _track_object_ids(store: RareMineStore, tc: TrackCandidate) -> List[tuple]:
    """(scene, object) pairs across the track's frames."""
    out = []
    for scene in store.where("scenes", sequence_id=tc.sequence_id):
        for obj in scene.objects:
            if obj.track_id == tc.track_id:
                out.append((scene, obj))
    return out


def observed_metrics(store: RareMineStore, tc: TrackCandidate) -> Optional[Dict[str, Any]]:
    """None when no baseline predictions exist anywhere on the track."""
    pairs = [(s, o) for s, o in _track_object_ids(store, tc)
             if s.modalities.get("baseline_predictions")]
    if not pairs:
        return None
    confidences: List[float] = []
    ious: List[float] = []
    misses = 0
    confusion: Dict[str, int] = {}
    unstable = 0
    correct = 0
    for scene, obj in pairs:
        pred = next((p for p in scene.baseline_predictions if p.object_id == obj.object_id), None)
        if pred is None:
            misses += 1
            continue
        confidences.append(pred.confidence)
        if pred.iou_with_gt is not None:
            ious.append(pred.iou_with_gt)
        if pred.predicted_class != "pedestrian":
            confusion[pred.predicted_class] = confusion.get(pred.predicted_class, 0) + 1
        else:
            correct += 1
        if not pred.track_stable:
            unstable += 1
    n = len(pairs)
    mean_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    mean_iou = round(sum(ious) / len(ious), 3) if ious else None
    miss_rate = round(misses / n, 3)
    confused_rate = round(sum(confusion.values()) / n, 3)
    stability = round(1.0 - unstable / max(n - misses, 1), 3)

    # observed difficulty grade from measured behavior
    if miss_rate > 0.5 or confused_rate > 0.5:
        grade = "EXTREME"
    elif miss_rate > 0.0 or confusion or unstable:
        grade = "HARD"
    elif mean_conf < 0.4 or (mean_iou is not None and mean_iou < 0.5):
        grade = "MODERATE" if mean_conf >= 0.25 else "HARD"
    else:
        grade = "EASY"

    return {
        "frames_with_predictions": n,
        "mean_baseline_confidence": mean_conf,
        "mean_iou": mean_iou,
        "miss_rate": miss_rate,
        "class_confusion": confusion,
        "confused_rate": confused_rate,
        "track_stability": stability,
        "per_example_correct": correct,
        "observed_difficulty": grade,
        "exposes_model_failure": miss_rate > 0 or bool(confusion) or unstable > 0
        or mean_conf < 0.35 or (mean_iou is not None and mean_iou < 0.5),
    }


def quantitative_report(store: RareMineStore, bank_id: str) -> Dict[str, Any]:
    tcs = [t for t in store.where("track_candidates", bank_id=bank_id)
           if t.duplicate_of is None and t.representative.edge_case_detected]
    rows = []
    matrix: Dict[str, Dict[str, int]] = {d: {d2: 0 for d2 in DIFF_ORDER} for d in DIFF_ORDER}
    agree = 0
    with_obs = 0
    for tc in sorted(tcs, key=lambda t: t.track_candidate_id):
        obs = observed_metrics(store, tc)
        predicted = tc.representative.perception_difficulty
        row = {
            "track_candidate_id": tc.track_candidate_id,
            "costume_type": tc.representative.costume_type,
            "predicted_difficulty": predicted,
            "observed": obs,
        }
        if obs is not None:
            with_obs += 1
            matrix[predicted][obs["observed_difficulty"]] += 1
            if abs(DIFF_ORDER.index(predicted) - DIFF_ORDER.index(obs["observed_difficulty"])) <= 1:
                agree += 1
        rows.append(row)
    return {
        "candidates_evaluated": len(rows),
        "with_model_outputs": with_obs,
        "agreement_matrix": matrix,
        "within_one_level_agreement": round(agree / with_obs, 3) if with_obs else None,
        "rows": rows,
        "note": "predicted difficulty is the miner's forecast; observed difficulty is "
                "measured from actual baseline model behavior — they are compared, "
                "never merged",
    }
