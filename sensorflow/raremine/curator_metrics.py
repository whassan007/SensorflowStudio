"""Curator-quality statistics: how good is the MINER, measured against the
planted ground truth. Statistics determine importance — not the miner itself.

  - mining precision / recall / false-discovery rate at track level
  - confidence calibration (binned reliability: stated confidence vs truth)
  - curation yield (validated / reviewed)
  - model-value (curated examples that expose a real baseline failure)
  - recurring-miss analysis -> improvement report feeding the next run config
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sensorflow.raremine.models import RareMineStore, Scene, SceneObject, TrackCandidate

CAL_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def _truth_by_track(store: RareMineStore, bank_id: str) -> Dict[Tuple[str, str], SceneObject]:
    """Planted truth per (sequence, track): the frame-0 object."""
    truth: Dict[Tuple[str, str], SceneObject] = {}
    for scene in store.where("scenes", bank_id=bank_id):
        if scene.near_duplicate_of is not None:
            continue
        for obj in scene.objects:
            key = (scene.sequence_id, obj.track_id)
            if key not in truth:
                truth[key] = obj
    return truth


def _track_candidates(store: RareMineStore, bank_id: str) -> List[TrackCandidate]:
    return [t for t in store.where("track_candidates", bank_id=bank_id)
            if t.duplicate_of is None]


def curator_report(store: RareMineStore, bank_id: str) -> Dict[str, Any]:
    truth = _truth_by_track(store, bank_id)
    tcs = _track_candidates(store, bank_id)

    tp = fp = fn = tn = 0
    calibration = [{"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "true": 0} for lo, hi in CAL_BINS]
    proposed_tracks = set()

    for tc in tcs:
        key = (tc.sequence_id, tc.track_id)
        proposed_tracks.add(key)
        obj = truth.get(key)
        truly_rare = obj is not None and obj.truth_kind == "costumed_pedestrian"
        detected = tc.representative.edge_case_detected
        if detected and truly_rare:
            tp += 1
        elif detected and not truly_rare:
            fp += 1
        elif not detected and truly_rare:
            fn += 1
        else:
            tn += 1
        conf = tc.representative.confidence_rare_event
        for i, (lo, hi) in enumerate(CAL_BINS):
            if lo <= conf < hi:
                calibration[i]["n"] += 1
                calibration[i]["true"] += int(truly_rare)
                break

    # planted positives the miner never even surfaced as track candidates
    for key, obj in truth.items():
        if obj.truth_kind == "costumed_pedestrian" and key not in proposed_tracks:
            fn += 1

    total_planted = sum(1 for o in truth.values() if o.truth_kind == "costumed_pedestrian")
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    fdr = round(fp / (tp + fp), 4) if (tp + fp) else None

    for row in calibration:
        row["observed_rate"] = round(row["true"] / row["n"], 3) if row["n"] else None

    # curation yield + model value over human-reviewed examples
    reviewed = [t for t in tcs if t.human_validation is not None]
    approved = [t for t in reviewed if t.human_validation.get("action") == "approve"]
    curated_failure = 0
    for t in approved:
        if _exposes_planted_failure(store, t):
            curated_failure += 1
    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "planted_positives": total_planted,
        "mining_precision": precision,
        "mining_recall": recall,
        "false_discovery_rate": fdr,
        "calibration": calibration,
        "curation_yield": {
            "reviewed": len(reviewed),
            "approved": len(approved),
            "yield": round(len(approved) / len(reviewed), 3) if reviewed else None,
        },
        "model_value": {
            "curated": len(approved),
            "expose_model_failure": curated_failure,
            "fraction": round(curated_failure / len(approved), 3) if approved else None,
        },
    }


def _exposes_planted_failure(store: RareMineStore, tc: TrackCandidate) -> bool:
    for scene in store.where("scenes", sequence_id=tc.sequence_id):
        for pred in scene.baseline_predictions:
            if pred.planted_failure:
                return True
        # FALSE_NEGATIVE plants remove the prediction entirely
        if scene.modalities.get("baseline_predictions"):
            covered = {p.object_id for p in scene.baseline_predictions}
            for obj in scene.objects:
                if obj.track_id == tc.track_id and obj.object_id not in covered:
                    return True
    return False


def improvement_report(store: RareMineStore, bank_id: str) -> Dict[str, Any]:
    """Recurring-miss analysis: what the miner misses or over-fires on, plus a
    config for the next run. Continuous-learning loop, statistics-driven."""
    truth = _truth_by_track(store, bank_id)
    tcs = {(t.sequence_id, t.track_id): t for t in _track_candidates(store, bank_id)}

    missed: List[Dict[str, Any]] = []
    overfired: List[Dict[str, Any]] = []
    for key, obj in truth.items():
        tc = tcs.get(key)
        detected = tc is not None and tc.representative.edge_case_detected
        if obj.truth_kind == "costumed_pedestrian" and not detected:
            missed.append({
                "sequence_id": key[0],
                "costume_type": obj.truth_costume_type,
                "silhouette": obj.truth_silhouette_deviation,
                "occlusion_env": obj.truth_occlusion_env,
                "context": obj.context,
                "reason": ("below confidence thresholds"
                           if tc is not None else "never surfaced as a candidate"),
                "miner_confidence": tc.representative.confidence_rare_event if tc else None,
            })
        elif obj.truth_kind == "confounder" and detected and \
                tc.representative.confidence_rare_event >= 0.5:
            overfired.append({
                "sequence_id": key[0],
                "confounder_type": obj.truth_costume_type,
                "context": obj.context,
                "miner_confidence": tc.representative.confidence_rare_event,
            })

    def _group(rows: List[Dict], field: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            k = str(r.get(field))
            out[k] = out.get(k, 0) + 1
        return out

    # translate recurring misses into next-run sensitivity boosts
    boost: Dict[str, float] = {}
    for ct, n in _group(missed, "costume_type").items():
        if ct != "None" and n >= 1:
            boost[ct] = round(min(0.05 * n, 0.15), 3)

    next_config: Dict[str, Any] = {"sensitivity_boost": boost}
    if overfired:
        next_config["rare_event_threshold"] = 0.5  # tighten when over-firing

    return {
        "recurring_misses": {
            "total": len(missed),
            "by_costume_type": _group(missed, "costume_type"),
            "by_occlusion": _group(missed, "occlusion_env"),
            "by_context": _group(missed, "context"),
            "examples": missed[:20],
        },
        "over_fires": {
            "total": len(overfired),
            "by_confounder_type": _group(overfired, "confounder_type"),
            "examples": overfired[:20],
        },
        "next_run_config": next_config,
        "note": "feed next_run_config into the next mining run; boosts raise "
                "sensitivity for costume families the miner recurrently missed",
    }
