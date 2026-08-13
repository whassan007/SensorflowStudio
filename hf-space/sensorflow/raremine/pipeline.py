"""Staged validation pipeline + review queue.

Lifecycle: Raw -> Mined -> Deduplicated -> Automated Validation ->
Human/GT Validation -> Curated (Evaluation Set | Training Candidate),
with an Archive for rejections.

The automated stage MEASURES candidate coherence against planted ground truth
where available; it never approves for training. Human validation is modeled
as an explicit API action (approve/reject with a note). CRITICAL candidates
surface at the top of the review queue; there is no clean hook to push foreign
entities into sensorflow.evaluation's ReviewTask queue (its tasks are keyed to
annotation/frame/dataset ids of that platform), so the miner exposes its own
queue instead — noted integration gap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sensorflow.raremine import dedup as dedup_mod
from sensorflow.raremine import lineage as lineage_mod
from sensorflow.raremine import miner as miner_mod
from sensorflow.raremine import tracks as tracks_mod
from sensorflow.raremine.models import (
    MiningRun,
    RareMineStore,
    TrackCandidate,
    new_id,
    now_iso,
)

PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def run_full_pipeline(store: RareMineStore, bank_id: str,
                      config: Optional[Dict[str, Any]] = None,
                      diversity_budget: int = 12) -> MiningRun:
    """Mine -> consolidate -> dedup -> diversity -> auto-validate -> lineage."""
    bank = store.get("banks", bank_id)
    if bank is None:
        raise KeyError(f"unknown scene bank {bank_id}")
    run_id = new_id("rmrun")
    # a fresh run replaces previous mining products for the bank
    store.clear("candidates")
    store.clear("track_candidates")
    store.clear("lineage")

    frame_cands = miner_mod.run_mining(store, bank_id, config, run_id=run_id)
    track_cands = tracks_mod.consolidate_tracks(store, bank_id, run_id=run_id)
    dedup_report = dedup_mod.deduplicate(store, bank_id, run_id=run_id)
    diversity_report = dedup_mod.select_diverse(store, bank_id, budget=diversity_budget,
                                                run_id=run_id)

    validated = 0
    for tc in store.where("track_candidates", bank_id=bank_id):
        if tc.stage != "DEDUPLICATED":
            continue
        auto_validate(store, tc)
        validated += 1

    run = MiningRun(
        run_id=run_id,
        bank_id=bank_id,
        config=config or {},
        num_frame_candidates=len(frame_cands),
        num_track_candidates=len(track_cands),
        num_duplicates=dedup_report["duplicates_archived"],
        num_diversity_selected=len(diversity_report["selected_ids"]),
    )
    store.meta["last_run"] = {
        "run_id": run_id,
        "bank_id": bank_id,
        "dedup_report": dedup_report,
        "diversity_report": diversity_report,
        "auto_validated": validated,
        "finished_at": now_iso(),
    }
    store.put("runs", run)
    store.audit("mining_run", "MiningRun", run_id,
                f"{len(frame_cands)} frame candidates -> {len(track_cands)} tracks, "
                f"{dedup_report['duplicates_archived']} duplicates archived")
    store.save()
    return run


def auto_validate(store: RareMineStore, tc: TrackCandidate) -> Dict[str, Any]:
    """AUTOMATED validation: measure the proposal against ground truth when the
    scene provides it. Records coherence — it does not confirm importance."""
    rep = tc.representative
    scene = store.get("scenes", rep.scene_id)
    result: Dict[str, Any] = {"checked_at": now_iso()}
    if scene is None or not scene.modalities.get("gt_annotations"):
        result["status"] = "NO_GT"
        result["detail"] = "no ground-truth annotations available for this scene"
    else:
        gt = next((g for g in scene.gt_boxes if g.object_id == rep.object_id), None)
        if gt is None:
            result["status"] = "AUTO_INCOHERENT"
            result["detail"] = "candidate object has no ground-truth counterpart"
        else:
            truly_rare = gt.class_name == "pedestrian" and gt.is_costumed
            coherent = rep.edge_case_detected == truly_rare
            result["status"] = "AUTO_COHERENT" if coherent else "AUTO_INCOHERENT"
            result["gt_class"] = gt.class_name
            result["gt_is_costumed"] = gt.is_costumed
            result["gt_costume_type"] = gt.costume_type
            result["detail"] = (
                "proposal agrees with ground truth" if coherent else
                f"proposal ({'detected' if rep.edge_case_detected else 'rejected'}) disagrees "
                f"with GT (class={gt.class_name}, costumed={gt.is_costumed})")
    tc.auto_validation = result
    tc.stage = "AUTO_VALIDATED"
    store.put("track_candidates", tc)
    rec = lineage_mod.create_lineage(store, tc)
    rec.validation_status = result["status"]
    store.put("lineage", rec)
    return result


def review_queue(store: RareMineStore, bank_id: str,
                 priority: Optional[str] = None,
                 status: Optional[str] = None,
                 costume: Optional[str] = None,
                 difficulty: Optional[str] = None) -> List[TrackCandidate]:
    """Ranked review queue over post-dedup candidates (CRITICAL first)."""
    items = [t for t in store.where("track_candidates", bank_id=bank_id)
             if t.stage not in ("ARCHIVED",) and t.duplicate_of is None]
    if priority:
        items = [t for t in items if t.representative.curation_priority == priority]
    if status:
        items = [t for t in items if t.stage == status]
    if costume:
        items = [t for t in items if costume in t.representative.costume_type]
    if difficulty:
        items = [t for t in items if t.representative.perception_difficulty == difficulty]
    items.sort(key=lambda t: (PRIORITY_ORDER.index(t.representative.curation_priority),
                              t.representative.confidence_rare_event),
               reverse=True)
    return items


def human_review(store: RareMineStore, track_candidate_id: str, action: str,
                 note: str = "", reviewer: str = "human-reviewer",
                 destination: Optional[str] = None) -> TrackCandidate:
    """HUMAN validation: approve into a destination or reject to the archive."""
    tc = store.get("track_candidates", track_candidate_id)
    if tc is None:
        raise KeyError(f"unknown track candidate {track_candidate_id}")
    if action not in ("approve", "reject"):
        raise ValueError("action must be approve|reject")
    rec = lineage_mod.get_lineage(store, track_candidate_id) or lineage_mod.create_lineage(store, tc)
    tc.human_validation = {"action": action, "note": note, "reviewer": reviewer,
                           "reviewed_at": now_iso()}
    if action == "approve":
        rec.validation_status = "APPROVED"
        store.put("lineage", rec)
        target = destination or tc.representative.recommended_dataset_destination
        if target in ("NO_ACTION", "REVIEW_QUEUE"):
            target = "RARE_EVENT_DATASET"
        lineage_mod.set_destination(store, tc, target, curator=reviewer)
        tc.stage = "CURATED"
    else:
        rec.validation_status = "REJECTED"
        rec.training_eligible = False
        rec.evaluation_eligible = False
        store.put("lineage", rec)
        tc.stage = "ARCHIVED"
        tc.destination = "NO_ACTION"
    store.put("track_candidates", tc)
    store.audit(f"human_{action}", "TrackCandidate", track_candidate_id, note, actor=reviewer)
    store.save()
    return tc


def destinations_report(store: RareMineStore, bank_id: str) -> Dict[str, Any]:
    tcs = [t for t in store.where("track_candidates", bank_id=bank_id)
           if t.duplicate_of is None]
    buckets: Dict[str, List[Dict]] = {}
    for t in tcs:
        rec = lineage_mod.get_lineage(store, t.track_candidate_id)
        buckets.setdefault(t.destination, []).append({
            "track_candidate_id": t.track_candidate_id,
            "stage": t.stage,
            "priority": t.representative.curation_priority,
            "costume_type": t.representative.costume_type,
            "lineage": rec.model_dump() if rec else None,
        })
    return {"destinations": buckets,
            "counts": {k: len(v) for k, v in buckets.items()}}
