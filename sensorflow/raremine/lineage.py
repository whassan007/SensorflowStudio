"""Lineage + governance: who curated what, where it may be used, and the
leakage guard that makes silent eval->training promotion impossible.

Rules:
  - every track candidate that survives dedup gets a LineageRecord;
  - destinations in PROTECTED_EVAL_DESTINATIONS set protected_evaluation=True
    and FORCE training_eligible=False;
  - promoting a protected example to training raises LeakageError unless an
    explicit governance override (who + why, audited) exists;
  - unvalidated candidates can never become training-eligible at all.
"""

from __future__ import annotations

from typing import Dict, Optional

from sensorflow.raremine.models import (
    PROTECTED_EVAL_DESTINATIONS,
    LineageRecord,
    RareMineStore,
    TrackCandidate,
    new_id,
    now_iso,
)


class LeakageError(RuntimeError):
    """Raised on any attempt to silently move a protected evaluation example
    into training."""


def create_lineage(store: RareMineStore, tc: TrackCandidate,
                   dataset_version: str = "scenebank-v1") -> LineageRecord:
    rec = LineageRecord(
        lineage_id=new_id("lin"),
        track_candidate_id=tc.track_candidate_id,
        source_frame_id=tc.representative.scene_id,
        source_sequence_id=tc.sequence_id,
        dataset_version=dataset_version,
        destination=tc.destination,
    )
    _apply_destination_rules(rec, tc.destination)
    store.put("lineage", rec)
    return rec


def get_lineage(store: RareMineStore, track_candidate_id: str) -> Optional[LineageRecord]:
    recs = store.where("lineage", track_candidate_id=track_candidate_id)
    return recs[0] if recs else None


def _apply_destination_rules(rec: LineageRecord, destination: str) -> None:
    rec.destination = destination
    if destination in PROTECTED_EVAL_DESTINATIONS:
        rec.protected_evaluation = True
        rec.evaluation_eligible = True
        rec.training_eligible = False  # leakage guard: unconditional
    elif destination in ("RARE_EVENT_DATASET", "HARD_EXAMPLE_DATASET"):
        rec.evaluation_eligible = True
    elif destination == "TRAINING_CANDIDATE":
        rec.evaluation_eligible = False


def set_destination(store: RareMineStore, tc: TrackCandidate, destination: str,
                    curator: str = "raremine-pipeline") -> LineageRecord:
    """The ONLY path that changes an example's destination. TRAINING_CANDIDATE
    is refused here unless validation approved it AND it is not protected."""
    rec = get_lineage(store, tc.track_candidate_id)
    if rec is None:
        rec = create_lineage(store, tc)
    if destination == "TRAINING_CANDIDATE":
        if rec.validation_status != "APPROVED":
            raise LeakageError(
                "unverified candidates never route to training: candidate "
                f"{tc.track_candidate_id} has validation_status={rec.validation_status}")
        if rec.protected_evaluation and not rec.governance_overrides:
            raise LeakageError(
                f"candidate {tc.track_candidate_id} belongs to a protected evaluation set "
                "(training_eligible=false); promoting it requires an explicit governance "
                "override recording who and why")
    _apply_destination_rules(rec, destination)
    if destination == "TRAINING_CANDIDATE":
        rec.training_eligible = True  # only reachable via the guarded path above
    rec.curator = curator
    rec.curation_timestamp = now_iso()
    store.put("lineage", rec)
    tc.destination = destination
    store.put("track_candidates", tc)
    store.audit("destination_set", "TrackCandidate", tc.track_candidate_id,
                f"-> {destination}", actor=curator)
    return rec


def governance_override(store: RareMineStore, track_candidate_id: str,
                        actor: str, reason: str) -> LineageRecord:
    """Explicitly recorded exception allowing a protected eval example to be
    considered for training. Requires who + why; fully audited."""
    if not actor.strip() or not reason.strip():
        raise ValueError("governance override requires a named actor and a reason")
    rec = get_lineage(store, track_candidate_id)
    if rec is None:
        raise KeyError(f"no lineage for {track_candidate_id}")
    rec.governance_overrides.append(
        {"actor": actor, "reason": reason, "timestamp": now_iso()})
    store.put("lineage", rec)
    store.audit("governance_override", "TrackCandidate", track_candidate_id,
                f"reason: {reason}", actor=actor)
    return rec


def promote_to_training(store: RareMineStore, track_candidate_id: str,
                        curator: str = "raremine-pipeline") -> LineageRecord:
    tc = store.get("track_candidates", track_candidate_id)
    if tc is None:
        raise KeyError(f"unknown track candidate {track_candidate_id}")
    return set_destination(store, tc, "TRAINING_CANDIDATE", curator=curator)


def lineage_report(store: RareMineStore, bank_id: str) -> Dict:
    tcs = {t.track_candidate_id: t for t in store.where("track_candidates", bank_id=bank_id)}
    rows = []
    for rec in store.all("lineage"):
        if rec.track_candidate_id not in tcs:
            continue
        rows.append(rec.model_dump())
    by_destination: Dict[str, int] = {}
    for r in rows:
        by_destination[r["destination"]] = by_destination.get(r["destination"], 0) + 1
    return {
        "records": sorted(rows, key=lambda r: r["track_candidate_id"]),
        "by_destination": by_destination,
        "protected_count": sum(1 for r in rows if r["protected_evaluation"]),
        "training_eligible_count": sum(1 for r in rows if r["training_eligible"]),
    }
