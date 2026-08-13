"""Track-level consolidation: N frame-level proposals of the same physical
event become ONE track candidate with representative frames.

Representative frames:
  - best_evidence: frame with the strongest evidence quality / rare confidence
  - worst_case: hardest frame (max difficulty, min visibility)
  - model_failure: a frame where a baseline failure was actually observed
  - minimal_set: the deduplicated union of the above (the recommended frames
    to curate — never the whole track)
"""

from __future__ import annotations

from typing import Dict, List

from sensorflow.raremine.models import (
    Candidate,
    RareMineStore,
    RepresentativeFrames,
    TrackCandidate,
    new_id,
)

DIFF_ORDER = ["EASY", "MODERATE", "HARD", "EXTREME"]
QUALITY_ORDER = ["LOW", "MEDIUM", "HIGH"]
PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _evidence_score(c: Candidate) -> float:
    return QUALITY_ORDER.index(c.evidence_quality) + c.confidence_rare_event


def _hardness_score(c: Candidate) -> float:
    occ = ["NONE", "PARTIAL", "HEAVY", "EXTREME"].index(c.occlusion_level)
    return DIFF_ORDER.index(c.perception_difficulty) * 2 + occ


def consolidate_tracks(store: RareMineStore, bank_id: str, run_id: str = "") -> List[TrackCandidate]:
    """Group frame candidates by (sequence, track) into track candidates."""
    frame_cands = [c for c in store.where("candidates", bank_id=bank_id)
                   if not run_id or c.run_id == run_id]
    groups: Dict[str, List[Candidate]] = {}
    for c in frame_cands:
        groups.setdefault(f"{c.sequence_id}:{c.track_id}", []).append(c)

    out: List[TrackCandidate] = []
    for key, cands in sorted(groups.items()):
        cands.sort(key=lambda c: c.frame_index)
        detected = [c for c in cands if c.edge_case_detected]
        # a track is a candidate if ANY frame proposed it; representative is the
        # best-evidence detected frame (or best overall for rejected tracks)
        pool = detected or cands
        best = max(pool, key=_evidence_score)
        worst = max(pool, key=_hardness_score)
        failure_frames = [c for c in pool
                          if c.observed_model_behavior and c.observed_model_behavior.failure_observed]
        failure = max(failure_frames, key=_evidence_score) if failure_frames else None

        minimal: List[str] = []
        for c in [best, worst, failure]:
            if c is not None and c.scene_id not in minimal:
                minimal.append(c.scene_id)

        max_diff = max(pool, key=lambda c: DIFF_ORDER.index(c.perception_difficulty))
        max_vis = max(pool, key=lambda c: QUALITY_ORDER.index(c.evidence_quality))

        # the representative carries the track-level judgment: max priority
        # across frames, an observed failure from ANY frame
        rep = best.model_copy(deep=True)
        rep.curation_priority = max(
            (c.curation_priority for c in pool), key=PRIORITY_ORDER.index)
        if failure is not None and (rep.observed_model_behavior is None
                                    or not rep.observed_model_behavior.failure_observed):
            rep.observed_model_behavior = failure.observed_model_behavior
            rep.recommended_dataset_destination = failure.recommended_dataset_destination
            rep.priority_reason = failure.priority_reason

        tc = TrackCandidate(
            track_candidate_id=new_id("tcand"),
            bank_id=bank_id,
            run_id=run_id,
            sequence_id=cands[0].sequence_id,
            track_id=cands[0].track_id,
            object_id=best.object_id,
            frame_count=len(cands),
            duration_frames=cands[-1].frame_index - cands[0].frame_index + 1,
            frame_candidate_ids=[c.candidate_id for c in cands],
            representative=rep,
            representative_frames=RepresentativeFrames(
                best_evidence=best.scene_id,
                worst_case=worst.scene_id,
                model_failure=failure.scene_id if failure else None,
                minimal_set=minimal,
            ),
            max_visibility=max_vis.evidence_quality,
            max_difficulty=max_diff.perception_difficulty,
            stage="MINED",
            destination=rep.recommended_dataset_destination,
        )
        store.put("track_candidates", tc)
        out.append(tc)
    return out
