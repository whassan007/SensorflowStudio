"""Track consolidation, near-duplicate collapse, and diversity selection."""

from __future__ import annotations

from sensorflow.raremine.dedup import coverage_score
from sensorflow.raremine.models import RareMineStore


def _tracks(store: RareMineStore, bank):
    return [t for t in store.where("track_candidates", bank_id=bank.bank_id)]


def test_long_track_consolidates_to_one_candidate(store, bank):
    """A 20-frame costumed track becomes ONE track candidate covering all frames."""
    # locate the 20-frame sequence via its scenes
    seq_frames = {}
    for s in store.where("scenes", bank_id=bank.bank_id):
        if s.near_duplicate_of is None:
            seq_frames[s.sequence_id] = max(seq_frames.get(s.sequence_id, 0), s.frame_index + 1)
    long_seqs = [sid for sid, n in seq_frames.items() if n == 20]
    assert long_seqs, "bank must contain a 20-frame sequence"
    sid = long_seqs[0]
    costumed = [t for t in _tracks(store, bank)
                if t.sequence_id == sid and t.representative.edge_case_detected]
    assert len(costumed) == 1, "one physical event must yield exactly one track candidate"
    tc = costumed[0]
    assert tc.frame_count == 20
    assert tc.duration_frames == 20
    assert len(tc.frame_candidate_ids) == 20


def test_representative_frames_are_correct(store, bank):
    detected = [t for t in _tracks(store, bank)
                if t.representative.edge_case_detected and t.frame_count > 1]
    assert detected
    for tc in detected:
        frames = [store.get("candidates", cid) for cid in tc.frame_candidate_ids]
        scene_ids = {f.scene_id for f in frames}
        rf = tc.representative_frames
        assert rf.best_evidence in scene_ids
        assert rf.worst_case in scene_ids
        assert rf.minimal_set and set(rf.minimal_set) <= scene_ids
        assert len(rf.minimal_set) <= 3, "minimal set must stay minimal"
        if rf.model_failure is not None:
            assert rf.model_failure in scene_ids
            failing = next(f for f in frames if f.scene_id == rf.model_failure)
            assert failing.observed_model_behavior is not None
            assert failing.observed_model_behavior.failure_observed


def test_near_duplicates_are_archived(store, bank):
    """Repeated drives of the same event collapse to one kept candidate."""
    report = store.meta["last_run"]["dedup_report"]
    assert report["duplicates_archived"] >= 2
    dupes = [t for t in _tracks(store, bank) if t.duplicate_of is not None]
    assert len(dupes) == report["duplicates_archived"]
    for d in dupes:
        assert d.stage == "ARCHIVED"
        keeper = store.get("track_candidates", d.duplicate_of)
        assert keeper is not None
        assert keeper.duplicate_of is None


def test_diversity_selection_beats_naive_topk_on_coverage(store, bank):
    report = store.meta["last_run"]["diversity_report"]
    assert report["coverage_selected"] >= report["coverage_naive_topk"], (
        "diversity-aware selection must never cover less than naive top-k")
    assert report["coverage_selected"] > 0
    assert len(report["selected_ids"]) == report["budget"]
    # a strictly better selection exists in this bank (dupe drives concentrate
    # the naive top-k in one region of the failure surface)
    pool = [t for t in _tracks(store, bank) if t.stage in ("DEDUPLICATED", "AUTO_VALIDATED")
            and t.duplicate_of is None]
    if report["coverage_naive_topk"] < 1.0:
        assert report["coverage_selected"] > report["coverage_naive_topk"] or \
            report["coverage_selected"] == 1.0


def test_coverage_score_bounds(store, bank):
    pool = [t for t in _tracks(store, bank) if t.duplicate_of is None
            and t.representative.edge_case_detected]
    assert coverage_score(pool, pool) == 1.0
    assert coverage_score([], pool) == 0.0
