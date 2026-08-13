"""Near-duplicate detection + diversity-aware selection.

Dedup: track candidates describing the same physical event (same signature of
costume family x context x lighting x weather x quantized location, across
frames and across drives) collapse into one representative; the rest are
archived as duplicates.

Diversity selection: within a budget, greedily maximize coverage of the
failure surface (costume x lighting x weather x occlusion x distance x
context cells) instead of taking the raw top-k by priority. Selection
optimizes failure-surface coverage, not candidate count.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sensorflow.raremine.models import RareMineStore, TrackCandidate

PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
QUALITY_ORDER = ["LOW", "MEDIUM", "HIGH"]

COVERAGE_DIMS = ["costume", "lighting", "weather", "occlusion", "distance_bin", "context"]


def _distance_bin(d: float) -> str:
    return "near" if d < 15 else "mid" if d < 35 else "far"


def _signature(tc: TrackCandidate) -> Tuple:
    rep = tc.representative
    costume = rep.costume_type[0] if rep.costume_type else "unknown"
    loc = rep.location
    return (
        costume,
        loc.get("context"),
        loc.get("lighting"),
        loc.get("weather"),
        _distance_bin(float(loc.get("distance_m", 0.0))),
        rep.silhouette_deviation,
    )


def _cells(tc: TrackCandidate) -> Dict[str, str]:
    rep = tc.representative
    loc = rep.location
    return {
        "costume": rep.costume_type[0] if rep.costume_type else "unknown",
        "lighting": str(loc.get("lighting")),
        "weather": str(loc.get("weather")),
        "occlusion": rep.occlusion_level,
        "distance_bin": _distance_bin(float(loc.get("distance_m", 0.0))),
        "context": str(loc.get("context")),
    }


def _pair_cells(tc: TrackCandidate) -> set:
    """Coverage cells: all (dim_a x dim_b) pairs the candidate occupies."""
    cells = _cells(tc)
    out = set()
    dims = COVERAGE_DIMS
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            out.add((dims[i], cells[dims[i]], dims[j], cells[dims[j]]))
    return out


def _quality_key(tc: TrackCandidate) -> Tuple:
    rep = tc.representative
    return (PRIORITY_ORDER.index(rep.curation_priority),
            QUALITY_ORDER.index(rep.evidence_quality),
            rep.confidence_rare_event,
            tc.frame_count)


def deduplicate(store: RareMineStore, bank_id: str, run_id: str = "") -> Dict:
    """Collapse same-event track candidates; mark the losers as duplicates."""
    tracks = [t for t in store.where("track_candidates", bank_id=bank_id)
              if (not run_id or t.run_id == run_id) and t.representative.edge_case_detected]
    groups: Dict[Tuple, List[TrackCandidate]] = {}
    for t in tracks:
        groups.setdefault(_signature(t), []).append(t)

    kept, duplicates = [], []
    for sig, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        members.sort(key=_quality_key, reverse=True)
        keeper = members[0]
        keeper.stage = "DEDUPLICATED"
        store.put("track_candidates", keeper)
        kept.append(keeper)
        for dupe in members[1:]:
            dupe.stage = "ARCHIVED"
            dupe.duplicate_of = keeper.track_candidate_id
            store.put("track_candidates", dupe)
            duplicates.append(dupe)

    store.audit("dedup", "MiningRun", run_id or bank_id,
                f"{len(kept)} kept, {len(duplicates)} duplicates archived")
    return {
        "groups": len(groups),
        "kept": len(kept),
        "duplicates_archived": len(duplicates),
        "dedup_savings": len(duplicates),
        "signatures": [
            {"signature": {"costume": s[0], "context": s[1], "lighting": s[2],
                           "weather": s[3], "distance_bin": s[4], "silhouette": s[5]},
             "members": len(m), "kept": m[0].track_candidate_id}
            for s, m in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        ],
    }


def coverage_score(selected: List[TrackCandidate], universe: List[TrackCandidate]) -> float:
    """Fraction of occupied failure-surface cells covered by the selection."""
    all_cells = set()
    for t in universe:
        all_cells |= _pair_cells(t)
    if not all_cells:
        return 0.0
    got = set()
    for t in selected:
        got |= _pair_cells(t)
    return round(len(got & all_cells) / len(all_cells), 4)


def select_diverse(store: RareMineStore, bank_id: str, budget: int = 12,
                   run_id: str = "") -> Dict:
    """Greedy max-coverage selection over deduplicated candidates, with the
    naive top-k-by-priority baseline for comparison."""
    pool = [t for t in store.where("track_candidates", bank_id=bank_id)
            if (not run_id or t.run_id == run_id) and t.stage == "DEDUPLICATED"]
    pool.sort(key=_quality_key, reverse=True)
    budget = min(budget, len(pool))

    naive = pool[:budget]

    selected: List[TrackCandidate] = []
    covered: set = set()
    remaining = list(pool)
    while len(selected) < budget and remaining:
        best: Optional[TrackCandidate] = None
        best_gain = (-1, ())
        for t in remaining:
            gain = len(_pair_cells(t) - covered)
            key = (gain, _quality_key(t))
            if key > best_gain:
                best_gain = key
                best = t
        assert best is not None
        selected.append(best)
        covered |= _pair_cells(best)
        remaining.remove(best)

    for t in pool:
        t.diversity_selected = t in selected
        store.put("track_candidates", t)

    matrix: Dict[str, Dict[str, int]] = {}
    for t in selected:
        cells = _cells(t)
        row = matrix.setdefault(cells["costume"], {})
        for dim in ("lighting", "weather", "occlusion", "context"):
            key = f"{dim}:{cells[dim]}"
            row[key] = row.get(key, 0) + 1

    return {
        "budget": budget,
        "pool_size": len(pool),
        "selected_ids": [t.track_candidate_id for t in selected],
        "coverage_selected": coverage_score(selected, pool),
        "coverage_naive_topk": coverage_score(naive, pool),
        "coverage_matrix": matrix,
        "note": "selection maximizes failure-surface coverage "
                "(costume x lighting x weather x occlusion x distance x context), "
                "not raw candidate count",
    }
