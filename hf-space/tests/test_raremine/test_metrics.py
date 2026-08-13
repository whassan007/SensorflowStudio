"""Curator statistics measured against planted truth + priority sanity +
the continuous-learning improvement loop."""

from __future__ import annotations

from sensorflow.raremine import curator_metrics, pipeline, quantval

PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _truth_kind(store, bank, tc):
    for scene in store.where("scenes", sequence_id=tc.sequence_id):
        for obj in scene.objects:
            if obj.track_id == tc.track_id:
                return obj.truth_kind
    return None


def test_precision_recall_against_planted_truth(store, bank):
    report = curator_metrics.curator_report(store, bank.bank_id)
    conf = report["confusion"]

    # independent recomputation from the planted truth
    tp = fp = 0
    tcs = [t for t in store.where("track_candidates", bank_id=bank.bank_id)
           if t.duplicate_of is None]
    for tc in tcs:
        if not tc.representative.edge_case_detected:
            continue
        if _truth_kind(store, bank, tc) == "costumed_pedestrian":
            tp += 1
        else:
            fp += 1
    assert conf["tp"] == tp and conf["fp"] == fp
    assert conf["tp"] + conf["fn"] == report["planted_positives"]
    if tp + fp:
        assert report["mining_precision"] == round(tp / (tp + fp), 4)
        assert report["false_discovery_rate"] == round(fp / (tp + fp), 4)
    assert report["mining_recall"] == round(tp / (tp + conf["fn"]), 4)
    # this bank is mined conservatively: no confounder should sneak through
    assert report["mining_precision"] >= 0.8
    assert report["mining_recall"] >= 0.3


def test_calibration_bins_are_reliability_measurements(store, bank):
    report = curator_metrics.curator_report(store, bank.bank_id)
    cal = report["calibration"]
    assert sum(row["n"] for row in cal) == len(
        [t for t in store.where("track_candidates", bank_id=bank.bank_id)
         if t.duplicate_of is None])
    for row in cal:
        if row["n"]:
            assert row["observed_rate"] == round(row["true"] / row["n"], 3)
    # high-stated-confidence bins must not be less reliable than the lowest bin
    filled = [r for r in cal if r["n"] >= 2]
    if len(filled) >= 2:
        assert filled[-1]["observed_rate"] >= filled[0]["observed_rate"]


def test_priority_ordering_sanity(store, bank):
    """Observed-failure + EXTREME deviation + safety context must outrank the
    easy fully-visible day case."""
    queue = [t for t in pipeline.review_queue(store, bank.bank_id)
             if t.representative.edge_case_detected]
    crit = [t for t in queue
            if t.representative.silhouette_deviation == "EXTREME"
            and t.representative.location.get("context") in ("crosswalk", "road_edge")
            and t.representative.observed_model_behavior is not None
            and t.representative.observed_model_behavior.failure_observed]
    assert crit, "bank plants an EXTREME crosswalk failure exemplar"
    easy = [t for t in queue
            if t.representative.location.get("context") == "sidewalk"
            and t.representative.location.get("lighting") == "day"
            and (t.representative.observed_model_behavior is None
                 or not t.representative.observed_model_behavior.failure_observed)]
    assert easy, "bank plants an easy visible day case"
    for c in crit:
        assert c.representative.curation_priority == "CRITICAL"
        for e in easy:
            assert PRIORITY_ORDER.index(c.representative.curation_priority) > \
                PRIORITY_ORDER.index(e.representative.curation_priority)
            assert queue.index(c) < queue.index(e)


def test_quantval_agreement_matrix(store, bank):
    report = quantval.quantitative_report(store, bank.bank_id)
    assert report["candidates_evaluated"] > 0
    assert report["with_model_outputs"] > 0
    total = sum(v for row in report["agreement_matrix"].values() for v in row.values())
    assert total == report["with_model_outputs"]
    # observed metrics exist only where the track actually has model outputs
    for row in report["rows"]:
        if row["observed"] is not None:
            assert row["observed"]["frames_with_predictions"] > 0
            assert row["observed"]["observed_difficulty"] in ("EASY", "MODERATE", "HARD", "EXTREME")


def test_improvement_loop_feeds_next_run(tmp_path):
    """Recurring-miss analysis produces a config that raises recall on rerun."""
    from sensorflow.raremine import scenes
    from sensorflow.raremine.models import reset_store
    store = reset_store(tmp_path)
    try:
        bank = scenes.generate_scene_bank(store, n_scenes=60, seed=7)
        pipeline.run_full_pipeline(store, bank.bank_id)
        before = curator_metrics.curator_report(store, bank.bank_id)
        imp = curator_metrics.improvement_report(store, bank.bank_id)
        assert imp["recurring_misses"]["total"] == before["confusion"]["fn"]
        assert imp["next_run_config"]["sensitivity_boost"], "misses must produce boosts"
        pipeline.run_full_pipeline(store, bank.bank_id, config=imp["next_run_config"])
        after = curator_metrics.curator_report(store, bank.bank_id)
        assert after["mining_recall"] >= before["mining_recall"]
        assert after["confusion"]["fp"] <= before["confusion"]["fp"] + 1
    finally:
        reset_store()
