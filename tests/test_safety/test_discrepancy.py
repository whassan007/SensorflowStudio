"""Auto-label discrepancy mining: injected online misses are found, cohort
rates are consistent, and findings feed the rare-event store + scenario DB."""

from __future__ import annotations

import pytest

from sensorflow.safety import discrepancy
from sensorflow.safety.scenario_db import get_db

AGGRESSIVE = {"extra_miss_rate": 0.5, "class_confusion_rate": 0.2,
              "position_noise_m": 1.2}


def test_mine_finds_injected_online_misses(eval_env, fresh_safety_root):
    store, ds = eval_env
    report = discrepancy.mine(store, ds.dataset_id, profile=AGGRESSIVE, seed=5)

    t = report["totals"]
    assert t["objects"] > 0
    assert t["discrepancies"] > 0
    assert t["discrepancy_rate"] == pytest.approx(
        t["discrepancies"] / t["objects"], abs=1e-3)
    # a 50% extra miss rate must produce MISSED_ONLINE findings
    assert report["by_type"].get("MISSED_ONLINE", 0) > 0
    assert sum(report["by_type"].values()) == t["discrepancies"]
    assert report["simulated"] is True

    # cohort accounting: rates in [0,1], hits never exceed objects
    for c in report["cohorts"]:
        assert 0.0 <= c["discrepancy_rate"] <= 1.0
        assert c["discrepancies"] <= c["objects"]
    assert sum(c["objects"] for c in report["cohorts"]) == t["objects"]

    # every discrepancy is a queryable scenario-mining record
    for d in report["discrepancies"][:10]:
        assert {"discrepancy_id", "frame_id", "gt_id", "type", "class_name",
                "weather", "time_of_day", "severity"} <= set(d)


def test_mine_is_deterministic(eval_env, fresh_safety_root):
    store, ds = eval_env
    r1 = discrepancy.mine(store, ds.dataset_id, profile=AGGRESSIVE, seed=5,
                          persist=False)
    r2 = discrepancy.mine(store, ds.dataset_id, profile=AGGRESSIVE, seed=5,
                          persist=False)
    assert r1["by_type"] == r2["by_type"]
    assert [d["gt_id"] for d in r1["discrepancies"]] == \
           [d["gt_id"] for d in r2["discrepancies"]]


def test_mine_feeds_rare_events_and_scenario_db(eval_env, fresh_safety_root):
    store, ds = eval_env
    before_events = len(store.all("rare_events"))
    report = discrepancy.mine(store, ds.dataset_id, profile=AGGRESSIVE, seed=5)

    critical = [d for d in report["discrepancies"] if d["severity"] == "critical"]
    if critical:  # pedestrian/cyclist online misses
        mined_events = [e for e in store.all("rare_events")
                        if e.scenario_type == "online_perception_miss"]
        assert len(store.all("rare_events")) > before_events
        assert mined_events and all(e.severity == "critical" for e in mined_events)

    recs = get_db().search(source="discrepancy", limit=1000)
    high = [d for d in report["discrepancies"] if d["severity"] in ("high", "critical")]
    assert len(recs) == len({(d["gt_id"], d["type"]) for d in high})


def test_latest_summary_roundtrip(eval_env, fresh_safety_root):
    store, ds = eval_env
    assert discrepancy.latest_summary(ds.dataset_id) is None
    discrepancy.mine(store, ds.dataset_id, profile=AGGRESSIVE, seed=5)
    summary = discrepancy.latest_summary(ds.dataset_id)
    assert summary is not None
    assert summary["dataset_id"] == ds.dataset_id
    assert "discrepancies" not in summary  # slimmed
    assert len(summary["sample_discrepancies"]) <= 25


def test_mine_unknown_dataset_raises(eval_env, fresh_safety_root):
    store, _ = eval_env
    with pytest.raises(KeyError):
        discrepancy.mine(store, "ds-nonexistent")
