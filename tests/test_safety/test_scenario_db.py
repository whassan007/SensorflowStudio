"""Scenario database: populate (idempotently) from rare events, search/filter,
export bundle."""

from __future__ import annotations

from sensorflow.evaluation.records import RareEvent
from sensorflow.safety.scenario_db import ScenarioRecord, get_db


def _make_rare_event(store, ds, frame, severity="high"):
    ev = RareEvent(
        event_id=f"evt-test-{severity}",
        dataset_id=ds.dataset_id,
        scenario_type="near_collision",
        severity=severity,
        rarity_score=0.9,
        confidence=0.8,
        evidence_frames=[frame.frame_id],
        description="test rare event",
    )
    store.put("rare_events", ev)
    return ev


def test_populate_from_rare_events_idempotent(eval_env, fresh_safety_root):
    store, ds = eval_env
    frame = store.where("frames", dataset_id=ds.dataset_id)[0]
    _make_rare_event(store, ds, frame)

    db = get_db()
    assert db.add_from_rare_events(store) == 1
    assert db.add_from_rare_events(store) == 0  # idempotent

    recs = db.search(source="rare_event")
    assert len(recs) == 1
    r = recs[0]
    assert r.scenario_type == "near_collision"
    assert r.odd_tags.get("weather") == frame.weather
    assert f"rare_event:{'evt-test-high'}" in r.evidence_refs


def test_search_filters_and_ordering(fresh_safety_root):
    db = get_db()
    db.add(ScenarioRecord(scenario_id="s1", scenario_type="cut_in", source="mined",
                          severity="low", odd_tags={"weather": "rain"},
                          description="cut-in in rain"))
    db.add(ScenarioRecord(scenario_id="s2", scenario_type="cut_in", source="synthetic",
                          severity="critical", odd_tags={"weather": "clear"},
                          description="synthetic cut-in"))
    db.add(ScenarioRecord(scenario_id="s3", scenario_type="jaywalk", source="mined",
                          severity="high", odd_tags={"weather": "rain"},
                          description="pedestrian jaywalking at night"))

    assert {r.scenario_id for r in db.search(scenario_type="cut_in")} == {"s1", "s2"}
    assert {r.scenario_id for r in db.search(source="mined")} == {"s1", "s3"}
    assert {r.scenario_id for r in db.search(severity="critical")} == {"s2"}
    assert {r.scenario_id for r in db.search(odd_tags={"weather": "rain"})} == {"s1", "s3"}
    assert {r.scenario_id for r in db.search(text="jaywalk")} == {"s3"}
    # ordering: highest severity first
    assert [r.scenario_id for r in db.search()][:2] == ["s2", "s3"]

    counts = db.counts()
    assert counts["total"] == 3
    assert counts["by_source"] == {"mined": 2, "synthetic": 1}


def test_persistence_across_reload(fresh_safety_root):
    db = get_db()
    db.add(ScenarioRecord(scenario_id="sp1", scenario_type="cut_in", source="mined",
                          severity="medium"))
    from sensorflow.safety import scenario_db as mod
    mod.reset_db()  # force reload from disk
    assert [r.scenario_id for r in get_db().search()] == ["sp1"]


def test_export_bundle(fresh_safety_root):
    db = get_db()
    db.add(ScenarioRecord(scenario_id="e1", scenario_type="cut_in", source="mined",
                          severity="high"))
    db.add(ScenarioRecord(scenario_id="e2", scenario_type="jaywalk",
                          source="discrepancy", severity="low"))
    bundle = db.export_bundle(source="mined")
    assert bundle["bundle_format"] == "sensorflow-scenario-bundle/v1"
    assert bundle["count"] == 1
    assert bundle["scenarios"][0]["scenario_id"] == "e1"
    full = db.export_bundle()
    assert full["count"] == 2
