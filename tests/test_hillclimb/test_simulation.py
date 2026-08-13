"""Simulation: determinism, safety-floor incidents, keep/reject, debrief mapping."""

from sensorflow.hillclimb.simulation import (create_simulation, get_simulation,
                                             step_simulation)

STRONG_HYP = "Investing in monitoring will improve safety by 5 points within two turns"


def _run(store, seed, moves):
    sim = create_simulation(seed=seed, max_turns=len(moves), store=store)
    for hyp, iid in moves:
        sim = step_simulation(sim.sim_id, hyp, iid, store=store)
    return sim


def test_deterministic_given_seed(isolated_store):
    moves = [(STRONG_HYP, "add_monitoring"),
             ("Pausing launches will raise safety", "pause_launches"),
             ("1:1s will recover team morale by 5", "team_1on1s")]
    a = _run(isolated_store, 42, moves)
    b = _run(isolated_store, 42, moves)
    assert a.objective_history == b.objective_history
    assert a.metrics == b.metrics
    c = _run(isolated_store, 43, moves)
    assert c.objective_history != a.objective_history


def test_safety_floor_triggers_incident_event(isolated_store):
    sim = create_simulation(seed=1, max_turns=6, store=isolated_store)
    for _ in range(5):
        sim = step_simulation(sim.sim_id, "Shipping fast will improve schedule",
                              "ship_feature_fast", store=isolated_store)
        if any(e["type"] == "SAFETY_INCIDENT" for e in sim.events):
            break
    incidents = [e for e in sim.events if e["type"] == "SAFETY_INCIDENT"]
    assert incidents, "repeated corner-cutting must breach the safety floor"
    assert "regression escaped" in incidents[0]["detail"]


def test_keep_reject_reverts_previous_effects(isolated_store):
    sim = create_simulation(seed=9, max_turns=4, store=isolated_store)
    sim = step_simulation(sim.sim_id, "Shipping fast raises schedule", "ship_feature_fast",
                          store=isolated_store)
    applied = sim.history[0].applied_effects
    sim = step_simulation(sim.sim_id, "Reverting; monitoring will improve safety",
                          "add_monitoring", revert_previous=True, store=isolated_store)
    assert sim.history[1].reverted_previous
    assert any("Rejected previous intervention" in e for e in sim.history[1].events)
    assert applied  # effects existed and were rolled back


def test_delayed_second_order_effects_land_later(isolated_store):
    sim = create_simulation(seed=3, max_turns=4, store=isolated_store)
    sim = step_simulation(sim.sim_id, "Test infra will raise reliability", "invest_test_infra",
                          store=isolated_store)
    assert sim.pending, "delayed effects must be queued"
    sim = step_simulation(sim.sim_id, "Waiting for infra to land", "team_1on1s", store=isolated_store)
    sim = step_simulation(sim.sim_id, "Infra lands this turn", "team_1on1s", store=isolated_store)
    landed = [n for t in sim.history for n in t.delayed_landed]
    assert any("Test infra lands" in n for n in landed)


def test_debrief_maps_decisions_to_competency_evidence(isolated_store):
    moves = [(STRONG_HYP, "add_monitoring"),
             ("Pausing launches will raise safety above 55", "pause_launches"),
             ("Test infra will improve reliability by 7", "invest_test_infra"),
             ("Expectation reset will improve schedule by 6", "exec_expectation_reset")]
    sim = _run(isolated_store, 42, moves)
    assert sim.status == "complete"
    assert sim.debrief is not None
    mapped = {m["competency_id"]: m for m in sim.debrief["competency_mappings"]}
    assert "p4.hill_climbing" in mapped
    assert mapped["p4.hill_climbing"]["verdict"] == "evidenced"  # falsifiable hypotheses
    assert "p2.reliability_tradeoffs" in mapped
    # evidenced mappings persist Evidence artifacts
    ev = isolated_store.where("evidence", artifact_type="simulation_debrief")
    assert ev, "debrief must store evidence"
    # state is retrievable
    assert get_simulation(sim.sim_id, isolated_store).status == "complete"


def test_vague_hypotheses_marked_as_gap(isolated_store):
    moves = [("do stuff", "add_monitoring"), ("things", "team_1on1s"),
             ("whatever", "cut_scope"), ("meh", "exec_expectation_reset")]
    sim = _run(isolated_store, 5, moves)
    mapped = {m["competency_id"]: m for m in sim.debrief["competency_mappings"]}
    assert mapped["p4.hill_climbing"]["verdict"] == "gap"
