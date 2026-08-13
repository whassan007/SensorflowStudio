"""Stateful hill-climbing simulation: multi-objective EM decision exercise.

State vector: system_performance, safety, reliability, cost, velocity,
maintainability, team_morale, customer_impact, risk, schedule (0-100; cost and
risk are "higher is worse"). A seeded scenario applies per-turn pressure; each
turn the user writes a hypothesis, picks an intervention (with probabilistic-
but-seeded effects, delays and second-order consequences), measures the result,
and keeps or rejects the change. The objective is a balanced multi-objective
score with HARD FLOORS (safety below threshold triggers an incident event).
Fully deterministic given the seed and the same choice sequence.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from sensorflow.hillclimb.models import Evidence, Store, get_store, new_id, now_iso

METRICS = ["system_performance", "safety", "reliability", "cost", "velocity",
           "maintainability", "team_morale", "customer_impact", "risk", "schedule"]
INVERTED = {"cost", "risk"}  # higher is worse

OBJECTIVE_WEIGHTS = {
    "system_performance": 0.14, "safety": 0.20, "reliability": 0.10, "cost": 0.06,
    "velocity": 0.10, "maintainability": 0.08, "team_morale": 0.12,
    "customer_impact": 0.10, "risk": 0.05, "schedule": 0.05,
}

SAFETY_FLOOR = 30.0
MORALE_FLOOR = 25.0


class InterventionSpec(BaseModel):
    id: str
    label: str
    description: str
    immediate: Dict[str, float]  # mean effect; jitter applied per-turn
    delayed: List[Dict] = Field(default_factory=list)  # {delay: int, effects: {metric: delta}, note}
    second_order: str = ""


INTERVENTIONS: List[InterventionSpec] = [
    InterventionSpec(
        id="add_regression_gate", label="Add a release regression gate",
        description="Block releases whose eval deltas regress safety-critical slices.",
        immediate={"safety": 7, "reliability": 4, "velocity": -6, "schedule": -4},
        delayed=[{"delay": 2, "effects": {"system_performance": 5, "customer_impact": 3},
                  "note": "Gate catches a bad build before it ships"}],
        second_order="Slower releases short-term; fewer production regressions later."),
    InterventionSpec(
        id="pause_launches", label="Pause launches for stabilization",
        description="Freeze feature launches for a sprint; all hands on regressions.",
        immediate={"safety": 10, "reliability": 6, "schedule": -10, "customer_impact": -6, "risk": -8},
        delayed=[{"delay": 1, "effects": {"team_morale": -4},
                  "note": "Exec pressure builds during the freeze"}],
        second_order="Buys stability at the price of roadmap credibility."),
    InterventionSpec(
        id="hire_senior", label="Hire a senior ML-infra engineer",
        description="Open and close a senior req to deepen the bench.",
        immediate={"cost": 8, "velocity": -2},
        delayed=[{"delay": 3, "effects": {"velocity": 8, "system_performance": 5, "team_morale": 4},
                  "note": "New senior hire reaches productivity"}],
        second_order="Expensive and slow to pay off, then compounding."),
    InterventionSpec(
        id="invest_test_infra", label="Invest in test & eval infrastructure",
        description="Two engineers build slice-level eval + CI hardening for a sprint.",
        immediate={"velocity": -5, "schedule": -3, "cost": 3},
        delayed=[{"delay": 2, "effects": {"reliability": 7, "maintainability": 8, "velocity": 6, "safety": 4},
                  "note": "Test infra lands: regressions caught pre-merge"}],
        second_order="Classic J-curve: worse before durably better."),
    InterventionSpec(
        id="ship_feature_fast", label="Ship the exec-requested feature now",
        description="Cut the corner: ship the feature leadership is pressing for.",
        immediate={"schedule": 10, "customer_impact": 7, "velocity": 3,
                   "safety": -7, "maintainability": -6, "risk": 10},
        delayed=[{"delay": 2, "effects": {"reliability": -6, "team_morale": -3},
                  "note": "Corner-cutting surfaces as flaky behavior"}],
        second_order="Relief now; debt and risk arrive on a delay."),
    InterventionSpec(
        id="team_1on1s", label="Deep-dive 1:1s across the team",
        description="Spend the week understanding people: load, friction, morale.",
        immediate={"team_morale": 7, "velocity": -2},
        delayed=[{"delay": 1, "effects": {"velocity": 4, "risk": -3},
                  "note": "Friction removed after 1:1 follow-ups"}],
        second_order="Costs a week; surfaces problems before they explode."),
    InterventionSpec(
        id="exec_expectation_reset", label="Reset expectations with leadership",
        description="Present the regression data and renegotiate the ship date.",
        immediate={"schedule": 6, "team_morale": 5, "risk": -4, "customer_impact": -3},
        delayed=[],
        second_order="Spends political capital to buy engineering room."),
    InterventionSpec(
        id="incident_review_process", label="Institute blameless incident reviews",
        description="Every escaped regression gets a review with tracked actions.",
        immediate={"velocity": -3, "maintainability": 3},
        delayed=[{"delay": 2, "effects": {"safety": 6, "reliability": 5},
                  "note": "Incident-review actions close systemic holes"}],
        second_order="Culture change: slow, then decisive."),
    InterventionSpec(
        id="add_monitoring", label="Add production model monitoring",
        description="Slice-level drift + regression monitoring on live traffic.",
        immediate={"reliability": 5, "cost": 4, "velocity": -2},
        delayed=[{"delay": 1, "effects": {"safety": 6, "system_performance": 4},
                  "note": "Monitoring catches a drift you didn't know about"}],
        second_order="Visibility first; fixes follow the visibility."),
    InterventionSpec(
        id="refactor_pipeline", label="Refactor the training/eval pipeline",
        description="Pay down the pipeline debt that makes every change slow.",
        immediate={"velocity": -7, "schedule": -5, "cost": 4},
        delayed=[{"delay": 3, "effects": {"maintainability": 10, "velocity": 9, "system_performance": 4},
                  "note": "Refactor lands: iteration speed jumps"}],
        second_order="The biggest J-curve on the menu."),
    InterventionSpec(
        id="cut_scope", label="Cut scope to protect the date",
        description="Descope the release to the safety-critical core.",
        immediate={"schedule": 8, "risk": -5, "customer_impact": -5, "velocity": 2},
        delayed=[],
        second_order="Trades breadth for certainty."),
]

INTERVENTION_INDEX = {i.id: i for i in INTERVENTIONS}


class Scenario(BaseModel):
    scenario_id: str
    title: str
    narrative: str
    initial: Dict[str, float]
    drift: Dict[str, float]  # per-turn pressure applied before interventions


SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id="inherited_perception_team",
        title="Inherited Perception Team Under Pressure",
        narrative=("You just inherited a 14-engineer perception team. The regression rate on "
                   "safety-critical slices has risen for three consecutive releases, on-call is "
                   "burning people out, and an exec has publicly committed your team to shipping "
                   "a major feature in 8 weeks. Every objective is in tension: climb carefully."),
        initial={"system_performance": 55, "safety": 48, "reliability": 50, "cost": 45,
                 "velocity": 52, "maintainability": 40, "team_morale": 44,
                 "customer_impact": 55, "risk": 55, "schedule": 50},
        drift={"safety": -2.0, "team_morale": -1.5, "maintainability": -1.0,
               "risk": 1.5, "schedule": -1.0},
    ),
]


class TurnRecord(BaseModel):
    turn: int
    hypothesis: str
    hypothesis_assessment: Dict
    intervention_id: str
    reverted_previous: bool = False
    applied_effects: Dict[str, float] = Field(default_factory=dict)
    delayed_landed: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    metrics_after: Dict[str, float] = Field(default_factory=dict)
    objective_after: float = 0.0
    objective_delta: float = 0.0
    verdict: str = ""  # keep_recommended | reject_recommended
    timestamp: str = Field(default_factory=now_iso)


class SimulationState(BaseModel):
    sim_id: str = Field(default_factory=lambda: new_id("sim"))
    user_id: str = "default"
    scenario_id: str = "inherited_perception_team"
    seed: int = 42
    max_turns: int = 8
    turn: int = 0
    status: str = "active"  # active | complete
    metrics: Dict[str, float] = Field(default_factory=dict)
    pending: List[Dict] = Field(default_factory=list)  # {due_turn, effects, note}
    history: List[TurnRecord] = Field(default_factory=list)
    events: List[Dict] = Field(default_factory=list)
    objective_history: List[float] = Field(default_factory=list)
    debrief: Optional[Dict] = None


def objective_score(metrics: Dict[str, float]) -> float:
    total = 0.0
    for m, w in OBJECTIVE_WEIGHTS.items():
        v = metrics.get(m, 50.0)
        total += w * ((100.0 - v) if m in INVERTED else v)
    return round(total, 2)


def create_simulation(user_id: str = "default", scenario_id: str = "inherited_perception_team",
                      seed: int = 42, max_turns: int = 8,
                      store: Optional[Store] = None) -> SimulationState:
    store = store or get_store()
    scenario = next((s for s in SCENARIOS if s.scenario_id == scenario_id), SCENARIOS[0])
    sim = SimulationState(user_id=user_id, scenario_id=scenario.scenario_id, seed=seed,
                          max_turns=max_turns, metrics=dict(scenario.initial))
    sim.objective_history.append(objective_score(sim.metrics))
    store.put("simulations", sim.sim_id, sim)
    return sim


def _assess_hypothesis(hypothesis: str, spec: InterventionSpec) -> Dict:
    """Rule-based check: is the hypothesis falsifiable and targeted?"""
    low = hypothesis.lower()
    touched = set(spec.immediate) | {m for d in spec.delayed for m in d["effects"]}
    mentions_metric = [m for m in METRICS if m.replace("_", " ") in low or m in low]
    targeted = any(m in touched for m in mentions_metric)
    directional = bool(any(w in low for w in ["increase", "decrease", "improve", "reduce", "raise",
                                              "lower", "recover", "drop", "+", "-"]))
    falsifiable = directional and (bool(mentions_metric) or any(ch.isdigit() for ch in low))
    quality = "strong" if (targeted and falsifiable) else ("directional" if directional else "vague")
    return {"mentions_metrics": mentions_metric, "targets_intervention_effects": targeted,
            "directional": directional, "falsifiable": falsifiable, "quality": quality}


def step_simulation(sim_id: str, hypothesis: str, intervention_id: str,
                    revert_previous: bool = False,
                    store: Optional[Store] = None) -> SimulationState:
    store = store or get_store()
    raw = store.get("simulations", sim_id)
    if raw is None:
        raise ValueError(f"unknown simulation '{sim_id}'")
    sim = SimulationState(**raw)
    if sim.status != "active":
        raise ValueError("simulation already complete")
    spec = INTERVENTION_INDEX.get(intervention_id)
    if spec is None:
        raise ValueError(f"unknown intervention '{intervention_id}'")

    scenario = next(s for s in SCENARIOS if s.scenario_id == sim.scenario_id)
    sim.turn += 1
    rng = random.Random(sim.seed * 1000 + sim.turn)
    record = TurnRecord(turn=sim.turn, hypothesis=hypothesis,
                        hypothesis_assessment=_assess_hypothesis(hypothesis, spec),
                        intervention_id=intervention_id)

    # keep/reject mechanic: reject the previous turn's intervention (undo its
    # immediate effects) before applying the new one.
    if revert_previous and sim.history:
        prev = sim.history[-1]
        for m, dv in prev.applied_effects.items():
            sim.metrics[m] = sim.metrics.get(m, 50.0) - dv
        record.reverted_previous = True
        record.events.append(f"Rejected previous intervention '{prev.intervention_id}' — its immediate effects were rolled back.")

    # scenario pressure (drift)
    for m, dv in scenario.drift.items():
        sim.metrics[m] = sim.metrics.get(m, 50.0) + dv

    # intervention immediate effects with seeded jitter (±30%)
    applied: Dict[str, float] = {}
    for m, mean in spec.immediate.items():
        jitter = 1.0 + (rng.random() - 0.5) * 0.6
        dv = round(mean * jitter, 2)
        sim.metrics[m] = sim.metrics.get(m, 50.0) + dv
        applied[m] = dv
    record.applied_effects = applied

    # enqueue delayed / second-order effects
    for d in spec.delayed:
        sim.pending.append({"due_turn": sim.turn + int(d["delay"]),
                            "effects": d["effects"], "note": d.get("note", "")})

    # deliver due delayed effects
    still_pending: List[Dict] = []
    for p in sim.pending:
        if p["due_turn"] <= sim.turn:
            for m, dv in p["effects"].items():
                jitter = 1.0 + (rng.random() - 0.5) * 0.4
                sim.metrics[m] = sim.metrics.get(m, 50.0) + round(dv * jitter, 2)
            record.delayed_landed.append(p["note"] or "Delayed effect landed")
        else:
            still_pending.append(p)
    sim.pending = still_pending

    # clamp
    for m in METRICS:
        sim.metrics[m] = round(max(0.0, min(100.0, sim.metrics.get(m, 50.0))), 2)

    # hard floors → events
    if sim.metrics["safety"] < SAFETY_FLOOR:
        incident = {
            "turn": sim.turn, "type": "SAFETY_INCIDENT",
            "detail": (f"Safety fell to {sim.metrics['safety']:.0f} (< floor {SAFETY_FLOOR:.0f}): a "
                       "regression escaped to the fleet. Forced remediation: launches frozen, "
                       "customer trust and morale take the hit."),
        }
        sim.events.append(incident)
        record.events.append(incident["detail"])
        sim.metrics["customer_impact"] = max(0.0, sim.metrics["customer_impact"] - 12)
        sim.metrics["team_morale"] = max(0.0, sim.metrics["team_morale"] - 8)
        sim.metrics["risk"] = min(100.0, sim.metrics["risk"] + 10)
        sim.metrics["safety"] = min(100.0, sim.metrics["safety"] + 12)  # forced remediation
    if sim.metrics["team_morale"] < MORALE_FLOOR:
        attrition = {
            "turn": sim.turn, "type": "ATTRITION",
            "detail": (f"Morale fell to {sim.metrics['team_morale']:.0f} (< floor {MORALE_FLOOR:.0f}): "
                       "a senior engineer resigns; velocity drops."),
        }
        sim.events.append(attrition)
        record.events.append(attrition["detail"])
        sim.metrics["velocity"] = max(0.0, sim.metrics["velocity"] - 10)
        sim.metrics["team_morale"] = min(100.0, sim.metrics["team_morale"] + 6)

    # measure & evaluate
    obj = objective_score(sim.metrics)
    prev_obj = sim.objective_history[-1]
    record.objective_after = obj
    record.objective_delta = round(obj - prev_obj, 2)
    record.verdict = "keep_recommended" if record.objective_delta >= 0 else "reject_recommended"
    record.metrics_after = dict(sim.metrics)
    sim.objective_history.append(obj)
    sim.history.append(record)

    if sim.turn >= sim.max_turns:
        sim.status = "complete"
        sim.debrief = build_debrief(sim, store)

    store.put("simulations", sim.sim_id, sim)
    return sim


def build_debrief(sim: SimulationState, store: Optional[Store] = None) -> Dict:
    """Map the decision history onto competency evidence."""
    store = store or get_store()
    mappings: List[Dict] = []

    strong_hyps = [t for t in sim.history if t.hypothesis_assessment.get("quality") == "strong"]
    if len(strong_hyps) >= max(1, len(sim.history) // 2):
        mappings.append({
            "competency_id": "p4.hill_climbing",
            "verdict": "evidenced",
            "reason": (f"{len(strong_hyps)}/{len(sim.history)} turns used falsifiable, targeted "
                       "hypotheses before intervening."),
            "quotes": [t.hypothesis for t in strong_hyps[:3]],
        })
    else:
        mappings.append({
            "competency_id": "p4.hill_climbing",
            "verdict": "gap",
            "reason": "Most hypotheses were vague (no metric, no direction) — hill climbing without a hypothesis is guessing.",
            "quotes": [t.hypothesis for t in sim.history[:2]],
        })

    rejects = [t for t in sim.history if t.reverted_previous]
    negative_turns = [t for t in sim.history if t.objective_delta < 0]
    if rejects:
        mappings.append({
            "competency_id": "p3.closed_loop_execution",
            "verdict": "evidenced",
            "reason": f"Rejected {len(rejects)} intervention(s) after measurement showed regression — real keep/reject discipline.",
            "quotes": [f"turn {t.turn}: rejected previous after Δobjective {sim.history[t.turn-2].objective_delta:+.1f}"
                       for t in rejects[:2] if t.turn >= 2],
        })
    elif len(negative_turns) >= 2:
        mappings.append({
            "competency_id": "p3.closed_loop_execution",
            "verdict": "gap",
            "reason": f"{len(negative_turns)} turns measured negative but nothing was ever rejected/rolled back.",
            "quotes": [],
        })

    incidents = [e for e in sim.events if e["type"] == "SAFETY_INCIDENT"]
    safety_interventions = {"add_regression_gate", "pause_launches", "incident_review_process", "add_monitoring"}
    used_safety = [t for t in sim.history if t.intervention_id in safety_interventions]
    if incidents and any(t.turn > incidents[0]["turn"] for t in used_safety):
        mappings.append({"competency_id": "p3.safety_culture", "verdict": "evidenced",
                         "reason": "Responded to the safety incident with systemic safety interventions.",
                         "quotes": [t.intervention_id for t in used_safety[:3]]})
    elif incidents:
        mappings.append({"competency_id": "p3.safety_culture", "verdict": "gap",
                         "reason": "A safety incident occurred and no safety-focused intervention followed.",
                         "quotes": []})
    elif used_safety:
        mappings.append({"competency_id": "p3.safety_culture", "verdict": "evidenced",
                         "reason": "Kept safety above the floor all game via proactive safety investment.",
                         "quotes": [t.intervention_id for t in used_safety[:3]]})

    final = sim.metrics
    balanced = all((100 - final[m] if m in INVERTED else final[m]) >= 35 for m in METRICS)
    mappings.append({
        "competency_id": "p2.reliability_tradeoffs",
        "verdict": "evidenced" if balanced else "gap",
        "reason": ("Ended with every objective above the balance threshold — no metric was sacrificed."
                   if balanced else
                   "At least one objective ended critically low — the climb over-optimized a subset."),
        "quotes": [],
    })

    start_obj, end_obj = sim.objective_history[0], sim.objective_history[-1]
    summary = {
        "objective_start": start_obj,
        "objective_end": end_obj,
        "objective_delta": round(end_obj - start_obj, 2),
        "incidents": len(incidents),
        "turns": sim.turn,
        "balanced_finish": balanced,
        "competency_mappings": mappings,
    }

    # persist evidence for the evidenced mappings
    for m in mappings:
        if m["verdict"] != "evidenced":
            continue
        ev = Evidence(
            user_id=sim.user_id,
            competency_ids=[m["competency_id"]],
            artifact_type="simulation_debrief",
            source=f"Simulation {sim.sim_id} ({sim.scenario_id})",
            summary=m["reason"],
            quotes=[q for q in m["quotes"] if isinstance(q, str)][:3],
            score=4.0 if balanced else 3.0,
            confidence=0.6,
            payload={"sim_id": sim.sim_id, "objective_delta": summary["objective_delta"]},
        )
        store.put("evidence", ev.evidence_id, ev)
        m["evidence_id"] = ev.evidence_id

    return summary


def get_simulation(sim_id: str, store: Optional[Store] = None) -> Optional[SimulationState]:
    raw = (store or get_store()).get("simulations", sim_id)
    return SimulationState(**raw) if raw else None
