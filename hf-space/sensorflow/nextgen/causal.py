"""Causal counterfactual replay: does a perception error actually MATTER?

The SAME scenario is run twice through the closed loop (closedloop.py):

  actual     perception as-is (engine failure model + any injected faults)
  corrected  ground-truth-injected perception (perfect detection)

The two runs are diffed on planner output, trajectory, velocity, braking,
minimum separation and collision probability, and the causal chain is
answered stepwise:

  Q1  Would correct perception change the planner output?
  Q2  Would that change the vehicle's behavior (trajectory/velocity)?
  Q3  Would that change the safety outcome (separation / collision)?

Verdict (deterministic policy):
  BEHAVIORALLY_CONSEQUENTIAL  iff Q3 is yes (the perception difference
                              propagates all the way to a safety outcome)
  METRIC_ONLY                 otherwise — the error shows up in open-loop
                              metrics but does not change what the vehicle
                              does in any safety-relevant way (e.g. a
                              cosmetic class flip: the planner reacts to
                              geometry, not class).

Both assessments carry the scenario's data label; nothing here upgrades
simulation to reality.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sensorflow.nextgen.closedloop import DEFAULT_ENGINE, run_closed_loop
from sensorflow.nextgen.models import (
    CAUSAL_BEHAVIORAL, CAUSAL_METRIC_ONLY, BehavioralAssessment,
    CausalReplayResult, DataLabel,
)
from sensorflow.nextgen.worldmodel import ActorTrack

# Deterministic thresholds of the causal policy (versioned via lineage).
PLANNER_DIFF_MPS2 = 0.5       # commanded-accel divergence that counts as "changed plan"
TRAJECTORY_DIFF_M = 0.5       # positional divergence that counts as "changed behavior"
VELOCITY_DIFF_MPS = 1.0
SEPARATION_DIFF_M = 0.5       # safety-outcome deltas
COLLISION_PROB_DIFF = 0.15
TTC_DIFF_S = 0.5


def _aligned(actual: List[Dict], corrected: List[Dict]):
    n = min(len(actual), len(corrected))
    return actual[:n], corrected[:n]


def causal_replay(actors: List[ActorTrack], environment: Dict[str, str],
                  scenario_id: str, data_label: DataLabel,
                  engine: str = DEFAULT_ENGINE, seed: int = 0,
                  faults: Optional[List[Dict]] = None) -> CausalReplayResult:
    actual = run_closed_loop(actors, environment, scenario_id, data_label,
                             engine=engine, seed=seed, corrected=False,
                             faults=faults)
    corrected = run_closed_loop(actors, environment, scenario_id, data_label,
                                engine=engine, seed=seed, corrected=True)

    ta, tc = _aligned(actual.trajectory, corrected.trajectory)
    if ta:
        pos_div = float(max(abs(a["x"] - c["x"]) + abs(a["y"] - c["y"])
                            for a, c in zip(ta, tc)))
        vel_div = float(max(abs(a["v"] - c["v"]) for a, c in zip(ta, tc)))
        acc_div = float(max(abs(a["a"] - c["a"]) for a, c in zip(ta, tc)))
    else:
        pos_div = vel_div = acc_div = 0.0
    # A truncated run means one branch collided and stopped early — that is
    # itself maximal behavioral divergence.
    if len(actual.trajectory) != len(corrected.trajectory):
        pos_div = max(pos_div, TRAJECTORY_DIFF_M * 2)

    ma, mc = actual.metrics, corrected.metrics
    sep_diff = _diff(ma.min_separation_m, mc.min_separation_m)
    col_prob_diff = ma.collision_probability - mc.collision_probability
    ttc_diff = _diff(ma.min_ttc_s, mc.min_ttc_s)
    brake_diff = ma.max_deceleration_mps2 - mc.max_deceleration_mps2

    q1 = acc_div > PLANNER_DIFF_MPS2
    q2 = q1 and (pos_div > TRAJECTORY_DIFF_M or vel_div > VELOCITY_DIFF_MPS)
    q3 = q2 and (ma.collision != mc.collision
                 or (sep_diff is not None and abs(sep_diff) > SEPARATION_DIFF_M)
                 or abs(col_prob_diff) > COLLISION_PROB_DIFF
                 or (ttc_diff is not None and abs(ttc_diff) > TTC_DIFF_S))

    chain = [
        {"question": "Would correct perception change the planner output?",
         "answer": q1,
         "evidence": f"max commanded-accel divergence {acc_div:.2f} m/s^2 "
                     f"(threshold {PLANNER_DIFF_MPS2})"},
        {"question": "Would that change the vehicle's behavior?",
         "answer": q2,
         "evidence": f"max trajectory divergence {pos_div:.2f} m, "
                     f"max velocity divergence {vel_div:.2f} m/s "
                     f"(thresholds {TRAJECTORY_DIFF_M} m / {VELOCITY_DIFF_MPS} m/s)"},
        {"question": "Would that change the safety outcome?",
         "answer": q3,
         "evidence": f"collision {ma.collision} vs {mc.collision}; "
                     f"min-separation diff "
                     f"{'n/a' if sep_diff is None else f'{sep_diff:+.2f} m'}; "
                     f"collision-probability diff {col_prob_diff:+.2f}; "
                     f"min-TTC diff "
                     f"{'n/a' if ttc_diff is None else f'{ttc_diff:+.2f} s'}"},
    ]

    return CausalReplayResult(
        scenario_id=scenario_id, data_label=data_label,
        actual=actual, corrected=corrected,
        diffs={
            "max_trajectory_divergence_m": round(pos_div, 3),
            "max_velocity_divergence_mps": round(vel_div, 3),
            "max_planner_divergence_mps2": round(acc_div, 3),
            "min_separation_diff_m": _round(sep_diff),
            "collision_probability_diff": round(col_prob_diff, 4),
            "min_ttc_diff_s": _round(ttc_diff),
            "max_braking_diff_mps2": round(brake_diff, 3),
            "collision_actual": ma.collision,
            "collision_corrected": mc.collision,
        },
        causal_chain=chain,
        verdict=CAUSAL_BEHAVIORAL if q3 else CAUSAL_METRIC_ONLY)


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _round(v: Optional[float], nd: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), nd)
