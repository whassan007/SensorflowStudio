"""Closed-loop behavioral evaluation: planted-scenario sanity + provenance."""

from __future__ import annotations

from sensorflow.nextgen.causal import causal_replay
from sensorflow.nextgen.closedloop import demo_emergence_scenario, run_closed_loop
from sensorflow.nextgen.models import (
    CAUSAL_BEHAVIORAL, CAUSAL_METRIC_ONLY, DataLabel,
)


def _run(demo, **kw):
    return run_closed_loop(demo["actors"], demo["environment"],
                           demo["scenario_id"], DataLabel.COUNTERFACTUAL, **kw)


def test_late_detection_shortens_ttc_and_margin():
    """Planted case: the same scenario with the critical pedestrian missed
    until late must produce shorter TTC and smaller safety margin."""
    demo = demo_emergence_scenario()
    ped = demo["critical_instance_id"]
    on_time = _run(demo, seed=0, corrected=True)
    late = _run(demo, seed=0,
                faults=[{"type": "miss", "instance_id": ped,
                         "from_s": 0.0, "until_s": 3.2}])
    assert late.metrics.min_ttc_s <= on_time.metrics.min_ttc_s
    assert late.metrics.min_separation_m < on_time.metrics.min_separation_m
    assert late.metrics.safety_margin_m < on_time.metrics.safety_margin_m


def test_behavioral_metrics_sane_and_deterministic():
    demo = demo_emergence_scenario()
    a1 = _run(demo, seed=4)
    a2 = _run(demo, seed=4)
    assert a1.metrics == a2.metrics                      # seeded determinism
    assert a1.metrics.max_deceleration_mps2 <= 8.0 + 1e-6
    assert a1.metrics.planner_interventions >= 1         # ped triggers braking
    assert a1.metrics.min_separation_m is not None
    assert a1.open_loop["frame_recall"] is not None      # open-loop attached
    assert a1.data_label == DataLabel.COUNTERFACTUAL     # label carried


def test_corrected_mode_is_perfect_perception():
    demo = demo_emergence_scenario()
    corrected = _run(demo, seed=0, corrected=True)
    assert corrected.open_loop["frame_recall"] == 1.0
    assert corrected.perception_mode == "corrected"


def test_causal_replay_separates_metric_only_from_behavioral():
    """Planted pair: missed crossing pedestrian is BEHAVIORALLY_CONSEQUENTIAL;
    a cosmetic classification flip is METRIC_ONLY."""
    demo = demo_emergence_scenario()
    ped = demo["critical_instance_id"]

    missed = causal_replay(demo["actors"], demo["environment"],
                           demo["scenario_id"], DataLabel.COUNTERFACTUAL,
                           seed=0,
                           faults=[{"type": "miss", "instance_id": ped,
                                    "from_s": 0.0, "until_s": 4.0}])
    assert missed.verdict == CAUSAL_BEHAVIORAL
    assert missed.diffs["collision_actual"] or \
        missed.diffs["min_separation_diff_m"] < -0.5
    answers = [c["answer"] for c in missed.causal_chain]
    assert answers == [True, True, True]   # full causal chain

    cosmetic = causal_replay(demo["actors"], demo["environment"],
                             demo["scenario_id"], DataLabel.COUNTERFACTUAL,
                             seed=0,
                             faults=[{"type": "misclassify",
                                      "instance_id": ped,
                                      "as_class": "cyclist"}])
    assert cosmetic.verdict == CAUSAL_METRIC_ONLY
    assert cosmetic.causal_chain[2]["answer"] is False
    # the flip IS visible open-loop — that's the point of METRIC_ONLY
    assert cosmetic.actual.open_loop["n_misclassified"] > 0


def test_causal_result_carries_data_label_through_report():
    demo = demo_emergence_scenario()
    res = causal_replay(demo["actors"], demo["environment"],
                        demo["scenario_id"], DataLabel.COUNTERFACTUAL, seed=0)
    assert res.data_label == DataLabel.COUNTERFACTUAL
    assert res.actual.data_label == DataLabel.COUNTERFACTUAL
    assert res.corrected.data_label == DataLabel.COUNTERFACTUAL
