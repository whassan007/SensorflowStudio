"""Safety-Critical Recall region math + the recall-up/SCR-down demonstration."""

from __future__ import annotations

import pytest

from sensorflow.nextgen.safety_metrics import (
    SafetyRegionParams, divergence_demo, in_safety_critical_region,
    risk_weight, safety_report,
)


@pytest.mark.parametrize("v,t_react,brake,mu,expected_d_crit", [
    # d_crit = v*t + v^2/(2*min(brake, mu*g)) + buffer
    (10.0, 0.9, 7.0, 0.9, 10.0 * 0.9 + 100.0 / (2 * 7.0) + 2.0),
    (20.0, 1.2, 7.0, 0.9, 20.0 * 1.2 + 400.0 / (2 * 7.0) + 2.0),
    # friction-limited: mu*g = 4.905 < brake capability 7.0
    (15.0, 0.9, 7.0, 0.5, 15.0 * 0.9 + 225.0 / (2 * 0.5 * 9.81) + 2.0),
])
def test_safety_region_distance_parameterized(v, t_react, brake, mu, expected_d_crit):
    p = SafetyRegionParams(reaction_time_s=t_react, brake_capability_mps2=brake,
                           friction_mu=mu)
    region = in_safety_critical_region(
        {"x": 5.0, "y": 0.0, "class_name": "pedestrian"}, v, p)
    assert region["d_crit_m"] == pytest.approx(expected_d_crit, abs=1e-3)


def test_region_uses_friction_limited_decel_not_naive_capability():
    wet = SafetyRegionParams(friction_mu=0.5, brake_capability_mps2=7.0)
    dry = SafetyRegionParams(friction_mu=0.9, brake_capability_mps2=7.0)
    assert wet.effective_decel == pytest.approx(0.5 * 9.81)
    assert dry.effective_decel == pytest.approx(7.0)  # capability binds
    assert wet.d_crit(15.0) > dry.d_crit(15.0)


def test_lateral_band_scales_with_class_encroachment_speed():
    p = SafetyRegionParams()
    ped = in_safety_critical_region({"x": 10.0, "y": 5.0,
                                     "class_name": "pedestrian"}, 10.0, p)
    cyc = in_safety_critical_region({"x": 10.0, "y": 5.0,
                                     "class_name": "cyclist"}, 10.0, p)
    assert cyc["lateral_band_m"] > ped["lateral_band_m"]


def test_far_off_corridor_object_not_critical_without_ttc_conflict():
    region = in_safety_critical_region(
        {"x": 70.0, "y": 10.0, "class_name": "vehicle", "vx": 8.0, "vy": 0.0},
        10.0)
    assert region["safety_critical"] is False


def test_ttc_criterion_captures_crossing_actor_outside_corridor():
    # Pedestrian outside the lateral band but crossing into a collision path.
    region = in_safety_critical_region(
        {"x": 18.0, "y": 6.5, "class_name": "pedestrian",
         "vx": 0.0, "vy": -2.4, "dims": [0.6, 0.6]}, 10.0)
    assert region["ttc_critical"] is True
    assert region["safety_critical"] is True


def test_risk_weight_orders_vulnerable_close_fast():
    near_ped = risk_weight({"x": 8.0, "y": 0.5, "class_name": "pedestrian",
                            "vx": 0.0, "vy": 1.0}, 12.0)
    far_truck = risk_weight({"x": 70.0, "y": 8.0, "class_name": "truck",
                             "vx": 10.0, "vy": 0.0}, 12.0)
    assert near_ped > 5 * far_truck


def test_recall_up_scr_down_demonstration():
    demo = divergence_demo()
    d = demo["deltas"]
    assert d["overall_recall"] > 0            # overall recall IMPROVES
    assert d["safety_critical_recall"] < 0    # SCR DEGRADES
    assert d["risk_weighted_recall"] < 0
    # both views present in both reports (complementarity, never replaced)
    for rep in (demo["baseline"], demo["candidate"]):
        assert rep["open_loop"]["recall"] is not None
        assert rep["safety_informed"]["safety_critical_recall"] is not None
        assert rep["region_params"]["reaction_time_s"] == 0.9


def test_safety_report_by_class_and_labels():
    objs = [{"x": 6.0, "y": 0.2, "class_name": "pedestrian", "vx": 0.0,
             "vy": 0.0, "detected": True},
            {"x": 60.0, "y": 9.0, "class_name": "vehicle", "vx": 8.0,
             "vy": 0.0, "detected": False}]
    rep = safety_report(objs, ego_speed=10.0, data_label="COUNTERFACTUAL")
    assert rep["data_label"] == "COUNTERFACTUAL"
    assert rep["open_loop"]["recall"] == 0.5
    assert rep["by_class"]["pedestrian"]["safety_critical_recall"] == 1.0
