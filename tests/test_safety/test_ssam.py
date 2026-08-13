"""Extended SSAM measures against hand-computed values (TTC, DRAC, collision
probability, DeltaS/MaxS, CSI aggregation, zone-grid PET) plus the synthetic
scenario suite and model-conditioned CSI."""

from __future__ import annotations

import math

import pytest

from sensorflow.safety import ssam_ext

CAR = (4.5, 1.9)


def _state(t, x, y, speed, heading=0.0):
    return {"t": t, "x": x, "y": y, "speed": speed, "heading": heading}


def test_projected_ttc_hand_computed():
    # A at x=0 doing 10 m/s toward stationary B at x=20, both 4.5 m long:
    # bumper gap = 20 - 2.25 - 2.25 = 15.5 m -> TTC = 1.55 s.
    a = _state(0.0, 0.0, 0.0, 10.0)
    b = _state(0.0, 20.0, 0.0, 0.0)
    ttc = ssam_ext.projected_ttc(a, CAR, b, CAR, look_ahead=3.0, dt=0.05)
    assert ttc == pytest.approx(1.55, abs=0.02)

    # diverging pair -> no TTC
    b_away = _state(0.0, 20.0, 0.0, 15.0)  # same heading, faster, pulling away
    assert ssam_ext.projected_ttc(a, CAR, b_away, CAR, 3.0, 0.05) is None


def test_drac_hand_computed():
    # closing speed 10 m/s, gap 15.5 m -> DRAC = 10^2 / (2*15.5) = 3.2258 m/s^2
    a = _state(0.0, 0.0, 0.0, 10.0)
    b = _state(0.0, 20.0, 0.0, 0.0)
    assert ssam_ext.drac(a, CAR, b, CAR) == pytest.approx(100 / 31, abs=1e-6)
    # opening geometry -> None
    a_rev = _state(0.0, 0.0, 0.0, 10.0, heading=math.pi)
    assert ssam_ext.drac(a_rev, CAR, b, CAR) is None


def test_collision_probability_formula():
    assert ssam_ext.collision_probability(None, 1.5) == 0.0
    assert ssam_ext.collision_probability(1.5, 1.5) == 0.0
    assert ssam_ext.collision_probability(0.75, 1.5) == pytest.approx(0.5)
    assert ssam_ext.collision_probability(0.0, 1.5) == 1.0
    assert ssam_ext.collision_probability(9.0, 1.5) == 0.0  # clipped


def _head_on_pair(dt=0.1, duration=2.0):
    """A drives at 10 m/s toward stationary B at x=30 (same lane)."""
    a = {"vehicle_id": "A", "vehicle_type": "car", "length": 4.5, "width": 1.9,
         "states": [_state(round(i * dt, 6), 10.0 * i * dt, 0.0, 10.0)
                    for i in range(int(duration / dt) + 1)]}
    b = {"vehicle_id": "B", "vehicle_type": "car", "length": 4.5, "width": 1.9,
         "states": [_state(round(i * dt, 6), 30.0, 0.0, 0.0)
                    for i in range(int(duration / dt) + 1)]}
    return a, b


def test_analyze_trajectories_measures_and_csi_aggregation():
    a, b = _head_on_pair()
    res = ssam_ext.analyze_trajectories([a, b])
    agg = res["aggregate"]
    assert agg["num_conflicts"] == 1
    conflict = res["conflicts"][0]

    # severity proxies: relative speed 10, max abs speed 10, rear-end geometry
    assert conflict["delta_s_mps"] == pytest.approx(10.0, abs=1e-6)
    assert conflict["max_s_mps"] == pytest.approx(10.0, abs=1e-6)
    assert conflict["conflict_type"] == "rear_end"
    assert 0.0 < conflict["collision_probability"] <= 1.0

    # CSI aggregation re-derived with an independent loop over the same
    # primitives: CSI = sum 0.5 * mu * DeltaS^2 * p(t) * dt   (mu reduced mass)
    p = res["params"]
    mu = 0.5  # two cars: 1*1/(1+1)
    expected = 0.0
    for sa, sb in zip(a["states"], b["states"]):
        ttc = ssam_ext.projected_ttc(sa, CAR, sb, CAR,
                                     p["look_ahead_s"], p["projection_dt_s"])
        if ttc is None or ttc > p["ttc_threshold_s"]:
            continue
        prob = ssam_ext.collision_probability(ttc, p["ttc_threshold_s"])
        expected += 0.5 * mu * 10.0 ** 2 * prob * 0.1
    assert agg["aggregate_csi"] == pytest.approx(expected, rel=1e-6)
    assert agg["aggregate_csi"] > 0


def test_zone_grid_pet_hand_computed():
    # A crosses the origin cell around t=1.0 (moving +x through x=0),
    # B crosses the same spot around t=3.0 (moving +y through y=0):
    # PET ~= 2.0 s (grid quantization tolerance from finite footprints).
    dt = 0.1
    a = {"vehicle_id": "A", "length": 1.0, "width": 1.0,
         "states": [_state(round(i * dt, 6), -10.0 + 10.0 * i * dt, 0.0, 10.0)
                    for i in range(21)]}
    b = {"vehicle_id": "B", "length": 1.0, "width": 1.0,
         "states": [{"t": round(2.0 + i * dt, 6), "x": 0.0, "y": -10.0 + 10.0 * i * dt,
                     "speed": 10.0, "heading": math.pi / 2} for i in range(21)]}
    pet = ssam_ext.zone_grid_pet(a, b, cell=0.5)
    assert pet is not None
    assert pet == pytest.approx(2.0, abs=0.35)


def test_scenario_suite_produces_conflicts():
    for scenario in ("crossing", "rear_end", "lane_change"):
        trajs = ssam_ext.generate_trajectories(seed=0, scenario=scenario,
                                               reaction_delay_s=0.6)
        res = ssam_ext.analyze_trajectories(trajs)
        assert res["aggregate"]["num_conflicts"] >= 1, scenario
        assert res["aggregate"]["aggregate_csi"] > 0, scenario
    with pytest.raises(ValueError):
        ssam_ext.generate_trajectories(scenario="teleportation")


def test_reaction_delay_monotonic_and_csi_increases():
    base = ssam_ext.reaction_delay_for_model("model-v41")
    worse = ssam_ext.reaction_delay_for_model(
        "model-v42-regressed", overrides={"night_penalty": 0.35, "vru_penalty": 0.12})
    assert worse["reaction_delay_s"] > base["reaction_delay_s"]
    assert base["simulated"] is True

    csis = []
    for delay in (base["reaction_delay_s"], worse["reaction_delay_s"]):
        total = 0.0
        for scenario in ("crossing", "rear_end", "lane_change"):
            trajs = ssam_ext.generate_trajectories(seed=0, scenario=scenario,
                                                   reaction_delay_s=delay)
            total += ssam_ext.analyze_trajectories(trajs)["aggregate"]["aggregate_csi"]
        csis.append(total)
    assert csis[1] > csis[0]


def test_csi_for_run_cached_and_ordered(mega_env, safety_root):
    good = ssam_ext.csi_for_run(mega_env["good"])
    bad = ssam_ext.csi_for_run(mega_env["bad"])
    assert good["aggregate_csi"] > 0
    assert bad["aggregate_csi"] > good["aggregate_csi"]
    assert good["simulated"] is True
    # cached second read is identical
    again = ssam_ext.csi_for_run(mega_env["good"])
    assert again == good
