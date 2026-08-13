"""ODD taxonomy + combinatorial coverage math (exact expectations on a
hand-built cube) + run-level coverage and gap filling."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from sensorflow.megaeval.cube import STAT_COLS
from sensorflow.megaeval.population import DIM_NAMES, DIMENSIONS
from sensorflow.safety import odd


def test_taxonomy_structure():
    tax = odd.taxonomy()
    dims = tax["dimensions"]
    for name in DIM_NAMES:
        assert dims[name]["values"] == DIMENSIONS[name]
        assert dims[name]["instrumented"] is True
        assert dims[name]["iso34503_category"] in (
            "scenery", "environmental_conditions", "dynamic_elements",
            "sensor_configuration")
    # declared-but-not-instrumented dimension is honest about itself
    assert dims["geography"]["instrumented"] is False
    assert "ISO 34503" in tax["standard_basis"]


def _mini_cube() -> pd.DataFrame:
    """Cube with data in exactly two weather cells:
    clear: tp=90 fn=10 (n=100), rain: tp=8 fn=2 (n=10). fog/snow empty."""
    rows = []
    for weather_code, tp, fn in ((0, 90, 10), (1, 8, 2)):
        row = {d: 0 for d in DIM_NAMES}
        row["weather"] = weather_code
        row.update({c: 0 for c in STAT_COLS})
        row.update({"n": tp + fn, "tp": tp, "fn": fn, "sum_iou": 0.8 * tp})
        rows.append(row)
    return pd.DataFrame(rows)


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def test_coverage_exact_math_on_known_population():
    dim_counts = {"weather": {"clear": 700, "rain": 200, "fog": 80, "snow": 20}}
    cov = odd.compute_coverage(_mini_cube(), dims=["weather"], dim_counts=dim_counts,
                               min_samples=20, max_ci_width=0.25, target_recall=0.9)
    cells = {c["cell"]["weather"]: c for c in cov["cells"]}
    assert set(cells) == {"clear", "rain", "fog", "snow"}

    # clear: n=100, recall 0.9, Wilson CI hand-checked, adequate
    clear = cells["clear"]
    lo, hi = _wilson(90, 100)
    assert clear["n"] == 100
    assert clear["recall"] == pytest.approx(0.9)
    assert clear["wilson_ci"][0] == pytest.approx(lo, abs=2e-3)
    assert clear["wilson_ci"][1] == pytest.approx(hi, abs=2e-3)
    assert clear["adequate"] and not clear["is_gap"]
    assert clear["production_share"] == pytest.approx(0.7)
    # risk = share * max(0, target - wilson_lo)
    assert clear["risk"] == pytest.approx(0.7 * max(0.0, 0.9 - lo), abs=2e-3)

    # rain: n=10 < 20 -> insufficient samples gap
    rain = cells["rain"]
    assert rain["is_gap"] and "insufficient_samples" in rain["gap_reasons"]
    assert rain["recall"] == pytest.approx(0.8)

    # empty cells: both gap reasons, CI width 1
    fog = cells["fog"]
    assert fog["n"] == 0 and fog["recall"] is None
    assert set(fog["gap_reasons"]) == {"insufficient_samples", "ci_too_wide"}
    assert fog["ci_width"] == pytest.approx(1.0)

    # summary: exactly 1 adequate of 4 cells; weighted coverage = clear share
    s = cov["summary"]
    assert s["total_cells"] == 4
    assert s["adequate_cells"] == 1 and s["gap_cells"] == 3
    assert s["coverage_rate"] == pytest.approx(0.25)
    assert s["production_weighted_coverage"] == pytest.approx(0.7)

    # gaps ranked by risk = share * deficit: rain(share .2) > fog(.08) > snow(.02)
    gap_order = [g["cell"]["weather"] for g in cov["gaps"]]
    assert gap_order == ["rain", "fog", "snow"]
    for g in cov["gaps"]:
        assert g["risk"] == pytest.approx(
            g["production_share"] * g["performance_deficit"], abs=1e-4)

    # every gap gets a fulfillable request
    assert len(cov["fill_requests"]) == 3
    req = cov["fill_requests"][0]
    assert req["cell"]["weather"] == "rain"
    assert req["needed_samples"] == 10
    assert req["generator"] == "sensorflow.evaluation.synthetic.generate_dataset"


def test_coverage_rejects_oversized_cell_product():
    with pytest.raises(ValueError):
        odd.compute_coverage(_mini_cube(), dims=DIM_NAMES)  # way over 2000 cells


def test_coverage_for_run_consistency(mega_env, safety_root):
    cov = odd.coverage_for_run(mega_env["store"], mega_env["good"],
                               dims=["weather", "lighting"], include_cells=True)
    s = cov["summary"]
    assert s["total_cells"] == len(DIMENSIONS["weather"]) * len(DIMENSIONS["lighting"])
    assert s["populated_cells"] + s["empty_cells"] == s["total_cells"]
    assert s["adequate_cells"] + s["gap_cells"] == s["total_cells"]
    # cell mass equals the population's object count (every object in one cell)
    assert sum(c["n"] for c in cov["cells"]) == mega_env["meta"]["num_objects"]
    # adequacy consistent with the thresholds echoed in the response
    th = cov["thresholds"]
    for c in cov["cells"]:
        expected = c["n"] >= th["min_samples"] and c["ci_width"] <= th["max_ci_width"]
        assert c["adequate"] == expected
    # gaps sorted by descending risk
    risks = [g["risk"] for g in cov["gaps"]]
    assert risks == sorted(risks, reverse=True)


def test_fill_gap_generates_data_and_improves_cell(mega_env, fresh_safety_root, tmp_path):
    from sensorflow.evaluation.records import reset_store
    reset_store(tmp_path / "eval")

    store, run = mega_env["store"], mega_env["good"]
    cov = odd.coverage_for_run(store, run, dims=["weather", "lighting"],
                               include_cells=True, min_samples=10_000)
    gap = cov["gaps"][0]  # min_samples=10k guarantees gaps exist

    result = odd.fill_gap(store, run, gap["cell"], num_sequences=1,
                          frames_per_sequence=10)
    assert result["objects_added"] > 0
    assert result["cell_after"]["n"] > result["cell_before"]["n"]
    assert result["cell_after"]["n"] == (result["cell_before"]["n"]
                                         + result["objects_added"])
    assert result["supplement"]["source"] == "synthetic"

    # supplement persisted and folded into subsequent coverage computations
    supplements = odd.load_supplements(run.run_id)
    assert len(supplements) == 1
    cov2 = odd.coverage_for_run(store, run, dims=["weather", "lighting"],
                                include_cells=True, min_samples=10_000)
    cell2 = next(c for c in cov2["cells"] if c["cell_id"] == gap["cell_id"])
    assert cell2["n"] == result["cell_after"]["n"]
    assert cell2["synthetic_supplement"]["n_added"] == result["objects_added"]

    # generated frames carry the requested conditions
    from sensorflow.evaluation.records import get_store
    legacy = get_store()
    frames = legacy.where("frames", dataset_id=result["generated_dataset_id"])
    assert frames
    weather = gap["cell"].get("weather")
    if weather:
        assert all(f.weather == weather for f in frames)

    # scenario DB registered the synthetic gap-fill scenario
    from sensorflow.safety.scenario_db import get_db
    recs = get_db().search(source="synthetic")
    assert any(r.scenario_type == "odd_gap_fill" for r in recs)


def test_fill_gap_rejects_unknown_cell(mega_env, fresh_safety_root):
    with pytest.raises(ValueError):
        odd.fill_gap(mega_env["store"], mega_env["good"], {"weather": "hurricane"})
