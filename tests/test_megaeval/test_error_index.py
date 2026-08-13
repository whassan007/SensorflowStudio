"""Error index: multi-criteria search correctness and worst-container ranking."""

from __future__ import annotations

import numpy as np

from sensorflow.megaeval import errors as errors_mod
from sensorflow.megaeval.population import DIMENSIONS
from sensorflow.megaeval.runs import ERROR_TYPES


def test_multi_criteria_search_matches_brute_filter(mega_env):
    store, run = mega_env["store"], mega_env["good"]
    art = store.artifacts(run.run_id)
    errs, containers = art["errors"], art["containers"]

    res = errors_mod.search_errors(
        errs, containers, error_types=["FN"],
        filters={"class": ["pedestrian"], "lighting": ["night"]},
        confidence_max=0.4, risk_min=0.5)

    mask = ((errs["error_type"] == ERROR_TYPES.index("FN"))
            & (errs["class"] == DIMENSIONS["class"].index("pedestrian"))
            & (errs["lighting"] == DIMENSIONS["lighting"].index("night"))
            & (errs["confidence"] <= 0.4)
            & (errs["risk_score"] >= 0.5))
    assert res["matched_errors"] == int(mask.sum())
    assert res["matched_errors"] > 0

    for ex in res["examples"]:
        assert ex["error_type"] == "FN"
        assert ex["class"] == "pedestrian"
        assert ex["lighting"] == "night"
        assert ex["confidence"] <= 0.4
        assert ex["risk_score"] >= 0.5


def test_worst_containers_ranked_and_capped(mega_env):
    store, run = mega_env["store"], mega_env["good"]
    art = store.artifacts(run.run_id)
    res = errors_mod.search_errors(art["errors"], art["containers"],
                                   error_types=["FN", "FP"], limit_containers=10)
    worst = res["worst_containers"]
    assert 0 < len(worst) <= 10
    scores = [w["error_count"] * w["mean_risk"] for w in worst]
    assert scores == sorted(scores, reverse=True)
    # returns aggregated containers with context dims, not raw record dumps
    assert "weather" in worst[0] and "n_objects" in worst[0]


def test_safety_only_filter(mega_env):
    store, run = mega_env["store"], mega_env["good"]
    art = store.artifacts(run.run_id)
    res = errors_mod.search_errors(art["errors"], art["containers"],
                                   error_types=["FN"], safety_only=True)
    assert res["matched_errors"] > 0
    assert all(ex["safety_critical"] for ex in res["examples"])


def test_by_type_totals_are_consistent(mega_env):
    store, run = mega_env["store"], mega_env["good"]
    art = store.artifacts(run.run_id)
    res = errors_mod.search_errors(art["errors"], art["containers"])
    assert sum(res["by_type"].values()) == res["matched_errors"] == len(art["errors"])
