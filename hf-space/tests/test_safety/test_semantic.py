"""Neuro-symbolic semantic mining: deterministic offline ranking with per-stage
explanations, symbolic filters, and hybrid retrieval over containers +
scenario-DB records."""

from __future__ import annotations

from sensorflow.safety import semantic
from sensorflow.safety.scenario_db import ScenarioRecord, get_db


def test_container_search_deterministic_and_explained(mega_env, safety_root):
    store, run = mega_env["store"], mega_env["good"]
    q = "pedestrian near miss at night in rain"
    r1 = semantic.search_containers(store, run, q, k=10, use_llm=False)
    r2 = semantic.search_containers(store, run, q, k=10, use_llm=False)
    assert r1["results"] == r2["results"]  # deterministic offline ranking

    assert r1["stage2"]["provider"] == "offline_deterministic"
    results = r1["results"]
    assert results
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    top = results[0]
    # concept terms must be reflected in structured evidence
    assert top["lighting"] == "night"
    assert top["weather"] != "clear"
    reasons = top["explanations"]["stage2_reasoning"]
    assert reasons and all({"term", "meaning", "evidence"} <= set(m) for m in reasons)
    # hybrid retrieval blended embedding similarity
    assert r1["hybrid"]["embedding_weight"] > 0
    assert top["embedding_similarity"] is not None


def test_container_search_symbolic_filter(mega_env, safety_root):
    store, run = mega_env["store"], mega_env["good"]
    res = semantic.search_containers(store, run, "risky scenes",
                                     filters={"lighting": "night"}, k=20,
                                     use_llm=False)
    assert res["stage1"]["applied_filters"] == {"lighting": ["night"]}
    assert res["stage1"]["candidates_after"] < res["stage1"]["candidates_before"]
    assert all(r["lighting"] == "night" for r in res["results"])


def test_container_search_fallback_on_unmatched_concept(mega_env, safety_root):
    store, run = mega_env["store"], mega_env["good"]
    res = semantic.search_containers(store, run, "xylophone zeitgeist", k=5,
                                     use_llm=False)
    assert res["results"]
    reasons = res["results"][0]["explanations"]["stage2_reasoning"]
    assert any("no lexicon term matched" in m["meaning"] for m in reasons)


def test_scenario_search_ranks_matching_odd_tags(fresh_safety_root):
    db = get_db()
    db.add(ScenarioRecord(scenario_id="n1", scenario_type="near_miss",
                          source="mined", severity="high",
                          odd_tags={"weather": "rain", "lighting": "night",
                                    "actor_class": "pedestrian"},
                          description="pedestrian near miss in night rain"))
    db.add(ScenarioRecord(scenario_id="d1", scenario_type="nominal",
                          source="synthetic", severity="low",
                          odd_tags={"weather": "clear", "lighting": "day"},
                          description="clear day nominal drive"))

    res = semantic.search_scenarios("pedestrian near miss at night in rain", k=5)
    assert res["results"][0]["scenario_id"] == "n1"
    assert res["results"][0]["score"] > res["results"][-1]["score"]
    assert res["stage2"]["provider"] == "offline_deterministic"

    filtered = semantic.search_scenarios("anything", filters={"source": "synthetic"})
    assert [r["scenario_id"] for r in filtered["results"]] == ["d1"]
