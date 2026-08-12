"""Model-vs-model comparison, distribution shift, why-layer, similarity."""

from __future__ import annotations

from sensorflow.megaeval import analysis


def test_compare_detects_injected_regression(mega_env):
    store = mega_env["store"]
    cmp = analysis.compare_runs(store, mega_env["bad"], mega_env["good"])
    assert cmp["recommendation"] == "DO_NOT_PROMOTE"
    assert cmp["blockers"]
    hd = {h["metric"]: h["delta"] for h in cmp["headline_deltas"]}
    assert hd["recall"] < 0
    assert hd["safety_recall"] < 0
    # injected night+VRU penalty must surface night cohorts among worst regressions
    assert any("night" in c["cohort"] for c in cmp["worst_cohorts"][:8])
    assert any("REGRESSION" in b for b in cmp["blockers"])


def test_compare_promotes_equivalent_model(mega_env):
    store, meta = mega_env["store"], mega_env["meta"]
    good = mega_env["good"]
    twin = store.create_run(population_id=meta["population_id"],
                            model_version=good.model_version, worker_delay_s=0.0)
    store.execute_sync(twin)
    cmp = analysis.compare_runs(store, twin, good)
    assert cmp["recommendation"] == "PROMOTE"
    assert cmp["blockers"] == []


def test_compare_writes_legacy_regression_result(mega_env):
    from sensorflow.evaluation.records import get_store as legacy_store
    analysis.compare_runs(mega_env["store"], mega_env["bad"], mega_env["good"])
    results = legacy_store().all("regressions")
    ours = [r for r in results if r.model_version == mega_env["bad"].model_version]
    assert ours and ours[-1].regression_detected


def test_distribution_shift_flags_underrepresented_cohorts(mega_env):
    sh = analysis.distribution_shift(mega_env["store"], mega_env["good"])
    assert sh["shifts"], "training mix was deliberately shifted; must be detected"
    top = sh["shifts"][0]
    assert abs(top["relative_change"]) >= 0.35
    assert top["eval_count"] >= 300
    # the train mix under-represents night/bad weather, so eval share must be higher
    assert any(s["relative_change"] > 0 and s["lighting"] == "night" for s in sh["shifts"])


def test_why_decomposition_sums_to_one(mega_env):
    w = analysis.why(mega_env["store"], mega_env["good"],
                     {"class": ["pedestrian"]}, "recall")
    assert w["failure_count"] > 0
    total_share = sum(f["share"] for f in w["factors"])
    assert abs(total_share - 1.0) < 0.01
    assert sum(f["count"] for f in w["factors"]) == w["failure_count"]
    names = {f["factor"] for f in w["factors"]}
    assert names <= {"occlusion", "low_illumination", "long_range",
                     "sensor_disagreement", "other"}
    assert w["top_cohorts"], "must list affected cohorts with metric values"
    assert all("recall" in c for c in w["top_cohorts"])


def test_similarity_hybrid_search(mega_env):
    store, run = mega_env["store"], mega_env["good"]
    containers = store.artifacts(run.run_id)["containers"]
    risky = containers.sort_values("risk_score", ascending=False).iloc[0]
    cid = int(risky["container_id"])
    res = analysis.similar_containers(store, run, cid, k=8)
    assert len(res["results"]) == 8
    sims = [r["similarity"] for r in res["results"]]
    assert sims == sorted(sims, reverse=True)
    assert all(r["container_id"] != cid for r in res["results"])
    # hybrid: structured filter restricts results
    filt = analysis.similar_containers(store, run, cid, filters={"lighting": ["night"]}, k=8)
    assert all(r["lighting"] == "night" for r in filt["results"])
