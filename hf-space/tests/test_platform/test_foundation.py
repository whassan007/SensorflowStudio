"""Phase 1 platform foundation tests — metrics, container quality, compare, gates, evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sensorflow.platform.metrics_engine import (
    aggregate_container_rows,
    container_quality_metrics,
    delta_metrics,
    precision_recall_f1,
    verification_rates,
)
from sensorflow.platform.container_quality import profile_from_counts
from sensorflow.platform.levels import AggregateLevel, EvaluationScope, summarize_scope
from sensorflow.platform.gates import (
    DEFAULT_GATE_CONFIG,
    evaluate_multi_gates,
    list_gate_defs,
    load_gate_config,
)
from sensorflow.platform.evidence import build_evidence_package, export_evidence_package
from sensorflow.platform.compare import compare_models
from sensorflow.platform.entities import EvidencePackage, Provenance


# ------------------------------------------------------------------ metrics


def test_precision_recall_f1_basic():
    out = precision_recall_f1(tp=8, fp=2, fn=2)
    assert out["precision"] == 0.8
    assert out["recall"] == 0.8
    assert out["f1"] == 0.8


def test_precision_recall_empty():
    out = precision_recall_f1(0, 0, 0)
    assert out["precision"] is None
    assert out["recall"] is None
    assert out["f1"] is None


def test_verification_rates():
    v = verification_rates(
        n_objects=100,
        verified=40,
        reviewed=55,
        disputed=5,
        auto_accepted=30,
        hitl=15,
    )
    assert v["verified_rate"] == 0.4
    assert v["hitl_rate"] == 0.15
    assert v["disputed_rate"] == 0.05
    assert v["auto_accept_rate"] == 0.3
    assert v["unverified"] == 60


def test_container_quality_metrics_bundle():
    m = container_quality_metrics(
        tp=10, fp=5, fn=5, sum_iou=7.0, n_objects=20,
        verified=8, reviewed=12, hitl=4, anomalies=2,
    )
    assert m["precision"] == pytest.approx(10 / 15, rel=1e-3)
    assert m["mean_iou"] == 0.7
    assert m["verification"]["verified_rate"] == 0.4
    assert m["anomaly_rate"] == 0.1


def test_aggregate_container_rows():
    rows = [
        {"tp": 5, "fp": 1, "fn": 1, "sum_iou": 4.0, "n_objects": 10, "verified": 3, "reviewed": 4},
        {"tp": 5, "fp": 1, "fn": 1, "sum_iou": 3.5, "n_objects": 10, "verified": 5, "reviewed": 6},
    ]
    agg = aggregate_container_rows(rows)
    assert agg["tp"] == 10
    assert agg["verification"]["verified"] == 8


def test_delta_metrics():
    deltas = delta_metrics(
        {"precision": 0.9, "recall": 0.8},
        {"precision": 0.88, "recall": 0.85},
        keys=["precision", "recall"],
    )
    assert deltas[0]["delta"] == pytest.approx(-0.02)
    assert deltas[1]["delta"] == pytest.approx(0.05)


def test_profile_from_counts():
    p = profile_from_counts(
        container_id="c-1", tp=4, fp=1, fn=1, sum_iou=3.2,
        n_objects=10, verified=2, reviewed=5, hitl=3,
    )
    assert p["container_id"] == "c-1"
    assert p["f1"] is not None
    assert p["verification"]["hitl"] == 3


# ------------------------------------------------------------------ levels


def test_aggregate_ladder():
    ladder = AggregateLevel.ladder()
    assert ladder[0] == AggregateLevel.FRAME
    assert ladder[-1] == AggregateLevel.POPULATION
    assert AggregateLevel.CONTAINER.parent() == AggregateLevel.DATASET
    assert AggregateLevel.FRAME.child() is None


def test_evaluation_scope_summary():
    scope = EvaluationScope(
        level=AggregateLevel.CONTAINER,
        run_id="run-1",
        mega_container_id=42,
        model_version="model-v1",
    )
    s = summarize_scope(scope)
    assert s["backend"] == "megaeval"
    assert s["refs"]["mega_container_id"] == 42


# ------------------------------------------------------------------ gates


def test_gate_defs_include_multi_gate_skeleton(tmp_path: Path):
    cfg_path = tmp_path / "gates.json"
    cfg = load_gate_config(cfg_path)
    assert cfg_path.exists()
    defs = list_gate_defs(cfg)
    types = {d.gate_type for d in defs}
    for required in ("scenario", "coverage", "regression", "safety", "release", "quality", "launch"):
        assert required in types


def test_evaluate_multi_gates_skeleton(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "gates.json"
    monkeypatch.setattr(
        "sensorflow.platform.gates.DEFAULT_GATE_CONFIG_PATH", cfg_path
    )
    # rewrite load to use tmp
    from sensorflow.platform import gates as gates_mod
    monkeypatch.setattr(gates_mod, "DEFAULT_GATE_CONFIG_PATH", cfg_path)
    out = gates_mod.evaluate_multi_gates()
    assert out["summary"]["unwired_count"] >= 3
    assert out["summary"]["release_ready"] is False
    types = [g["gate_type"] for g in out["gates"]]
    assert "scenario" in types and "safety" in types
    skeleton = [g for g in out["gates"] if g["gate_type"] == "safety"][0]
    assert skeleton["ready"] is False
    assert skeleton["passed"] is None


def test_gate_config_thresholds_are_config_driven():
    assert "map_3d" in DEFAULT_GATE_CONFIG["gates"]["quality"]["thresholds"]
    assert DEFAULT_GATE_CONFIG["gates"]["safety"]["ready"] is False


# ------------------------------------------------------------------ evidence


def test_evidence_package_stub(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "sensorflow.platform.evidence.EVIDENCE_DIR", tmp_path
    )
    pkg = build_evidence_package(evaluation_id="eval-test")
    assert isinstance(pkg, EvidencePackage)
    assert pkg.evaluation_id == "eval-test"
    assert "model_versions" in pkg.placeholders or pkg.model_versions
    path = export_evidence_package(pkg, tmp_path / f"{pkg.package_id}.json")
    assert path.exists()
    raw = json.loads(path.read_text())
    assert raw["evaluation_id"] == "eval-test"
    assert "gate_results" in raw
    assert "compute_usage" in raw
    assert "provenance" in raw

    # append-only: second export with same path gets a revision file
    path2 = export_evidence_package(pkg, path)
    assert path2 != path
    assert path2.exists()


# ------------------------------------------------------------------ compare (unit, no mega store)


def test_compare_models_requires_two_runs():
    with pytest.raises(ValueError):
        compare_models(["only-one"])


class _FakeRun:
    def __init__(self, run_id, model_version, headline, status="published"):
        self.run_id = run_id
        self.model_version = model_version
        self.headline = headline
        self.status = status
        self.population_id = "pop-1"
        self.label_version = "labels-v1"


class _FakeStore:
    def __init__(self, runs):
        self.runs = {r.run_id: r for r in runs}

    def artifacts(self, run_id):
        raise NotImplementedError


def test_compare_models_matrix(monkeypatch):
    a = _FakeRun("run-a", "model-A", {
        "precision": 0.9, "recall": 0.8, "f1": 0.85,
        "mean_iou": 0.7, "safety_recall": 0.75, "anomaly_rate": 0.1,
    })
    b = _FakeRun("run-b", "model-B", {
        "precision": 0.88, "recall": 0.82, "f1": 0.85,
        "mean_iou": 0.71, "safety_recall": 0.74, "anomaly_rate": 0.09,
    })
    c = _FakeRun("run-c", "model-C", {
        "precision": 0.91, "recall": 0.79, "f1": 0.845,
        "mean_iou": 0.69, "safety_recall": 0.78, "anomaly_rate": 0.11,
    })
    store = _FakeStore([a, b, c])

    def fake_compare(store, cand, base, policy=None):
        return {
            "recommendation": "PROMOTE",
            "blockers": [],
            "headline_deltas": [],
            "per_class": [],
            "regressions": [],
        }

    monkeypatch.setattr("sensorflow.megaeval.analysis.compare_runs", fake_compare)
    out = compare_models(
        ["run-a", "run-b", "run-c"],
        baseline_run_id="run-a",
        store=store,
    )
    assert len(out["models"]) == 3
    assert len(out["pairwise"]) == 2
    assert len(out["metric_matrix"]) >= 3
    recall_row = next(r for r in out["metric_matrix"] if r["metric"] == "recall")
    assert recall_row["model-B"] == 0.82
    assert recall_row["deltas_vs_baseline"]["model-B"] == pytest.approx(0.02)


# ------------------------------------------------------------------ API smoke (optional FastAPI)


def test_platform_router_mounts():
    from fastapi.testclient import TestClient
    from sensorflow.platform.api import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    levels = client.get("/api/evaluations/levels")
    assert levels.status_code == 200
    body = levels.json()
    assert "population" in body["levels"]
    assert body["backends"]["container"] == "megaeval"

    gates = client.get("/api/gates")
    assert gates.status_code == 200
    assert any(g["gate_type"] == "scenario" for g in gates.json()["gates"])

    status = client.get("/api/gates/status")
    assert status.status_code == 200
    assert status.json()["summary"]["release_ready"] is False

    evidence = client.post("/api/evaluations/evidence", json={
        "evaluation_id": "api-eval-1",
        "persist": False,
    })
    assert evidence.status_code == 200
    assert evidence.json()["package"]["evaluation_id"] == "api-eval-1"

    bad = client.post("/api/models/compare", json={"run_ids": ["x"]})
    assert bad.status_code == 422 or bad.status_code == 400
