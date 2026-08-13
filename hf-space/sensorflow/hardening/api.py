"""/api/hardening — audit browser, readiness scorecard, funnel, demos.

Read-only over other packages' data (funnel counts come from the existing
runs/ stores). Demo endpoints run the hardening primitives on SEEDED
SYNTHETIC fixtures and label them as such.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from sensorflow.hardening import hitl, power, quality, readiness, safety_config, sampling
from sensorflow.hardening.interfaces import describe_implementations

router = APIRouter(prefix="/api/hardening", tags=["hardening"])

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "hardening"
_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


# ------------------------------------------------------------------ audit & readiness


@router.get("/audit")
def get_audit() -> Dict:
    path = _DOCS_DIR / "audit.json"
    if not path.exists():
        raise HTTPException(404, "audit.json not found")
    return json.loads(path.read_text())


@router.get("/audit.md", response_class=PlainTextResponse)
def get_audit_markdown() -> str:
    path = _DOCS_DIR / "audit.md"
    if not path.exists():
        raise HTTPException(404, "audit.md not found")
    return path.read_text()


@router.get("/readiness")
def get_readiness() -> Dict:
    try:
        return readiness.scorecard()
    except FileNotFoundError:
        raise HTTPException(404, "audit.json not found; run the audit first")


@router.get("/thresholds")
def get_thresholds() -> Dict:
    return {
        "config_version": safety_config.CONFIG_VERSION,
        "thresholds": safety_config.registry_summary(),
    }


@router.get("/interfaces")
def get_interfaces() -> Dict:
    return {"implementations": describe_implementations()}


# ------------------------------------------------------------------ funnel / observability


@router.get("/funnel")
def get_funnel() -> Dict:
    """Data-funnel counts pulled from the existing labeleval store (read-only).

    Stages: frames -> annotations -> validated -> triaged (by outcome) ->
    review tasks -> human reviews. Counts are from the local prototype store;
    the `store` field says exactly where they came from.
    """
    store_path = _RUNS_DIR / "labeleval" / "store.json"
    if not store_path.exists():
        return {"available": False, "store": str(store_path),
                "note": "labeleval store not present; run the labeleval pipeline"}
    data = json.loads(store_path.read_text())

    def count(kind: str) -> int:
        v = data.get(kind, {})
        return len(v)

    triage_statuses: Counter = Counter()
    for rec in (data.get("triage_decisions") or {}).values():
        triage_statuses[str(rec.get("status", rec.get("decision", "unknown")))] += 1

    review_statuses: Counter = Counter()
    for rec in (data.get("review_tasks") or {}).values():
        review_statuses[str(rec.get("status", "unknown"))] += 1

    return {
        "available": True,
        "store": "runs/labeleval/store.json (LOCAL prototype store)",
        "stages": [
            {"stage": "frames", "count": count("frames")},
            {"stage": "annotations", "count": count("annotations")},
            {"stage": "validations", "count": count("validations")},
            {"stage": "anomaly_checks", "count": count("anomalies")},
            {"stage": "grader_comparisons", "count": count("grader_comparisons")},
            {"stage": "triage_decisions", "count": count("triage_decisions")},
            {"stage": "review_tasks", "count": count("review_tasks")},
            {"stage": "human_reviews", "count": count("human_reviews")},
        ],
        "triage_breakdown": dict(triage_statuses),
        "review_breakdown": dict(review_statuses),
        "alerts": count("alerts"),
        "audit_events": count("audit_events"),
    }


# ------------------------------------------------------------------ demos (seeded synthetic)


@router.get("/demo/sampling")
def demo_sampling(seed: int = 7, budget: int = 60) -> Dict:
    """Stratified sampling + HT reweighting on a SEEDED SYNTHETIC population."""
    rng = np.random.default_rng(seed)
    pop_sizes = {
        sampling.Stratum.REPRESENTATIVE: 500,
        sampling.Stratum.NOMINAL: 300,
        sampling.Stratum.RARE: 60,
        sampling.Stratum.SAFETY_CRITICAL: 40,
        sampling.Stratum.KNOWN_FAILURES: 25,
        sampling.Stratum.NOVEL: 50,
        sampling.Stratum.DISTRIBUTION_SHIFTED: 25,
    }
    # Per-stratum true failure rates: rare/safety strata fail far more often.
    true_rates = {
        sampling.Stratum.REPRESENTATIVE: 0.02,
        sampling.Stratum.NOMINAL: 0.03,
        sampling.Stratum.RARE: 0.20,
        sampling.Stratum.SAFETY_CRITICAL: 0.30,
        sampling.Stratum.KNOWN_FAILURES: 0.60,
        sampling.Stratum.NOVEL: 0.15,
        sampling.Stratum.DISTRIBUTION_SHIFTED: 0.25,
    }
    population = {s: [f"{s.value}-{i}" for i in range(n)] for s, n in pop_sizes.items()}
    failures = {}
    for s, ids in population.items():
        fails = rng.random(len(ids)) < true_rates[s]
        for item_id, f in zip(ids, fails):
            failures[item_id] = float(f)

    sample = sampling.stratified_sample(population, budget=budget, seed=seed)
    measured = {it.item_id: failures[it.item_id] for it in sample.items}
    ht_estimate = sample.estimate_population_rate(measured)
    naive_estimate = float(np.mean(list(measured.values()))) if measured else None
    total = sum(pop_sizes.values())
    true_rate = sum(pop_sizes[s] * true_rates[s] for s in pop_sizes) / total

    return {
        "simulated": True,
        "seed": seed,
        "quotas": {s.value: n for s, n in sample.quotas.items()},
        "sampled": len(sample.items),
        "true_population_rate": round(true_rate, 4),
        "ht_reweighted_estimate": round(ht_estimate, 4) if ht_estimate is not None else None,
        "naive_sample_mean": round(naive_estimate, 4) if naive_estimate is not None else None,
        "note": "The naive mean is inflated by oversampling rare/safety strata; "
                "the stratum-weighted estimate recovers the population rate.",
    }


@router.get("/demo/quality")
def demo_quality() -> Dict:
    """Explainable routing on fixed synthetic fixtures (incl. required cases)."""
    fixtures = {
        "clean_vehicle": quality.QualityEvidence(
            geometric_validity=0.95, sensor_agreement=0.9, temporal_consistency=0.92,
            semantic_agreement=0.9, confidence=0.88, uncertainty=0.1,
            occlusion=0.1, safety_relevance=0.1),
        "vru_miss_in_boundary": quality.QualityEvidence(
            geometric_validity=0.6, sensor_agreement=0.25, temporal_consistency=0.5,
            semantic_agreement=0.4, confidence=0.3, uncertainty=0.8,
            occlusion=0.6, safety_relevance=0.95),
        "phantom_obstacle": quality.QualityEvidence(
            geometric_validity=0.8, sensor_agreement=0.15, temporal_consistency=0.3,
            semantic_agreement=0.5, confidence=0.7, uncertainty=0.6,
            occlusion=0.2, safety_relevance=0.7),
        "incomplete_telemetry": quality.QualityEvidence(
            geometric_validity=0.9, sensor_agreement=None, temporal_consistency=None,
            semantic_agreement=0.85, confidence=0.9, uncertainty=0.2,
            occlusion=0.1, safety_relevance=0.2),
    }
    routed = {name: quality.route(ev) for name, ev in fixtures.items()}

    profiles = [
        quality.GraderProfile(grader_id="g1", backbone="vit-l14", training_data="webds-3b"),
        quality.GraderProfile(grader_id="g2", backbone="vit-l14", training_data="webds-3b"),
        quality.GraderProfile(grader_id="g3", backbone="convnext", training_data="internal-av"),
    ]
    consensus = quality.consensus_with_dependence(
        {"g1": "pedestrian", "g2": "pedestrian", "g3": "pedestrian"}, profiles)

    return {"simulated": True, "routing": {
        name: {"decision": r["decision"].value, "explanation": r["explanation"]}
        for name, r in routed.items()
    }, "grader_dependence_example": consensus}


@router.get("/demo/hitl")
def demo_hitl(budget: int = 5) -> Dict:
    """HITL prioritization comparison on fixed synthetic candidates."""
    candidates = [
        hitl.ReviewCandidate(item_id="max_risk_certain", risk=0.99, uncertainty=0.05,
                             novelty=0.1, training_value=0.2),
        hitl.ReviewCandidate(item_id="balanced_high", risk=0.7, uncertainty=0.7,
                             novelty=0.6, training_value=0.7),
        hitl.ReviewCandidate(item_id="novel_uncertain", risk=0.3, uncertainty=0.8,
                             novelty=0.95, training_value=0.8),
        hitl.ReviewCandidate(item_id="routine", risk=0.2, uncertainty=0.2,
                             novelty=0.1, training_value=0.2),
        hitl.ReviewCandidate(item_id="risky_uncertain", risk=0.9, uncertainty=0.85,
                             novelty=0.3, training_value=0.5),
        hitl.ReviewCandidate(item_id="training_gold", risk=0.4, uncertainty=0.6,
                             novelty=0.7, training_value=0.95),
    ]
    ranked = hitl.prioritize(candidates, budget=budget)
    product_only = sorted(candidates, key=lambda c: -hitl.information_gain_score(c))
    metrics = hitl.acceptance_metrics(
        routed_ids=[r["item_id"] for r in ranked],
        true_problem_ids=["balanced_high", "risky_uncertain", "max_risk_certain",
                          "novel_uncertain"],
        critical_ids=["max_risk_certain", "risky_uncertain"],
        total_items=len(candidates),
    )
    return {
        "simulated": True,
        "pareto_then_product": ranked,
        "product_only_order": [c.item_id for c in product_only],
        "comparison_note": "Pareto ranking keeps max_risk_certain (extreme on one "
                           "axis) on the first front; the pure product buries it.",
        "acceptance_metrics": metrics,
    }


@router.get("/demo/power")
def demo_power() -> Dict:
    """Tier sizing derived from decision parameters (no fixed counts)."""
    return {
        "tiers": [power.tier_sizing(spec) for spec in power.default_tiers().values()],
        "example_rare_event": power.required_events_rare(
            prevalence=0.01, baseline_rate=0.95, mde=0.02),
    }


@router.get("/summary")
def get_summary() -> Dict:
    """One-call overview for the frontend page."""
    audit = get_audit()
    return {
        "summary": audit.get("summary", {}),
        "strengths": audit.get("strengths", []),
        "readiness": readiness.scorecard(audit),
    }
