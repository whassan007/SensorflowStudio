"""Layered release gating + Safety Evidence Package.

Industry concept: staged release gates for AV perception stacks, producing an
auditable evidence trail structured after three families of standards:

- ISO 26262 (functional safety): deterministic, repeatable verification —
  geometric plausibility checks, calibration validation, regression testing
  against a pinned baseline with pinned thresholds.
- ISO 21448 (SOTIF): performance-limitation risk — ODD coverage of the
  scenario space, under-covered cells, surrogate-safety (SSAM-style) analysis
  of near-miss severity.
- UL 4600 (safety case): the Release Readiness Gate compiles all gate
  evidence, methodology and lineage into a claim/evidence-structured Safety
  Evidence Package (JSON + rendered markdown).

IMPORTANT WORDING: the package SUPPORTS a safety case by organizing
verifiable evidence. It does NOT certify or demonstrate compliance with any
standard, and several inputs are simulated (marked per-record).

Gates (evaluated in order, block-on-fail; every gate always produces an
evidence record even when an earlier gate blocked):

1. scenario_quality  — geometric plausibility (evaluation.validation) +
                       multi-sensor calibration validation (safety.calibration)
2. coverage          — ODD combinatorial coverage thresholds (safety.odd)
3. regression        — model-vs-model compare (megaeval.analysis.compare_runs,
                       not duplicated here)
4. safety            — surrogate-safety: aggregate CSI must not increase vs
                       baseline beyond tolerance (safety.ssam_ext)
5. release_readiness — all of the above + compiles the Safety Evidence Package
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sensorflow.safety import calibration as calib_mod
from sensorflow.safety import odd as odd_mod
from sensorflow.safety import ssam_ext
from sensorflow.safety.store import read_json, write_json

DEFAULT_POLICY: Dict = {
    "scenario_quality": {
        "enabled": True,
        "min_geometric_pass_rate": 0.70,
        "geometric_sample_size": 24,
        "block_on_miscalibration": True,
    },
    "coverage": {
        "enabled": True,
        "dims": ["weather", "lighting", "road_type"],
        "min_samples": 50,
        "max_ci_width": 0.25,
        "min_coverage_rate": 0.60,
        "min_production_weighted_coverage": 0.80,
    },
    "regression": {
        "enabled": True,
        # thresholds delegated to megaeval.analysis.DEFAULT_PROMOTION_POLICY;
        # entries here override it (e.g. max_recall_drop)
        "overrides": {},
    },
    "safety": {
        "enabled": True,
        "max_csi_increase_ratio": 0.15,  # candidate CSI <= baseline * (1 + ratio)
        "max_csi_absolute": None,        # optional hard ceiling
    },
}

_POLICY_FILE = ("gate_policy.json",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def get_policy() -> Dict:
    stored = read_json(*_POLICY_FILE) or {}
    return _merge(DEFAULT_POLICY, stored)


def set_policy(overrides: Dict) -> Dict:
    stored = _merge(read_json(*_POLICY_FILE) or {}, overrides or {})
    write_json(stored, *_POLICY_FILE)
    return get_policy()


def _check(name: str, actual, threshold, passed: bool, direction: str = "") -> Dict:
    return {"check": name, "actual": actual, "threshold": threshold,
            "direction": direction, "passed": bool(passed)}


def _gate_record(gate: str, name: str, standard_refs: List[str], checks: List[Dict],
                 evidence: Dict, skipped: bool = False, notes: str = "") -> Dict:
    if skipped:
        status = "SKIPPED"
    else:
        status = "PASS" if all(c["passed"] for c in checks) else "BLOCK"
    return {"gate": gate, "name": name, "status": status,
            "standard_refs": standard_refs, "checks": checks,
            "evidence": evidence, "notes": notes, "evaluated_at": _now()}


# ------------------------------------------------------------------ gate 1


def scenario_quality_gate(policy: Dict, eval_store=None) -> Dict:
    """Geometric plausibility sample + calibration validation.

    Geometric part: validates a deterministic sample of auto-label annotations
    from the newest labeleval dataset (generating a small probe dataset if the
    store is empty) through the existing strict validation engine. Calibration
    part: latest calibration validation status, refreshed with a clean run if
    none exists.
    """
    p = policy["scenario_quality"]
    if not p.get("enabled", True):
        return _gate_record("scenario_quality", "Scenario Quality Gate",
                            ["ISO 26262-style deterministic verification"],
                            [], {}, skipped=True, notes="disabled by policy")

    from sensorflow.evaluation import synthetic
    from sensorflow.evaluation.records import get_store
    from sensorflow.evaluation.validation import validate_annotation

    store = eval_store or get_store()
    datasets = store.all("datasets")
    probe_generated = False
    if datasets:
        ds = max(datasets, key=lambda d: d.created_at)
    else:
        ds = synthetic.generate_dataset(store, name="gate-probe", num_sequences=2,
                                        frames_per_sequence=6, seed=101)
        synthetic.generate_labels(store, ds, seed_offset=1)
        probe_generated = True

    anns = sorted(store.where("annotations", dataset_id=ds.dataset_id),
                  key=lambda a: a.annotation_id)
    sample = anns[:max(1, int(p["geometric_sample_size"]))]
    frames = {f.frame_id: f for f in store.where("frames", dataset_id=ds.dataset_id)}
    points_cache: Dict[str, object] = {}
    n_pass = 0
    fail_hist: Dict[str, int] = {}
    for ann in sample:
        frame = frames.get(ann.frame_id)
        if frame is None:
            continue
        if ann.frame_id not in points_cache:
            points_cache[ann.frame_id] = synthetic.frame_points(store, frame)
        res = validate_annotation(store, ann, frame, points=points_cache[ann.frame_id])
        if res.passed:
            n_pass += 1
        else:
            for c in res.checks:
                if c.applicable and not c.passed:
                    fail_hist[c.gate] = fail_hist.get(c.gate, 0) + 1
    pass_rate = n_pass / max(len(sample), 1)

    calib = calib_mod.latest_status()
    if calib is None:
        calib = calib_mod.run_validation(mode="clean", persist=True)
    calib_ok = (calib["status"] != "MISCALIBRATED"
                if p["block_on_miscalibration"] else True)

    checks = [
        _check("geometric_pass_rate", round(pass_rate, 4),
               p["min_geometric_pass_rate"], pass_rate >= p["min_geometric_pass_rate"], ">="),
        _check("calibration_status", calib["status"], "not MISCALIBRATED", calib_ok, "=="),
    ]
    return _gate_record(
        "scenario_quality", "Scenario Quality Gate",
        ["ISO 26262-style deterministic verification (geometric plausibility, "
         "calibration validation)"],
        checks,
        evidence={
            "dataset_id": ds.dataset_id,
            "probe_dataset_generated": probe_generated,
            "annotations_sampled": len(sample),
            "annotations_passed": n_pass,
            "failing_check_histogram": fail_hist,
            "calibration": {"status": calib["status"], "metrics": calib.get("metrics"),
                            "mode": calib.get("mode"), "diagnosis": calib.get("diagnosis")},
        },
        notes="geometric checks run on the strict validation engine; calibration "
              "scenes are simulated (see safety.calibration)")


# ------------------------------------------------------------------ gate 2


def coverage_gate(mega_store, candidate, policy: Dict) -> Dict:
    p = policy["coverage"]
    if not p.get("enabled", True):
        return _gate_record("coverage", "Coverage Gate",
                            ["ISO 21448/SOTIF-style ODD coverage"],
                            [], {}, skipped=True, notes="disabled by policy")
    cov = odd_mod.coverage_for_run(
        mega_store, candidate, dims=p["dims"], min_samples=p["min_samples"],
        max_ci_width=p["max_ci_width"], max_gaps=10)
    s = cov["summary"]
    checks = [
        _check("coverage_rate", s["coverage_rate"], p["min_coverage_rate"],
               s["coverage_rate"] >= p["min_coverage_rate"], ">="),
        _check("production_weighted_coverage", s["production_weighted_coverage"],
               p["min_production_weighted_coverage"],
               s["production_weighted_coverage"] >= p["min_production_weighted_coverage"],
               ">="),
    ]
    return _gate_record(
        "coverage", "Coverage Gate",
        ["ISO 21448/SOTIF-style performance-limitation risk (ODD coverage per "
         "ISO 34503-inspired taxonomy)"],
        checks,
        evidence={"dims": cov["dims"], "thresholds": cov["thresholds"],
                  "summary": s, "top_gaps": cov["gaps"][:5],
                  "fill_requests_available": len(cov["fill_requests"])},
        notes="coverage over exact metric-cube counts; production shares use a "
              "marginal-independence assumption")


# ------------------------------------------------------------------ gate 3


def regression_gate(mega_store, candidate, baseline, policy: Dict) -> Dict:
    p = policy["regression"]
    if not p.get("enabled", True):
        return _gate_record("regression", "Regression Gate",
                            ["ISO 26262-style regression verification"],
                            [], {}, skipped=True, notes="disabled by policy")
    from sensorflow.megaeval.analysis import compare_runs
    cmp = compare_runs(mega_store, candidate, baseline, policy=p.get("overrides") or None)
    ok = cmp["recommendation"] == "PROMOTE"
    checks = [_check("compare_recommendation", cmp["recommendation"], "PROMOTE", ok, "==")]
    return _gate_record(
        "regression", "Regression Gate",
        ["ISO 26262-style deterministic regression testing vs pinned baseline"],
        checks,
        evidence={"headline_deltas": cmp["headline_deltas"],
                  "regressions": cmp["regressions"],
                  "blockers": cmp["blockers"],
                  "policy": cmp["policy"],
                  "worst_cohorts": cmp["worst_cohorts"][:5]},
        notes="delegates to megaeval model-vs-model compare (single source of truth)")


# ------------------------------------------------------------------ gate 4


def safety_gate(candidate, baseline, policy: Dict) -> Dict:
    p = policy["safety"]
    if not p.get("enabled", True):
        return _gate_record("safety", "Safety Gate",
                            ["ISO 21448/SOTIF-style surrogate safety"],
                            [], {}, skipped=True, notes="disabled by policy")
    cand = ssam_ext.csi_for_run(candidate)
    base = ssam_ext.csi_for_run(baseline)
    cand_csi, base_csi = cand["aggregate_csi"], base["aggregate_csi"]
    limit = base_csi * (1.0 + p["max_csi_increase_ratio"])
    increase_ratio = ((cand_csi - base_csi) / base_csi) if base_csi > 1e-9 else (
        0.0 if cand_csi <= 1e-9 else float("inf"))
    checks = [
        _check("aggregate_csi_vs_baseline", round(cand_csi, 4),
               round(limit, 4), cand_csi <= limit + 1e-9, "<="),
    ]
    if p.get("max_csi_absolute") is not None:
        checks.append(_check("aggregate_csi_absolute", round(cand_csi, 4),
                             p["max_csi_absolute"], cand_csi <= p["max_csi_absolute"], "<="))
    return _gate_record(
        "safety", "Safety Gate",
        ["ISO 21448/SOTIF-style surrogate safety (FHWA SSAM: TTC/PET/DRAC/CSI)"],
        checks,
        evidence={
            "candidate_csi": cand_csi, "baseline_csi": base_csi,
            "csi_increase_ratio": (round(increase_ratio, 4)
                                   if increase_ratio != float("inf") else "inf"),
            "candidate_reaction_model": cand["reaction_model"],
            "baseline_reaction_model": base["reaction_model"],
            "candidate_min_ttc_s": cand["min_ttc_s"],
            "candidate_scenarios": cand["scenarios"],
        },
        notes="CSI from a deterministic simulated scenario suite conditioned on the "
              "model's perception profile (marked simulated); the SSAM formulas "
              "themselves are real math")


# ------------------------------------------------------------------ orchestration


GATE_ORDER = ["scenario_quality", "coverage", "regression", "safety"]


def evaluate_gates(mega_store, candidate, baseline,
                   policy_overrides: Optional[Dict] = None, eval_store=None) -> Dict:
    """Run all gates for candidate vs baseline; compile + persist the Safety
    Evidence Package. Layered semantics: overall decision blocks on the first
    failing gate, but every gate is still evaluated for evidence."""
    policy = _merge(get_policy(), policy_overrides or {})
    gates = [
        scenario_quality_gate(policy, eval_store=eval_store),
        coverage_gate(mega_store, candidate, policy),
        regression_gate(mega_store, candidate, baseline, policy),
        safety_gate(candidate, baseline, policy),
    ]
    blocking = [g["gate"] for g in gates if g["status"] == "BLOCK"]
    release_ready = not blocking
    readiness = _gate_record(
        "release_readiness", "Release Readiness Gate",
        ["UL 4600-style safety-case evidence compilation"],
        [_check("all_gates_pass", f"{len(gates) - len(blocking)}/{len(gates)}",
                f"{len(gates)}/{len(gates)}", release_ready, "==")],
        evidence={"blocking_gates": blocking,
                  "first_blocking_gate": blocking[0] if blocking else None},
        notes="compiles the Safety Evidence Package below")
    gates.append(readiness)

    package = build_evidence_package(mega_store, candidate, baseline, policy, gates,
                                     release_ready, blocking)
    write_json(package, "evidence", f"{candidate.run_id}.json")
    from sensorflow.safety.store import safety_path
    md = render_markdown(package)
    with open(safety_path("evidence", f"{candidate.run_id}.md"), "w") as f:
        f.write(md)

    result = {
        "candidate_run_id": candidate.run_id,
        "baseline_run_id": baseline.run_id,
        "decision": "RELEASE_READY" if release_ready else "BLOCKED",
        "blocking_gates": blocking,
        "gates": gates,
        "policy": policy,
        "evidence_package_id": package["package_id"],
        "evaluated_at": _now(),
    }
    write_json(result, "gates", f"{candidate.run_id}.json")
    return result


def latest_gate_result(run_id: str) -> Optional[Dict]:
    return read_json("gates", f"{run_id}.json")


def load_evidence(run_id: str) -> Optional[Dict]:
    return read_json("evidence", f"{run_id}.json")


# ------------------------------------------------------------------ evidence package


DISCLAIMER = (
    "This package organizes verifiable evidence to SUPPORT a safety case. It does "
    "not certify, demonstrate or imply compliance with ISO 26262, ISO 21448, "
    "UL 4600 or any other standard. Several evidence sources are simulated and "
    "are marked as such on the individual records.")

STANDARD_MAPPINGS = [
    {"standard": "ISO 26262 (functional safety)",
     "style": "deterministic, repeatable verification with pinned thresholds",
     "evidence": ["gates.scenario_quality (geometric plausibility, calibration)",
                  "gates.regression (model-vs-model compare vs pinned baseline)",
                  "lineage (pinned dataset/model/label/metric/code versions + seed)"]},
    {"standard": "ISO 21448 / SOTIF (safety of the intended functionality)",
     "style": "performance-limitation risk in the operational domain",
     "evidence": ["gates.coverage + odd_coverage_summary (under-covered ODD cells, "
                  "risk-ranked gaps)",
                  "gates.safety + surrogate_safety_analysis (SSAM-style near-miss "
                  "severity, CSI vs baseline)"]},
    {"standard": "UL 4600 (safety case for autonomous products)",
     "style": "claim -> argument -> evidence structure",
     "evidence": ["decision + gates[] (claims with thresholds, actuals and "
                  "pass/block outcomes)",
                  "sampling_methodology (statistical argument: stratified "
                  "risk-weighted review, Wilson CIs)",
                  "the package itself as the evidence index"]},
]


def build_evidence_package(mega_store, candidate, baseline, policy: Dict,
                           gates: List[Dict], release_ready: bool,
                           blocking: List[str]) -> Dict:
    cov_gate = next(g for g in gates if g["gate"] == "coverage")
    reg_gate = next(g for g in gates if g["gate"] == "regression")
    saf_gate = next(g for g in gates if g["gate"] == "safety")

    review = None
    try:
        review = mega_store.artifacts(candidate.run_id).get("review")
    except Exception:
        pass

    supplementary: Dict = {}
    try:
        from sensorflow.safety.scenario_db import get_db
        supplementary["scenario_db"] = get_db().counts()
    except Exception:
        pass
    try:
        cal = calib_mod.latest_status()
        if cal:
            supplementary["calibration_status"] = {
                "status": cal["status"], "metrics": cal.get("metrics")}
    except Exception:
        pass

    return {
        "package_id": f"sep-{candidate.run_id}",
        "package_format": "sensorflow-safety-evidence/v1",
        "generated_at": _now(),
        "disclaimer": DISCLAIMER,
        "candidate": {"run_id": candidate.run_id, "model_version": candidate.model_version,
                      "population_id": candidate.population_id,
                      "headline": candidate.headline},
        "baseline": {"run_id": baseline.run_id, "model_version": baseline.model_version,
                     "headline": baseline.headline},
        "decision": {"release_ready": release_ready, "blocking_gates": blocking},
        "gates": gates,
        "gate_policy": policy,
        "odd_coverage_summary": cov_gate["evidence"],
        "regression_analysis": reg_gate["evidence"],
        "surrogate_safety_analysis": saf_gate["evidence"],
        "sampling_methodology": {
            "config": candidate.lineage.get("sampling_config"),
            "confidence_intervals": "Wilson score intervals on per-cell and "
                                    "review-sample proportions",
            "review_artifact": ({"target_n": review.get("target_n"),
                                 "achieved_n": review.get("achieved_n")}
                                if isinstance(review, dict) else None),
        },
        "lineage": {"candidate": candidate.lineage, "baseline": baseline.lineage},
        "standard_mappings": STANDARD_MAPPINGS,
        "supplementary": supplementary,
    }


def render_markdown(pkg: Dict) -> str:
    d = pkg["decision"]
    lines = [
        f"# Safety Evidence Package — {pkg['candidate']['model_version']}",
        "",
        f"Package `{pkg['package_id']}` generated {pkg['generated_at']}",
        "",
        f"> {pkg['disclaimer']}",
        "",
        f"**Decision: {'RELEASE READY' if d['release_ready'] else 'BLOCKED'}**"
        + (f" — blocking gates: {', '.join(d['blocking_gates'])}" if d["blocking_gates"] else ""),
        "",
        f"Candidate `{pkg['candidate']['run_id']}` ({pkg['candidate']['model_version']}) "
        f"vs baseline `{pkg['baseline']['run_id']}` ({pkg['baseline']['model_version']})",
        "",
        "## Gate results",
        "",
        "| Gate | Status | Checks (actual vs threshold) |",
        "|---|---|---|",
    ]
    for g in pkg["gates"]:
        checks = "; ".join(
            f"{c['check']}: {c['actual']} {c['direction'] or 'vs'} {c['threshold']} "
            f"({'ok' if c['passed'] else 'FAIL'})" for c in g["checks"]) or "—"
        lines.append(f"| {g['name']} | {g['status']} | {checks} |")

    s = pkg["odd_coverage_summary"].get("summary", {})
    lines += [
        "",
        "## ODD coverage (SOTIF-style)",
        "",
        f"- dims: {', '.join(pkg['odd_coverage_summary'].get('dims', []))}",
        f"- adequate cells: {s.get('adequate_cells')}/{s.get('total_cells')} "
        f"(coverage rate {s.get('coverage_rate')})",
        f"- production-weighted coverage: {s.get('production_weighted_coverage')}",
        f"- gap cells: {s.get('gap_cells')}",
        "",
        "## Surrogate safety (SSAM-style)",
        "",
        f"- candidate aggregate CSI: {pkg['surrogate_safety_analysis'].get('candidate_csi')}",
        f"- baseline aggregate CSI: {pkg['surrogate_safety_analysis'].get('baseline_csi')}",
        f"- increase ratio: {pkg['surrogate_safety_analysis'].get('csi_increase_ratio')}",
        f"- candidate min TTC: {pkg['surrogate_safety_analysis'].get('candidate_min_ttc_s')} s",
        "",
        "## Regression analysis",
        "",
    ]
    for h in pkg["regression_analysis"].get("headline_deltas", []):
        lines.append(f"- {h['metric']}: {h['baseline']} -> {h['candidate']} (Δ {h['delta']:+})")
    blockers = pkg["regression_analysis"].get("blockers", [])
    if blockers:
        lines += ["", "Blockers:"] + [f"- {b}" for b in blockers]

    lines += [
        "",
        "## Methodology",
        "",
        f"- sampling: {pkg['sampling_methodology'].get('config')}",
        f"- CIs: {pkg['sampling_methodology'].get('confidence_intervals')}",
        "",
        "## Standard mappings (evidence organization, not compliance)",
        "",
    ]
    for m in pkg["standard_mappings"]:
        lines.append(f"### {m['standard']}")
        lines.append(f"_{m['style']}_")
        lines += [f"- {e}" for e in m["evidence"]]
        lines.append("")

    lines += [
        "## Lineage",
        "",
        "```json",
        __import__("json").dumps(pkg["lineage"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)
