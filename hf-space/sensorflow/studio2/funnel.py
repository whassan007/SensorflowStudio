"""Unified observability funnel over the real stores.

One view of the whole system: the evaluation funnel (raw → selected →
simulated → evaluated → failed → HITL → regression suite), the safety panel
(SCR, VRU miss rate, TTC distribution), model comparison, drift, and compute.

Everything is aggregated best-effort per source package with an availability
flag. Absent sources report available=False with the reason — numbers are
NEVER fabricated. Reads go through the packages' persisted stores (plain
JSON/filesystem) so an unimportable in-flight package does not break the
funnel for the landed ones.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from sensorflow.studio2 import store
from sensorflow.studio2.registry import Registry, get_registry


def _panel(available: bool, source: str, data: Optional[Dict] = None,
           reason: str = "") -> Dict:
    out = {"available": available, "source": source}
    if available:
        out["data"] = data or {}
    else:
        out["status"] = "UNAVAILABLE"
        out["reason"] = reason
    return out


def _safe_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _mega_states(repo_root: str) -> List[Dict]:
    base = os.path.join(repo_root, "runs", "megaeval", "runs")
    out = []
    if os.path.isdir(base):
        for rid in sorted(os.listdir(base)):
            d = _safe_json(os.path.join(base, rid, "run.json"))
            state = (d or {}).get("state")
            if state:
                out.append(state)
    return out


# ------------------------------------------------------------------ stages


def _stage_raw(repo_root: str) -> Dict:
    base = os.path.join(repo_root, "runs", "megaeval", "populations")
    total, pops = 0, 0
    if os.path.isdir(base):
        for pid in os.listdir(base):
            meta = _safe_json(os.path.join(base, pid, "meta.json"))
            if meta:
                pops += 1
                total += int(meta.get("num_objects") or 0)
    if pops == 0:
        return _panel(False, "megaeval.populations", reason="no populations built")
    return _panel(True, "megaeval.populations",
                  {"populations": pops, "objects": total})


def _stage_selected(repo_root: str) -> Dict:
    base = os.path.join(repo_root, "runs", "seqeval", "runs")
    allocated, runs = 0, 0
    if os.path.isdir(base):
        for rid in sorted(os.listdir(base)):
            st = _safe_json(os.path.join(base, rid, "run.json"))
            if not st:
                continue
            plan = st.get("plan") or {}
            if plan.get("total_allocated"):
                runs += 1
                allocated += int(plan["total_allocated"])
    if runs == 0:
        return _panel(False, "seqeval.plans", reason="no sequential plans frozen")
    return _panel(True, "seqeval.plans",
                  {"plans": runs, "objects_allocated": allocated})


def _stage_simulated(repo_root: str) -> Dict:
    # nextgen counterfactual scenarios (in-flight package; read its store)
    candidates = [os.path.join(repo_root, "runs", "nextgen", "counterfactuals"),
                  os.path.join(repo_root, "runs", "nextgen", "scenarios")]
    n = 0
    found_dir = None
    for base in candidates:
        if os.path.isdir(base):
            found_dir = base
            n += sum(1 for f in os.listdir(base) if f.endswith(".json"))
    if found_dir is None:
        try:
            from sensorflow.nextgen import counterfactual as cf
            n = len(cf.list_scenarios())
            found_dir = "nextgen.counterfactual.list_scenarios"
        except Exception:
            return _panel(False, "nextgen.counterfactuals",
                          reason="nextgen counterfactual store absent "
                                 "(in-flight package)")
    return _panel(True, str(found_dir), {"counterfactual_scenarios": n})


def _stage_evaluated(repo_root: str) -> Dict:
    states = _mega_states(repo_root)
    published = [s for s in states if s.get("status") == "published"]
    if not published:
        return _panel(False, "megaeval.runs", reason="no published evaluation runs")
    return _panel(True, "megaeval.runs", {
        "runs": len(published),
        "objects_evaluated": sum(int(s.get("objects_processed") or 0)
                                 for s in published)})


def _stage_failed(repo_root: str) -> Dict:
    states = _mega_states(repo_root)
    published = [s for s in states if s.get("status") == "published"]
    fp = sum(int((s.get("headline") or {}).get("fp") or 0) for s in published)
    fn = sum(int((s.get("headline") or {}).get("fn") or 0) for s in published)
    agentic_dir = os.path.join(repo_root, "runs", "agentic", "failures")
    agentic_failures = (len([f for f in os.listdir(agentic_dir) if f.endswith(".json")])
                        if os.path.isdir(agentic_dir) else None)
    if not published and agentic_failures is None:
        return _panel(False, "megaeval.errors + agentic.failures",
                      reason="no evaluation errors recorded")
    return _panel(True, "megaeval.errors + agentic.failures",
                  {"false_positives": fp, "false_negatives": fn,
                   "agentic_failure_events": agentic_failures})


def _stage_hitl(repo_root: str) -> Dict:
    try:
        from sensorflow.evaluation.records import get_store
        st = get_store()
        tasks = st.all("review_tasks")
        reviews = st.all("human_reviews")
        return _panel(True, "evaluation.review_queue",
                      {"review_tasks": len(tasks),
                       "completed_reviews": len(reviews),
                       "open_tasks": sum(1 for t in tasks
                                         if getattr(t, "status", "") not in
                                         ("resolved", "completed"))})
    except Exception as e:
        return _panel(False, "evaluation.review_queue",
                      reason=f"labeleval store unavailable: {type(e).__name__}")


def _stage_regression(repo_root: str, registry: Registry) -> Dict:
    suites_dir = os.path.join(repo_root, "runs", "agentic", "suites")
    suites = (len([f for f in os.listdir(suites_dir) if f.endswith(".json")])
              if os.path.isdir(suites_dir) else 0)
    reg_datasets = [d for d in registry.list("datasets")
                    if d.get("role") == "REGRESSION"]
    if suites == 0 and not reg_datasets:
        return _panel(False, "agentic.suites + studio2.registry",
                      reason="no regression suites registered yet")
    return _panel(True, "agentic.suites + studio2.registry",
                  {"agentic_evaluation_suites": suites,
                   "regression_role_datasets": len(reg_datasets)})


# ------------------------------------------------------------------ panels


def _safety_panel(repo_root: str) -> Dict:
    data: Dict = {}
    sources: List[str] = []

    # SCR / risk-weighted metrics need a nextgen safety report (in-flight)
    try:
        from sensorflow.nextgen.safety_metrics import divergence_demo
        rep = divergence_demo()
        base = ((rep.get("baseline") or {}).get("safety_informed") or {})
        cand = ((rep.get("candidate") or {}).get("safety_informed") or {})
        data["scr"] = {
            "safety_critical_recall": {
                "baseline": base.get("safety_critical_recall"),
                "candidate": cand.get("safety_critical_recall")},
            "risk_weighted_recall": {
                "baseline": base.get("risk_weighted_recall"),
                "candidate": cand.get("risk_weighted_recall")},
            "deltas": rep.get("deltas"),
            "note": "nextgen safety-region analysis (deterministic demo scene, "
                    "data_label SIMULATED)"}
        sources.append("nextgen.safety_metrics")
    except Exception:
        data["scr"] = {"status": "UNAVAILABLE",
                       "reason": "nextgen safety metrics not importable (in-flight)"}

    # VRU miss rate from the real megaeval per-class recall
    states = [s for s in _mega_states(repo_root) if s.get("status") == "published"]
    if states:
        latest = max(states, key=lambda s: s.get("published_at") or "")
        vru = {}
        for cls in ("pedestrian", "cyclist"):
            pc = (latest.get("per_class") or {}).get(cls) or {}
            if pc.get("recall") is not None:
                vru[cls] = {"miss_rate": round(1.0 - pc["recall"], 4),
                            "n": pc.get("n")}
        if vru:
            data["vru_miss_rate"] = {"run_id": latest.get("run_id"), **vru}
            sources.append("megaeval.per_class")
    if "vru_miss_rate" not in data:
        data["vru_miss_rate"] = {"status": "UNAVAILABLE",
                                 "reason": "no published megaeval run with per-class stats"}

    # TTC distribution from the safety SSAM store
    ssam_dir = os.path.join(repo_root, "runs", "safety", "ssam")
    ttcs: List[float] = []
    if os.path.isdir(ssam_dir):
        for name in os.listdir(ssam_dir):
            doc = _safe_json(os.path.join(ssam_dir, name))
            if not doc:
                continue
            for sc in doc.get("scenarios") or []:
                v = sc.get("min_ttc_s")
                if isinstance(v, (int, float)):
                    ttcs.append(float(v))
    if ttcs:
        ttcs.sort()
        data["ttc"] = {"n_scenarios": len(ttcs), "min": round(ttcs[0], 3),
                       "p50": round(ttcs[len(ttcs) // 2], 3),
                       "max": round(ttcs[-1], 3), "values": ttcs[:50]}
        sources.append("safety.ssam")
    else:
        data["ttc"] = {"status": "UNAVAILABLE",
                       "reason": "no SSAM analyses persisted under runs/safety/ssam"}

    available = any(k for k in ("scr", "vru_miss_rate", "ttc")
                    if "status" not in data[k])
    return _panel(available, " + ".join(sources) or "safety sources", data,
                  reason="no safety metric source available")


def _comparison_panel(repo_root: str) -> Dict:
    bev = _safe_json(os.path.join(repo_root, "runs", "bevfusion", "latest.json"))
    if bev:
        return _panel(True, "bevfusion.latest", {
            "baseline": (bev.get("engines") or {}).get("baseline"),
            "candidate": (bev.get("engines") or {}).get("candidate"),
            "headline_deltas": bev.get("headline_deltas"),
            "recommendation": bev.get("recommendation"),
            "blockers": bev.get("blockers")})
    states = [s for s in _mega_states(repo_root) if s.get("status") == "published"]
    if len(states) >= 2:
        states.sort(key=lambda s: s.get("published_at") or "")
        b, c = states[-2], states[-1]
        deltas = []
        for m in ("precision", "recall", "safety_recall", "mean_iou"):
            bv, cv = (b.get("headline") or {}).get(m), (c.get("headline") or {}).get(m)
            if bv is not None and cv is not None:
                deltas.append({"metric": m, "baseline": round(bv, 4),
                               "candidate": round(cv, 4),
                               "delta": round(cv - bv, 4)})
        return _panel(True, "megaeval.headlines",
                      {"baseline": b.get("model_version"),
                       "candidate": c.get("model_version"),
                       "headline_deltas": deltas})
    return _panel(False, "bevfusion + megaeval",
                  reason="fewer than two comparable runs")


def _drift_panel(repo_root: str) -> Dict:
    base = os.path.join(repo_root, "runs", "megaeval", "runs")
    if os.path.isdir(base):
        for rid in sorted(os.listdir(base), reverse=True):
            shift = _safe_json(os.path.join(base, rid, "shift.json"))
            if shift:
                worst = (shift.get("shifts") or [])[:5]
                return _panel(True, f"megaeval.shift ({rid})",
                              {"run_id": rid, "method": shift.get("method"),
                               "worst_shifts": worst,
                               "n_shifts": len(shift.get("shifts") or [])})
    return _panel(False, "megaeval.shift",
                  reason="no cached distribution-shift report; run "
                         "megaeval distribution_shift for a published run")


def _compute_panel(repo_root: str) -> Dict:
    data: Dict = {}
    sources: List[str] = []
    comp = _safe_json(os.path.join(repo_root, "runs", "nextgen", "compute",
                                   "latest.json"))
    if comp:
        data["dedup"] = {k: comp.get(k) for k in
                         ("cache_hit_rate", "hit_rate", "saved_units",
                          "baseline_units", "dedup_units", "speedup")
                         if comp.get(k) is not None} or comp
        sources.append("nextgen.compute")
    try:
        from sensorflow.nextgen.cache import get_feature_cache
        stats = get_feature_cache().stats
        data["feature_cache"] = {"hits": stats.hits, "misses": stats.misses,
                                 "hit_rate": round(stats.hits / max(1, stats.hits + stats.misses), 4)}
        sources.append("nextgen.cache")
    except Exception:
        pass
    gaunt = _safe_json(os.path.join(repo_root, "runs", "nextgen", "gauntlets",
                                    "latest.json"))
    if gaunt:
        elapsed = gaunt.get("elapsed_s") or gaunt.get("wall_s")
        units = gaunt.get("units_evaluated") or gaunt.get("samples_used")
        data["gauntlet"] = {"run_id": gaunt.get("run_id"),
                            "units_evaluated": units,
                            "elapsed_s": elapsed,
                            "scenarios_per_sec": (round(units / elapsed, 2)
                                                  if units and elapsed else None),
                            "halted": gaunt.get("halted")}
        sources.append("nextgen.gauntlets")
    # megaeval throughput is always real when runs exist
    states = [s for s in _mega_states(repo_root) if s.get("status") == "published"]
    if states:
        latest = max(states, key=lambda s: s.get("published_at") or "")
        data["megaeval_throughput"] = {
            "run_id": latest.get("run_id"),
            "objects_per_sec": latest.get("throughput_objs_per_s")}
        sources.append("megaeval.runs")
    if not data:
        return _panel(False, "nextgen + megaeval",
                      reason="no compute telemetry persisted")
    return _panel(True, " + ".join(sources), data)


# ------------------------------------------------------------------ assembly


def build_funnel(repo_root: str = ".", registry: Optional[Registry] = None) -> Dict:
    registry = registry or get_registry()
    stages = [
        {"stage": "raw", "label": "Raw population objects",
         **_stage_raw(repo_root)},
        {"stage": "selected", "label": "Selected for evaluation (frozen plans)",
         **_stage_selected(repo_root)},
        {"stage": "simulated", "label": "Counterfactual / simulated scenarios",
         **_stage_simulated(repo_root)},
        {"stage": "evaluated", "label": "Objects evaluated",
         **_stage_evaluated(repo_root)},
        {"stage": "failed", "label": "Failures detected",
         **_stage_failed(repo_root)},
        {"stage": "hitl", "label": "Human-in-the-loop review",
         **_stage_hitl(repo_root)},
        {"stage": "regression", "label": "Regression suite entries",
         **_stage_regression(repo_root, registry)},
    ]
    funnel = {
        "generated_at": store.now_iso(),
        "stages": stages,
        "safety": _safety_panel(repo_root),
        "model_comparison": _comparison_panel(repo_root),
        "drift": _drift_panel(repo_root),
        "compute": _compute_panel(repo_root),
        "availability": {s["stage"]: s["available"] for s in stages},
    }
    return funnel
