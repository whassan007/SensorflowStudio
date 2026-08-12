"""Multi-gate readiness skeleton — config-driven thresholds.

Extends existing QualityGate + LaunchGateEvaluator toward:
  Scenario | Coverage | Regression | Safety | Release

Phases 4–6 fill Safety/ODD/Release deeply; Phase 1 returns honest ready=False
for unwired gates and real results for quality/launch when sequence artifacts exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensorflow.platform.entities import GateResult, QualityGateDef, Provenance, _id

DEFAULT_GATE_CONFIG_PATH = Path("runs/platform/gate_config.json")

# Config-driven defaults — never hard-code safety cutoffs in callers.
DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "schema_version": "platform-gates-1.0",
    "gates": {
        "scenario": {
            "enabled": True,
            "ready": False,  # Phase 5/6
            "thresholds": {"min_scenario_types_covered": 5},
            "message": "TODO(Phase 5): scenario coverage vs ODDDefinition",
        },
        "coverage": {
            "enabled": True,
            "ready": False,
            "thresholds": {"min_cohort_support": 200, "min_dim_coverage": 0.8},
            "message": "TODO(Phase 5): ODD combinatorial coverage via MegaEval cube",
        },
        "regression": {
            "enabled": True,
            "ready": True,  # can use MegaEval compare / LabelEval regression when run_ids provided
            "thresholds": {
                "max_recall_drop": 0.010,
                "max_precision_drop": 0.015,
                "max_safety_recall_drop": 0.005,
            },
            "message": "Uses MegaEval promotion policy when candidate/baseline runs supplied",
        },
        "safety": {
            "enabled": True,
            "ready": False,
            "thresholds": {},  # filled in Phase 4 from config only
            "message": "TODO(Phase 4): SSAM DRAC/DeltaS conflict engine",
        },
        "release": {
            "enabled": True,
            "ready": False,
            "thresholds": {"require_all_prior_passed": 1.0},
            "message": "TODO(Phase 6): release orchestration after Safety+Regression",
        },
        "quality": {
            "enabled": True,
            "ready": True,
            "thresholds": {
                "map_3d": 0.65,
                "orientation_error_deg": 5.0,
                "id_swap_rate": 0.02,
                "track_fragmentation_rate": 0.05,
                "position_error_m": 2.0,
            },
            "message": "Delegates to sensorflow.quality_gate.QualityGate when sequence_id set",
        },
        "launch": {
            "enabled": True,
            "ready": True,
            "thresholds": {
                "map_3d": 0.65,
                "orientation_error_deg": 5.0,
                "id_swap_rate": 0.02,
                "track_fragmentation_rate": 0.05,
            },
            "message": "Delegates to LaunchGateEvaluator when sequence_id set",
        },
    },
}


def load_gate_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or DEFAULT_GATE_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            cfg = json.load(f)
        # merge missing keys from defaults without wiping user thresholds
        merged = json.loads(json.dumps(DEFAULT_GATE_CONFIG))
        for gname, gcfg in (cfg.get("gates") or {}).items():
            if gname in merged["gates"]:
                merged["gates"][gname].update(gcfg)
            else:
                merged["gates"][gname] = gcfg
        return merged
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(DEFAULT_GATE_CONFIG, f, indent=2)
    return json.loads(json.dumps(DEFAULT_GATE_CONFIG))


def save_gate_config(cfg: Dict[str, Any], path: Optional[Path] = None) -> Path:
    path = path or DEFAULT_GATE_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


GATE_ORDER = ["scenario", "coverage", "regression", "safety", "quality", "launch", "release"]


def list_gate_defs(cfg: Optional[Dict[str, Any]] = None) -> List[QualityGateDef]:
    cfg = cfg or load_gate_config()
    out = []
    for name in GATE_ORDER:
        g = cfg["gates"].get(name, {})
        out.append(QualityGateDef(
            gate_id=f"gate-{name}",
            name=name.title(),
            gate_type=name,  # type: ignore[arg-type]
            thresholds=dict(g.get("thresholds") or {}),
            config_path=str(DEFAULT_GATE_CONFIG_PATH),
            enabled=bool(g.get("enabled", True)),
            version="v1",
            provenance=Provenance(source_system="platform", notes=g.get("message", "")),
        ))
    return out


def evaluate_multi_gates(
    *,
    sequence_id: Optional[str] = None,
    candidate_run_id: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = load_gate_config(config_path)
    results: List[GateResult] = []

    for name in GATE_ORDER:
        g = cfg["gates"][name]
        if not g.get("enabled", True):
            continue
        ready = bool(g.get("ready", False))
        thresholds = dict(g.get("thresholds") or {})
        result = GateResult(
            result_id=_id("gateres"),
            gate_id=f"gate-{name}",
            gate_type=name,
            thresholds=thresholds,
            ready=ready,
            message=g.get("message", ""),
            scope_ref=sequence_id or candidate_run_id,
            passed=None,
        )

        if name == "quality" and sequence_id and ready:
            result = _eval_quality(sequence_id, thresholds, result)
        elif name == "launch" and sequence_id and ready:
            result = _eval_launch(sequence_id, result)
        elif name == "regression" and candidate_run_id and baseline_run_id and ready:
            result = _eval_regression(candidate_run_id, baseline_run_id, thresholds, result)
        elif not ready:
            result.passed = None
            result.message = g.get("message") or f"{name} gate skeleton — not yet wired"
        else:
            result.passed = None
            result.message = f"{name} gate ready but missing required inputs"

        results.append(result)

    # Overall readiness: all ready gates that produced a boolean must pass;
    # unwired gates block "release ready" but do not fail the suite silently.
    wired = [r for r in results if r.ready and r.passed is not None]
    unwired = [r for r in results if not r.ready]
    all_wired_passed = all(r.passed for r in wired) if wired else False

    return {
        "config_path": str(config_path or DEFAULT_GATE_CONFIG_PATH),
        "sequence_id": sequence_id,
        "candidate_run_id": candidate_run_id,
        "baseline_run_id": baseline_run_id,
        "gates": [r.model_dump() for r in results],
        "summary": {
            "wired_count": len(wired),
            "unwired_count": len(unwired),
            "wired_all_passed": all_wired_passed,
            "release_ready": False,  # Phase 6
            "unwired_gate_types": [r.gate_type for r in unwired],
        },
    }


def _eval_quality(sequence_id: str, thresholds: Dict[str, float], result: GateResult) -> GateResult:
    try:
        from sensorflow.quality_gate import QualityGate
        from sensorflow.schemas.unified_frame import UnifiedSequence

        manifest = Path("runs/pipeline") / sequence_id / "manifest.json"
        tracks_path = Path("runs/pipeline") / sequence_id / "tracks.json"
        if not manifest.exists() or not tracks_path.exists():
            result.passed = False
            result.failures = [{"metric": "artifacts", "message": "manifest/tracks missing"}]
            result.message = "Quality gate: run ingest/auto-label/track first"
            return result
        sequence = UnifiedSequence.load(manifest)
        pred_tracks = json.loads(tracks_path.read_text())
        gate = QualityGate(thresholds=thresholds)
        out = gate.evaluate(sequence, pred_tracks)
        result.passed = bool(out.get("passed"))
        result.metrics = out.get("metric_card") or {}
        result.failures = out.get("quality_report", {}).get("failures") or out.get("failures") or []
        result.message = "QualityGate evaluated"
    except Exception as exc:
        result.passed = False
        result.failures = [{"error": str(exc)}]
        result.message = f"Quality gate error: {exc}"
    return result


def _eval_launch(sequence_id: str, result: GateResult) -> GateResult:
    try:
        from sensorflow.launch_gate_evaluator import LaunchGateEvaluator
        evaluator = LaunchGateEvaluator()
        out = evaluator.evaluate(sequence_id)
        result.passed = bool(out.get("passed"))
        result.failures = out.get("failures") or []
        result.metrics = out.get("metrics") or {}
        result.thresholds = evaluator.thresholds
        result.message = "LaunchGateEvaluator evaluated"
    except Exception as exc:
        result.passed = False
        result.failures = [{"error": str(exc)}]
        result.message = f"Launch gate error: {exc}"
    return result


def _eval_regression(
    candidate_run_id: str,
    baseline_run_id: str,
    thresholds: Dict[str, float],
    result: GateResult,
) -> GateResult:
    try:
        from sensorflow.platform.compare import compare_models
        cmp = compare_models(
            [baseline_run_id, candidate_run_id],
            baseline_run_id=baseline_run_id,
            policy=thresholds,
        )
        pair = (cmp.get("pairwise") or [{}])[0]
        rec = pair.get("recommendation")
        result.passed = rec == "PROMOTE"
        result.failures = [{"blocker": b} for b in (pair.get("blockers") or [])]
        result.metrics = {"recommendation": rec, "headline": pair.get("headline")}
        result.message = f"Regression gate: {rec}"
    except Exception as exc:
        result.passed = False
        result.failures = [{"error": str(exc)}]
        result.message = f"Regression gate error: {exc}"
    return result
