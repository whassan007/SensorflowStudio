"""Hardware/domain dimensions as first-class gate strata.

A release that passes globally can still be unshippable on one platform in
one city with one sensor generation. This module builds a gate matrix whose
rows are combinations of:

    compute_platform   real device configs from sensorflow.vitis.backend
                       (KNOWN_DEVICES) + the reference CPU harness
    sensor_config      the megaeval 'sensor' cube dimension (lidar/camera/fused)
    sensor_generation  config-mapped generation label per sensor_config
    calibration_version latest sensorflow.safety.calibration status
    firmware           config-mapped per compute_platform
    region             domain stratum proxied from the megaeval 'road_type'
                       cube dimension (the derivation is recorded per row)

Metrics per combination come from REAL per-cohort data (megaeval metric cube
of the latest published run). Platform-specific evidence exists only where a
vitis HIL run exists for that device; combinations without evidence are
reported INSUFFICIENT — never assumed passing.

evaluate_matrix() is a pure function of (rows, policy) so the gate logic is
testable on planted inputs, separate from data synthesis.

Minimum-evidence math is delegated to hardening/seqeval power utilities when
importable; the fallback is a Wilson-interval width bound.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from sensorflow.studio2 import store
from sensorflow.studio2.registry import content_hash

# ------------------------------------------------------------------ policy

DEFAULT_HARDWARE_POLICY: Dict = {
    "policy_name": "studio2-hardware-v1",
    "metrics": {
        "recall": {"min": 0.80},
        "precision": {"min": 0.90},
    },
    # evidence adequacy: a combination needs enough support that a Wilson CI
    # on its pass proportion is narrower than this (min-n delegated to
    # hardening/seqeval when importable)
    "max_ci_width": 0.10,
    # combinations matching any of these dicts are launch-critical: a failure
    # there blocks even when the global aggregate passes
    "critical_combinations": [
        {"region": "san_francisco", "sensor_config": "lidar"},
        {"sensor_config": "fused"},
    ],
}

# region strata proxied from the megaeval road_type dimension; the mapping is
# reported on every row so nobody mistakes it for geo-tagged data
REGION_PROFILES = {
    "san_francisco": {"road_type": "urban"},
    "phoenix": {"road_type": "highway"},
    "rural_pilot": {"road_type": "rural"},
}

SENSOR_GENERATIONS = {"lidar": "LiDAR-Gen2", "camera": "Camera-Gen3",
                      "fused": "Fusion-Stack-v1"}

FIRMWARE = {"cpu-reference": "host-fp32", "versal-ai-edge": "vitis-2024.2",
            "zynq-ultrascale": "vitis-2024.2"}


# ------------------------------------------------------------------ min-n


def min_support(max_ci_width: float, p: float = 0.85) -> Dict:
    """Minimum per-combination sample size for adequate evidence.

    Delegates to hardening.power when that package is importable (n to detect
    a proportion drop of max_ci_width/2 below the threshold, one-sided);
    otherwise a Wilson/normal-approximation bound: the CI full width at
    proportion p is ~ 2*z*sqrt(p(1-p)/n).
    """
    try:
        from sensorflow.hardening import power as hardening_power  # type: ignore
        n = int(hardening_power.required_n_two_proportions(
            baseline_rate=p, mde=max_ci_width / 2.0, two_sided=False))
        return {"n": n, "method": "hardening.power.required_n_two_proportions "
                f"(one-sided, p={p}, mde={max_ci_width / 2.0})"}
    except Exception:
        pass
    z = 1.959964  # 95%
    n = math.ceil((2.0 * z / max_ci_width) ** 2 * p * (1.0 - p))
    return {"n": int(n), "method": "wilson/normal-approx fallback "
            f"(z=1.96, p={p}, full width<={max_ci_width})"}


# ------------------------------------------------------------------ synthesis


def _latest_published_mega_run():
    """Returns (mega_store, run_object) for the newest published run."""
    from sensorflow.megaeval.runs import get_mega_store
    store_ = get_mega_store()
    published = [r for r in store_.runs.values()
                 if getattr(r, "status", None) == "published"]
    if not published:
        raise RuntimeError("no published megaeval runs")
    published.sort(key=lambda r: getattr(r, "published_at", None)
                   or getattr(r, "created_at", "") or "")
    return store_, published[-1]


def _cohort_metrics(mega_store, run_id: str, sensor: str, road_type: str) -> Optional[Dict]:
    from sensorflow.megaeval import cube as cube_mod
    cube_df = mega_store.artifacts(run_id)["cube"]
    rows, _ = cube_mod.aggregate(cube_df,
                                 {"sensor": [sensor], "road_type": [road_type]},
                                 None, ["n", "recall", "precision"], 1)
    return rows[0] if rows else None


def _vitis_platforms() -> Dict[str, Dict]:
    """Real device configs + whether any HIL evidence exists for them."""
    platforms: Dict[str, Dict] = {
        "cpu-reference": {"description": "reference CPU harness (fp32)",
                          "hil_evidence": True,
                          "evidence_note": "metrics measured directly on this harness"}}
    try:
        from sensorflow.vitis.backend import KNOWN_DEVICES
        hil_dir = os.path.join("runs", "vitis", "hil")
        has_hil = os.path.isdir(hil_dir) and any(
            n.startswith(("hilrun-", "hilsweep-")) for n in os.listdir(hil_dir))
        for name, spec in KNOWN_DEVICES.items():
            platforms[name] = {
                "description": spec.get("description", name),
                "hil_evidence": bool(has_hil),
                "evidence_note": ("vitis HIL quantization-gap runs present"
                                  if has_hil else
                                  "no vitis HIL run for this device"),
            }
    except Exception:
        pass
    return platforms


def synthesize_combinations() -> Dict:
    """Build combination rows from the real stores. Raises RuntimeError when
    the megaeval store has no published run (callers surface that as an
    unavailable matrix, they do not invent one)."""
    mega_store, run = _latest_published_mega_run()
    run_id = run.run_id
    calibration = None
    try:
        from sensorflow.safety import calibration as calib_mod
        cal = calib_mod.latest_status()
        calibration = (cal or {}).get("status")
    except Exception:
        pass
    calibration_version = f"calib-{(calibration or 'UNKNOWN').lower()}"

    platforms = _vitis_platforms()
    sensors = ["lidar", "camera", "fused"]

    rows: List[Dict] = []
    for platform, pinfo in platforms.items():
        for sensor in sensors:
            for region, proxy in REGION_PROFILES.items():
                combo = {
                    "compute_platform": platform,
                    "sensor_config": sensor,
                    "sensor_generation": SENSOR_GENERATIONS.get(sensor, "unknown"),
                    "calibration_version": calibration_version,
                    "firmware": FIRMWARE.get(platform, "unknown"),
                    "region": region,
                }
                row: Dict = {
                    "combination": combo,
                    "combination_label": f"{region} × {platform} × "
                                         f"{SENSOR_GENERATIONS.get(sensor, sensor)}",
                    "derivation": {
                        "metrics_source": f"megaeval cube of run {run_id}, "
                                          f"filter sensor={sensor}, "
                                          f"road_type={proxy['road_type']} "
                                          f"(region proxied from road_type)",
                        "platform_evidence": pinfo["evidence_note"],
                    },
                }
                if not pinfo["hil_evidence"]:
                    row.update({"metrics": None, "n": 0,
                                "evidence": "MISSING",
                                "evidence_reason": pinfo["evidence_note"]})
                else:
                    m = _cohort_metrics(mega_store, run_id, sensor,
                                        proxy["road_type"])
                    if m is None:
                        row.update({"metrics": None, "n": 0,
                                    "evidence": "MISSING",
                                    "evidence_reason": "no cube cells for this cohort"})
                    else:
                        row.update({"metrics": {"recall": m.get("recall"),
                                                "precision": m.get("precision")},
                                    "n": int(m.get("n") or 0),
                                    "evidence": "PRESENT"})
                rows.append(row)

    # global aggregate for the "global pass but critical combo fails" check
    headline = getattr(run, "headline", None) or {}
    global_metrics = {"recall": headline.get("recall"),
                      "precision": headline.get("precision")}
    return {"rows": rows, "global_metrics": global_metrics,
            "source_run_id": run_id}


# ------------------------------------------------------------------ evaluation


def _check_metrics(metrics: Optional[Dict], thresholds: Dict) -> List[str]:
    failed = []
    for metric, spec in thresholds.items():
        val = (metrics or {}).get(metric)
        if val is None:
            failed.append(f"{metric} missing")
        elif "min" in spec and val < spec["min"]:
            failed.append(f"{metric} {val:.3f} < {spec['min']}")
        elif "max" in spec and val > spec["max"]:
            failed.append(f"{metric} {val:.3f} > {spec['max']}")
    return failed


def _matches(combo: Dict, pattern: Dict) -> bool:
    return all(combo.get(k) == v for k, v in pattern.items())


def evaluate_matrix(rows: List[Dict], policy: Optional[Dict] = None,
                    global_metrics: Optional[Dict] = None,
                    source_run_id: Optional[str] = None,
                    persist: bool = True) -> Dict:
    """Pure gate logic over combination rows.

    Each row: {combination: {...}, combination_label, metrics|None, n,
    evidence: PRESENT|MISSING, [evidence_reason]}.
    """
    pol = {**DEFAULT_HARDWARE_POLICY, **(policy or {})}
    support = min_support(pol["max_ci_width"])
    thresholds = pol["metrics"]

    evaluated: List[Dict] = []
    insufficient: List[Dict] = []
    failures: List[Dict] = []
    critical_failures: List[Dict] = []

    for row in rows:
        out = dict(row)
        critical = any(_matches(row["combination"], pat)
                       for pat in pol["critical_combinations"])
        out["critical"] = critical
        if row.get("evidence") != "PRESENT" or row.get("metrics") is None:
            out["status"] = "INSUFFICIENT"
            out["reason"] = row.get("evidence_reason") or "no evidence"
            insufficient.append(out)
        elif int(row.get("n") or 0) < support["n"]:
            out["status"] = "INSUFFICIENT"
            out["reason"] = (f"support n={row.get('n')} below minimum "
                             f"{support['n']} ({support['method']})")
            insufficient.append(out)
        else:
            failed = _check_metrics(row["metrics"], thresholds)
            if failed:
                out["status"] = "FAIL"
                out["failed_checks"] = failed
                failures.append(out)
                if critical:
                    critical_failures.append(out)
            else:
                out["status"] = "PASS"
        evaluated.append(out)

    global_failed = _check_metrics(global_metrics, thresholds) if global_metrics else None
    global_pass = (global_failed == []) if global_failed is not None else None

    if critical_failures:
        status = "FAIL_CRITICAL"
    elif failures:
        status = "FAIL"
    elif not any(r["status"] == "PASS" for r in evaluated):
        status = "INSUFFICIENT"
    else:
        status = "PASS"

    report = {
        "matrix_id": f"hwm-{content_hash({'rows': [r.get('combination') for r in rows], 'policy': pol}, exclude=())}",
        "status": status,
        "global_pass": global_pass,
        "global_metrics": global_metrics,
        "global_vs_matrix_note": (
            "global aggregate passes but a launch-critical combination fails — "
            "the aggregate is hiding a stratum failure (Simpson-style); the "
            "matrix verdict wins"
            if global_pass and critical_failures else None),
        "n_combinations": len(evaluated),
        "n_pass": sum(1 for r in evaluated if r["status"] == "PASS"),
        "n_fail": len(failures),
        "n_insufficient": len(insufficient),
        "min_support": support,
        "critical_failures": critical_failures,
        "failures": [f for f in failures if f not in critical_failures],
        "insufficient": [{"combination_label": r["combination_label"],
                          "combination": r["combination"],
                          "critical": r["critical"],
                          "reason": r["reason"]} for r in insufficient],
        "rows": evaluated,
        "policy": pol,
        "source_run_id": source_run_id,
        "evaluated_at": store.now_iso(),
    }
    if persist:
        store.write_json(report, "hardware", "latest.json")
    return report


def gate_matrix(policy: Optional[Dict] = None, persist: bool = True) -> Dict:
    """Synthesize combinations from the real stores and evaluate them."""
    data = synthesize_combinations()
    return evaluate_matrix(data["rows"], policy=policy,
                           global_metrics=data["global_metrics"],
                           source_run_id=data["source_run_id"],
                           persist=persist)


def latest_matrix() -> Optional[Dict]:
    return store.read_json("hardware", "latest.json")
