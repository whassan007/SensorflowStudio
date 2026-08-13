"""Per-stage diagnostic computations.

Every function takes a ScenarioBundle and returns (data, findings):
    data      JSON-serializable payload for the UI table/visualization
    findings  list of Finding records (auto evidence)

Nothing here is invented: every number is computed from the investigation's
offline/shadow unit data or its recorded configs.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sensorflow.rca import stats as st
from sensorflow.rca.models import Finding, make_finding
from sensorflow.rca.scenario import FEATURES, SEGMENT_DIMS, ScenarioBundle

LARGE_N_CAVEAT = (
    "With thousands of units almost any difference is 'statistically "
    "significant'; act on the practical-effect magnitude (PSI / JS labels), "
    "not the p-value.")

CONF_BANDS = ((0.0, 0.35), (0.35, 0.50), (0.50, 0.70), (0.70, 1.01))


def _scored(bundle: ScenarioBundle) -> pd.DataFrame:
    sh = bundle.shadow
    return sh[sh["sampled"]]


def _delta_pp(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float((df["b_correct"].mean() - df["a_correct"].mean()) * 100)


def _parse_window(win: str) -> Optional[Tuple[date, date]]:
    ds = re.findall(r"(\d{4})-(\d{2})-(\d{2})", win or "")
    if len(ds) < 2:
        return None
    return (date(*map(int, ds[0])), date(*map(int, ds[1])))


# ================================================== stage 0: comparison validity

# (field, label, criticality-on-mismatch, expected-to-differ)
_CV_FIELDS = [
    ("metric_definition", "Metric definition", "CRITICAL", False),
    ("aggregation", "Aggregation", "WARN", False),
    ("eval_window", "Evaluation window", "WARN", True),
    ("population_source", "Population source", "WARN", True),
    ("confidence_threshold", "Confidence threshold", "CRITICAL", False),
    ("quantization", "Numeric precision / quantization", "CRITICAL", False),
    ("runtime_version", "Runtime version", "WARN", False),
    ("nms_iou", "NMS IoU", "WARN", False),
    ("feature_pipeline_version", "Feature pipeline version", "CRITICAL", False),
    ("label_policy_version", "Labeling policy", "CRITICAL", False),
    ("label_maturity_policy", "Label maturity policy", "WARN", True),
    ("sampling_policy", "Sampling policy", "WARN", True),
]


def comparison_validity(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    off_cfg = bundle.meta["offline_config"]
    sh_cfg = bundle.meta["shadow_config"]
    rows: List[Dict] = []
    findings: List[Finding] = []
    stage = "comparison_validity"

    for field, label, crit, expected_diff in _CV_FIELDS:
        ov, sv = off_cfg.get(field), sh_cfg.get(field)
        if ov is None or sv is None:
            status = "unknown"
            findings.append(make_finding(
                stage, f"CV_UNKNOWN:{field}", f"{label} is unrecorded",
                "UNKNOWN", crit,
                f"Offline={ov!r}, shadow={sv!r}. You cannot certify the two "
                f"numbers are comparable while {label.lower()} is unknown."))
        elif ov == sv:
            status = "comparable"
        elif expected_diff:
            status = "expected_difference"
        else:
            status = "mismatch"
            findings.append(make_finding(
                stage, f"CV_MISMATCH:{field}", f"{label} differs",
                "MISMATCH", crit,
                f"Offline={ov!r} vs shadow={sv!r}. The two measurements are "
                "not measuring the same thing along this dimension.",
                data={"offline": ov, "shadow": sv}))
        rows.append({"field": field, "label": label, "offline": ov,
                     "shadow": sv, "status": status,
                     "criticality": crit})

    # Window staleness: a large gap between offline eval window and the live
    # shadow window is itself a validity hazard (world may have moved).
    ow, sw = _parse_window(off_cfg.get("eval_window", "")), \
        _parse_window(sh_cfg.get("eval_window", ""))
    if ow and sw:
        gap_days = (sw[0] - ow[1]).days
        if gap_days > 45:
            findings.append(make_finding(
                stage, "CV_WINDOW_STALE", "Offline eval window is stale",
                "MISMATCH", "WARN",
                f"Offline window ends {ow[1].isoformat()}, shadow starts "
                f"{sw[0].isoformat()} ({gap_days} days later). Data drift in "
                "the gap can make both numbers correct on their own windows."))
        rows.append({"field": "window_gap_days", "label": "Window gap (days)",
                     "offline": ow[1].isoformat(), "shadow": sw[0].isoformat(),
                     "status": "mismatch" if gap_days > 45 else "comparable",
                     "criticality": "WARN", "gap_days": gap_days})

    if not any(f.status != "PASS" for f in findings):
        findings.append(make_finding(
            stage, "CV_COMPARABLE", "No blocking comparability mismatch",
            "PASS", "INFO",
            "All recorded dimensions are comparable or expected differences."))

    data = {"rows": rows,
            "explainer": ("Dimension-by-dimension check that '+5% offline' and "
                          "'-2% shadow' are even commensurable claims. Any "
                          "mismatch row is a candidate explanation that must "
                          "be ruled out before believing either number.")}
    return data, findings


# ======================================================= stage 1: offline audit


def offline_audit(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "offline_audit"
    findings: List[Finding] = []
    off = bundle.offline
    repro = bundle.meta["reproducibility"]
    model_b = bundle.meta["model_b"]

    orig = float(repro["original_offline_metric_delta"])
    rerun = float(repro["rerun_metric_delta"])
    diff = abs(rerun - orig)
    reproduces = diff < 0.005
    if reproduces:
        findings.append(make_finding(
            stage, "OA_REPRO_OK", "Offline result reproduces", "PASS", "INFO",
            f"Re-run delta {rerun*100:+.2f}pp vs original {orig*100:+.2f}pp."))
    else:
        findings.append(make_finding(
            stage, "OA_REPRO_FAIL", "Offline result does NOT reproduce",
            "MISMATCH", "CRITICAL",
            f"Re-running the baseline evaluation gives {rerun*100:+.2f}pp vs "
            f"the claimed {orig*100:+.2f}pp (|diff| {diff*100:.2f}pp). An "
            "irreproducible +5% is not evidence of anything."))
    if repro["pins_present"]:
        findings.append(make_finding(
            stage, "OA_PINS_OK", "Version pins recorded", "PASS", "INFO",
            repro["environment_lock"]))
    else:
        findings.append(make_finding(
            stage, "OA_PINS_MISSING", "Version pins missing", "UNKNOWN", "WARN",
            f"Environment: {repro['environment_lock']}. Cannot certify the "
            "offline run's exact code/data state."))

    # Leakage scan: near-duplicate flags + entity overlap with B's train set.
    dup_mask = off["dup_similarity"] > 0.90
    n_dup = int(dup_mask.sum())
    train_ids = set(model_b.get("train_entity_ids", []))
    overlap_entities = sorted(set(off["entity_id"]) & train_ids)
    dup_frac = n_dup / max(1, len(off))
    dup_delta = _delta_pp(off[dup_mask]) if n_dup else 0.0
    clean_delta = _delta_pp(off[~dup_mask])
    if n_dup > 0 or overlap_entities:
        findings.append(make_finding(
            stage, "OA_LEAKAGE_DUPLICATES", "Train/eval leakage detected",
            "MISMATCH", "CRITICAL",
            f"{n_dup} eval units ({dup_frac:.1%}) are near-duplicates "
            f"(similarity>0.90) of candidate training data; "
            f"{len(overlap_entities)} eval entities appear in B's training "
            f"set. Delta on leaked units: {dup_delta:+.1f}pp vs "
            f"{clean_delta:+.1f}pp on clean units.",
            data={"n_duplicates": n_dup, "dup_fraction": dup_frac,
                  "overlap_entity_count": len(overlap_entities),
                  "dup_delta_pp": dup_delta, "clean_delta_pp": clean_delta}))
    else:
        findings.append(make_finding(
            stage, "OA_LEAKAGE_CLEAN", "No leakage detected", "PASS", "INFO",
            "No near-duplicates of candidate training data in the eval set; "
            "no entity overlap with B's training entities."))

    # Temporal leakage: eval window must start after B's training window ends.
    eval_win = _parse_window(bundle.meta["offline_config"].get("eval_window", ""))
    train_end = model_b.get("train_window", "").split("..")[-1].strip()
    temporal_ok = True
    if eval_win and re.match(r"\d{4}-\d{2}", train_end):
        y, m = map(int, train_end.split("-")[:2])
        temporal_ok = eval_win[0] >= date(y, m, 1)
    findings.append(make_finding(
        stage, "OA_TEMPORAL_OK" if temporal_ok else "OA_TEMPORAL_LEAK",
        "Temporal split check", "PASS" if temporal_ok else "MISMATCH",
        "INFO" if temporal_ok else "CRITICAL",
        f"Eval window vs B train window ({model_b.get('train_window')})."))

    split = model_b.get("split_strategy", "unknown")
    if split == "row_level":
        findings.append(make_finding(
            stage, "OA_SPLIT_ROW_LEVEL", "Row-level split (correlated entities)",
            "MISMATCH", "WARN",
            "Train/eval split was row-level, so frames from the same drive "
            "segment can straddle the boundary; correlated near-duplicates "
            "inflate offline scores."))
    else:
        findings.append(make_finding(
            stage, "OA_SPLIT_ENTITY_LEVEL", "Entity-level split", "PASS", "INFO",
            "Split respects drive-segment boundaries."))

    data = {
        "reproducibility": {"original_delta_pp": orig * 100,
                            "rerun_delta_pp": rerun * 100,
                            "abs_diff_pp": diff * 100,
                            "reproduces": reproduces,
                            "pins_present": bool(repro["pins_present"]),
                            "environment_lock": repro["environment_lock"]},
        "leakage": {"n_duplicates": n_dup, "dup_fraction": dup_frac,
                    "overlap_entity_count": len(overlap_entities),
                    "overlap_examples": overlap_entities[:8],
                    "dup_delta_pp": dup_delta, "clean_delta_pp": clean_delta,
                    "offline_n": int(len(off))},
        "split": {"strategy": split,
                  "n_entities": int(off["entity_id"].nunique()),
                  "mean_units_per_entity": float(len(off) / max(1, off["entity_id"].nunique()))},
        "temporal": {"ok": temporal_ok,
                     "b_train_window": model_b.get("train_window")},
        "explainer": ("Audits the +5%% claim itself: does it reproduce, is the "
                      "environment pinned, and is the eval set contaminated by "
                      "the candidate's training data?"),
    }
    return data, findings


# ================================================ stage 2: population validation


def population_validation(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "population_validation"
    findings: List[Finding] = []
    off, sh = bundle.offline, bundle.shadow
    scored = _scored(bundle)
    traffic = bundle.meta["traffic"]

    dim_tables = {}
    for dim in SEGMENT_DIMS:
        o = off[dim].value_counts(normalize=True)
        s = scored[dim].value_counts(normalize=True)
        cats = sorted(set(o.index) | set(s.index))
        dim_tables[dim] = [{"value": c,
                            "offline_share": float(o.get(c, 0.0)),
                            "shadow_share": float(s.get(c, 0.0))}
                           for c in cats]

    overlap = len(set(off["entity_id"]) & set(sh["entity_id"]))
    if len(scored) < 1000:
        findings.append(make_finding(
            stage, "POP_VOLUME_LOW", "Shadow scored volume is small",
            "UNKNOWN", "CRITICAL",
            f"Only {len(scored)} scored shadow units "
            f"({sh['entity_id'].nunique()} drive segments). Aggregate deltas "
            "at this volume are dominated by sampling noise; treat -2% as a "
            "hypothesis, not a measurement."))
    if len(off) < 1000:
        findings.append(make_finding(
            stage, "POP_SMALL_OFFLINE", "Offline eval set is small",
            "UNKNOWN", "WARN",
            f"Offline eval has only {len(off)} units."))
    if len(scored) >= 1000 and len(off) >= 1000:
        findings.append(make_finding(
            stage, "POP_VOLUME_OK", "Volumes adequate", "PASS", "INFO",
            f"Offline n={len(off)}, shadow scored n={len(scored)} "
            f"(of {traffic['eligible_count']} eligible)."))

    data = {
        "volumes": {"offline_n": int(len(off)),
                    "shadow_eligible_n": int(traffic["eligible_count"]),
                    "shadow_scored_n": int(len(scored)),
                    "sampling_rate": traffic["sampling_rate"],
                    "offline_entities": int(off["entity_id"].nunique()),
                    "shadow_entities": int(sh["entity_id"].nunique()),
                    "entity_overlap": overlap},
        "dimensions": dim_tables,
        "explainer": ("Confirms both evaluations have enough volume to say "
                      "anything, and shows the raw population composition "
                      "side-by-side before any shift statistics."),
    }
    return data, findings


# ================================================= stage 3: distribution shift


def distribution_shift(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "distribution_shift"
    findings: List[Finding] = []
    off, scored = bundle.offline, _scored(bundle)
    volume_ok = min(len(off), len(scored)) >= 800

    seg_rows, feat_rows = [], []
    seg_psis = {}
    for dim in SEGMENT_DIMS:
        psi = st.psi_categorical(off[dim], scored[dim])
        js = st.js_divergence_categorical(off[dim], scored[dim])
        chi, p = st.chi2_test(off[dim], scored[dim])
        seg_psis[dim] = psi
        seg_rows.append({"dimension": dim, "kind": "segment", "psi": psi,
                         "js": js, "test": "chi2", "stat": chi, "p_value": p,
                         "magnitude": st.shift_magnitude_label(psi)})

    for feat in FEATURES:
        a = off[feat].dropna().to_numpy()
        b = scored[feat].dropna().to_numpy()
        psi = st.psi_continuous(a, b)
        js = st.js_divergence_continuous(a, b)
        ks, p = st.ks_test(a, b)
        feat_rows.append({"dimension": feat, "kind": "feature", "psi": psi,
                          "js": js, "test": "ks", "stat": ks, "p_value": p,
                          "magnitude": st.shift_magnitude_label(psi),
                          "offline_mean": float(np.mean(a)),
                          "shadow_mean": float(np.mean(b))})

    if not volume_ok:
        findings.append(make_finding(
            stage, "DS_VOLUME_TOO_SMALL",
            "Too little data to assess distribution shift",
            "UNKNOWN", "WARN",
            f"Only {len(off)} offline / {len(scored)} scored shadow units. "
            "PSI estimates are upward-biased at this volume; shift can be "
            "neither confirmed nor excluded."))
        data = {"segments": seg_rows, "features": feat_rows,
                "caveat": LARGE_N_CAVEAT, "volume_ok": False,
                "explainer": ("Per-dimension PSI + Jensen-Shannon + KS/"
                              "chi-square with BOTH significance and practical "
                              "magnitude.")}
        return data, findings

    high_segs = [d for d, v in seg_psis.items() if v >= 0.25]
    for dim in high_segs:
        findings.append(make_finding(
            stage, f"DS_SHIFT_HIGH:{dim}", f"Large population shift on {dim}",
            "MISMATCH", "CRITICAL" if seg_psis[dim] >= 0.5 else "WARN",
            f"PSI={seg_psis[dim]:.2f} ({st.shift_magnitude_label(seg_psis[dim])}). "
            "Offline and shadow are scoring materially different populations; "
            "aggregate deltas are not directly comparable. " + LARGE_N_CAVEAT,
            data={"psi": seg_psis[dim]}))

    if not high_segs and all(v < 0.10 for v in seg_psis.values()):
        findings.append(make_finding(
            stage, "DS_SHIFT_LOW", "Population mix is stable", "PASS", "INFO",
            f"All segment-dimension PSIs < 0.10 "
            f"({', '.join(f'{d}={v:.03f}' for d, v in seg_psis.items())})."))

    # A single feature shifting hugely while the population mix is stable is
    # the signature of a pipeline artifact, not a population change.
    if all(v < 0.10 for v in seg_psis.values()):
        for row in feat_rows:
            if row["psi"] >= 0.5:
                findings.append(make_finding(
                    stage, f"DS_FEATURE_ONLY_SHIFT:{row['dimension']}",
                    f"Isolated shift in {row['dimension']} despite stable mix",
                    "MISMATCH", "WARN",
                    f"{row['dimension']} PSI={row['psi']:.2f} while all segment "
                    "mixes are stable -- consistent with a feature-pipeline "
                    "artifact rather than a real-world change. Check feature "
                    "parity (stage 7).",
                    data={"psi": row["psi"],
                          "offline_mean": row["offline_mean"],
                          "shadow_mean": row["shadow_mean"]}))

    data = {"segments": seg_rows, "features": feat_rows,
            "caveat": LARGE_N_CAVEAT, "volume_ok": True,
            "explainer": ("Per-dimension PSI + Jensen-Shannon + KS/chi-square "
                          "with BOTH significance and practical magnitude. "
                          "Large shift means the -2% may be measured on a "
                          "different world than the +5%.")}
    return data, findings


# ============================================ stage 4: conditional performance


def conditional_performance(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "conditional_performance"
    findings: List[Finding] = []
    off, scored = bundle.offline, _scored(bundle)

    rows = []
    agg_off, agg_sh = _delta_pp(off), _delta_pp(scored)
    total_neg_contrib = 0.0
    contribs: List[Tuple[str, float]] = []
    consistent_volume = 0
    neg_volume = 0

    for (scene, tod), o_seg in off.groupby(["scene", "time_of_day"]):
        s_seg = scored[(scored["scene"] == scene) & (scored["time_of_day"] == tod)]
        od, sd = _delta_pp(o_seg), _delta_pp(s_seg)
        off_share = len(o_seg) / len(off)
        sh_share = len(s_seg) / max(1, len(scored))
        seg_name = f"{scene}/{tod}"
        low_vol = len(s_seg) < 60 or len(o_seg) < 60
        if low_vol:
            interp = "low_volume"
        elif od > 0.5 and sd > 0.5:
            interp = "consistent_improvement"
        elif od < -0.5 and sd < -0.5:
            interp = "consistent_regression"
        elif od > 0.5 and sd < -0.5:
            interp = "sign_flip"
        else:
            interp = "flat"
        rows.append({"segment": seg_name, "scene": scene, "time_of_day": tod,
                     "offline_delta_pp": od, "shadow_delta_pp": sd,
                     "offline_n": int(len(o_seg)), "shadow_n": int(len(s_seg)),
                     "offline_share": off_share, "shadow_share": sh_share,
                     "shift_ratio": sh_share / max(1e-9, off_share),
                     "interpretation": interp})
        if not low_vol:
            if np.sign(od) == np.sign(sd) and abs(od) > 0.5 and abs(sd) > 0.5:
                consistent_volume += len(s_seg)
            if sd < 0:
                neg_volume += len(s_seg)
                contrib = sh_share * sd
                total_neg_contrib += contrib
                contribs.append((seg_name, contrib))

    rows.sort(key=lambda r: r["shadow_delta_pp"])
    consistency = consistent_volume / max(1, len(scored))

    # Simpson's requires the mix to have actually moved: within-segment sign
    # agreement plus an aggregate flip is only a mix artifact if the segment
    # composition differs between the two evaluations.
    mix_psi = max(st.psi_categorical(off["scene"], scored["scene"]),
                  st.psi_categorical(off["time_of_day"], scored["time_of_day"]))
    simpsons = (agg_off > 0.5 and agg_sh < -0.5 and consistency >= 0.55
                and mix_psi >= 0.25)
    if simpsons:
        findings.append(make_finding(
            stage, "CP_SIMPSONS_DETECTED",
            "Simpson's paradox: aggregate flip explained by mix",
            "MISMATCH", "CRITICAL",
            f"Aggregate offline {agg_off:+.1f}pp vs shadow {agg_sh:+.1f}pp, "
            f"yet within-segment deltas agree in sign on {consistency:.0%} of "
            "scored volume. The sign flip comes from WHERE the traffic is, "
            "not from the model changing behavior.",
            data={"consistency": consistency, "agg_offline_pp": agg_off,
                  "agg_shadow_pp": agg_sh}))
    elif contribs:
        contribs.sort(key=lambda c: c[1])
        # Concentration is checked at segment level and along each single
        # dimension (all-highway regression should register even when split
        # across highway/day, highway/dusk, highway/night).
        grouped: Dict[str, float] = {}
        for seg_name, contrib in contribs:
            scene, tod = seg_name.split("/")
            grouped[seg_name] = grouped.get(seg_name, 0.0) + contrib
            grouped[f"scene={scene}"] = grouped.get(f"scene={scene}", 0.0) + contrib
            grouped[f"time={tod}"] = grouped.get(f"time={tod}", 0.0) + contrib
        top_seg, top_contrib = min(grouped.items(), key=lambda kv: kv[1])
        top_share = top_contrib / total_neg_contrib if total_neg_contrib < 0 else 0.0
        if top_share >= 0.55:
            findings.append(make_finding(
                stage, f"CP_SEGMENT_CONCENTRATED:{top_seg}",
                f"Shadow regression concentrated in {top_seg}",
                "MISMATCH", "WARN",
                f"{top_share:.0%} of the total shadow regression comes from "
                f"{top_seg}. A concentrated regression points at something "
                "that segment depends on (a feature, a sensor mode), not a "
                "global model change.",
                data={"segment": top_seg, "share_of_regression": top_share}))
        elif neg_volume / max(1, len(scored)) >= 0.6:
            findings.append(make_finding(
                stage, "CP_UNIFORM_REGRESSION",
                "Shadow regression is broad-based",
                "MISMATCH", "WARN",
                f"{neg_volume / max(1, len(scored)):.0%} of scored volume sits "
                "in segments with negative shadow delta and no single segment "
                "dominates -- consistent with a genuine model-level change.",
                data={"negative_volume_share": neg_volume / max(1, len(scored))}))

    if not findings:
        findings.append(make_finding(
            stage, "CP_NO_PATTERN", "No dominant conditional pattern",
            "PASS", "INFO",
            "Per-segment deltas show no Simpson flip, concentration, or "
            "uniform regression above thresholds."))

    data = {"rows": rows, "aggregate": {"offline_delta_pp": agg_off,
                                        "shadow_delta_pp": agg_sh,
                                        "sign_consistency": consistency},
            "explainer": ("Segment x (offline delta, shadow delta, volume, "
                          "shift) matrix. This is where Simpson's paradox "
                          "becomes visible: per-segment agreement with an "
                          "aggregate sign flip means the mix, not the model.")}
    return data, findings


# ================================================ stage 5: paired comparison


def paired_comparison(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "paired_comparison"
    findings: List[Finding] = []
    scored = _scored(bundle)
    off = bundle.offline

    def matrix(df: pd.DataFrame) -> Dict:
        a, b = df["a_correct"] == 1, df["b_correct"] == 1
        return {"both_correct": int((a & b).sum()),
                "regressions": int((a & ~b).sum()),
                "improvements": int((~a & b).sum()),
                "both_wrong": int((~a & ~b).sum()),
                "n": int(len(df)),
                "net_pp": _delta_pp(df)}

    overall = matrix(scored)
    offline_overall = matrix(off)

    by_segment = []
    for (scene, tod), g in scored.groupby(["scene", "time_of_day"]):
        m = matrix(g)
        m["segment"] = f"{scene}/{tod}"
        by_segment.append(m)
    by_class = []
    for cls, g in scored.groupby("object_class"):
        m = matrix(g)
        m["segment"] = cls
        by_class.append(m)

    by_band = []
    conf_col = "b_conf_serving" if "b_conf_serving" in scored.columns else "b_conf"
    total_reg = max(1, overall["regressions"])
    band_reg_share = {}
    for lo, hi in CONF_BANDS:
        g = scored[(scored[conf_col] >= lo) & (scored[conf_col] < hi)]
        m = matrix(g)
        m["segment"] = f"[{lo:.2f}, {hi:.2f})"
        m["unit_share"] = len(g) / max(1, len(scored))
        m["regression_share"] = m["regressions"] / total_reg
        m["lift"] = (m["regression_share"] / m["unit_share"]
                     if m["unit_share"] > 0 else 0.0)
        band_reg_share[(lo, hi)] = m
        by_band.append(m)

    focus = band_reg_share.get((0.35, 0.50))
    if focus and focus["lift"] >= 1.5 and focus["regression_share"] >= 0.30:
        findings.append(make_finding(
            stage, "PC_CONF_BAND_CONCENTRATION",
            "Regressions concentrated in the 0.35-0.50 confidence band",
            "MISMATCH", "CRITICAL",
            f"{focus['regression_share']:.0%} of A-correct->B-incorrect "
            f"transitions sit in the [0.35, 0.50) serving-confidence band, "
            f"{focus['lift']:.1f}x its traffic share. That band is exactly "
            "where a serving-threshold difference would delete detections.",
            data={"lift": focus["lift"],
                  "regression_share": focus["regression_share"]}))

    if overall["regressions"] > overall["improvements"] * 1.15:
        findings.append(make_finding(
            stage, "PC_NET_REGRESSION", "Net regression transitions in shadow",
            "MISMATCH", "WARN",
            f"{overall['regressions']} regression transitions vs "
            f"{overall['improvements']} improvements on identical units "
            f"(net {overall['net_pp']:+.1f}pp).",
            data={"regressions": overall["regressions"],
                  "improvements": overall["improvements"]}))
    else:
        findings.append(make_finding(
            stage, "PC_BALANCED", "Transitions roughly balanced", "PASS", "INFO",
            f"{overall['regressions']} regressions vs "
            f"{overall['improvements']} improvements."))

    data = {"shadow": overall, "offline": offline_overall,
            "by_segment": by_segment, "by_class": by_class, "by_band": by_band,
            "conf_col": conf_col,
            "explainer": ("Error-transition matrix on the SAME units: "
                          "A-correct->B-incorrect (regressions) vs "
                          "A-incorrect->B-correct (improvements), sliced by "
                          "segment, class, and serving-confidence band.")}
    return data, findings


# ============================================== stage 6: statistical significance


def _seqeval_snapshot(scored: pd.DataFrame, margin: float) -> Optional[Dict]:
    """Complementary anytime-valid evidence from sensorflow.seqeval, if the
    package is available. The fixed-n cluster-robust CI stays the verdict
    driver for this stage (one batch, one pre-committed look); the sequential
    snapshot answers the different question 'could we have stopped early /
    keep monitoring without alpha-spending'."""
    try:
        from sensorflow.seqeval import units as sq_units
        from sensorflow.seqeval.sequential import PairedSequentialTest
    except Exception:
        return None
    d = (scored["b_correct"] - scored["a_correct"]).to_numpy().astype(float)
    means, _sizes = sq_units.cluster_units(d, scored["entity_id"].to_numpy())
    order = np.random.default_rng(0).permutation(len(means))
    test = PairedSequentialTest(delta=margin, alpha=0.05)
    test.update_clusters(means[order])
    test.record_objects(scored["a_correct"].to_numpy().astype(bool),
                        scored["b_correct"].to_numpy().astype(bool))
    test.evaluate()
    snap = test.snapshot()
    snap["note"] = (
        "Anytime-valid empirical-Bernstein confidence sequence "
        "(sensorflow.seqeval). Wider than the fixed-n CI by design: it stays "
        "valid under continuous monitoring and optional stopping.")
    return snap


def statistical_significance(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "statistical_significance"
    findings: List[Finding] = []
    scored = _scored(bundle)
    off = bundle.offline
    margin = float(bundle.meta.get("practical_margin_pp", 1.0)) / 100.0

    seq_snap = _seqeval_snapshot(scored, margin)
    engine = "local+seqeval" if seq_snap is not None else "local"
    sh_res = st.paired_delta_cluster(scored)
    off_res = st.paired_delta_cluster(off)
    outcome = st.three_way_outcome(sh_res["delta"], sh_res["ci_low"],
                                   sh_res["ci_high"], margin)

    wilson = {}
    for name, df, col in (("offline_a", off, "a_correct"),
                          ("offline_b", off, "b_correct"),
                          ("shadow_a", scored, "a_correct"),
                          ("shadow_b", scored, "b_correct")):
        lo, hi = st.wilson_ci(int(df[col].sum()), len(df))
        wilson[name] = {"rate": float(df[col].mean()), "ci_low": lo, "ci_high": hi,
                        "n": int(len(df))}

    if outcome == "significant_regression":
        findings.append(make_finding(
            stage, "SS_SIGNIFICANT_REGRESSION",
            "Shadow regression is significant beyond the practical margin",
            "MISMATCH", "CRITICAL",
            f"Paired delta {sh_res['delta']*100:+.2f}pp, 95% CI "
            f"[{sh_res['ci_low']*100:+.2f}, {sh_res['ci_high']*100:+.2f}]pp, "
            f"entirely below the -{margin*100:.1f}pp practical margin "
            f"(cluster-aware, effective n={sh_res['effective_n']}).",
            data=sh_res))
    elif outcome == "no_significant_difference":
        findings.append(make_finding(
            stage, "SS_NO_SIG_DIFF",
            "No practically-important difference detectable",
            "PASS", "INFO",
            f"95% CI [{sh_res['ci_low']*100:+.2f}, {sh_res['ci_high']*100:+.2f}]pp "
            f"excludes a regression beyond -{margin*100:.1f}pp.",
            data=sh_res))
    else:
        findings.append(make_finding(
            stage, "SS_INSUFFICIENT_EVIDENCE",
            "Insufficient evidence to call the shadow delta",
            "UNKNOWN", "CRITICAL",
            f"Paired delta {sh_res['delta']*100:+.2f}pp with 95% CI "
            f"[{sh_res['ci_low']*100:+.2f}, {sh_res['ci_high']*100:+.2f}]pp "
            f"spans the practical margin. Effective n is only "
            f"{sh_res['effective_n']} (design effect "
            f"{sh_res['design_effect']:.1f} from {sh_res['n_clusters']} "
            "clusters). Neither 'regression' nor 'fine' is supported.",
            data=sh_res))

    if sh_res["effective_n"] < 800:
        findings.append(make_finding(
            stage, "SS_LOW_ESS", "Low effective sample size", "UNKNOWN", "WARN",
            f"Cluster correlation shrinks {sh_res['n']} units to an effective "
            f"n of {sh_res['effective_n']}.",
            data={"effective_n": sh_res["effective_n"],
                  "design_effect": sh_res["design_effect"]}))

    data = {"engine": engine,
            "engine_note": ("Verdict: fixed-n cluster-robust paired CI (valid "
                            "for this single pre-committed look). "
                            + ("Complementary anytime-valid confidence "
                               "sequence from sensorflow.seqeval attached."
                               if seq_snap is not None else
                               "sensorflow.seqeval not importable; local "
                               "Wilson/paired implementation only.")),
            "practical_margin_pp": margin * 100,
            "shadow_paired": sh_res, "offline_paired": off_res,
            "outcome": outcome, "wilson": wilson,
            "seqeval": seq_snap,
            "explainer": ("Paired B-A delta with cluster-aware effective "
                          "sample size and a minimum practically-important "
                          "regression margin. Three-way verdict: significant "
                          "regression / no significant difference / "
                          "insufficient evidence.")}
    return data, findings


# ==================================================== stage 7: feature parity


def feature_parity(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "feature_parity"
    findings: List[Finding] = []
    off, scored = bundle.offline, _scored(bundle)

    rows = []
    for feat in FEATURES:
        a, b = off[feat], scored[feat]
        a_clean, b_clean = a.dropna(), b.dropna()
        pooled_sd = float(np.sqrt((a_clean.std() ** 2 + b_clean.std() ** 2) / 2))
        smd = abs(float(b_clean.mean() - a_clean.mean())) / max(pooled_sd, 1e-9)

        # Within-segment SMD controls for population mix: a real-world shift
        # moves the mix, a pipeline skew moves the values inside segments too.
        seg_smds, seg_ws = [], []
        for (scene, tod), g_off in off.groupby(["scene", "time_of_day"]):
            g_sh = scored[(scored["scene"] == scene)
                          & (scored["time_of_day"] == tod)]
            if len(g_sh) < 40 or len(g_off) < 40:
                continue
            ac, bc = g_off[feat].dropna(), g_sh[feat].dropna()
            sd = float(np.sqrt((ac.std() ** 2 + bc.std() ** 2) / 2))
            seg_smds.append(abs(float(bc.mean() - ac.mean())) / max(sd, 1e-9))
            seg_ws.append(len(g_sh))
        within_smd = (float(np.average(seg_smds, weights=seg_ws))
                      if seg_smds else smd)

        q_off = a_clean.quantile([0.05, 0.5, 0.95])
        q_sh = b_clean.quantile([0.05, 0.5, 0.95])
        median_ratio = float(q_sh[0.5] / q_off[0.5]) if q_off[0.5] else float("nan")
        miss_off = float(a.isna().mean())
        miss_sh = float(b.isna().mean())
        skew = within_smd >= 1.0
        rows.append({"feature": feat,
                     "offline_mean": float(a_clean.mean()),
                     "shadow_mean": float(b_clean.mean()),
                     "offline_p5": float(q_off[0.05]), "offline_p50": float(q_off[0.5]),
                     "offline_p95": float(q_off[0.95]),
                     "shadow_p5": float(q_sh[0.05]), "shadow_p50": float(q_sh[0.5]),
                     "shadow_p95": float(q_sh[0.95]),
                     "smd": smd, "within_segment_smd": within_smd,
                     "median_ratio": median_ratio,
                     "offline_missing": miss_off, "shadow_missing": miss_sh,
                     "missing_delta": miss_sh - miss_off,
                     "skew_flag": bool(skew)})

    rows.sort(key=lambda r: r["within_segment_smd"], reverse=True)
    for r in rows:
        if r["skew_flag"]:
            findings.append(make_finding(
                stage, f"FP_FEATURE_SKEW:{r['feature']}",
                f"Training-serving skew on {r['feature']}",
                "MISMATCH", "CRITICAL",
                f"{r['feature']}: shadow mean {r['shadow_mean']:.2f} vs offline "
                f"{r['offline_mean']:.2f} (median ratio {r['median_ratio']:.2f}, "
                f"within-segment SMD {r['within_segment_smd']:.2f}). The value "
                "distribution differs INSIDE matched segments, so this is a "
                "pipeline difference (unit/normalization), not a population "
                "change.",
                data={"feature": r["feature"],
                      "median_ratio": r["median_ratio"],
                      "within_segment_smd": r["within_segment_smd"]}))
        elif abs(r["missing_delta"]) > 0.02:
            findings.append(make_finding(
                stage, f"FP_MISSINGNESS_DELTA:{r['feature']}",
                f"Missingness differs on {r['feature']}", "MISMATCH", "WARN",
                f"Missing offline {r['offline_missing']:.1%} vs shadow "
                f"{r['shadow_missing']:.1%}."))
    if not any(f.status == "MISMATCH" and f.severity == "CRITICAL"
               for f in findings):
        findings.append(make_finding(
            stage, "FP_CLEAN", "No training-serving feature skew", "PASS", "INFO",
            "All features have within-segment SMD < 1.0 and stable "
            "missingness."))

    data = {"rows": rows,
            "explainer": ("Per-feature offline vs shadow deltas ranked by "
                          "WITHIN-SEGMENT standardized mean difference -- the "
                          "statistic that separates 'the world changed' from "
                          "'the pipeline computes this feature differently "
                          "online'.")}
    return data, findings


# ==================================================== stage 8: serving parity

_SERVING_FIELDS = [
    ("confidence_threshold", "Confidence threshold", "CRITICAL"),
    ("quantization", "Quantization / precision", "CRITICAL"),
    ("runtime_version", "Runtime version", "WARN"),
    ("nms_iou", "NMS IoU threshold", "WARN"),
    ("feature_pipeline_version", "Feature pipeline version", "CRITICAL"),
]


def serving_parity(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "serving_parity"
    findings: List[Finding] = []
    off_cfg, sh_cfg = bundle.meta["offline_config"], bundle.meta["shadow_config"]

    rows = []
    for field, label, crit in _SERVING_FIELDS:
        ov, sv = off_cfg.get(field), sh_cfg.get(field)
        if ov is None or sv is None:
            status = "unknown"
            findings.append(make_finding(
                stage, f"SP_UNKNOWN:{field}", f"{label} unrecorded in shadow",
                "UNKNOWN", crit,
                f"Offline={ov!r}, shadow={sv!r}. Serving parity cannot be "
                "certified while this artifact/config is unrecorded."))
        elif ov != sv:
            status = "mismatch"
            findings.append(make_finding(
                stage, f"SP_CONFIG_DIFF:{field}", f"{label} differs in serving",
                "MISMATCH", crit,
                f"Offline={ov!r} vs shadow={sv!r}. The shadow stack is not "
                "running the model the offline harness evaluated.",
                data={"offline": ov, "shadow": sv}))
        else:
            status = "match"
        rows.append({"field": field, "label": label, "offline": ov,
                     "shadow": sv, "status": status, "criticality": crit})

    if not findings:
        findings.append(make_finding(
            stage, "SP_CLEAN", "Serving config at parity", "PASS", "INFO",
            "All recorded serving artifacts/configs match the offline harness."))

    data = {"rows": rows,
            "explainer": ("Artifact/config diff between the offline harness "
                          "and the shadow serving stack: threshold, precision "
                          "mode, runtime, NMS, feature pipeline.")}
    return data, findings


# ================================================== stage 9: shadow traffic


def shadow_traffic(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "shadow_traffic"
    findings: List[Finding] = []
    sh = bundle.shadow
    scored = _scored(bundle)
    unsampled = sh[~sh["sampled"]]
    traffic = dict(bundle.meta["traffic"])

    diff_psi = st.psi_continuous(sh["difficulty"].to_numpy(),
                                 scored["difficulty"].to_numpy())
    conf_psi = st.psi_continuous(sh["a_conf"].to_numpy(),
                                 scored["a_conf"].to_numpy())
    seg_psi = st.psi_categorical(sh["time_of_day"], scored["time_of_day"])
    sampled_delta = _delta_pp(scored)
    eligible_delta = _delta_pp(sh)
    unsampled_delta = _delta_pp(unsampled)
    gap = abs(sampled_delta - unsampled_delta)

    drop_rate = (traffic["dropped_count"] + traffic["fallback_count"]
                 + traffic["timeout_count"]) / max(1, traffic["eligible_count"])

    if diff_psi >= 0.1 or conf_psi >= 0.1 or (gap >= 4.0 and diff_psi >= 0.03):
        findings.append(make_finding(
            stage, "ST_SELECTION_BIAS", "Shadow sample is selection-biased",
            "MISMATCH", "CRITICAL",
            f"Scored sample differs from the eligible stream: difficulty "
            f"PSI={diff_psi:.2f}, confidence PSI={conf_psi:.2f}. Sampled delta "
            f"{sampled_delta:+.1f}pp vs unsampled {unsampled_delta:+.1f}pp "
            f"(eligible stream {eligible_delta:+.1f}pp). The -2% is a "
            "statement about the sampler, not the traffic.",
            data={"difficulty_psi": diff_psi, "conf_psi": conf_psi,
                  "sampled_delta_pp": sampled_delta,
                  "unsampled_delta_pp": unsampled_delta,
                  "eligible_delta_pp": eligible_delta,
                  "sampler": traffic.get("sampler")}))
    else:
        findings.append(make_finding(
            stage, "ST_SAMPLE_FAIR", "Scored sample matches eligible stream",
            "PASS", "INFO",
            f"Difficulty PSI={diff_psi:.3f}, confidence PSI={conf_psi:.3f}, "
            f"sampled vs unsampled delta gap {gap:.1f}pp."))

    if drop_rate > 0.05:
        findings.append(make_finding(
            stage, "ST_DROPS_HIGH", "High dropped/fallback/timeout rate",
            "MISMATCH", "WARN",
            f"{drop_rate:.1%} of eligible traffic was dropped, timed out, or "
            "ran the fallback stack -- those frames never enter the shadow "
            "metric.",
            data={"drop_rate": drop_rate}))

    data = {"traffic": traffic,
            "selection": {"difficulty_psi": diff_psi, "conf_psi": conf_psi,
                          "segment_psi": seg_psi,
                          "sampled_delta_pp": sampled_delta,
                          "unsampled_delta_pp": unsampled_delta,
                          "eligible_delta_pp": eligible_delta,
                          "gap_pp": gap, "drop_rate": drop_rate},
            "explainer": ("Audit of what the shadow sampler actually scored: "
                          "sampling rate, eligibility filters, drops, and "
                          "whether the scored sample statistically matches "
                          "the eligible stream.")}
    return data, findings


# ================================================ stage 10: label integrity


def label_integrity(bundle: ScenarioBundle) -> Tuple[Dict, List[Finding]]:
    stage = "label_integrity"
    findings: List[Finding] = []
    scored = _scored(bundle)
    maturity_h = float(bundle.meta.get("label_maturity_hours", 72.0))
    off_cfg, sh_cfg = bundle.meta["offline_config"], bundle.meta["shadow_config"]

    prov = scored["label_is_provisional"] == True  # noqa: E712 (CSV round-trip)
    prov_frac = float(prov.mean())
    mature = scored[~prov]
    provisional = scored[prov]
    mature_delta = _delta_pp(mature)
    prov_delta = _delta_pp(provisional)

    ages = scored["label_age_hours"]
    hist_edges = [0, 24, 48, 72, 120, 240, 480]
    hist = []
    for lo, hi in zip(hist_edges[:-1], hist_edges[1:]):
        hist.append({"bucket": f"{lo}-{hi}h",
                     "count": int(((ages >= lo) & (ages < hi)).sum())})
    hist.append({"bucket": f">{hist_edges[-1]}h",
                 "count": int((ages >= hist_edges[-1]).sum())})

    by_quartile = []
    q = pd.qcut(scored["difficulty"], 4, labels=["q1", "q2", "q3", "q4"],
                duplicates="drop")
    for label, g in scored.groupby(q, observed=False):
        gp = g[g["label_is_provisional"] == True]  # noqa: E712
        by_quartile.append({
            "quartile": str(label),
            "provisional_frac": float((g["label_is_provisional"] == True).mean()),
            "delta_pp": _delta_pp(g),
            "provisional_delta_pp": _delta_pp(gp) if len(gp) >= 30 else None,
            "n": int(len(g))})

    policy_diff = off_cfg.get("label_policy_version") != sh_cfg.get("label_policy_version")

    if prov_frac > 0.2:
        findings.append(make_finding(
            stage, "LI_PROVISIONAL_HIGH", "Large provisional-label fraction",
            "MISMATCH", "WARN",
            f"{prov_frac:.0%} of scored shadow units are graded against "
            f"provisional labels younger than {maturity_h:.0f}h.",
            data={"provisional_fraction": prov_frac}))
    if prov_frac > 0.1 and len(mature) > 200 and (mature_delta - prov_delta) > 4.0:
        findings.append(make_finding(
            stage, "LI_MATURE_DIVERGES",
            "Verdict flips on mature labels",
            "MISMATCH", "CRITICAL",
            f"On mature labels the paired delta is {mature_delta:+.1f}pp; on "
            f"provisional labels it is {prov_delta:+.1f}pp. The -2% is a "
            "property of the provisional labels, not of the model.",
            data={"mature_delta_pp": mature_delta,
                  "provisional_delta_pp": prov_delta}))
    if len(by_quartile) == 4:
        q1, q4 = by_quartile[0], by_quartile[-1]
        if (q1["provisional_delta_pp"] is not None
                and q4["provisional_delta_pp"] is not None
                and (q1["provisional_delta_pp"] - q4["provisional_delta_pp"]) > 6.0):
            findings.append(make_finding(
                stage, "LI_DIFFICULTY_CORRELATED",
                "Provisional-label penalty concentrates on hard cases",
                "MISMATCH", "WARN",
                f"Provisional-subset delta falls from "
                f"{q1['provisional_delta_pp']:+.1f}pp (easiest quartile) to "
                f"{q4['provisional_delta_pp']:+.1f}pp (hardest) -- the classic "
                "signature of pseudo-labels that are wrong on hard cases."))
    if policy_diff:
        findings.append(make_finding(
            stage, "LI_POLICY_DIFF", "Labeling policy version differs",
            "MISMATCH", "WARN",
            f"Offline: {off_cfg.get('label_policy_version')!r} vs shadow: "
            f"{sh_cfg.get('label_policy_version')!r}."))
    if not findings:
        findings.append(make_finding(
            stage, "LI_CLEAN", "Labels look trustworthy", "PASS", "INFO",
            f"Provisional fraction {prov_frac:.1%}; mature-label delta "
            f"({mature_delta:+.1f}pp) consistent with the overall shadow "
            "verdict; same labeling policy."))

    data = {"provisional_fraction": prov_frac,
            "maturity_hours": maturity_h,
            "age_histogram": hist,
            "mature_delta_pp": mature_delta,
            "provisional_delta_pp": prov_delta,
            "mature_n": int(len(mature)), "provisional_n": int(len(provisional)),
            "by_difficulty_quartile": by_quartile,
            "policy": {"offline": off_cfg.get("label_policy_version"),
                       "shadow": sh_cfg.get("label_policy_version"),
                       "differs": bool(policy_diff)},
            "explainer": ("Are shadow 'ground truth' labels actually ground "
                          "truth? Label age distribution, provisional "
                          "fraction, difficulty correlation, and the "
                          "mature-vs-provisional verdict split.")}
    return data, findings


# ------------------------------------------------------------------ registry

STAGE_DIAGNOSTICS = {
    "comparison_validity": comparison_validity,
    "offline_audit": offline_audit,
    "population_validation": population_validation,
    "distribution_shift": distribution_shift,
    "conditional_performance": conditional_performance,
    "paired_comparison": paired_comparison,
    "statistical_significance": statistical_significance,
    "feature_parity": feature_parity,
    "serving_parity": serving_parity,
    "shadow_traffic": shadow_traffic,
    "label_integrity": label_integrity,
}


def run_all(bundle: ScenarioBundle) -> Dict[str, Tuple[Dict, List[Finding]]]:
    """Run the full diagnostic battery (used by scoring, reports, tests)."""
    return {key: fn(bundle) for key, fn in STAGE_DIAGNOSTICS.items()}
