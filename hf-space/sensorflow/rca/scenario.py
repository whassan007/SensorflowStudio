"""Synthetic investigation scenario generator.

Produces a coherent offline + shadow dataset pair for models A (baseline) and
B (candidate) with a PLANTED root cause. The aggregate story is always the
same -- offline says B is ~+5pp better, shadow says B is ~-2pp worse -- but
the unit-level data carries the planted cause's downstream signatures, so the
diagnostic battery (not the generator) is what identifies the cause.

All randomness flows from a single seed (numpy default_rng), so scenarios are
deterministic and test-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sensorflow.rca.models import ROOT_CAUSES

SEGMENT_DIMS = ("scene", "time_of_day", "object_class")
SCENES = ("urban", "highway", "suburban")
TIMES = ("day", "dusk", "night")
CLASSES = ("vehicle", "pedestrian", "cyclist")

FEATURES = ("obj_distance_m", "ego_speed_mps", "occlusion_ratio",
            "points_in_box", "track_age_s", "ambient_lux")

OFFLINE_MIX = {
    "scene": {"urban": 0.45, "highway": 0.30, "suburban": 0.25},
    "time_of_day": {"day": 0.62, "dusk": 0.24, "night": 0.14},
    "object_class": {"vehicle": 0.60, "pedestrian": 0.25, "cyclist": 0.15},
}

# Shadow mix drifts slightly from offline for every cause (production is never
# identical); DISTRIBUTION_SHIFT replaces this with a heavy night/urban tilt.
SHADOW_MIX_DEFAULT = {
    "scene": {"urban": 0.48, "highway": 0.28, "suburban": 0.24},
    "time_of_day": {"day": 0.58, "dusk": 0.25, "night": 0.17},
    "object_class": {"vehicle": 0.58, "pedestrian": 0.26, "cyclist": 0.16},
}

SHADOW_MIX_SHIFTED = {
    "scene": {"urban": 0.58, "highway": 0.18, "suburban": 0.24},
    "time_of_day": {"day": 0.28, "dusk": 0.22, "night": 0.50},
    "object_class": {"vehicle": 0.52, "pedestrian": 0.30, "cyclist": 0.18},
}

LABEL_MATURITY_HOURS = 72.0

CAUSE_EXPLANATIONS: Dict[str, str] = {
    "TRUE_MODEL_REGRESSION":
        "B genuinely regresses on current production data (concept drift since "
        "the offline eval window). The offline +5% is real for June data, but "
        "stale. Signature: uniform within-segment shadow regression, all parity "
        "checks clean, significance stage confirms a practical regression.",
    "DISTRIBUTION_SHIFT":
        "Production traffic is night/urban heavy while the offline set is "
        "day-dominated. B is better in daytime segments and worse at night IN "
        "BOTH environments -- a Simpson's paradox: the aggregate sign flips "
        "purely from the population mix.",
    "FEATURE_SKEW":
        "The online feature pipeline emits obj_distance_m in feet (x3.281) "
        "while offline uses meters. B depends on that feature more than A, so "
        "shadow degrades B most where distance matters (highway). Signature: "
        "one feature tops the parity report with a huge within-segment delta.",
    "SERVING_MISMATCH":
        "Shadow serving applies confidence threshold 0.50 + int8 quantization "
        "vs the offline harness's 0.35/fp32. B's (recalibrated, lower) "
        "confidences fall into the suppressed band far more often. Signature: "
        "config diff + regression transitions concentrated in the 0.35-0.50 "
        "confidence band.",
    "LABEL_LATENCY":
        "40% of shadow labels are provisional pseudo-labels from the incumbent "
        "auto-labeler (which mirrors A) and are wrong mostly on hard cases. A "
        "gets credit for matching wrong labels; B is penalized for disagreeing. "
        "Signature: mature-label subset shows B ahead; provisional fraction is "
        "difficulty-correlated.",
    "SAMPLING_BIAS":
        "The shadow sampler oversamples hard, low-confidence traffic (triage-"
        "driven selection). B is slightly weaker only on very hard cases. The "
        "full eligible stream agrees with offline (+); the sampled subset "
        "flips negative. Signature: sampled-vs-eligible distribution mismatch "
        "in the traffic audit.",
    "STATISTICAL_NOISE":
        "True delta is ~0 in both environments. Small n plus cluster-"
        "correlated drive segments make the effective sample tiny, so +5%/-2% "
        "are both within noise. Signature: overlapping CIs, low effective n, "
        "significance stage returns insufficient evidence.",
    "OFFLINE_CONTAMINATION":
        "A quarter of the offline eval set are near-duplicates of B's training "
        "data (row-level split leak), inflating B's offline score. Signature: "
        "leakage scan fires, baseline re-run fails to reproduce, version pins "
        "missing; shadow shows no reliable regression.",
}


@dataclass
class ScenarioBundle:
    cause: str
    seed: int
    offline: pd.DataFrame
    shadow: pd.DataFrame
    meta: Dict


# ------------------------------------------------------------------ helpers


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pick(rng: np.random.Generator, table: Dict[str, float], n: int) -> np.ndarray:
    keys = list(table.keys())
    probs = np.array([table[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)


def _sample_units(rng: np.random.Generator, n: int, mix: Dict, cluster_size: int,
                  entity_prefix: str) -> pd.DataFrame:
    """Sample units grouped into drive-segment entities. Scene and time of day
    are entity-level (a drive segment is one place, one time); object class is
    unit-level."""
    n_entities = max(2, int(np.ceil(n / cluster_size)))
    ent_scene = _pick(rng, mix["scene"], n_entities)
    ent_time = _pick(rng, mix["time_of_day"], n_entities)
    ent_ids = np.array([f"{entity_prefix}-{i:05d}" for i in range(n_entities)])

    idx = rng.integers(0, n_entities, size=n)
    scene = ent_scene[idx]
    time_of_day = ent_time[idx]
    entity_id = ent_ids[idx]
    object_class = _pick(rng, mix["object_class"], n)

    dist_mu = np.where(scene == "urban", np.log(18.0),
                       np.where(scene == "highway", np.log(45.0), np.log(28.0)))
    obj_distance = np.exp(rng.normal(dist_mu, 0.45))

    speed_mu = np.where(scene == "urban", 9.0,
                        np.where(scene == "highway", 27.0, 14.0))
    ego_speed = np.clip(rng.normal(speed_mu, 3.5), 0.0, None)

    occ_a = np.where(scene == "urban", 2.5, np.where(scene == "highway", 1.5, 2.0))
    occ_b = np.where(scene == "urban", 3.5, np.where(scene == "highway", 6.0, 5.0))
    occlusion = rng.beta(occ_a, occ_b)

    points = np.exp(rng.normal(6.8 - 1.1 * np.log(obj_distance / 20.0), 0.5))
    track_age = rng.exponential(4.0, size=n)

    lux_mu = np.where(time_of_day == "day", 10.3,
                      np.where(time_of_day == "dusk", 7.5, 3.6))
    lux_sd = np.where(time_of_day == "day", 0.3,
                      np.where(time_of_day == "dusk", 0.4, 0.5))
    ambient_lux = np.exp(rng.normal(lux_mu, lux_sd))

    difficulty = np.clip(
        rng.beta(2.2, 3.2, size=n)
        + 0.20 * (time_of_day == "night")
        + 0.12 * occlusion
        + 0.08 * (obj_distance > 40.0)
        - 0.10, 0.0, 1.0)

    return pd.DataFrame({
        "entity_id": entity_id,
        "scene": scene,
        "time_of_day": time_of_day,
        "object_class": object_class,
        "obj_distance_m": obj_distance,
        "ego_speed_mps": ego_speed,
        "occlusion_ratio": occlusion,
        "points_in_box": points,
        "track_age_s": track_age,
        "ambient_lux": ambient_lux,
        "difficulty": difficulty,
    })


def _segment_adj(df: pd.DataFrame) -> np.ndarray:
    """Shared (model-A) segment difficulty adjustments on the logit scale."""
    adj = np.zeros(len(df))
    adj += np.where(df["time_of_day"] == "night", -0.50,
                    np.where(df["time_of_day"] == "dusk", -0.15, 0.0))
    adj += np.where(df["object_class"] == "pedestrian", -0.30,
                    np.where(df["object_class"] == "cyclist", -0.45, 0.0))
    adj += np.where(df["scene"] == "urban", -0.15,
                    np.where(df["scene"] == "highway", 0.05, 0.0))
    return adj


def _cluster_effects(rng: np.random.Generator, entities: pd.Series,
                     sigma_shared: float, sigma_b: float) -> Tuple[np.ndarray, np.ndarray]:
    uniq = entities.unique()
    shared = dict(zip(uniq, rng.normal(0.0, sigma_shared, size=len(uniq))))
    b_only = dict(zip(uniq, rng.normal(0.0, sigma_b, size=len(uniq))))
    return (entities.map(shared).to_numpy(), entities.map(b_only).to_numpy())


def _draw_correctness(rng: np.random.Generator, df: pd.DataFrame,
                      b_edge: np.ndarray, sigma_shared: float = 0.4,
                      sigma_b_cluster: float = 0.15) -> pd.DataFrame:
    shared_eff, b_cluster = _cluster_effects(rng, df["entity_id"],
                                             sigma_shared, sigma_b_cluster)
    logit_a = 1.9 - 3.0 * df["difficulty"].to_numpy() + _segment_adj(df) + shared_eff
    logit_b = logit_a + b_edge + b_cluster
    p_a, p_b = _sigmoid(logit_a), _sigmoid(logit_b)
    df = df.copy()
    df["a_correct"] = (rng.random(len(df)) < p_a).astype(int)
    df["b_correct"] = (rng.random(len(df)) < p_b).astype(int)
    df["a_conf"] = np.clip(p_a + rng.normal(0, 0.07, len(df)), 0.02, 0.99)
    df["b_conf"] = np.clip(p_b + rng.normal(0, 0.07, len(df)), 0.02, 0.99)
    return df


def _calibrate(df: pd.DataFrame, target_delta: float, rng: np.random.Generator,
               flip_mask: Optional[pd.Series] = None,
               metric_mask: Optional[pd.Series] = None,
               col_a: str = "a_correct", col_b: str = "b_correct") -> pd.DataFrame:
    """Flip a minimal number of candidate-model outcomes so the measured
    aggregate delta lands on target. Flips are allocated proportionally across
    scene x time_of_day strata so planted per-segment structure survives."""
    df = df.copy()
    if metric_mask is None:
        metric_mask = pd.Series(True, index=df.index)
    if flip_mask is None:
        flip_mask = pd.Series(True, index=df.index)

    m = df[metric_mask]
    n_metric = len(m)
    if n_metric == 0:
        return df
    cur = m[col_b].mean() - m[col_a].mean()
    need = target_delta - cur
    k = int(round(abs(need) * n_metric))
    if k == 0:
        return df

    want_value = 1 if need > 0 else 0
    cand = df[flip_mask & metric_mask & (df[col_b] != want_value)]
    if len(cand) == 0:
        cand = df[metric_mask & (df[col_b] != want_value)]
    k = min(k, len(cand))

    strata = cand.groupby(["scene", "time_of_day"]).indices
    chosen: List[int] = []
    total = len(cand)
    for _, idx in strata.items():
        take = int(round(k * len(idx) / total))
        take = min(take, len(idx))
        if take > 0:
            sel = rng.choice(idx, size=take, replace=False)
            chosen.extend(cand.index[sel])
    # Round-off top-up.
    remaining = k - len(chosen)
    if remaining > 0:
        pool = cand.index.difference(pd.Index(chosen))
        if len(pool) > 0:
            extra = rng.choice(pool.to_numpy(), size=min(remaining, len(pool)),
                               replace=False)
            chosen.extend(extra.tolist())
    df.loc[chosen, col_b] = want_value
    return df


# --------------------------------------------------------------- meta builder


def _base_configs() -> Tuple[Dict, Dict]:
    offline = {
        "metric_definition": "object-level accuracy (IoU>=0.5 match, class-aware)",
        "aggregation": "micro-average over objects",
        "eval_window": "2026-06-01 .. 2026-06-30",
        "population_source": "curated holdout eval set (holdout-2026Q2)",
        "confidence_threshold": 0.35,
        "quantization": "fp32",
        "runtime_version": "trt-9.2.1",
        "nms_iou": 0.50,
        "feature_pipeline_version": "fp-2.4.1",
        "label_policy_version": "human-audited-v3",
        "label_maturity_policy": "labels frozen post-audit",
        "sampling_policy": "full eval set, no sampling",
    }
    shadow = dict(offline)
    shadow.update({
        "eval_window": "2026-08-01 .. 2026-08-10 (live)",
        "population_source": "live shadow traffic",
        "sampling_policy": "uniform 60% shadow sampler",
        "label_maturity_policy": "mixed: audited after 72h maturity",
    })
    return offline, shadow


def _traffic_meta(rng: np.random.Generator, eligible: int, sampled: int,
                  biased: bool) -> Dict:
    if biased:
        return {
            "eligible_count": eligible,
            "sampled_count": sampled,
            "sampling_rate": round(sampled / eligible, 4),
            "dropped_count": int(eligible * 0.041),
            "timeout_count": int(eligible * 0.018),
            "fallback_count": int(eligible * 0.062),
            "eligibility_filters": [
                "exclude frames with fallback stack engaged",
                "triage-priority sampler v2 (uplifts low-confidence / "
                "disagreement traffic)",
            ],
            "sampler": "engaged-triage-sampler-v2",
        }
    return {
        "eligible_count": eligible,
        "sampled_count": sampled,
        "sampling_rate": round(sampled / eligible, 4),
        "dropped_count": int(eligible * 0.012),
        "timeout_count": int(eligible * 0.006),
        "fallback_count": int(eligible * 0.009),
        "eligibility_filters": ["exclude frames with fallback stack engaged"],
        "sampler": "uniform-shadow-sampler-v1",
    }


# ------------------------------------------------------------------ generator


def generate_scenario(cause: str, seed: int = 7) -> ScenarioBundle:
    if cause not in ROOT_CAUSES:
        raise ValueError(f"Unknown root cause {cause!r}; expected one of {ROOT_CAUSES}")
    rng = np.random.default_rng(seed)

    n_off, off_cluster = 3000, 8
    n_sh, sh_cluster = 6000, 10
    sampled_rate = 0.6
    sigma_b_cluster = 0.15
    shadow_mix = SHADOW_MIX_DEFAULT
    off_target, sh_target = 0.05, -0.02

    if cause == "STATISTICAL_NOISE":
        n_off, off_cluster = 500, 25
        n_sh, sh_cluster = 420, 20
        sampled_rate = 1.0
        sigma_b_cluster = 0.60
    elif cause == "TRUE_MODEL_REGRESSION":
        n_sh = 20000
        sh_target = -0.023
    elif cause == "DISTRIBUTION_SHIFT":
        shadow_mix = SHADOW_MIX_SHIFTED
    elif cause == "OFFLINE_CONTAMINATION":
        sh_target = -0.010

    off = _sample_units(rng, n_off, OFFLINE_MIX, off_cluster, "off-seg")
    sh = _sample_units(rng, n_sh, shadow_mix, sh_cluster, "sh-seg")

    # ---------------------------------------------------------- B's true edge
    def zeros(df):
        return np.zeros(len(df))

    b_edge_off, b_edge_sh = zeros(off), zeros(sh)
    if cause == "TRUE_MODEL_REGRESSION":
        b_edge_off = np.full(n_off, 0.35)
        b_edge_sh = np.full(n_sh, -0.18)      # concept drift hits B uniformly
    elif cause == "DISTRIBUTION_SHIFT":
        def mix_edge(df):
            return np.where(df["time_of_day"] == "day", 0.60,
                            np.where(df["time_of_day"] == "dusk", 0.10, -0.55))
        b_edge_off, b_edge_sh = mix_edge(off), mix_edge(sh)
    elif cause == "FEATURE_SKEW":
        b_edge_off = np.full(n_off, 0.35)
        w = np.where(sh["scene"] == "highway", 1.0,
                     np.where(sh["scene"] == "suburban", 0.5, 0.2))
        # Skewed distance feature degrades both models online, B far more.
        b_edge_sh = 0.35 - 1.25 * w
    elif cause == "SERVING_MISMATCH":
        b_edge_off = np.full(n_off, 0.35)
        b_edge_sh = np.full(n_sh, 0.35)       # B is genuinely better...
    elif cause == "LABEL_LATENCY":
        b_edge_off = np.full(n_off, 0.35)
        b_edge_sh = np.full(n_sh, 0.32)
    elif cause == "SAMPLING_BIAS":
        # B slightly weaker only on very hard cases, in BOTH environments.
        b_edge_off = 0.55 - 1.1 * off["difficulty"].to_numpy()
        b_edge_sh = 0.55 - 1.1 * sh["difficulty"].to_numpy()
    elif cause == "STATISTICAL_NOISE":
        pass                                   # true delta ~ 0 everywhere
    elif cause == "OFFLINE_CONTAMINATION":
        b_edge_off = np.full(n_off, 0.02)
        b_edge_sh = np.full(n_sh, -0.05)

    off = _draw_correctness(rng, off, b_edge_off, sigma_b_cluster=sigma_b_cluster)
    sh = _draw_correctness(rng, sh, b_edge_sh, sigma_b_cluster=sigma_b_cluster)

    # -------------------------------------------------- offline contamination
    off["is_near_duplicate"] = False
    off["dup_similarity"] = 0.0
    train_entity_ids: List[str] = []
    if cause == "OFFLINE_CONTAMINATION":
        entities = off["entity_id"].unique()
        n_dup_ent = int(len(entities) * 0.25)
        dup_entities = set(rng.choice(entities, size=n_dup_ent, replace=False))
        dup_mask = off["entity_id"].isin(dup_entities)
        off.loc[dup_mask, "is_near_duplicate"] = True
        off.loc[dup_mask, "dup_similarity"] = rng.uniform(0.93, 0.999, dup_mask.sum())
        # B memorized these rows during training.
        mem = dup_mask & (rng.random(n_off) < 0.985)
        off.loc[mem, "b_correct"] = 1
        off.loc[dup_mask, "b_conf"] = np.clip(
            off.loc[dup_mask, "b_conf"] + 0.15, 0.02, 0.995)
        train_entity_ids = sorted(dup_entities)

    # ------------------------------------------------------ shadow mechanisms
    sh["a_correct_true"] = sh["a_correct"]
    sh["b_correct_true"] = sh["b_correct"]
    sh["a_conf_serving"] = sh["a_conf"]
    sh["b_conf_serving"] = sh["b_conf"]

    if cause == "FEATURE_SKEW":
        # Online pipeline logs/consumes distance in feet.
        sh["obj_distance_m"] = sh["obj_distance_m"] * 3.281

    if cause == "SERVING_MISMATCH":
        # int8 quantization nudges confidences down slightly (B more).
        sh["a_conf_serving"] = np.clip(sh["a_conf"] - 0.01, 0.02, 0.99)
        sh["b_conf_serving"] = np.clip(sh["b_conf"] - 0.05, 0.02, 0.99)
        band = (sh["b_conf_serving"] >= 0.35) & (sh["b_conf_serving"] < 0.50)
        suppressed = band & (rng.random(n_sh) < 0.9)
        sh.loc[suppressed, "b_correct"] = 0

    sh["label_is_provisional"] = False
    sh["label_age_hours"] = rng.uniform(80.0, 400.0, n_sh)
    if cause == "LABEL_LATENCY":
        prov = rng.random(n_sh) < 0.40
        sh["label_is_provisional"] = prov
        sh.loc[prov, "label_age_hours"] = rng.uniform(1.0, 60.0, prov.sum())
        diff = sh["difficulty"].to_numpy()
        pseudo_wrong = prov & (rng.random(n_sh) < np.clip(0.65 * diff, 0, 0.9))
        # Wrong pseudo-labels mirror the incumbent (A-like) auto-labeler:
        # A "matches" the wrong label often; B rarely does.
        a_match = pseudo_wrong & (rng.random(n_sh) < 0.75)
        b_match = pseudo_wrong & (rng.random(n_sh) < 0.12)
        sh.loc[pseudo_wrong, "a_correct"] = 0
        sh.loc[pseudo_wrong & a_match, "a_correct"] = 1
        sh.loc[pseudo_wrong, "b_correct"] = 0
        sh.loc[pseudo_wrong & b_match, "b_correct"] = 1

    # ---------------------------------------------------------- shadow sampling
    if cause == "SAMPLING_BIAS":
        disagree = (sh["a_correct_true"] != sh["b_correct_true"]).to_numpy()
        w = 0.12 + 0.55 * disagree + 0.55 * sh["difficulty"].to_numpy() \
            + 0.25 * (sh["a_conf"].to_numpy() < 0.5)
        p_sample = np.clip(w, 0.02, 0.98)
        sh["sampled"] = rng.random(n_sh) < p_sample
    else:
        sh["sampled"] = rng.random(n_sh) < sampled_rate

    # ------------------------------------------------------------- calibration
    off_flip_mask = off["is_near_duplicate"] if cause == "OFFLINE_CONTAMINATION" else None
    off = _calibrate(off, off_target, rng, flip_mask=off_flip_mask)

    if cause == "SERVING_MISMATCH":
        flip = (sh["b_conf_serving"] >= 0.33) & (sh["b_conf_serving"] < 0.52)
    elif cause == "LABEL_LATENCY":
        flip = sh["label_is_provisional"]
    else:
        flip = None
    if cause == "SAMPLING_BIAS":
        # Only the triage-tilted sample regresses; the unsampled remainder of
        # the eligible stream agrees with offline (B ahead).
        sh = _calibrate(sh, sh_target, rng, metric_mask=sh["sampled"])
        sh = _calibrate(sh, 0.06, rng, metric_mask=~sh["sampled"])
    else:
        # Calibrate the full stream, then a small touch-up on the scored
        # sample, so sampled vs unsampled stay consistent (no spurious
        # selection-bias signature).
        sh = _calibrate(sh, sh_target, rng, flip_mask=flip)
        sh = _calibrate(sh, sh_target, rng, flip_mask=flip,
                        metric_mask=sh["sampled"])

    # Light feature missingness (parity report exercises missingness deltas).
    for col, frac_off, frac_sh in (("points_in_box", 0.004, 0.009),
                                   ("track_age_s", 0.002, 0.004)):
        off.loc[off.sample(frac=frac_off, random_state=seed).index, col] = np.nan
        sh.loc[sh.sample(frac=frac_sh, random_state=seed + 1).index, col] = np.nan

    off.insert(0, "unit_id", [f"off-{i:06d}" for i in range(len(off))])
    sh.insert(0, "unit_id", [f"sh-{i:06d}" for i in range(len(sh))])

    # ------------------------------------------------------------------- meta
    off_cfg, sh_cfg = _base_configs()
    if cause == "SERVING_MISMATCH":
        sh_cfg["confidence_threshold"] = 0.50
        sh_cfg["quantization"] = "int8"
        sh_cfg["runtime_version"] = "trt-9.3.0"
    if cause == "FEATURE_SKEW":
        sh_cfg["feature_pipeline_version"] = None   # nobody recorded it
    if cause == "LABEL_LATENCY":
        sh_cfg["label_policy_version"] = "auto-provisional-v1 (mixed)"
        sh_cfg["label_maturity_policy"] = "provisional pseudo-labels until 72h audit"
    if cause == "SAMPLING_BIAS":
        sh_cfg["sampling_policy"] = "engaged-triage-sampler-v2 (priority traffic)"
    if cause == "TRUE_MODEL_REGRESSION":
        off_cfg["eval_window"] = "2026-05-20 .. 2026-06-10 (8 weeks stale)"

    repro_ok = cause != "OFFLINE_CONTAMINATION"
    sampled_count = int(sh["sampled"].sum())

    measured_off = float(off["b_correct"].mean() - off["a_correct"].mean())
    m = sh[sh["sampled"]]
    measured_sh = float(m["b_correct"].mean() - m["a_correct"].mean())

    meta = {
        "cause": cause,
        "seed": seed,
        "offline_config": off_cfg,
        "shadow_config": sh_cfg,
        "model_a": {
            "version": "detr3d-a-v41",
            "train_window": "2025-11 .. 2026-04",
            "train_entity_ids": [],
            "split_strategy": "entity_level",
        },
        "model_b": {
            "version": "detr3d-b-v42",
            "train_window": "2025-12 .. 2026-05",
            "train_entity_ids": train_entity_ids,
            "split_strategy": ("row_level" if cause == "OFFLINE_CONTAMINATION"
                               else "entity_level"),
        },
        "reproducibility": {
            "original_offline_metric_delta": measured_off,
            "rerun_metric_delta": (measured_off + (0.001 if repro_ok else -0.031)),
            "pins_present": repro_ok,
            "environment_lock": ("conda-lock + image digest recorded" if repro_ok
                                 else "requirements.txt without pins; image tag "
                                      "'latest'"),
        },
        "traffic": _traffic_meta(rng, n_sh, sampled_count,
                                 biased=(cause == "SAMPLING_BIAS")),
        "label_maturity_hours": LABEL_MATURITY_HOURS,
        "practical_margin_pp": 1.0,
        "measured": {
            "offline_delta": measured_off,
            "shadow_delta": measured_sh,
            "offline_n": len(off),
            "shadow_scored_n": int(m.shape[0]),
        },
        "explanation": CAUSE_EXPLANATIONS[cause],
    }
    return ScenarioBundle(cause=cause, seed=seed, offline=off, shadow=sh, meta=meta)
