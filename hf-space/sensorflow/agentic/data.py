"""Deterministic synthetic evaluation campaign for the agentic subsystem.

Two coupled data sources, both seeded and reproducible:

1. SCENE CAMPAIGN — reuses sensorflow.bevfusion.scenes.generate_sequences as
   the scene/track source. Every sequence gets a scene context (construction
   zone flag, geo bucket) and every ground-truth box gets a baseline and a
   candidate model prediction. Classification flips are PLANTED as contiguous
   windows (mirroring the scenes module's occlusion-window convention) so that
   temporal context (correct-before / failing / correct-after) genuinely
   exists in the data.

2. RATE POPULATION — a large vectorized paired outcome log (default 240k
   pedestrian observations per model) that provides the denominator for
   frequency claims. Pairing follows the seqeval recipe (shared per-object
   latent + per-model flip stream) so the statistical agent can feed genuine
   paired cluster means into seqeval's anytime-valid machinery.

Numbers are chosen so the motivating failure reproduces: the candidate flips
pedestrian->construction_cone at ~1.2e-4 overall (~0.012%), concentrated in
construction-zone strata, while the baseline sits near 2e-5.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from sensorflow.bevfusion.scenes import SceneSequence, generate_sequences

PRED_CLASSES = ["pedestrian", "cyclist", "motorcycle", "vehicle", "truck",
                "construction_cone", "background"]
GEO_BUCKETS = ["bay_area_urban", "bay_area_suburban", "highway_101"]

DEFAULT_SEED = 101
DEFAULT_N_SEQUENCES = 12
DEFAULT_FRAMES = 24

# Rate-population physics (per-observation flip probabilities).
RATE_N = 240_000
RATE_CLUSTER_SIZE = 100
BASELINE_FLIP_P = 2.0e-5
CANDIDATE_FLIP_P_NORMAL = 3.0e-5
CANDIDATE_FLIP_P_CONSTRUCTION = 8.0e-4
CONSTRUCTION_EXPOSURE = 0.12
NIGHT_SHARE = 0.35


class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    bbox_3d: List[float]
    track_id: str


class ObservationRecord(BaseModel):
    """One GT object in one frame with both models' predictions."""

    sequence_id: str
    frame_id: str
    frame_index: int
    timestamp_us: int
    object_instance_id: str
    gt_class: str
    gt_bbox_3d: List[float]
    distance_m: float
    occluded: bool
    baseline: Prediction
    candidate: Prediction
    construction_zone: bool
    time_of_day: str
    weather: str
    geo_bucket: str
    has_planner_trace: bool = False


class SequenceContext(BaseModel):
    sequence_id: str
    construction_zone: bool
    geo_bucket: str
    time_of_day: str
    weather: str


class Campaign(BaseModel):
    campaign_id: str
    seed: int
    baseline_model: str = "baseline-v1"
    candidate_model: str = "candidate-v2"
    feature_pipeline_version: str = "featpipe-3.4.1"
    software_config: Dict[str, str] = Field(default_factory=lambda: {
        "perception_stack": "sensorflow-bevfusion-sim",
        "build": "synthetic-campaign",
    })
    hardware_config: Dict[str, str] = Field(default_factory=lambda: {
        "compute": "synthetic-replay-node", "sensors": "cam+lidar (simulated)",
    })
    contexts: List[SequenceContext] = Field(default_factory=list)
    observations: List[ObservationRecord] = Field(default_factory=list)


class RatePopulation(BaseModel):
    """Vectorized paired outcome log (arrays kept out of pydantic)."""

    population_id: str
    seed: int
    n: int
    n_clusters: int
    description: str

    model_config = {"arbitrary_types_allowed": True}


# ------------------------------------------------------------------ scene campaign


def _noisy_box(rng: np.random.Generator, bbox: List[float], sigma: float) -> List[float]:
    out = list(bbox)
    out[0] = round(out[0] + float(rng.normal(0, sigma)), 3)
    out[1] = round(out[1] + float(rng.normal(0, sigma)), 3)
    return out


def build_campaign(seed: int = DEFAULT_SEED, n_sequences: int = DEFAULT_N_SEQUENCES,
                   frames_per_sequence: int = DEFAULT_FRAMES) -> Campaign:
    sequences: List[SceneSequence] = generate_sequences(
        n_sequences=n_sequences, frames_per_sequence=frames_per_sequence, seed=seed)
    rng = np.random.default_rng([seed, 0xA6E])

    campaign = Campaign(campaign_id=f"campaign-{seed}-{n_sequences}x{frames_per_sequence}",
                        seed=seed)

    for qi, seq in enumerate(sequences):
        construction = qi % 4 == 0  # every 4th sequence is a construction zone
        ctx = SequenceContext(
            sequence_id=seq.sequence_id,
            construction_zone=construction,
            geo_bucket=GEO_BUCKETS[qi % len(GEO_BUCKETS)],
            time_of_day=seq.time_of_day,
            weather=seq.weather,
        )
        campaign.contexts.append(ctx)

        # Plant candidate flip windows on pedestrians in construction zones:
        # contiguous frame windows where pedestrian -> construction_cone.
        flip_windows: Dict[str, tuple] = {}
        if construction:
            ped_ids = sorted({gt.instance_id for fr in seq.frames for gt in fr.gt
                              if gt.class_name == "pedestrian"})
            for k, iid in enumerate(ped_ids[:2]):
                start = 6 + 5 * k
                flip_windows[iid] = (start, start + 3)

        for frame in seq.frames:
            for gt in frame.gt:
                b_cls = gt.class_name
                c_cls = gt.class_name
                win = flip_windows.get(gt.instance_id)
                flipped = bool(win and win[0] <= frame.index < win[1])
                if flipped:
                    c_cls = "construction_cone"
                # tiny background rate of unrelated flips for both models
                elif rng.random() < 0.002:
                    c_cls = "vehicle" if gt.class_name != "vehicle" else "truck"
                if rng.random() < 0.001:
                    b_cls = "vehicle" if gt.class_name != "vehicle" else "truck"

                conf_c = 0.62 if flipped else float(np.clip(rng.normal(0.88, 0.05), 0.3, 0.99))
                conf_b = float(np.clip(rng.normal(0.87, 0.05), 0.3, 0.99))
                obs = ObservationRecord(
                    sequence_id=seq.sequence_id,
                    frame_id=frame.frame_id,
                    frame_index=frame.index,
                    timestamp_us=frame.index * 100_000,
                    object_instance_id=gt.instance_id,
                    gt_class=gt.class_name,
                    gt_bbox_3d=gt.bbox_3d,
                    distance_m=gt.distance,
                    occluded=gt.occluded,
                    baseline=Prediction(predicted_class=b_cls, confidence=round(conf_b, 3),
                                        bbox_3d=_noisy_box(rng, gt.bbox_3d, 0.12),
                                        track_id=f"trk-b-{gt.instance_id}"),
                    candidate=Prediction(predicted_class=c_cls, confidence=round(conf_c, 3),
                                         bbox_3d=_noisy_box(rng, gt.bbox_3d, 0.15),
                                         track_id=f"trk-c-{gt.instance_id}"),
                    construction_zone=construction,
                    time_of_day=seq.time_of_day,
                    weather=seq.weather,
                    geo_bucket=ctx.geo_bucket,
                    # replay planner-response traces exist for half the planted
                    # flip windows (deterministically: first planted object)
                    has_planner_trace=bool(flipped and win == sorted(flip_windows.values())[0]),
                )
                campaign.observations.append(obs)
    return campaign


# ------------------------------------------------------------------ rate population


class RateArrays:
    """Raw numpy arrays for the rate population (paired outcomes + strata)."""

    def __init__(self, seed: int, n: int = RATE_N):
        rng_strata = np.random.default_rng([seed, 11])
        self.n = int(n)
        self.cluster_id = np.arange(self.n) // RATE_CLUSTER_SIZE
        self.n_clusters = int(self.cluster_id.max()) + 1
        self.construction = rng_strata.random(self.n) < CONSTRUCTION_EXPOSURE
        self.night = rng_strata.random(self.n) < NIGHT_SHARE
        self.geo = rng_strata.integers(0, len(GEO_BUCKETS), size=self.n)

        p_base = np.full(self.n, BASELINE_FLIP_P)
        p_cand = np.where(self.construction, CANDIDATE_FLIP_P_CONSTRUCTION,
                          CANDIDATE_FLIP_P_NORMAL)
        p_cand = p_cand * np.where(self.night, 1.3, 1.0)

        # seqeval-style pairing: shared latent + per-model flip stream.
        u_pair = np.random.default_rng([seed, 0x5E9]).random(self.n)
        b_noise = np.random.default_rng([seed, 21]).random(self.n)
        c_noise = np.random.default_rng([seed, 22]).random(self.n)
        u_b = np.where(b_noise < 0.05, np.random.default_rng([seed, 31]).random(self.n), u_pair)
        u_c = np.where(c_noise < 0.05, np.random.default_rng([seed, 32]).random(self.n), u_pair)
        self.baseline_flip = (u_b < p_base)
        self.candidate_flip = (u_c < p_cand)

    def fingerprint(self) -> str:
        blob = json.dumps({"n": self.n, "b": int(self.baseline_flip.sum()),
                           "c": int(self.candidate_flip.sum())}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


_CACHE: Dict[int, Dict] = {}
_LOCK = threading.Lock()


def get_campaign(seed: int = DEFAULT_SEED) -> Campaign:
    with _LOCK:
        entry = _CACHE.setdefault(seed, {})
        if "campaign" not in entry:
            entry["campaign"] = build_campaign(seed)
        return entry["campaign"]


def get_rate_arrays(seed: int = DEFAULT_SEED) -> RateArrays:
    with _LOCK:
        entry = _CACHE.setdefault(seed, {})
        if "rates" not in entry:
            entry["rates"] = RateArrays(seed)
        return entry["rates"]


def reset_data_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def campaign_fingerprint(campaign: Campaign) -> str:
    blob = json.dumps({"id": campaign.campaign_id, "seed": campaign.seed,
                       "obs": len(campaign.observations)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def observations_for_instance(campaign: Campaign, sequence_id: str,
                              object_instance_id: str) -> List[ObservationRecord]:
    return sorted((o for o in campaign.observations
                   if o.sequence_id == sequence_id
                   and o.object_instance_id == object_instance_id),
                  key=lambda o: o.frame_index)


def context_for(campaign: Campaign, sequence_id: str) -> Optional[SequenceContext]:
    for c in campaign.contexts:
        if c.sequence_id == sequence_id:
            return c
    return None
