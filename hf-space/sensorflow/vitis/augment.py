"""Synthetic rare-edge-case generation on the acceleration backends.

Parameterized augmentations (sensor noise, low light, HDR extremes, lens
distortion, chromatic aberration, motion blur, glare) mint stress variants
of evaluation frames. Heavy image math routes through the selected
`VisionBackend`, so the same recipes run on the emulated FPGA path.

Platform integration:
* Variants are written as evaluation-set supplements shaped like the
  evaluation package's frame records (frame_id / dataset_id / scene tags /
  gt_boxes with class_name + bbox_3d), with FULL lineage (recipe, seed,
  source frame) on every record.
* Leakage guard (mirrors raremine's PROTECTED_EVAL_DESTINATIONS concept):
  every variant is `evaluation_only=True, training_eligible=False` by
  default and destined for REGRESSION_EVALUATION_SET.
* If sensorflow.raremine is importable, costume-pedestrian-like stress
  variants are also routed into its scene-bank candidate store; when it is
  not (mid-edit), the hook degrades gracefully and the run notes it.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from sensorflow.bevfusion.scenes import generate_sequences
from sensorflow.vitis.backend import DeviceConfig, PipelineConfig, get_backend
from sensorflow.vitis.png import png_data_uri
from sensorflow.vitis.render import render_bev_rgb
from sensorflow.vitis.store import is_default_root, new_id, save_run, vitis_root

PROTECTED_DESTINATION = "REGRESSION_EVALUATION_SET"


# --------------------------------------------------------------------------
# Augmentation library
# --------------------------------------------------------------------------

def _aug_sensor_noise(img, params, rng, be):
    shot = float(params.get("shot", 0.06))
    read = float(params.get("read", 0.02))
    noisy = img + rng.normal(0.0, read, img.shape) + \
        rng.normal(0.0, 1.0, img.shape) * shot * np.sqrt(np.maximum(img, 0.0))
    return be.gain_exposure(np.clip(noisy, 0.0, 1.0).astype(np.float32), gain=1.0)


def _aug_low_light(img, params, rng, be):
    ev = float(params.get("ev_drop", 3.0))  # stops of underexposure
    dark = be.gain_exposure(img, gain=float(2.0 ** -ev))
    noisy = dark + rng.normal(0.0, 0.03, dark.shape).astype(np.float32)
    return be.gain_exposure(np.clip(noisy, 0.0, 1.0).astype(np.float32),
                            gain=float(2.0 ** (ev * 0.6)))  # partial re-gain


def _aug_hdr_extreme(img, params, rng, be):
    boost = float(params.get("highlight_boost", 6.0))
    hot = np.clip(img * boost, 0.0, boost).astype(np.float32)
    return be.hdr_tone_map(hot, white_point=float(params.get("white_point", 2.0)))


def _aug_lens_distortion(img, params, rng, be):
    return be.lens_distortion(img, k1=float(params.get("k1", 0.35)),
                              k2=float(params.get("k2", 0.1)), mode="apply")


def _aug_chromatic_aberration(img, params, rng, be):
    if img.ndim != 3:
        img = np.stack([img] * 3, axis=-1)
    shift = float(params.get("strength", 0.06))
    r = be.lens_distortion(img[..., 0], k1=shift, mode="apply")
    b = be.lens_distortion(img[..., 2], k1=-shift, mode="apply")
    return np.stack([r, img[..., 1], b], axis=-1).astype(np.float32)


def _aug_motion_blur(img, params, rng, be):
    length = max(2, int(params.get("length_px", 7)))
    angle = float(params.get("angle_deg", 0.0))
    k = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) / 2.0
    for t in np.linspace(-c, c, length * 4):
        y = int(round(c + t * np.sin(np.radians(angle))))
        x = int(round(c + t * np.cos(np.radians(angle))))
        if 0 <= y < length and 0 <= x < length:
            k[y, x] = 1.0
    k /= max(k.sum(), 1.0)
    if img.ndim == 3:
        out = np.stack([ndimage.convolve(img[..., ch], k, mode="nearest")
                        for ch in range(img.shape[2])], axis=-1)
    else:
        out = ndimage.convolve(img, k, mode="nearest")
    return be.gain_exposure(out.astype(np.float32), gain=1.0)


def _aug_glare(img, params, rng, be):
    strength = float(params.get("strength", 0.8))
    cy = rng.uniform(0.2, 0.8) * img.shape[0]
    cx = rng.uniform(0.2, 0.8) * img.shape[1]
    yy, xx = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]),
                         indexing="ij")
    sigma = float(params.get("radius_px", 28.0))
    bloom = strength * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    bloom = bloom.astype(np.float32)
    if img.ndim == 3:
        bloom = bloom[..., None]
    flared = np.clip(img + bloom, 0.0, 4.0).astype(np.float32)
    return be.hdr_tone_map(flared, white_point=3.0)


AUGMENTATIONS = {
    "sensor_noise": {"fn": _aug_sensor_noise,
                     "description": "Shot + read noise at configurable levels.",
                     "defaults": {"shot": 0.06, "read": 0.02}},
    "low_light": {"fn": _aug_low_light,
                  "description": "EV underexposure, noise floor, partial re-gain.",
                  "defaults": {"ev_drop": 3.0}},
    "hdr_extreme": {"fn": _aug_hdr_extreme,
                    "description": "Blown highlights pushed through tone mapping.",
                    "defaults": {"highlight_boost": 6.0, "white_point": 2.0}},
    "lens_distortion": {"fn": _aug_lens_distortion,
                        "description": "Radial barrel/pincushion distortion.",
                        "defaults": {"k1": 0.35, "k2": 0.1}},
    "chromatic_aberration": {"fn": _aug_chromatic_aberration,
                             "description": "Opposed radial shifts of R/B channels.",
                             "defaults": {"strength": 0.06}},
    "motion_blur": {"fn": _aug_motion_blur,
                    "description": "Directional blur kernel (length, angle).",
                    "defaults": {"length_px": 7, "angle_deg": 0.0}},
    "glare": {"fn": _aug_glare,
              "description": "Localized bloom pushed through tone mapping.",
              "defaults": {"strength": 0.8, "radius_px": 28.0}},
}


def list_augmentations() -> List[Dict]:
    return [{"name": k, "description": v["description"],
             "defaults": v["defaults"]} for k, v in sorted(AUGMENTATIONS.items())]


# --------------------------------------------------------------------------
# raremine integration hook (in-progress package: import behind try/except)
# --------------------------------------------------------------------------

def _route_to_raremine(variants: List[Dict]) -> Dict:
    """Route pedestrian-bearing stress variants into raremine's candidate
    store if that package is importable; otherwise report the hook point."""
    ped_variants = [v for v in variants
                    if any(g["class_name"] == "pedestrian" for g in v["gt_boxes"])]
    try:
        from sensorflow.raremine.models import Candidate, Evidence, RareMineStore  # type: ignore
        store = RareMineStore()
        routed = 0
        for v in ped_variants:
            store.put("candidates", Candidate(
                candidate_id=f"vitis-{v['variant_id']}",
                bank_id="vitis-augment",
                scene_id=v["source"]["scene_id"],
                sequence_id=v["source"]["sequence_id"],
                object_id="",
                track_id="",
                frame_index=v["source"]["frame_index"],
                run_id=v["lineage"]["batch_id"],
                edge_case_detected=True,
                event_type="synthetic_stress_variant",
                costume_type=["augmented"],
                visual_evidence=[Evidence(
                    modality="camera",
                    description=(f"Vitis-generated stress variant "
                                 f"({', '.join(a['aug'] for a in v['recipe'])})"),
                )],
                recommended_dataset_destination=PROTECTED_DESTINATION,
                requires_human_validation=True,
            ))
            routed += 1
        return {"available": True, "routed_candidates": routed}
    except Exception as e:
        return {"available": False, "routed_candidates": 0,
                "note": ("sensorflow.raremine not importable/compatible right "
                         f"now ({type(e).__name__}); hook point: route "
                         f"{len(ped_variants)} pedestrian-bearing variants "
                         "into RareMineStore('candidates') once it lands.")}


# --------------------------------------------------------------------------
# Batch generation
# --------------------------------------------------------------------------

def generate_batch(recipes: Optional[List[Dict]] = None, n_variants: int = 12,
                   seed: int = 23, backend_name: str = "vitis_emulated",
                   width_bits: int = 12, int_bits: int = 4,
                   device: str = "versal-ai-edge",
                   include_thumbnails: bool = True,
                   persist: bool = True) -> Dict:
    """Mint stress variants of evaluation frames with full lineage.

    recipes: list of {"aug": name, "params": {...}}; omitted -> one variant
    per registered augmentation, cycling.
    """
    t0 = time.perf_counter()
    if recipes:
        for r in recipes:
            if r.get("aug") not in AUGMENTATIONS:
                raise ValueError(f"Unknown augmentation {r.get('aug')!r}; "
                                 f"known: {sorted(AUGMENTATIONS)}")
    else:
        recipes = [{"aug": name, "params": {}} for name in sorted(AUGMENTATIONS)]

    config = PipelineConfig(precision={"default": (width_bits, int_bits)},
                            device=DeviceConfig(name=device))
    be = get_backend(backend_name, config)

    sequences = generate_sequences(
        n_sequences=max(2, (n_variants + 3) // 4), frames_per_sequence=6,
        seed=seed)
    frames = [(seq, fr) for seq in sequences for fr in seq.frames if fr.gt]

    batch_id = new_id("augbatch")
    rng = np.random.default_rng(seed)
    variants: List[Dict] = []
    for vi in range(n_variants):
        seq, frame = frames[vi % len(frames)]
        recipe = [dict(recipes[vi % len(recipes)])]
        # Occasionally stack a second augmentation for compound stress.
        if vi % 3 == 2 and len(recipes) > 1:
            recipe.append(dict(recipes[(vi + 1) % len(recipes)]))
        variant_seed = int(rng.integers(0, 2**31))
        vrng = np.random.default_rng(variant_seed)
        img = render_bev_rgb(frame, seq, seed)
        applied = []
        for step in recipe:
            spec = AUGMENTATIONS[step["aug"]]
            params = {**spec["defaults"], **(step.get("params") or {})}
            img = spec["fn"](img, params, vrng, be)
            applied.append({"aug": step["aug"], "params": params})
        variant_id = new_id("vaug")
        record = {
            # Evaluation-supplement record: mirrors the evaluation package's
            # frame shape (frame_id/dataset_id/scene tags/gt boxes).
            "variant_id": variant_id,
            "frame_id": f"{variant_id}-frame",
            "dataset_id": "vitis-augment-supplement",
            "scene_id": seq.sequence_id,
            "sequence_id": seq.sequence_id,
            "weather": seq.weather,
            "time_of_day": seq.time_of_day,
            "scenario_tags": ["synthetic_stress", "vitis_augment"] +
                             [a["aug"] for a in applied],
            "gt_boxes": [{"gt_id": g.instance_id, "class_name": g.class_name,
                          "bbox_3d": g.bbox_3d,
                          "track_instance_id": g.instance_id,
                          "gt_type": "synthetic"} for g in frame.gt],
            # Leakage guard: evaluation-only by default, never training data.
            "evaluation_only": True,
            "training_eligible": False,
            "protected_evaluation": True,
            "recommended_dataset_destination": PROTECTED_DESTINATION,
            "lineage": {
                "batch_id": batch_id,
                "source_frame_id": frame.frame_id,
                "source_sequence_id": seq.sequence_id,
                "source_generator": "sensorflow.bevfusion.scenes.generate_sequences",
                "recipe": applied,
                "seed": variant_seed,
                "backend": be.name,
                "backend_emulated": be.is_emulated,
                "backend_config": config.to_dict(),
            },
            "source": {"scene_id": seq.sequence_id,
                       "sequence_id": seq.sequence_id,
                       "frame_index": frame.index},
            "recipe": applied,
        }
        if include_thumbnails:
            thumb = img if img.ndim == 3 else np.stack([img] * 3, axis=-1)
            record["thumbnail_png"] = png_data_uri(thumb[::2, ::2])
        variants.append(record)

    # Only route into raremine's real candidate store for real persisted
    # batches; test batches (persist=False or redirected vitis root) must not
    # touch raremine's shared store.
    if persist and is_default_root():
        raremine_hook = _route_to_raremine(variants)
    else:
        raremine_hook = {"available": False, "routed_candidates": 0,
                         "note": "routing skipped for ephemeral/test batch"}

    payload = {
        "run_id": batch_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "augment",
        "params": {"n_variants": n_variants, "seed": seed,
                   "backend": backend_name, "width_bits": width_bits,
                   "device": device,
                   "recipes": [r.get("aug") for r in recipes]},
        "variants": variants,
        "raremine_hook": raremine_hook,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": {"n_variants": len(variants),
                    "evaluation_only": True,
                    "raremine_routed": raremine_hook["routed_candidates"]},
        "emulation_note": ("Augmentations executed on the selected backend; "
                           "vitis_emulated is a CPU emulator of FPGA constraints."),
    }
    if persist:
        save_run("augment", batch_id, payload)
    return payload


def list_variants() -> List[Dict]:
    """Lineage listing across all persisted augmentation batches."""
    d = vitis_root() / "augment"
    out: List[Dict] = []
    if not d.exists():
        return out
    import json
    for p in sorted(d.glob("augbatch-*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for v in data.get("variants", []):
            out.append({k: v[k] for k in
                        ("variant_id", "frame_id", "dataset_id", "scenario_tags",
                         "evaluation_only", "training_eligible", "lineage")
                        if k in v})
    return out
