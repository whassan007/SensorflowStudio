"""Deterministic synthetic multimodal scene bank.

Generates driving scenes with per-scene modality availability flags and three
object populations:

  - normal pedestrians (should never become rare-event candidates),
  - PLANTED costumed pedestrians across the full costume taxonomy,
  - CONFOUNDERS (statues, decorations, ads, mannequins, signs, construction
    objects) that must NOT become confident candidates.

Objects expose per-modality `observables` derived from the planted truth with
seeded noise; the miner consumes ONLY those observables, never the truth.
Sequences include multi-frame tracks (5-20 frames), near-duplicate frames, and
scenes whose baseline model predictions contain planted real failures.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from sensorflow.raremine.models import (
    CONFOUNDER_TYPES,
    COSTUME_TYPES,
    CONTEXTS,
    LIGHTING,
    WEATHER,
    BaselinePrediction,
    GtBox,
    RareMineStore,
    Scene,
    SceneBank,
    SceneObject,
    new_id,
)

SIL_LEVELS = ["LOW", "MODERATE", "HIGH", "EXTREME"]
OCC_LEVELS = ["NONE", "PARTIAL", "HEAVY", "EXTREME"]

# How strongly each costume type distorts a human silhouette (typical range).
COSTUME_SILHOUETTE = {
    "mascot": ("HIGH", "EXTREME"),
    "inflatable": ("EXTREME", "EXTREME"),
    "animal": ("HIGH", "EXTREME"),
    "character": ("MODERATE", "HIGH"),
    "robot_armor": ("HIGH", "EXTREME"),
    "oversized": ("MODERATE", "HIGH"),
    "large_prop": ("HIGH", "EXTREME"),
}

COSTUME_TEXTURE = {
    "mascot": "fur",
    "inflatable": "vinyl",
    "animal": "fur",
    "character": "fabric",
    "robot_armor": "metallic",
    "oversized": "fabric",
    "large_prop": "cardboard",
}

CONFOUNDER_TEXTURE = {
    "mascot_statue": "fiberglass",
    "inflatable_decoration": "vinyl",
    "roadside_advertisement": "flat_print",
    "mannequin": "plastic",
    "sign": "flat_print",
    "construction_object": "plastic",
}

FAILURE_CYCLE = [
    "FALSE_NEGATIVE",
    "MISCLASSIFICATION_VEHICLE",
    "MISCLASSIFICATION_ANIMAL",
    "MISCLASSIFICATION_BACKGROUND",
    "LOCALIZATION_ERROR",
    "LOW_CONFIDENCE",
    "TRACKING_FAILURE",
]


def _level_ge(level: str, floor: str, scale: List[str]) -> bool:
    return scale.index(level) >= scale.index(floor)


def _combine_occ(a: str, b: str) -> str:
    return OCC_LEVELS[max(OCC_LEVELS.index(a), OCC_LEVELS.index(b))]


# ------------------------------------------------------------------ observables


def _rgb_observables(rng: random.Random, obj: SceneObject, lighting: str) -> Dict[str, Any]:
    """What a single RGB camera would plausibly report about this object."""
    occ = _combine_occ(obj.truth_occlusion_env, obj.truth_occlusion_costume)
    vis_penalty = {"NONE": 0.0, "PARTIAL": 0.25, "HEAVY": 0.55, "EXTREME": 0.8}[occ]
    vis_penalty += 0.15 if lighting == "night" else 0.0
    vis_penalty += min(obj.distance_m / 120.0, 0.3)
    visibility = max(0.05, 1.0 - vis_penalty + rng.uniform(-0.05, 0.05))

    kind = obj.truth_kind
    if kind == "normal_pedestrian":
        face = visibility > 0.35
        skin = visibility > 0.3
        limbs = visibility > 0.25
        sil = "LOW"
        texture = "fabric"
        bulky = False
    elif kind == "costumed_pedestrian":
        ct = obj.truth_costume_type or "mascot"
        # face/skin peeking through costume seams — rarer for full-body suits
        face = visibility > 0.55 and ct in ("character", "oversized") and rng.random() < 0.7
        skin = visibility > 0.5 and rng.random() < (0.55 if ct in ("character", "oversized") else 0.2)
        limbs = visibility > 0.3 and rng.random() < 0.8
        sil = obj.truth_silhouette_deviation
        texture = COSTUME_TEXTURE[ct]
        bulky = sil in ("HIGH", "EXTREME")
    else:  # confounder
        ct = obj.truth_costume_type or "sign"
        face = ct == "mannequin" and visibility > 0.5
        skin = False
        limbs = ct in ("mannequin", "mascot_statue") and visibility > 0.4
        sil = obj.truth_silhouette_deviation
        texture = CONFOUNDER_TEXTURE[ct]
        bulky = ct in ("mascot_statue", "inflatable_decoration")

    return {
        "visibility": round(visibility, 3),
        "silhouette_deviation_observed": sil,
        "face_visible": bool(face),
        "skin_visible": bool(skin),
        "limbs_visible": bool(limbs),
        "texture": texture,
        "bulky_shape": bool(bulky),
        "flat_appearance": texture == "flat_print",
    }


def _point_cloud_observables(rng: random.Random, obj: SceneObject) -> Dict[str, Any]:
    kind = obj.truth_kind
    ct = obj.truth_costume_type or ""
    points = max(6, int(1400 / max(obj.distance_m, 4.0) + rng.uniform(-8, 8)))
    if kind == "normal_pedestrian":
        shape = "humanoid"
        pedestal = False
    elif kind == "costumed_pedestrian":
        shape = "bulky_blob" if obj.truth_silhouette_deviation in ("HIGH", "EXTREME") else "humanoid"
        pedestal = False
    else:
        shape = {"roadside_advertisement": "flat_panel", "sign": "flat_panel",
                 "construction_object": "rigid_geometric"}.get(ct, "rigid_geometric")
        if ct in ("mascot_statue", "mannequin"):
            shape = "humanoid" if rng.random() < 0.4 else "rigid_geometric"
        pedestal = ct in ("mascot_statue", "mannequin") and rng.random() < 0.75
    return {"point_count": points, "cluster_shape": shape,
            "pedestal_detected": bool(pedestal),
            "height_m": round(rng.uniform(1.5, 2.6) if kind != "confounder" or ct != "sign"
                              else rng.uniform(0.8, 1.4), 2)}


def _temporal_observables(rng: random.Random, obj: SceneObject, frame_count: int) -> Dict[str, Any]:
    kind = obj.truth_kind
    moving = obj.truth_is_moving
    if kind == "confounder":
        articulated = False
        gait = 0.0
        displacement = round(rng.uniform(0.0, 0.05), 3)
        # tethered inflatables sway without displacing
        sway = obj.truth_costume_type == "inflatable_decoration"
    else:
        articulated = moving and rng.random() < 0.92
        gait = round(rng.uniform(0.6, 0.95), 3) if moving else round(rng.uniform(0.0, 0.15), 3)
        displacement = round(rng.uniform(1.0, 8.0), 2) if moving else round(rng.uniform(0.0, 0.2), 2)
        sway = False
    return {
        "frames_observed": frame_count,
        "articulated_motion": bool(articulated),
        "gait_periodicity": gait,
        "displacement_m": displacement,
        "stationary": displacement < 0.3,
        "sway_only": bool(sway),
    }


def _build_observables(rng: random.Random, obj: SceneObject, scene_modalities: Dict[str, bool],
                       lighting: str, frame_count: int) -> Dict[str, Dict[str, Any]]:
    obs: Dict[str, Dict[str, Any]] = {}
    if scene_modalities.get("rgb"):
        obs["rgb"] = _rgb_observables(rng, obj, lighting)
    if scene_modalities.get("multi_camera"):
        obs["multi_camera"] = {
            "views_confirming": rng.randint(1, 3),
            "parallax_consistent_3d": obj.truth_costume_type not in ("roadside_advertisement", "sign")
            or rng.random() < 0.15,
        }
    if scene_modalities.get("point_cloud"):
        obs["point_cloud"] = _point_cloud_observables(rng, obj)
    if scene_modalities.get("lidar_projection"):
        obs["lidar_projection"] = {
            "camera_lidar_extent_agree": rng.random() < 0.9,
            "projected_height_plausible_human": 1.4 <= (obs.get("point_cloud", {}).get("height_m") or rng.uniform(1.5, 2.4)) <= 2.7,
        }
    if scene_modalities.get("lidar_intensity"):
        obs["lidar_intensity"] = {
            "retroreflective": obj.truth_costume_type in ("sign", "roadside_advertisement") and rng.random() < 0.8,
            "intensity_uniform": obj.truth_kind == "confounder" and rng.random() < 0.7,
        }
    if scene_modalities.get("fusion_view"):
        obs["fusion_view"] = {
            "camera_lidar_agree": rng.random() < (0.65 if obj.truth_kind == "costumed_pedestrian" else 0.9),
        }
    if scene_modalities.get("temporal_sequence"):
        obs["temporal_sequence"] = _temporal_observables(rng, obj, frame_count)
    return obs


# ------------------------------------------------------------------ predictions / GT


def _plant_predictions(rng: random.Random, scene: Scene, failure_for: Dict[str, str]) -> None:
    for obj in scene.objects:
        failure = failure_for.get(obj.track_id)
        if failure == "FALSE_NEGATIVE":
            continue  # model emits nothing for this object
        pred = BaselinePrediction(
            prediction_id=new_id("pred"),
            object_id=obj.object_id,
            predicted_class="pedestrian",
            confidence=round(rng.uniform(0.75, 0.97), 3),
            localization_offset_m=round(rng.uniform(0.02, 0.3), 3),
            iou_with_gt=round(rng.uniform(0.72, 0.92), 3),
            track_stable=True,
            planted_failure=failure,
        )
        if failure == "MISCLASSIFICATION_VEHICLE":
            pred.predicted_class = "vehicle"
        elif failure == "MISCLASSIFICATION_ANIMAL":
            pred.predicted_class = "animal"
        elif failure == "MISCLASSIFICATION_BACKGROUND":
            pred.predicted_class = "background"
        elif failure == "LOCALIZATION_ERROR":
            pred.localization_offset_m = round(rng.uniform(1.4, 3.2), 3)
            pred.iou_with_gt = round(rng.uniform(0.05, 0.32), 3)
        elif failure == "LOW_CONFIDENCE":
            pred.confidence = round(rng.uniform(0.08, 0.28), 3)
        elif failure == "TRACKING_FAILURE":
            pred.track_stable = False
        scene.baseline_predictions.append(pred)


def _plant_gt(scene: Scene) -> None:
    for obj in scene.objects:
        if obj.truth_kind == "confounder":
            cls = {"roadside_advertisement": "advertisement", "sign": "sign",
                   "construction_object": "construction_object",
                   "mascot_statue": "statue", "mannequin": "mannequin",
                   "inflatable_decoration": "decoration"}.get(obj.truth_costume_type or "", "static_object")
            scene.gt_boxes.append(GtBox(object_id=obj.object_id, class_name=cls))
        else:
            scene.gt_boxes.append(GtBox(
                object_id=obj.object_id, class_name="pedestrian",
                is_costumed=obj.truth_kind == "costumed_pedestrian",
                costume_type=obj.truth_costume_type))


# ------------------------------------------------------------------ event templates


def _make_object(rng: random.Random, kind: str, costume_type: Optional[str],
                 context: str, track_id: str, moving: bool,
                 distance: Optional[float] = None,
                 sil: Optional[str] = None, occ_env: Optional[str] = None) -> SceneObject:
    if kind == "costumed_pedestrian":
        lo, hi = COSTUME_SILHOUETTE[costume_type or "mascot"]
        sil = sil or rng.choice(SIL_LEVELS[SIL_LEVELS.index(lo):SIL_LEVELS.index(hi) + 1])
        occ_costume = rng.choice(["NONE", "PARTIAL", "PARTIAL", "HEAVY"])
    elif kind == "confounder":
        sil = sil or rng.choice(["MODERATE", "HIGH", "EXTREME"])
        occ_costume = "NONE"
    else:
        sil = "LOW"
        occ_costume = "NONE"
    d = distance if distance is not None else round(rng.uniform(6, 55), 1)
    lateral = rng.uniform(-8, 8)
    return SceneObject(
        object_id=new_id("obj"),
        track_id=track_id,
        truth_kind=kind,  # type: ignore[arg-type]
        truth_costume_type=costume_type,
        truth_silhouette_deviation=sil,  # type: ignore[arg-type]
        truth_occlusion_env=occ_env or rng.choice(["NONE", "NONE", "PARTIAL", "HEAVY"]),  # type: ignore[arg-type]
        truth_occlusion_costume=occ_costume,  # type: ignore[arg-type]
        truth_is_moving=moving,
        distance_m=d,
        context=context,
        position=[round(d, 1), round(lateral, 1)],
        bbox_2d=[round(0.5 + lateral / 40.0, 3), round(0.45 + rng.uniform(-0.05, 0.05), 3),
                 round(max(0.03, 2.2 / d), 3), round(max(0.06, 5.0 / d), 3)],
    )


def _modality_flags(rng: random.Random, multi_frame: bool, with_predictions: bool,
                    with_gt: bool, lidar: Optional[bool] = None) -> Dict[str, bool]:
    has_lidar = lidar if lidar is not None else rng.random() < 0.7
    return {
        "rgb": True,
        "multi_camera": rng.random() < 0.6,
        "lidar_projection": has_lidar and rng.random() < 0.85,
        "lidar_intensity": has_lidar and rng.random() < 0.6,
        "point_cloud": has_lidar,
        "fusion_view": has_lidar and rng.random() < 0.7,
        "temporal_sequence": multi_frame,
        "baseline_predictions": with_predictions,
        "gt_annotations": with_gt,
    }


def generate_scene_bank(store: RareMineStore, n_scenes: int = 60, seed: int = 7) -> SceneBank:
    """Deterministic bank: ~n_scenes frames spread across event sequences."""
    rng = random.Random(seed)
    bank_id = f"bank-{seed}-{n_scenes}"
    # wipe any previous bank content for a clean regeneration
    for coll in ("scenes", "candidates", "track_candidates", "lineage", "runs"):
        store.clear(coll)
    for b in store.all("banks"):
        pass  # keep history of bank metadata; scenes are rebuilt

    scenes: List[Scene] = []
    seq_counter = 0
    planted_rare = 0
    confounders = 0

    def next_seq() -> str:
        nonlocal seq_counter
        seq_counter += 1
        return f"seq-{seed}-{seq_counter:03d}"

    def add_sequence(objects_spec: List[Tuple[str, Optional[str], bool]],
                     context: str, lighting: str, weather: str,
                     n_frames: int, with_predictions: bool, with_gt: bool,
                     failure_for_kind: Optional[str] = None,
                     lidar: Optional[bool] = None,
                     near_dupe_frames: int = 0,
                     sil: Optional[str] = None,
                     distance: Optional[float] = None,
                     occ_env: Optional[str] = None) -> None:
        nonlocal planted_rare, confounders
        seq_id = next_seq()
        modalities = _modality_flags(rng, n_frames > 1, with_predictions, with_gt, lidar=lidar)
        objs: List[SceneObject] = []
        failure_for: Dict[str, str] = {}
        for kind, ct, moving in objects_spec:
            track_id = new_id("trk")
            obj = _make_object(rng, kind, ct, context, track_id, moving,
                               sil=sil if kind == "costumed_pedestrian" else None,
                               distance=distance if kind == "costumed_pedestrian" else None,
                               occ_env=occ_env if kind == "costumed_pedestrian" else None)
            objs.append(obj)
            if kind == "costumed_pedestrian":
                planted_rare += 1
                if failure_for_kind:
                    failure_for[track_id] = failure_for_kind
            elif kind == "confounder":
                confounders += 1

        base_scene_id = None
        for fi in range(n_frames):
            scene = Scene(
                scene_id=new_id("scene"),
                bank_id=bank_id,
                sequence_id=seq_id,
                frame_index=fi,
                lighting=lighting,
                weather=weather,
                modalities=dict(modalities),
            )
            frame_objs = []
            for obj in objs:
                o = obj.model_copy(deep=True)
                o.object_id = f"{obj.object_id}-f{fi}"
                if obj.truth_is_moving:
                    o.position = [round(obj.position[0] - 0.4 * fi, 1), obj.position[1]]
                    o.distance_m = max(3.0, round(obj.distance_m - 0.4 * fi, 1))
                o.observables = _build_observables(rng, o, modalities, lighting, n_frames)
                frame_objs.append(o)
            scene.objects = frame_objs
            if with_gt:
                _plant_gt(scene)
            if with_predictions:
                _plant_predictions(rng, scene, failure_for)
            if fi == 0:
                base_scene_id = scene.scene_id
            scenes.append(scene)

        # explicit near-duplicate copies of the first frame (same event/drive)
        for _ in range(near_dupe_frames):
            dupe = scenes[-n_frames].model_copy(deep=True)
            dupe.scene_id = new_id("scene")
            dupe.near_duplicate_of = base_scene_id
            scenes.append(dupe)

    # --- 1) one long multi-frame track per costume type (full taxonomy),
    #        alternating planted model failures and no-model scenes
    for i, ct in enumerate(COSTUME_TYPES):
        n_frames = [20, 12, 8, 5, 15, 7, 10][i % 7]
        with_preds = i % 3 != 2  # every third costume track has no model outputs
        failure = FAILURE_CYCLE[i % len(FAILURE_CYCLE)] if with_preds else None
        add_sequence(
            [("costumed_pedestrian", ct, True), ("normal_pedestrian", None, True)],
            context=CONTEXTS[i % len(CONTEXTS)],
            lighting=LIGHTING[i % 2],
            weather=WEATHER[i % 3],
            n_frames=n_frames,
            with_predictions=with_preds,
            with_gt=True,
            failure_for_kind=failure,
        )

    # --- 2) single-frame costumed sightings (temporal NOT available)
    for i, ct in enumerate(COSTUME_TYPES):
        add_sequence(
            [("costumed_pedestrian", ct, i % 2 == 0)],
            context=CONTEXTS[(i + 1) % len(CONTEXTS)],
            lighting=LIGHTING[(i + 1) % 2],
            weather=WEATHER[(i + 1) % 3],
            n_frames=1,
            with_predictions=i % 2 == 0,
            with_gt=i % 2 == 0,
            failure_for_kind="LOW_CONFIDENCE" if i % 2 == 0 else None,
        )

    # --- 3) confounders (every type), mostly static, some in event areas
    for i, ct in enumerate(CONFOUNDER_TYPES):
        add_sequence(
            [("confounder", ct, False)],
            context="event_area" if i % 2 == 0 else CONTEXTS[i % len(CONTEXTS)],
            lighting=LIGHTING[i % 2],
            weather="clear",
            n_frames=6 if i % 2 == 0 else 1,
            with_predictions=False,
            with_gt=True,
        )

    # --- 4) a safety-critical CRITICAL exemplar: EXTREME inflatable at a
    #        crosswalk at night with an observed FALSE_NEGATIVE
    add_sequence(
        [("costumed_pedestrian", "inflatable", True)],
        context="crosswalk", lighting="night", weather="clear",
        n_frames=8, with_predictions=True, with_gt=True,
        failure_for_kind="FALSE_NEGATIVE", lidar=True,
        sil="EXTREME", distance=9.0, occ_env="PARTIAL",
    )
    # ... and an easy fully-visible day case for priority-ordering contrast
    add_sequence(
        [("costumed_pedestrian", "character", True)],
        context="sidewalk", lighting="day", weather="clear",
        n_frames=6, with_predictions=True, with_gt=True,
        failure_for_kind=None, lidar=True, sil="HIGH", distance=12.0,
        occ_env="NONE",
    )

    # --- 5) near-duplicate drives: same mascot event repeated (dedup fodder)
    for _ in range(3):
        add_sequence(
            [("costumed_pedestrian", "mascot", True)],
            context="event_area", lighting="day", weather="clear",
            n_frames=5, with_predictions=True, with_gt=True,
            failure_for_kind="MISCLASSIFICATION_ANIMAL",
            near_dupe_frames=2, sil="HIGH", distance=18.0,
        )

    # --- 6) a scene with NO lidar at all (modality-discipline fodder)
    add_sequence(
        [("costumed_pedestrian", "robot_armor", True), ("normal_pedestrian", None, True)],
        context="road_edge", lighting="night", weather="rain",
        n_frames=1, with_predictions=False, with_gt=False, lidar=False,
    )

    # --- 7) normal-pedestrian-only filler up to ~n_scenes frames
    while len(scenes) < n_scenes:
        add_sequence(
            [("normal_pedestrian", None, True), ("normal_pedestrian", None, rng.random() < 0.5)],
            context=rng.choice(CONTEXTS), lighting=rng.choice(LIGHTING),
            weather=rng.choice(WEATHER), n_frames=1,
            with_predictions=rng.random() < 0.5, with_gt=rng.random() < 0.5,
        )

    bank = SceneBank(
        bank_id=bank_id, seed=seed,
        num_scenes=len(scenes), num_sequences=seq_counter,
        num_planted_rare=planted_rare, num_confounders=confounders,
        generation_params={"n_scenes": n_scenes, "seed": seed},
    )
    store.put("banks", bank)
    for s in scenes:
        store.put("scenes", s)
    store.audit("scene_bank_generated", "SceneBank", bank_id,
                f"{len(scenes)} scenes, {seq_counter} sequences, "
                f"{planted_rare} planted rare, {confounders} confounders")
    store.save()
    return bank
