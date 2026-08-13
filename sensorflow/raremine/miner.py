"""Rule-based rare-event candidate proposer (the deterministic default).

The miner PROPOSES; it never validates, never grades, never decides training
usage. Hard rules enforced here:

  - consumes ONLY modalities flagged available on the scene — every evidence
    item carries its source modality and is checked against the scene flags;
  - three SEPARATE confidences (human identity, costume, rare event);
  - temporal validation is NOT_AVAILABLE on single frames — motion is never
    invented;
  - observed model behavior is populated ONLY when baseline predictions were
    supplied, and is kept strictly separate from the predicted failure mode;
  - unverified candidates are never routed to TRAINING_CANDIDATE.

An optional LLM/VLM narrative (Ollama, copilot pattern) is exposed at the API
layer; mining itself always runs deterministically without any model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sensorflow.raremine.models import (
    EVIDENCE_MODALITIES,
    Candidate,
    AlternativeHypothesis,
    Evidence,
    ObservedModelBehavior,
    RareMineStore,
    Scene,
    SceneObject,
    TemporalValidation,
    new_id,
)

SIL_ORDER = ["LOW", "MODERATE", "HIGH", "EXTREME"]
OCC_ORDER = ["NONE", "PARTIAL", "HEAVY", "EXTREME"]
DIFF_ORDER = ["EASY", "MODERATE", "HARD", "EXTREME"]
PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# texture -> plausible costume families (the miner's taxonomy inference)
TEXTURE_TO_COSTUME = {
    "fur": ["mascot", "animal"],
    "vinyl": ["inflatable"],
    "metallic": ["robot_armor"],
    "fabric": ["character", "oversized"],
    "cardboard": ["large_prop"],
}

MANUFACTURED_TEXTURES = {"fiberglass", "plastic", "flat_print"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "rare_event_threshold": 0.45,
    "min_human_identity": 0.30,
    "min_costume": 0.35,
    # additive rare-confidence boost per inferred costume family, fed back by
    # the improvement report of a previous run
    "sensitivity_boost": {},
    "insufficient_visibility": 0.15,
}


def _clamp(x: float, lo: float = 0.0, hi: float = 0.98) -> float:
    return round(max(lo, min(hi, x)), 3)


class _EvidenceCollector:
    """Evidence sink that HARD-ENFORCES modality discipline: any attempt to
    record evidence from a modality the scene does not provide raises."""

    def __init__(self, scene: Scene):
        self.available = {m for m, on in scene.modalities.items() if on}
        self.items: List[Evidence] = []

    def add(self, modality: str, description: str) -> None:
        if modality not in EVIDENCE_MODALITIES:
            raise ValueError(f"{modality} is not an evidence modality")
        if modality not in self.available:
            raise ValueError(
                f"modality discipline violation: evidence from unavailable modality '{modality}'")
        self.items.append(Evidence(modality=modality, description=description))


def validate_candidate_modalities(candidate: Candidate, scene: Scene) -> None:
    """Final guard called before persisting any candidate."""
    available = {m for m, on in scene.modalities.items() if on}
    for ev in candidate.visual_evidence + candidate.human_identity_evidence:
        if ev.modality not in available:
            raise ValueError(
                f"candidate {candidate.candidate_id} cites absent modality {ev.modality}")
    if candidate.temporal_validation.available and not scene.modalities.get("temporal_sequence"):
        raise ValueError("temporal validation claimed without temporal_sequence modality")
    if candidate.observed_model_behavior is not None and not scene.modalities.get("baseline_predictions"):
        raise ValueError("observed model behavior without baseline predictions")


# ------------------------------------------------------------------ sub-analyses


def _analyze_human_identity(obj: SceneObject, ev: _EvidenceCollector,
                            identity_ev: List[Evidence]) -> float:
    obs = obj.observables
    score = 0.15

    rgb = obs.get("rgb")
    if rgb:
        if rgb["face_visible"]:
            score += 0.25
            identity_ev.append(Evidence(modality="rgb", description="partial human face visible"))
        if rgb["skin_visible"]:
            score += 0.15
            identity_ev.append(Evidence(modality="rgb", description="exposed skin visible at costume seam"))
        if rgb["limbs_visible"]:
            score += 0.12
            identity_ev.append(Evidence(modality="rgb", description="articulated limb structure visible"))
        if rgb["flat_appearance"]:
            score -= 0.22
            ev.add("rgb", "flat printed appearance — inconsistent with a physical person")

    temporal = obs.get("temporal_sequence")
    if temporal:
        if temporal["articulated_motion"]:
            score += 0.25
            identity_ev.append(Evidence(
                modality="temporal_sequence",
                description=f"articulated body motion across {temporal['frames_observed']} frames"))
        if temporal["gait_periodicity"] > 0.5:
            score += 0.15
            identity_ev.append(Evidence(
                modality="temporal_sequence",
                description=f"periodic gait signature (periodicity {temporal['gait_periodicity']:.2f})"))
        if temporal["displacement_m"] > 0.5:
            score += 0.08
            identity_ev.append(Evidence(
                modality="temporal_sequence",
                description=f"self-propelled displacement of {temporal['displacement_m']:.1f} m"))
        if temporal["stationary"] and not temporal["articulated_motion"]:
            score -= 0.12
            ev.add("temporal_sequence", "object fully static with no articulation across sequence")
        if temporal.get("sway_only"):
            score -= 0.1
            ev.add("temporal_sequence", "sway without displacement — consistent with a tethered object")

    pc = obs.get("point_cloud")
    if pc:
        if pc["cluster_shape"] == "humanoid":
            score += 0.08
            identity_ev.append(Evidence(modality="point_cloud",
                                        description="LiDAR cluster has humanoid proportions"))
        if pc["pedestal_detected"]:
            score -= 0.28
            ev.add("point_cloud", "mounting pedestal detected under object")
        if pc["cluster_shape"] == "flat_panel":
            score -= 0.2
            ev.add("point_cloud", "flat panel geometry — no human volume")

    mc = obs.get("multi_camera")
    if mc and not mc["parallax_consistent_3d"]:
        score -= 0.2
        ev.add("multi_camera", "parallax inconsistent with a 3D physical object (planar surface)")

    li = obs.get("lidar_intensity")
    if li:
        if li["retroreflective"]:
            score -= 0.12
            ev.add("lidar_intensity", "retroreflective return typical of signage")
        if li["intensity_uniform"]:
            score -= 0.06
            ev.add("lidar_intensity", "uniform intensity typical of manufactured surfaces")

    return _clamp(score)


def _analyze_costume(obj: SceneObject, ev: _EvidenceCollector) -> tuple[float, List[str], List[str]]:
    """Returns (costume confidence, inferred costume types, unusual geometry)."""
    obs = obj.observables
    score = 0.05
    inferred: List[str] = []
    geometry: List[str] = []

    rgb = obs.get("rgb")
    sil = rgb["silhouette_deviation_observed"] if rgb else "LOW"
    if rgb:
        score += {"LOW": 0.0, "MODERATE": 0.25, "HIGH": 0.45, "EXTREME": 0.6}[sil]
        if sil != "LOW":
            ev.add("rgb", f"silhouette deviates {sil} from human reference shape")
            geometry.append(f"{sil.lower()} silhouette deviation from canonical human outline")
        texture = rgb["texture"]
        if texture in TEXTURE_TO_COSTUME:
            score += 0.15
            inferred = list(TEXTURE_TO_COSTUME[texture])
            ev.add("rgb", f"surface texture '{texture}' consistent with {'/'.join(inferred)} costume")
        if rgb["bulky_shape"]:
            score += 0.1
            geometry.append("bulk exceeding normal human body envelope")
            ev.add("rgb", "bulky non-anatomical body envelope")

    pc = obs.get("point_cloud")
    if pc:
        if pc["cluster_shape"] == "bulky_blob":
            score += 0.1
            geometry.append("LiDAR cluster volume inconsistent with human body")
            ev.add("point_cloud", "oversized LiDAR cluster volume for a pedestrian-height object")
        if pc["height_m"] > 2.2:
            geometry.append(f"height {pc['height_m']:.1f} m above normal pedestrian range")
            ev.add("point_cloud", f"object height {pc['height_m']:.1f} m exceeds typical pedestrian height")

    fv = obs.get("fusion_view")
    if fv and not fv["camera_lidar_agree"]:
        score += 0.05
        ev.add("fusion_view", "camera and LiDAR classify object shape differently — ambiguous geometry")

    lp = obs.get("lidar_projection")
    if lp and not lp["projected_height_plausible_human"]:
        geometry.append("projected extent outside plausible human range")
        ev.add("lidar_projection", "projected object extent outside plausible human range")

    return _clamp(score), inferred, geometry


def _alternative_hypotheses(obj: SceneObject, scene: Scene,
                            human_conf: float) -> List[AlternativeHypothesis]:
    obs = obj.observables
    rgb = obs.get("rgb") or {}
    pc = obs.get("point_cloud") or {}
    temporal = obs.get("temporal_sequence")
    mc = obs.get("multi_camera")
    li = obs.get("lidar_intensity") or {}
    alts: List[AlternativeHypothesis] = []

    def motion_rejects(name: str) -> Optional[AlternativeHypothesis]:
        """Static-object hypotheses die when articulated motion is observed."""
        if temporal and temporal["articulated_motion"]:
            return AlternativeHypothesis(
                hypothesis=name, confidence=0.05, status="REJECTED",
                reason=f"articulated motion observed across {temporal['frames_observed']} frames "
                       "rules out a static object")
        return None

    static_context = obj.context == "event_area"
    stationary = bool(temporal and temporal["stationary"])
    texture = rgb.get("texture", "")

    # statue — actively considered whenever context or rigidity suggests it
    statue_cues = sum([
        bool(pc.get("pedestal_detected")), stationary,
        texture == "fiberglass", bool(static_context)])
    if statue_cues or texture in MANUFACTURED_TEXTURES or static_context or temporal is None:
        rej = motion_rejects("mascot_statue")
        if rej:
            alts.append(rej)
        else:
            conf = _clamp(0.15 + 0.2 * statue_cues, hi=0.9)
            reason = ("no temporal data — cannot rule out a static replica"
                      if temporal is None else
                      f"{statue_cues} static-object cue(s): "
                      + ", ".join(c for c, on in [
                          ("pedestal", pc.get("pedestal_detected")),
                          ("stationary", stationary),
                          ("fiberglass surface", texture == "fiberglass"),
                          ("event-area context", static_context)] if on))
            alts.append(AlternativeHypothesis(
                hypothesis="mascot_statue", confidence=conf, status="RETAINED", reason=reason))

    # mannequin
    if texture == "plastic" or pc.get("pedestal_detected") or (temporal is None and human_conf < 0.6):
        rej = motion_rejects("mannequin")
        alts.append(rej or AlternativeHypothesis(
            hypothesis="mannequin",
            confidence=_clamp(0.2 + (0.25 if texture == "plastic" else 0.0)
                              + (0.2 if pc.get("pedestal_detected") else 0.0), hi=0.85),
            status="RETAINED",
            reason="rigid humanoid form cannot be distinguished from a display mannequin "
                   "with the available evidence"))

    # inflatable decoration (vs an inflatable costume with a person inside)
    if texture == "vinyl":
        if temporal and temporal["displacement_m"] > 0.5:
            alts.append(AlternativeHypothesis(
                hypothesis="inflatable_decoration", confidence=0.08, status="REJECTED",
                reason=f"self-propelled displacement of {temporal['displacement_m']:.1f} m — "
                       "tethered decorations do not translate"))
        else:
            sway = bool(temporal and temporal.get("sway_only"))
            alts.append(AlternativeHypothesis(
                hypothesis="inflatable_decoration",
                confidence=_clamp(0.35 + (0.3 if sway else 0.0) + (0.15 if temporal is None else 0.0), hi=0.9),
                status="RETAINED",
                reason="vinyl inflatable without observed self-propulsion — could be an anchored "
                       "decoration rather than a person in an inflatable costume"))

    # advertisement / sign (flat objects)
    flat = rgb.get("flat_appearance") or pc.get("cluster_shape") == "flat_panel" \
        or (mc is not None and not mc["parallax_consistent_3d"]) or li.get("retroreflective")
    if flat:
        confirmed_3d = (pc and pc.get("cluster_shape") in ("humanoid", "bulky_blob")) or \
            (mc is not None and mc["parallax_consistent_3d"])
        if confirmed_3d:
            alts.append(AlternativeHypothesis(
                hypothesis="roadside_advertisement", confidence=0.1, status="REJECTED",
                reason="3D volume confirmed by parallax/LiDAR — not a printed surface"))
        else:
            alts.append(AlternativeHypothesis(
                hypothesis="roadside_advertisement", confidence=0.6, status="RETAINED",
                reason="flat appearance with no confirmed 3D volume — likely printed figure"))

    # plain pedestrian (not a rare event at all)
    sil = rgb.get("silhouette_deviation_observed", "LOW")
    if sil in ("LOW", "MODERATE"):
        alts.append(AlternativeHypothesis(
            hypothesis="normal_pedestrian",
            confidence=_clamp(0.7 if sil == "LOW" else 0.3, hi=0.9),
            status="RETAINED" if sil == "LOW" else "RETAINED",
            reason="silhouette deviation is limited — could be bulky clothing rather than a costume"))
    else:
        alts.append(AlternativeHypothesis(
            hypothesis="normal_pedestrian", confidence=0.08, status="REJECTED",
            reason=f"{sil} silhouette deviation exceeds what ordinary clothing produces"))

    return alts


def _temporal_validation(scene: Scene, obj: SceneObject) -> TemporalValidation:
    if not scene.modalities.get("temporal_sequence"):
        # single frame: never invent motion
        return TemporalValidation(available=False, status="NOT_AVAILABLE", evidence=[])
    t = obj.observables.get("temporal_sequence") or {}
    evidence = []
    if t.get("articulated_motion"):
        evidence.append(f"articulated motion over {t.get('frames_observed', 0)} frames")
    if t.get("gait_periodicity", 0) > 0.5:
        evidence.append(f"gait periodicity {t['gait_periodicity']:.2f}")
    if t.get("displacement_m", 0) > 0.5:
        evidence.append(f"net displacement {t['displacement_m']:.1f} m")
    if evidence:
        return TemporalValidation(available=True, status="VALIDATED", evidence=evidence)
    return TemporalValidation(available=True, status="INCONCLUSIVE",
                              evidence=["no articulation or displacement observed across sequence"])


def _observed_behavior(scene: Scene, obj: SceneObject,
                       human_conf: float) -> Optional[ObservedModelBehavior]:
    """OBSERVED failures only — populated iff predictions were supplied."""
    if not scene.modalities.get("baseline_predictions"):
        return None
    pred = next((p for p in scene.baseline_predictions if p.object_id == obj.object_id), None)
    behavior = ObservedModelBehavior()
    if pred is None:
        behavior.failure_observed = True
        behavior.failure_modes = ["FALSE_NEGATIVE"]
        behavior.details = ["baseline model emitted no detection at this object's location"]
        return behavior
    if pred.predicted_class != "pedestrian" and human_conf >= 0.3:
        behavior.failure_observed = True
        behavior.failure_modes.append(f"MISCLASSIFICATION_{pred.predicted_class.upper()}")
        behavior.details.append(
            f"model classified object as '{pred.predicted_class}' despite human-identity evidence")
    if pred.confidence < 0.35:
        behavior.failure_observed = True
        behavior.failure_modes.append("LOW_CONFIDENCE")
        behavior.details.append(f"detection confidence {pred.confidence:.2f} below reliable range")
    if pred.localization_offset_m > 1.0:
        behavior.failure_observed = True
        behavior.failure_modes.append("LOCALIZATION_ERROR")
        behavior.details.append(
            f"predicted box offset {pred.localization_offset_m:.1f} m from sensed object cluster")
    if not pred.track_stable:
        behavior.failure_observed = True
        behavior.failure_modes.append("TRACKING_FAILURE")
        behavior.details.append("track identity unstable across the observed sequence")
    if not behavior.failure_observed:
        behavior.details = [f"model detected 'pedestrian' at confidence {pred.confidence:.2f} — no failure observed"]
    return behavior


def _difficulty(obj: SceneObject, scene: Scene, sil: str, occ: str) -> tuple[str, List[str]]:
    basis: List[str] = []
    score = SIL_ORDER.index(sil)
    basis.append(f"silhouette deviation {sil}")
    score += OCC_ORDER.index(occ)
    if occ != "NONE":
        basis.append(f"occlusion {occ}")
    if obj.distance_m > 35:
        score += 1
        basis.append(f"long range ({obj.distance_m:.0f} m)")
    if scene.lighting == "night":
        score += 1
        basis.append("night lighting")
    if scene.weather in ("rain", "fog"):
        score += 1
        basis.append(f"{scene.weather} degrades sensing")
    level = DIFF_ORDER[min(3, score // 2)]
    return level, basis


# ------------------------------------------------------------------ main entry


def propose_for_object(scene: Scene, obj: SceneObject,
                       config: Optional[Dict[str, Any]] = None,
                       run_id: str = "") -> Candidate:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ev = _EvidenceCollector(scene)
    identity_ev: List[Evidence] = []

    human_conf = _analyze_human_identity(obj, ev, identity_ev)
    costume_conf, inferred_types, geometry = _analyze_costume(obj, ev)
    alts = _alternative_hypotheses(obj, scene, human_conf)
    temporal = _temporal_validation(scene, obj)

    rgb = obj.observables.get("rgb") or {}
    sil = rgb.get("silhouette_deviation_observed", "LOW")
    visibility = rgb.get("visibility", 0.5)

    # occlusion: level from visibility, source from costume bulk vs environment
    occ_level = "NONE" if visibility > 0.75 else "PARTIAL" if visibility > 0.5 \
        else "HEAVY" if visibility > 0.25 else "EXTREME"
    bulky = bool(rgb.get("bulky_shape"))
    if occ_level == "NONE":
        occ_source = "NONE"
    elif bulky and visibility < 0.5:
        occ_source = "COMBINED"
    elif bulky:
        occ_source = "COSTUME_INDUCED"
    else:
        occ_source = "ENVIRONMENTAL"

    # rare-event confidence: requires BOTH a human and a costume — combined but
    # reported separately; discounted by the strongest retained alternative.
    rare = min(human_conf, costume_conf) * (0.75 + 0.25 * max(human_conf, costume_conf))
    strongest_alt = max((a.confidence for a in alts
                         if a.status == "RETAINED" and a.hypothesis != "normal_pedestrian"),
                        default=0.0)
    rare *= (1.0 - 0.45 * strongest_alt)
    for fam in inferred_types:
        rare += float(cfg["sensitivity_boost"].get(fam, 0.0))
    rare = _clamp(rare)

    # evidence quality from breadth of corroborating modalities + visibility
    n_modalities = len({e.modality for e in ev.items + identity_ev})
    quality = "HIGH" if (n_modalities >= 3 and visibility > 0.5) else \
        "MEDIUM" if n_modalities >= 2 else "LOW"

    insufficient = visibility < cfg["insufficient_visibility"] or n_modalities == 0
    detected = (not insufficient
                and rare >= cfg["rare_event_threshold"]
                and costume_conf >= cfg["min_costume"]
                and human_conf >= cfg["min_human_identity"])

    difficulty, diff_basis = _difficulty(obj, scene, sil, occ_level)
    behavior = _observed_behavior(scene, obj, human_conf)

    # predicted failure mode — the miner's forecast, independent of any model
    predicted_failure: Optional[str] = None
    if detected:
        if sil == "EXTREME":
            predicted_failure = "MISCLASSIFICATION_BACKGROUND" if occ_level in ("HEAVY", "EXTREME") \
                else "MISCLASSIFICATION_ANIMAL" if "animal" in inferred_types or "mascot" in inferred_types \
                else "FALSE_NEGATIVE"
        elif occ_level in ("HEAVY", "EXTREME"):
            predicted_failure = "FALSE_NEGATIVE"
        elif scene.lighting == "night" or obj.distance_m > 35:
            predicted_failure = "LOW_CONFIDENCE"
        elif sil in ("HIGH", "MODERATE"):
            predicted_failure = "LOW_CONFIDENCE"

    # priority: rarity x difficulty x impact x safety relevance (qualitative)
    safety_context = obj.context in ("crosswalk", "road_edge")
    observed_failure = bool(behavior and behavior.failure_observed)
    reasons: List[str] = []
    if detected:
        reasons.append("costumed-pedestrian events are rare in the fleet distribution")
        reasons.append(f"perception difficulty {difficulty} ({'; '.join(diff_basis)})")
        if observed_failure:
            reasons.append("baseline model failure OBSERVED on this example: "
                           + ", ".join(behavior.failure_modes))  # type: ignore[union-attr]
        elif predicted_failure:
            reasons.append(f"failure PREDICTED (not observed): {predicted_failure}")
        if safety_context:
            reasons.append(f"safety-relevant location: {obj.context} in the vehicle path")
    if observed_failure and sil == "EXTREME" and safety_context:
        priority = "CRITICAL"
    elif observed_failure or (detected and sil == "EXTREME" and safety_context):
        priority = "HIGH"
    elif detected and (DIFF_ORDER.index(difficulty) >= 2 or safety_context):
        priority = "MEDIUM" if not safety_context else "HIGH"
    elif detected:
        priority = "MEDIUM" if DIFF_ORDER.index(difficulty) >= 1 else "LOW"
    else:
        priority = "LOW"
        reasons = ["not a confirmed rare-event candidate"]

    # destination — a RECOMMENDATION only; never TRAINING_CANDIDATE pre-validation
    if not detected:
        destination = "REVIEW_QUEUE" if (insufficient and costume_conf >= 0.3) else "NO_ACTION"
    elif priority == "CRITICAL":
        destination = "SAFETY_CRITICAL_EVALUATION_SET"
    elif observed_failure:
        destination = "REGRESSION_EVALUATION_SET" if rare >= 0.6 else "HARD_EXAMPLE_DATASET"
    elif rare >= 0.6 and quality == "HIGH":
        destination = "RARE_EVENT_DATASET"
    else:
        destination = "REVIEW_QUEUE"

    requires_human = detected or (insufficient and costume_conf >= 0.3) or \
        (strongest_alt >= 0.4 and costume_conf >= 0.3)
    if detected:
        hv_reason = "miner proposals are hypotheses; dataset admission requires human/GT confirmation"
        if strongest_alt >= 0.4:
            hv_reason += " — a static-object alternative hypothesis remains plausible"
    elif requires_human:
        hv_reason = "ambiguous evidence: retained alternative hypotheses need human adjudication"
    else:
        hv_reason = ""

    visible_features = [d.description for d in identity_ev if d.modality == "rgb"]
    occluded_features = []
    if not rgb.get("face_visible", False):
        occluded_features.append("face (covered or not visible)")
    if not rgb.get("skin_visible", False):
        occluded_features.append("skin")
    if not rgb.get("limbs_visible", False):
        occluded_features.append("limb articulation points")

    cand = Candidate(
        candidate_id=new_id("cand"),
        bank_id=scene.bank_id,
        scene_id=scene.scene_id,
        sequence_id=scene.sequence_id,
        object_id=obj.object_id,
        track_id=obj.track_id,
        frame_index=scene.frame_index,
        run_id=run_id,
        edge_case_detected=detected,
        insufficient_visual_evidence=insufficient,
        event_type="costumed_pedestrian" if detected else "none",
        costume_type=inferred_types if detected else [],
        confidence_human_identity=human_conf,
        confidence_costume=costume_conf,
        confidence_rare_event=rare,
        visual_evidence=ev.items,
        human_identity_evidence=identity_ev,
        alternative_hypotheses=alts,
        silhouette_deviation=sil,  # type: ignore[arg-type]
        occlusion_level=occ_level,  # type: ignore[arg-type]
        occlusion_source=occ_source,  # type: ignore[arg-type]
        visible_human_features=visible_features,
        occluded_human_features=occluded_features,
        unusual_geometry=geometry,
        temporal_validation=temporal,
        location={"context": obj.context, "distance_m": obj.distance_m,
                  "position": obj.position, "lighting": scene.lighting,
                  "weather": scene.weather},
        perception_difficulty=difficulty,  # type: ignore[arg-type]
        difficulty_evidence=diff_basis,
        observed_model_behavior=behavior,
        predicted_failure_mode=predicted_failure,
        curation_priority=priority,  # type: ignore[arg-type]
        priority_reason="; ".join(reasons),
        recommended_dataset_destination=destination,
        requires_human_validation=requires_human,
        human_validation_reason=hv_reason,
        evidence_quality=quality,  # type: ignore[arg-type]
    )
    validate_candidate_modalities(cand, scene)
    return cand


def mine_scene(scene: Scene, config: Optional[Dict[str, Any]] = None,
               run_id: str = "") -> List[Candidate]:
    """Propose candidates for one scene, ranked within the scene."""
    candidates = [propose_for_object(scene, obj, config, run_id) for obj in scene.objects]
    ranked = sorted(
        candidates,
        key=lambda c: (PRIORITY_ORDER.index(c.curation_priority), c.confidence_rare_event),
        reverse=True)
    for i, c in enumerate(ranked):
        c.rank_in_scene = i + 1
    return ranked


def run_mining(store: RareMineStore, bank_id: str,
               config: Optional[Dict[str, Any]] = None, run_id: str = "") -> List[Candidate]:
    scenes = store.where("scenes", bank_id=bank_id)
    scenes.sort(key=lambda s: (s.sequence_id, s.frame_index, s.scene_id))
    out: List[Candidate] = []
    for scene in scenes:
        for cand in mine_scene(scene, config, run_id):
            store.put("candidates", cand)
            out.append(cand)
    return out
