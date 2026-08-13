"""Deterministic scripted synthesis (mock backend path).

Rule-driven hypothesis generation assembled from the ACTUAL evidence in the
artifact — the same pattern as the platform's offline MITL copilot
(sensorflow/evaluation/copilot.py). Never invents telemetry: hypotheses cite
supporting evidence keys and explicitly list the evidence that is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sensorflow.retro.scorecard import RootCauseHypothesis, Severity


def scripted_hypotheses(failure_type: str, log: Dict[str, Any],
                        missing_fields: List[str]) -> List[RootCauseHypothesis]:
    gt = log.get("ground_truth") or {}
    pred = log.get("prediction") or {}
    scenario = log.get("scenario") or {}
    sensors = log.get("sensor_metadata") or {}
    tracks = log.get("tracks") or []
    hyps: List[RootCauseHypothesis] = []

    gt_class = (gt.get("class") or "").lower()
    weather = (scenario.get("weather") or "").lower()
    flicker = any(t.get("class_flicker") for t in tracks)

    if failure_type == "FALSE_POSITIVE" and gt_class in (
            "plastic_bag", "debris", "cardboard", "balloon"):
        hyps.append(RootCauseHypothesis(
            hypothesis="Training-distribution gap for wind-blown deformable "
                       "low-mass objects: the detector assigned a hazardous "
                       "class with high confidence to a non-hazard object, "
                       "consistent with the historical phantom-braking cluster "
                       "(RETRO-2025-014 [SYNTHETIC]).",
            confidence=0.65,
            supporting_evidence_keys=["gt_class", "pred_class", "pred_confidence"]
                                     + (["track_class_flicker"] if flicker else []),
            missing_evidence=["camera embeddings for the misclassified frames",
                              "training-set frequency of deformable clutter"]))
        if "velocity_note" in pred or any(
                t.get("velocity_variance") == "high" for t in tracks):
            hyps.append(RootCauseHypothesis(
                hypothesis="Tracker velocity latch: bag flutter was integrated "
                           "into a spurious closing velocity, converting a "
                           "static-clutter FP into a phantom emergency brake.",
                confidence=0.55,
                supporting_evidence_keys=["pred_relative_velocity",
                                          "gt_relative_velocity",
                                          "track_velocity_variance"],
                missing_evidence=["raw radar doppler track for the object"]))

    elif failure_type == "FALSE_NEGATIVE" and gt_class in (
            "pedestrian", "cyclist", "motorcycle", "wheelchair"):
        if weather in ("rain", "snow", "fog"):
            cam = (sensors.get("camera") or {}).get("status", "")
            lid = (sensors.get("lidar") or {}).get("status", "")
            hyps.append(RootCauseHypothesis(
                hypothesis=f"Fusion confidence starvation under {weather}: "
                           f"camera '{cam}' and LiDAR '{lid}' each degraded, "
                           "and the fusion confidence stayed below the tracking "
                           "threshold instead of applying weather-conditioned "
                           "weighting (pattern matches RETRO-2025-021 "
                           "[SYNTHETIC]).",
                confidence=0.6,
                supporting_evidence_keys=["sensor_camera_status",
                                          "sensor_lidar_status",
                                          "pred_confidence",
                                          "pred_detection_delay"],
                missing_evidence=["per-frame fusion confidence trace",
                                  "confidence threshold config for this release"]))
        hyps.append(RootCauseHypothesis(
            hypothesis="Training-data coverage gap for this VRU appearance/"
                       "condition slice; the distribution-shift slice data "
                       "shows the regression concentrated rather than uniform.",
            confidence=0.5,
            supporting_evidence_keys=["distribution_shift", "metric_delta"],
            missing_evidence=["slice-level training data counts"]))

    elif failure_type == "FALSE_POSITIVE":
        hyps.append(RootCauseHypothesis(
            hypothesis="Static-scene classification ambiguity: a fixed roadside "
                       "asset was transiently classified as a dynamic hazard; "
                       "map-prior fusion did not veto the detection.",
            confidence=0.5,
            supporting_evidence_keys=["gt_class", "pred_class",
                                      "pred_confidence"],
            missing_evidence=["map-prior veto decision log"]))

    elif failure_type == "FALSE_NEGATIVE":
        hyps.append(RootCauseHypothesis(
            hypothesis="Detection miss on a low-salience static object; range "
                       "sparsity is the most consistent factor given nominal "
                       "sensor status.",
            confidence=0.45,
            supporting_evidence_keys=["gt_distance", "sensor_lidar_status"],
            missing_evidence=["point count inside the ground-truth box"]))

    if not hyps:
        hyps.append(RootCauseHypothesis(
            hypothesis="Insufficient patterns matched for a specific causal "
                       "hypothesis; requires human investigation.",
            confidence=0.3, supporting_evidence_keys=[],
            missing_evidence=["additional telemetry"]))

    if missing_fields:
        for h in hyps:
            for f in missing_fields:
                marker = f"artifact field {f}"
                if marker not in h.missing_evidence:
                    h.missing_evidence.append(marker)
    return hyps


def scripted_severity_proposal(log: Dict[str, Any],
                               failure_type: str) -> Optional[Severity]:
    """The mock 'LLM proposal' — a deliberately simpler heuristic than the
    policy engine, so the policy-validate/override flow is real."""
    collision = (log.get("collision") or {})
    planner = (log.get("planner_response") or {})
    traffic = ((log.get("traffic_context") or {}).get("following_vehicle") or {})
    if collision.get("occurred"):
        return Severity.FATAL
    if collision.get("near_miss"):
        return Severity.CRITICAL
    decel = planner.get("decel_mps2") or 0.0
    gap = traffic.get("time_gap_s")
    if decel >= 3.4 and gap is not None and gap < 1.5:
        return Severity.CRITICAL
    if decel >= 3.0:
        return Severity.DISRUPTIVE
    return Severity.BENIGN
