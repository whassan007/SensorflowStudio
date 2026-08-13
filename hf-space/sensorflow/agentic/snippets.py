"""Reproducible failure snippet packages.

A snippet carries everything needed to reproduce and triage one concrete
failure instance: scene/sequence/timestamp, model versions, prediction, GT,
confidence, tracking, planner state, environment, geo bucket, software and
hardware config, feature pipeline version — PLUS temporal context frames
(before / failure / after). Single-frame snippets are never emitted when
temporal context exists in the data, because classification flips are often
transient and the before/after frames are the evidence of that.
"""

from __future__ import annotations

from typing import List

from sensorflow.agentic import data as data_mod
from sensorflow.agentic.models import (FailureEvent, FailureInstance,
                                       FailureSnippet, TemporalFrameRef, new_id)

TEMPORAL_RADIUS = 2  # frames of context on each side when available


def _temporal_context(campaign: data_mod.Campaign, inst: FailureInstance
                      ) -> List[TemporalFrameRef]:
    history = data_mod.observations_for_instance(
        campaign, inst.sequence_id, inst.object_instance_id)
    refs: List[TemporalFrameRef] = []
    for obs in history:
        d = obs.frame_index - inst.frame_index
        if abs(d) > TEMPORAL_RADIUS:
            continue
        role = "failure" if d == 0 else ("before" if d < 0 else "after")
        refs.append(TemporalFrameRef(
            frame_id=obs.frame_id, frame_index=obs.frame_index, role=role,
            gt_class=obs.gt_class, predicted_class=obs.candidate.predicted_class,
            confidence=obs.candidate.confidence, bbox_3d=obs.candidate.bbox_3d))
    return refs


def build_snippet(failure: FailureEvent, inst: FailureInstance,
                  campaign: data_mod.Campaign,
                  similar_historical: List[dict] | None = None) -> FailureSnippet:
    history = data_mod.observations_for_instance(
        campaign, inst.sequence_id, inst.object_instance_id)
    obs = next((o for o in history if o.frame_id == inst.frame_id), None)
    if obs is None:
        raise KeyError(f"No observation for instance {inst.instance_id}")

    planner_state = (
        {"replay_trace": "available",
         "scenario": "crossing (synthetic replay)",
         "note": "quantified by the safety-impact agent"}
        if inst.has_planner_trace else
        {"replay_trace": "unavailable",
         "note": "no behavioral evidence exists for this instance"})

    return FailureSnippet(
        snippet_id=new_id("snip"),
        failure_id=failure.failure_id,
        instance_id=inst.instance_id,
        scene_id=inst.sequence_id,
        sequence_id=inst.sequence_id,
        timestamp_us=obs.timestamp_us,
        frame_index=inst.frame_index,
        model_version_candidate=failure.candidate_model,
        model_version_baseline=failure.baseline_model,
        prediction={"class": obs.candidate.predicted_class,
                    "confidence": obs.candidate.confidence,
                    "bbox_3d": obs.candidate.bbox_3d,
                    "track_id": obs.candidate.track_id,
                    "baseline_class": obs.baseline.predicted_class,
                    "baseline_confidence": obs.baseline.confidence},
        ground_truth={"class": obs.gt_class, "bbox_3d": obs.gt_bbox_3d,
                      "source": "synthetic_vendor_gt"},
        confidence=obs.candidate.confidence,
        tracking_state={"candidate_track_id": obs.candidate.track_id,
                        "baseline_track_id": obs.baseline.track_id,
                        "frames_observed": len(history)},
        planner_state=planner_state,
        environment={"time_of_day": inst.time_of_day, "weather": inst.weather,
                     "construction_zone": inst.construction_zone,
                     "occluded": inst.occluded,
                     "distance_m": inst.distance_m},
        geo_bucket=inst.geo_bucket,
        software_config=campaign.software_config,
        hardware_config=campaign.hardware_config,
        feature_pipeline_version=campaign.feature_pipeline_version,
        temporal_context=_temporal_context(campaign, inst),
        similar_historical=similar_historical or [],
    )
