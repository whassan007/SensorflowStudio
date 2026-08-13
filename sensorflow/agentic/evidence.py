"""Failure Evidence Graph.

Structured, typed nodes (Object / Environment / Sensors / Prediction /
GroundTruth / Tracking / Planner / HistoricalSimilarity / Frequency /
SafetyConsequence) each carrying an explicit evidence status:

    OBSERVED     directly measured from the campaign data / replay
    DERIVED      computed deterministically from observed data
    HYPOTHESIS   produced by an advisory agent; requires confirmation
    UNAVAILABLE  the evidence does not exist yet; the graph says so instead
                 of letting anything downstream pretend it does

The graph starts from deterministic scene data and is enriched as analysis
stages complete. Nothing here consults an LLM.
"""

from __future__ import annotations

from typing import Optional

from sensorflow.agentic import data as data_mod
from sensorflow.agentic.models import (EvidenceEdge, EvidenceGraph, EvidenceNode,
                                       FailureEvent, new_id)


def _node(node_type: str, status: str, summary: str, fields: dict,
          source: str, caveats: Optional[list] = None) -> EvidenceNode:
    return EvidenceNode(node_id=new_id("ev"), node_type=node_type, status=status,
                        summary=summary, fields=fields, source=source,
                        caveats=caveats or [])


def build_graph(failure: FailureEvent, campaign: data_mod.Campaign) -> EvidenceGraph:
    """Initial graph from deterministic campaign evidence for the failure's
    representative instance (first instance)."""
    graph = EvidenceGraph(failure_id=failure.failure_id)
    inst = failure.instances[0] if failure.instances else None

    if inst is None:
        graph.nodes.append(_node(
            "Object", "UNAVAILABLE",
            "No concrete failure instance is attached to this pattern",
            {}, "pipeline"))
        return graph

    history = data_mod.observations_for_instance(
        campaign, inst.sequence_id, inst.object_instance_id)
    obs = next((o for o in history if o.frame_id == inst.frame_id), None)

    graph.nodes.append(_node(
        "Object", "OBSERVED",
        f"{inst.gt_class} instance {inst.object_instance_id} at "
        f"{inst.distance_m:.1f} m",
        {"instance_id": inst.object_instance_id, "gt_class": inst.gt_class,
         "distance_m": inst.distance_m,
         "bbox_3d": obs.gt_bbox_3d if obs else None},
        "bevfusion synthetic scene ground truth"))

    graph.nodes.append(_node(
        "Environment", "OBSERVED",
        f"{inst.time_of_day}/{inst.weather}, construction_zone="
        f"{inst.construction_zone}, geo={inst.geo_bucket}",
        {"time_of_day": inst.time_of_day, "weather": inst.weather,
         "construction_zone": inst.construction_zone,
         "geo_bucket": inst.geo_bucket, "occluded": inst.occluded},
        "sequence context (campaign metadata)"))

    graph.nodes.append(_node(
        "Sensors", "DERIVED",
        "Camera/LiDAR coherence not yet verified — pending fusion agent",
        {"modalities": ["camera", "lidar"], "verified": False},
        "sensorflow.bevfusion simulated sensors",
        caveats=["fusion verification runs in FAILURE_ANALYSIS"]))

    if obs is not None:
        graph.nodes.append(_node(
            "Prediction", "OBSERVED",
            f"candidate predicted {obs.candidate.predicted_class} "
            f"(conf {obs.candidate.confidence}); baseline predicted "
            f"{obs.baseline.predicted_class} (conf {obs.baseline.confidence})",
            {"candidate": obs.candidate.model_dump(),
             "baseline": obs.baseline.model_dump(),
             "candidate_model": failure.candidate_model,
             "baseline_model": failure.baseline_model},
            "campaign prediction log"))
        graph.nodes.append(_node(
            "GroundTruth", "OBSERVED",
            f"GT class {obs.gt_class} (synthetic vendor-style GT)",
            {"gt_class": obs.gt_class, "gt_bbox_3d": obs.gt_bbox_3d,
             "gt_source": "synthetic_vendor_gt"},
            "bevfusion synthetic scene ground truth"))
        graph.nodes.append(_node(
            "Tracking", "OBSERVED",
            f"candidate track {obs.candidate.track_id}; "
            f"{len(history)} frames observed for this instance",
            {"candidate_track_id": obs.candidate.track_id,
             "baseline_track_id": obs.baseline.track_id,
             "frames_observed": len(history)},
            "campaign prediction log"))
    else:
        graph.nodes.append(_node("Prediction", "UNAVAILABLE",
                                 "prediction record missing", {}, "campaign"))
        graph.nodes.append(_node("GroundTruth", "UNAVAILABLE",
                                 "ground truth record missing", {}, "campaign"))
        graph.nodes.append(_node("Tracking", "UNAVAILABLE",
                                 "tracking record missing", {}, "campaign"))

    if inst.has_planner_trace:
        graph.nodes.append(_node(
            "Planner", "OBSERVED",
            "Replay planner-response trace exists for this instance",
            {"replay_available": True, "analyzed": False},
            "synthetic planner replay trace",
            caveats=["safety-impact agent quantifies behavior in FAILURE_ANALYSIS"]))
    else:
        graph.nodes.append(_node(
            "Planner", "UNAVAILABLE",
            "No behavioral replay evidence for this instance — downstream "
            "consequence cannot be observed, only hypothesized",
            {"replay_available": False},
            "synthetic planner replay trace"))

    graph.nodes.append(_node(
        "HistoricalSimilarity", "DERIVED",
        "Similar-failure retrieval pending (scenario mining agent)",
        {"retrieved": False}, "scenario mining (embedding + attribute retrieval)"))

    basis = failure.detection_basis
    graph.nodes.append(_node(
        "Frequency", "DERIVED",
        f"candidate {basis.candidate_events}/{basis.denominator} "
        f"({basis.candidate_rate:.6f}) vs baseline {basis.baseline_events}"
        f"/{basis.denominator} ({basis.baseline_rate:.6f}) — CI/significance "
        "pending statistical agent",
        {"candidate_events": basis.candidate_events,
         "baseline_events": basis.baseline_events,
         "denominator": basis.denominator,
         "candidate_rate": basis.candidate_rate,
         "baseline_rate": basis.baseline_rate},
        "deterministic detection scan"))

    graph.nodes.append(_node(
        "SafetyConsequence", "UNAVAILABLE",
        "Not yet assessed — requires safety-impact analysis over replay "
        "evidence; hypotheses are never recorded as consequences",
        {}, "safety impact agent (FAILURE_ANALYSIS)"))

    ids = {n.node_type: n.node_id for n in graph.nodes}
    for src, dst, rel in [
        ("Object", "Prediction", "was_predicted_as"),
        ("Object", "GroundTruth", "labeled_as"),
        ("Object", "Environment", "situated_in"),
        ("Prediction", "Tracking", "feeds"),
        ("Tracking", "Planner", "feeds"),
        ("Prediction", "Frequency", "aggregated_into"),
        ("Planner", "SafetyConsequence", "determines"),
        ("Object", "HistoricalSimilarity", "compared_against"),
        ("Sensors", "Prediction", "produced"),
    ]:
        if src in ids and dst in ids:
            graph.edges.append(EvidenceEdge(src=ids[src], dst=ids[dst], relation=rel))
    return graph


def set_node(graph: EvidenceGraph, node_type: str, status: str, summary: str,
             fields: dict, source: str, caveats: Optional[list] = None) -> None:
    """Replace (or add) a node after an analysis stage produced its evidence."""
    for i, n in enumerate(graph.nodes):
        if n.node_type == node_type:
            graph.nodes[i] = EvidenceNode(
                node_id=n.node_id, node_type=node_type, status=status,
                summary=summary, fields=fields, source=source,
                caveats=caveats or [])
            return
    graph.nodes.append(_node(node_type, status, summary, fields, source, caveats))
