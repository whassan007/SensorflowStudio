"""Evidence graph structure + snippet completeness and temporal context."""

from __future__ import annotations

from sensorflow.agentic import pipeline as pipeline_mod

REQUIRED_NODE_TYPES = {
    "Object", "Environment", "Sensors", "Prediction", "GroundTruth",
    "Tracking", "Planner", "HistoricalSimilarity", "Frequency",
    "SafetyConsequence",
}

SNIPPET_REQUIRED_FIELDS = [
    "scene_id", "sequence_id", "timestamp_us", "frame_index",
    "model_version_candidate", "model_version_baseline", "prediction",
    "ground_truth", "confidence", "tracking_state", "planner_state",
    "environment", "geo_bucket", "software_config", "hardware_config",
    "feature_pipeline_version", "temporal_context",
]


def test_evidence_graph_has_all_spec_node_types(walkthrough):
    graph = pipeline_mod.get_evidence_graph(walkthrough["failure_id"])
    assert graph is not None
    types = {n.node_type for n in graph.nodes}
    assert REQUIRED_NODE_TYPES <= types, REQUIRED_NODE_TYPES - types


def test_evidence_statuses_are_typed_and_honest(walkthrough):
    graph = pipeline_mod.get_evidence_graph(walkthrough["failure_id"])
    allowed = {"OBSERVED", "DERIVED", "HYPOTHESIS", "UNAVAILABLE"}
    by_type = {n.node_type: n for n in graph.nodes}
    for n in graph.nodes:
        assert n.status in allowed
        assert n.summary  # every node explains itself
    # GT comes from the synthetic vendor annotations -> OBSERVED
    assert by_type["GroundTruth"].status == "OBSERVED"
    # scene-context reasoning is a hypothesis, never ground truth
    assert by_type["Environment"].status in ("HYPOTHESIS", "OBSERVED")
    # every edge references real nodes
    ids = {n.node_id for n in graph.nodes}
    for e in graph.edges:
        assert e.src in ids and e.dst in ids


def test_snippet_completeness(walkthrough):
    snippets = pipeline_mod.get_snippets(walkthrough["failure_id"])
    assert snippets, "no snippets were built"
    for snip in snippets:
        for field in SNIPPET_REQUIRED_FIELDS:
            assert snip.get(field) not in (None, "", [], {}), \
                f"snippet missing {field}"
        assert snip["prediction"]["class"] == "construction_cone"
        assert snip["ground_truth"]["class"] == "pedestrian"


def test_snippets_carry_temporal_context_not_single_frames(walkthrough):
    snippets = pipeline_mod.get_snippets(walkthrough["failure_id"])
    for snip in snippets:
        ctx = snip["temporal_context"]
        roles = {c["role"] for c in ctx}
        assert "failure" in roles
        # never single-frame when temporal context exists in the data
        assert len(ctx) > 1, "single-frame snippet emitted"
        assert roles & {"before", "after"}
        # frames ordered by index and centered on the failure
        idxs = [c["frame_index"] for c in ctx]
        assert idxs == sorted(idxs)
