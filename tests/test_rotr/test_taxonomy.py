"""Taxonomy signatures, the spec's structured query, and clustering."""

from __future__ import annotations

from sensorflow.rotr.taxonomy import (
    build_clusters, cluster_key, matches, parse_query, signature,
)

SPEC_QUERY = ("failed to yield to pedestrian at uncontrolled intersection "
              "during low visibility")


class TestSignature:
    def test_fail_yield_signature(self, bank_v1, detections_v1):
        sc = next(s for s in bank_v1
                  if s.planted.kind == "fail_yield_pedestrian"
                  and s.planted.committed)
        v = detections_v1[sc.scenario_id][0]
        sig = signature(v, sc)
        assert sig["actor"] == "pedestrian"
        assert sig["vulnerability"] == "VRU"
        assert sig["legality"] == "YIELD"
        assert sig["behavior"] == "proceed_without_yield"
        assert sig["road_geometry"] == "uncontrolled"
        assert sig["interaction"] == "CROSSING"


class TestStructuredQuery:
    def test_spec_sentence_parses_to_filterable_object(self):
        q = parse_query(SPEC_QUERY)
        assert q.actor == "pedestrian"
        assert q.legality == "YIELD"
        assert q.road_geometry == "uncontrolled"
        assert q.visibility == "low"
        assert q.text == SPEC_QUERY

    def test_query_matches_only_the_planted_cohort(self, bank_v1, detections_v1):
        q = parse_query(SPEC_QUERY)
        matched, unmatched_kinds = [], set()
        for sc in bank_v1:
            for v in detections_v1[sc.scenario_id]:
                sig = signature(v, sc)
                if matches(q, sig, None, None,
                           {"lighting": sc.environment.lighting,
                            "weather": sc.environment.weather}):
                    matched.append((sc, v))
                else:
                    unmatched_kinds.add(sc.planted.kind)
        assert matched, "seeded bank must contain the spec cohort"
        for sc, v in matched:
            assert sc.planted.kind == "fail_yield_pedestrian"
            assert sc.environment.visibility == "low"
            assert sc.actual_context.intersection_type == "uncontrolled"

    def test_consequence_and_layer_filters(self):
        q = parse_query("safety critical")
        assert q.consequence_class == "SAFETY_CRITICAL"
        assert matches(q, {}, "SAFETY_CRITICAL", None, {})
        assert not matches(q, {}, "DEGRADED_COMFORT", None, {})


class TestClustering:
    def test_same_signature_clusters_together(self):
        items = [
            {"violation_id": "v1", "primary_layer": "planning",
             "consequence_class": "SAFETY_CRITICAL",
             "signature": {"legality": "YIELD", "actor": "pedestrian",
                           "traffic_control": "none",
                           "behavior": "proceed_without_yield",
                           "environment": "day/clear", "visibility": "low"}},
            {"violation_id": "v2", "primary_layer": "planning",
             "consequence_class": "DEGRADED_COMFORT",
             "signature": {"legality": "YIELD", "actor": "pedestrian",
                           "traffic_control": "none",
                           "behavior": "proceed_without_yield",
                           "environment": "night/rain", "visibility": "clear"}},
            {"violation_id": "v3", "primary_layer": "perception",
             "consequence_class": "SAFETY_CRITICAL",
             "signature": {"legality": "MERGE", "actor": "vehicle",
                           "traffic_control": "none",
                           "behavior": "insufficient_gap_merge",
                           "environment": "day/clear", "visibility": "clear"}},
        ]
        clusters = build_clusters(items)
        assert len(clusters) == 2
        big = clusters[0]
        assert big["count"] == 2
        assert set(big["member_violation_ids"]) == {"v1", "v2"}
        assert len(big["environment_spread"]) == 2
        assert big["consequence_distribution"] == {"SAFETY_CRITICAL": 1,
                                                   "DEGRADED_COMFORT": 1}

    def test_primary_layer_separates_clusters(self):
        sig = {"legality": "YIELD", "actor": "pedestrian",
               "traffic_control": "none", "behavior": "proceed_without_yield"}
        assert cluster_key(sig, "planning") != cluster_key(sig, "perception")
