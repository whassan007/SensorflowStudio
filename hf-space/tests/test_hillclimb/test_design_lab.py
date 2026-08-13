"""Design-lab grader: good vs broken architectures."""

from sensorflow.hillclimb.design_lab import (Component, DesignSubmission, Edge,
                                             grade_submission)

RATIONALES = {
    "Failure-mining trigger criteria": (
        "Monitoring computes slice metrics and alerts when drift exceeds 2 sigma; scalability comes "
        "from partitioned stream consumers handling 20000 qps with 30% headroom."),
    "Label routing & quality control": (
        "Redundant dual-region labeling queues (replicated, failover in under 5 minutes) with "
        "observability dashboards, logs and traces on every stage; cost is controlled by tiered "
        "storage and spot instances."),
    "Loop latency (failure → fixed model)": (
        "End-to-end loop latency target is 7 days; p99 inference latency stays under 100 ms. "
        "The tradeoff: faster loops cost more compute, at the expense of budget."),
    "How you detect the loop itself breaking": (
        "A dead letter queue plus a heartbeat monitor covers failure handling; if mining stalls we "
        "degrade gracefully to weekly batch replay and recover from checkpoints."),
}


def _good_submission():
    comps = [
        Component(id="src", type="source", name="Fleet frames", note="2 PB/week"),
        Component(id="inf1", type="inference", name="Serving A", note="10000 qps"),
        Component(id="inf2", type="inference", name="Serving B", note="10000 qps"),
        Component(id="mon", type="monitoring", name="Slice monitor"),
        Component(id="fb", type="feedback", name="Failure miner"),
        Component(id="train", type="training", name="Retrain"),
        Component(id="ev", type="eval", name="Eval gate"),
    ]
    edges = [Edge(source="src", target="inf1"), Edge(source="src", target="inf2"),
             Edge(source="inf1", target="mon"), Edge(source="inf2", target="mon"),
             Edge(source="mon", target="fb"), Edge(source="fb", target="train"),
             Edge(source="train", target="ev"), Edge(source="ev", target="inf1"),
             Edge(source="ev", target="inf2")]
    return DesignSubmission(challenge_id="feedback_loop", components=comps,
                            edges=edges, rationales=RATIONALES)


def _broken_submission():
    comps = [
        Component(id="src", type="source", name="Fleet frames"),
        Component(id="inf", type="inference", name="Serving"),
        Component(id="stor", type="storage", name="Orphan bucket"),  # never connected
    ]
    edges = [Edge(source="src", target="inf")]
    return DesignSubmission(challenge_id="feedback_loop", components=comps,
                            edges=edges, rationales={"why": "it works"})


def test_good_architecture_grades_well():
    grade = grade_submission(_good_submission(), save_evidence=False)
    s = grade.structural
    assert s.missing_stages == []
    assert s.orphan_components == []
    assert s.feedback_loop_closed
    assert s.capacity_math_found and s.capacity_quotes
    assert grade.overall_score >= 3


def test_broken_architecture_reports_specific_gaps():
    grade = grade_submission(_broken_submission(), save_evidence=False)
    s = grade.structural
    assert "monitoring" in " ".join(s.missing_stages)
    assert "Orphan bucket" in s.orphan_components
    assert not s.feedback_loop_closed
    assert not s.capacity_math_found
    joined = " ".join(grade.gaps).lower()
    assert "missing required stage" in joined
    assert "orphan" in joined
    assert "feedback loop is not closed" in joined
    assert "capacity" in joined
    assert grade.overall_score < grade_submission(_good_submission(), save_evidence=False).overall_score


def test_spof_detection():
    # single inference node shared by everything upstream/downstream = articulation point
    comps = [
        Component(id="src", type="source"), Component(id="inf", type="inference"),
        Component(id="mon", type="monitoring"), Component(id="fb", type="feedback"),
        Component(id="train", type="training"), Component(id="ev", type="eval"),
    ]
    edges = [Edge(source="src", target="inf"), Edge(source="inf", target="mon"),
             Edge(source="mon", target="fb"), Edge(source="fb", target="train"),
             Edge(source="train", target="ev"), Edge(source="ev", target="inf")]
    grade = grade_submission(DesignSubmission(challenge_id="feedback_loop", components=comps,
                                              edges=edges, rationales={}), save_evidence=False)
    assert "inf" in grade.structural.single_points_of_failure


def test_per_dimension_scores_with_rationale_evidence():
    grade = grade_submission(_good_submission(), save_evidence=False)
    dims = {d.dimension: d for d in grade.dimension_grades}
    assert set(dims) == {"scalability", "reliability", "latency", "cost",
                         "observability", "failure_handling", "tradeoffs"}
    assert dims["scalability"].score >= 3 and dims["scalability"].evidence
    assert dims["tradeoffs"].score >= 2


def test_grading_stores_evidence_artifact(isolated_store):
    grade = grade_submission(_good_submission(), store=isolated_store, save_evidence=True)
    assert grade.evidence_id
    raw = isolated_store.get("evidence", grade.evidence_id)
    assert raw["artifact_type"] == "design_submission"
    assert "p2.feedback_loops" in raw["competency_ids"]
