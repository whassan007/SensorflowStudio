"""Readiness matrix, state derivation, and leverage-based bottleneck analysis."""

from sensorflow.hillclimb import readiness
from sensorflow.hillclimb.blueprint import load_blueprint
from sensorflow.hillclimb.models import Attempt, EvaluationResult, ReadinessState


def _attempt(cid, score, kind, user="default"):
    return Attempt(user_id=user, competency_id=cid, kind=kind,
                   exercise_id="x", responses={},
                   evaluation=EvaluationResult(competency=cid, score=score, confidence=0.8,
                                               evidence=["quoted statement"]))


def _seed_scores(store, cid, score, user="default"):
    readiness.record_attempt(_attempt(cid, score, "diagnostic", user), store)
    readiness.record_attempt(_attempt(cid, score, "exercise", user), store)


def test_states_derive_from_scores(isolated_store):
    _seed_scores(isolated_store, "p1.precision_recall", 5)
    _seed_scores(isolated_store, "p1.iou", 3)
    _seed_scores(isolated_store, "p1.cusum", 2)
    matrix = readiness.compute_matrix(store=isolated_store)
    assert matrix["p1.precision_recall"].readiness_state == ReadinessState.STRONG
    assert matrix["p1.iou"].readiness_state == ReadinessState.COMPETENT
    assert matrix["p1.cusum"].readiness_state == ReadinessState.PRACTICING
    assert matrix["p1.ap_map"].readiness_state == ReadinessState.NOT_STARTED


def test_needs_review_on_knowledge_application_mismatch(isolated_store):
    readiness.record_attempt(_attempt("p1.iou", 5, "diagnostic"), isolated_store)
    readiness.record_attempt(_attempt("p1.iou", 2, "exercise"), isolated_store)
    matrix = readiness.compute_matrix(store=isolated_store)
    assert matrix["p1.iou"].readiness_state == ReadinessState.NEEDS_REVIEW


def test_scores_tracked_separately_never_collapsed(isolated_store):
    readiness.record_attempt(_attempt("p1.iou", 5, "diagnostic"), isolated_store)
    readiness.record_attempt(_attempt("p1.iou", 2, "exercise"), isolated_store)
    r = readiness.compute_matrix(store=isolated_store)["p1.iou"]
    assert r.knowledge_score == 5.0
    assert r.application_score == 2.0


def test_bottleneck_is_blocking_prerequisite_not_min_score(isolated_store):
    bp = load_blueprint(isolated_store)
    weak_hub = "p2.distributed_fundamentals"     # blocks much of phase 2
    lowest = "p3.mentorship"                      # lowest raw score, blocks nothing
    untouched = {"p2.petabyte_ingestion", "p2.streaming_batch", "p2.parallel_inference"}
    for comp in bp.competencies:
        if comp.id == weak_hub:
            _seed_scores(isolated_store, comp.id, 2)
        elif comp.id == lowest:
            _seed_scores(isolated_store, comp.id, 1)
        elif comp.id not in untouched:
            _seed_scores(isolated_store, comp.id, 5)

    bn = readiness.bottleneck_analysis(store=isolated_store)
    assert bn is not None
    assert bn["competency_id"] == weak_hub, (
        f"bottleneck must be the blocking prerequisite, got {bn['competency_id']}")
    assert bn["blocked_count"] >= 3
    assert set(bn["blocked_competencies"]) >= untouched
    # plain-language explanation names the leverage
    assert "limiting" in bn["explanation"]


def test_bottleneck_none_when_everything_competent(isolated_store):
    bp = load_blueprint(isolated_store)
    for comp in bp.competencies:
        _seed_scores(isolated_store, comp.id, 5)
    assert readiness.bottleneck_analysis(store=isolated_store) is None


def test_dimension_summary_covers_display_dimensions(isolated_store):
    _seed_scores(isolated_store, "p1.precision_recall", 4)
    dims = readiness.dimension_summary(store=isolated_store)
    names = {d["dimension"] for d in dims}
    assert names == {"Technical Depth", "System Design", "Execution", "Leadership",
                     "Communication", "Safety/Risk"}
    tech = next(d for d in dims if d["dimension"] == "Technical Depth")
    assert tech["avg_score"] > 0
