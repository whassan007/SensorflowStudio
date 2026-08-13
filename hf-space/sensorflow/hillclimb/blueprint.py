"""Hill Climbing EM Blueprint: normalized 4-phase model + competency graph.

IMPORTANT PROVENANCE NOTE: the original blueprint image was not provided.
This blueprint is RECONSTRUCTED FROM THE WRITTEN SPEC (source =
"reconstructed-from-spec"). It is seeded once into runs/hillclimb/blueprint.json
and is editable there / via the API — it is data, not hardcoded prose.

Model: Blueprint -> Phase -> {Objective, Topics, Skills, Exercises,
Assessments, CompletionCriteria}, plus a flat competency list forming a
prerequisite DAG. Every competency is tagged with exactly one dimension
(Knowledge / Technical Reasoning / Leadership / Execution) which is tracked
separately and never collapsed into a single score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from sensorflow.hillclimb.models import Dimension, Store, get_store


class Competency(BaseModel):
    id: str
    name: str
    phase: int
    dimension: Dimension
    description: str = ""
    prerequisites: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)


class PhaseSpec(BaseModel):
    phase: int
    title: str
    objective: str
    topics: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    exercises: List[str] = Field(default_factory=list)
    assessments: List[str] = Field(default_factory=list)
    completion_criteria: List[str] = Field(default_factory=list)


class Blueprint(BaseModel):
    version: int = 1
    source: str = "reconstructed-from-spec"
    note: str = (
        "The original Hill Climbing EM blueprint image was not provided; this "
        "structure was derived from the written specification and is editable."
    )
    phases: List[PhaseSpec] = Field(default_factory=list)
    competencies: List[Competency] = Field(default_factory=list)


def _c(cid: str, name: str, phase: int, dim: Dimension, desc: str,
       prereqs: Optional[List[str]] = None, topics: Optional[List[str]] = None) -> Competency:
    return Competency(id=cid, name=name, phase=phase, dimension=dim,
                      description=desc, prerequisites=prereqs or [], topics=topics or [])


def seed_blueprint() -> Blueprint:
    K, T, L, E = (Dimension.KNOWLEDGE, Dimension.TECHNICAL_REASONING,
                  Dimension.LEADERSHIP, Dimension.EXECUTION)

    competencies: List[Competency] = [
        # ----------------------------------------------------------- Phase 1
        _c("p1.precision_recall", "Precision & Recall", 1, K,
           "Definitions, confusion-matrix intuition, threshold effects, PR curves.",
           topics=["confusion matrix", "thresholding", "PR curves", "class imbalance"]),
        _c("p1.iou", "IoU & Localization Quality", 1, K,
           "2D/3D IoU, localization vs classification errors, matching criteria.",
           topics=["IoU", "3D IoU", "matching thresholds", "localization error"]),
        _c("p1.ap_map", "AP / mAP", 1, K,
           "Average precision, mAP across classes and IoU thresholds, ranking metrics.",
           ["p1.precision_recall", "p1.iou"], ["AP", "mAP", "interpolated precision", "COCO-style evaluation"]),
        _c("p1.macro_micro", "Macro vs Micro Averaging", 1, K,
           "When macro vs micro averaging changes conclusions; per-class safety weighting.",
           ["p1.precision_recall"], ["macro averaging", "micro averaging", "class weighting"]),
        _c("p1.statistical_evaluation", "Statistical Evaluation", 1, K,
           "Confidence intervals, significance, sample-size effects, multiple comparisons.",
           topics=["confidence intervals", "bootstrap", "significance", "power"]),
        _c("p1.temporal_effects", "Temporal Effects", 1, T,
           "Track-level metrics, flicker, latency vs accuracy, temporal consistency.",
           ["p1.precision_recall"], ["track continuity", "flicker", "latency", "temporal IoU"]),
        _c("p1.regression_detection", "Regression Detection", 1, T,
           "Detecting model regressions across versions; offline vs online divergence.",
           ["p1.precision_recall", "p1.statistical_evaluation"],
           ["A/B deltas", "shadow mode", "release gates", "offline-online gap"]),
        _c("p1.cusum", "CUSUM & Drift Detection", 1, T,
           "Sequential change detection, CUSUM charts, alerting thresholds.",
           ["p1.statistical_evaluation"], ["CUSUM", "change points", "control charts"]),
        _c("p1.model_monitoring", "Model Monitoring", 1, T,
           "Production monitoring design: metrics, slicing, canaries, alert fatigue.",
           ["p1.regression_detection"], ["canary", "slicing", "data drift", "alerting"]),
        _c("p1.failure_analysis", "Failure Analysis", 1, T,
           "Systematic diagnosis of model failures; error taxonomies; root-causing metric gaps.",
           ["p1.precision_recall", "p1.iou"], ["error taxonomy", "root cause", "failure buckets"]),
        _c("p1.safety_metrics", "Safety Metrics", 1, K,
           "Safety-weighted evaluation: severity weighting, VRU recall, worst-case slices.",
           ["p1.precision_recall"], ["severity weighting", "VRU recall", "worst-case slices"]),

        # ----------------------------------------------------------- Phase 2
        _c("p2.distributed_fundamentals", "Distributed-Systems Fundamentals", 2, K,
           "Partitioning, replication, consistency, backpressure, queueing, failure domains.",
           topics=["partitioning", "replication", "consistency", "backpressure", "queueing theory"]),
        _c("p2.petabyte_ingestion", "Petabyte-Scale Ingestion", 2, T,
           "Sensor-data ingestion at PB scale: fan-in, schema, compression, ordering, backfill.",
           ["p2.distributed_fundamentals"], ["fan-in", "schema evolution", "backfill", "compaction"]),
        _c("p2.streaming_batch", "Streaming vs Batch Processing", 2, T,
           "Lambda/kappa tradeoffs, exactly-once semantics, watermarking, late data.",
           ["p2.distributed_fundamentals"], ["exactly-once", "watermarks", "late data", "kappa"]),
        _c("p2.storage_design", "Storage Design", 2, T,
           "Hot/warm/cold tiers, columnar formats, indexing, cost-aware retention.",
           ["p2.distributed_fundamentals"], ["tiering", "parquet", "indexing", "retention"]),
        _c("p2.feature_generation", "Feature Generation", 2, T,
           "Offline/online feature parity, materialization, point-in-time correctness.",
           ["p2.storage_design"], ["feature store", "point-in-time", "online-offline parity"]),
        _c("p2.training_infra", "Training Infrastructure", 2, T,
           "Distributed training, data loading bottlenecks, checkpointing, experiment tracking.",
           ["p2.distributed_fundamentals"], ["data parallel", "checkpointing", "throughput"]),
        _c("p2.parallel_inference", "Parallel ML Inference Design", 2, T,
           "High-throughput serving: batching, sharding, autoscaling, tail latency, GPU economics.",
           ["p2.distributed_fundamentals"], ["dynamic batching", "sharding", "tail latency", "autoscaling"]),
        _c("p2.eval_platform", "Evaluation Platform Design", 2, T,
           "Continuous evaluation at scale: metric cubes, slicing, human-in-the-loop sampling.",
           ["p1.ap_map", "p2.storage_design"], ["metric aggregation", "sampling", "HITL"]),
        _c("p2.monitoring_observability", "Monitoring & Observability", 2, T,
           "SLOs, golden signals, distributed tracing, model+system observability in one plane.",
           ["p1.model_monitoring", "p2.distributed_fundamentals"], ["SLO", "golden signals", "tracing"]),
        _c("p2.feedback_loops", "Data Feedback Loops", 2, T,
           "Closing the loop: mining production failures back into training/eval sets.",
           ["p2.monitoring_observability", "p2.eval_platform"], ["active learning", "auto-mining", "flywheel"]),
        _c("p2.reliability_tradeoffs", "Cost / Latency / Fault-Tolerance Tradeoffs", 2, T,
           "Explicit tradeoff reasoning: capacity math, redundancy cost, degradation modes.",
           ["p2.distributed_fundamentals"], ["capacity planning", "redundancy", "graceful degradation"]),

        # ----------------------------------------------------------- Phase 3
        _c("p3.roadmap_ambiguity", "Roadmap from Ambiguity", 3, E,
           "Turning ambiguous goals into a sequenced, measurable roadmap.",
           topics=["problem framing", "sequencing", "milestones", "success metrics"]),
        _c("p3.cross_functional", "Cross-Functional Leadership", 3, L,
           "Leading across PM, research, ops, safety; alignment without authority.",
           topics=["alignment", "stakeholders", "communication cadence"]),
        _c("p3.technical_strategy", "Technical Strategy", 3, E,
           "Multi-quarter technical bets, platform vs product tension, build/buy.",
           ["p3.roadmap_ambiguity"], ["technical bets", "platform investment", "build vs buy"]),
        _c("p3.org_influence", "Organizational Influence", 3, L,
           "Influencing beyond your team: writing, reviews, coalition building.",
           ["p3.cross_functional"], ["influence", "writing", "coalitions"]),
        _c("p3.performance_management", "Performance Management", 3, L,
           "Expectations, feedback, underperformance, promotion cases.",
           topics=["expectations", "feedback", "PIP", "promotion"]),
        _c("p3.mentorship", "Mentorship & Growth", 3, L,
           "Growing engineers and leads; delegation as a growth tool.",
           topics=["coaching", "delegation", "growth plans"]),
        _c("p3.hiring", "Hiring & Team Building", 3, L,
           "Sourcing, calibration, closing, onboarding, team shape.",
           topics=["calibration", "closing", "onboarding", "team topology"]),
        _c("p3.conflict_resolution", "Conflict Resolution", 3, L,
           "Surfacing and resolving disagreement; disagree-and-commit done honestly.",
           topics=["mediation", "escalation", "disagree and commit"]),
        _c("p3.prioritization_risk", "Prioritization & Risk", 3, E,
           "Ruthless prioritization under constraints; explicit risk registers.",
           topics=["prioritization", "risk register", "cut lines"]),
        _c("p3.safety_culture", "Safety Ownership", 3, E,
           "Safety as a first-class engineering constraint; incident culture.",
           ["p1.safety_metrics"], ["incident review", "safety cases", "blameless postmortems"]),
        _c("p3.closed_loop_execution", "Closed-Loop Execution", 3, E,
           "Plan → execute → measure → correct; commitments that land.",
           ["p3.prioritization_risk"], ["execution cadence", "measurement", "course correction"]),
        _c("p3.business_impact", "Business Impact", 3, E,
           "Connecting engineering work to business/customer outcomes with numbers.",
           topics=["impact quantification", "cost of delay", "customer outcomes"]),

        # ----------------------------------------------------------- Phase 4
        _c("p4.hill_climbing", "Multi-Objective Hill Climbing", 4, E,
           "Iteratively improving a system under competing objectives (performance, safety, "
           "cost, velocity, morale) with hypotheses, measurements, and keep/reject decisions.",
           ["p2.reliability_tradeoffs", "p3.prioritization_risk", "p3.closed_loop_execution",
            "p1.model_monitoring"],
           ["multi-objective optimization", "hypothesis testing", "local optima", "hard constraints"]),
    ]

    phases = [
        PhaseSpec(
            phase=1, title="Technical / ML Depth",
            objective="Rebuild deep, interview-grade fluency in ML evaluation, monitoring, and failure analysis for perception systems.",
            topics=["precision/recall", "IoU", "AP/mAP", "macro vs micro", "temporal effects",
                    "regression detection", "CUSUM", "statistical evaluation", "model monitoring",
                    "failure analysis", "safety metrics"],
            skills=["explain metric tradeoffs from first principles", "diagnose offline/online divergence",
                    "design monitoring for a perception model", "quantify uncertainty in evaluations"],
            exercises=["metric-diagnosis scenarios (offline +5% / shadow −2%)", "CUSUM drift drills",
                       "monitoring-design critiques", "failure-taxonomy construction"],
            assessments=["adaptive diagnostic", "scenario exercises scored 1-5 against rubrics"],
            completion_criteria=["all Phase-1 competencies at COMPETENT+",
                                 "at least one STRONG in regression detection or failure analysis"],
        ),
        PhaseSpec(
            phase=2, title="Large-Scale System Design",
            objective="Design petabyte-scale ML data/serving/evaluation systems with explicit capacity math and tradeoff reasoning.",
            topics=["petabyte ingestion", "streaming/batch", "storage", "feature generation", "training",
                    "parallel inference", "evaluation platforms", "monitoring", "feedback loops",
                    "reliability", "observability", "cost/latency/fault-tolerance tradeoffs"],
            skills=["propose an end-to-end architecture", "state capacity math", "identify SPOFs",
                    "close data feedback loops", "argue tradeoffs quantitatively"],
            exercises=["design-lab challenges with typed component graphs + rationales"],
            assessments=["design-lab rubric grading per dimension"],
            completion_criteria=["two design-lab challenges at COMPETENT+",
                                 "no missing-stage or feedback-loop gaps on the final submission"],
        ),
        PhaseSpec(
            phase=3, title="Execution & People Management",
            objective="Evidence leadership and execution with quantified STAR stories mapped to EM competencies.",
            topics=["roadmap from ambiguity", "cross-functional leadership", "technical strategy",
                    "org influence", "performance management", "mentorship", "hiring",
                    "conflict resolution", "prioritization", "risk", "safety", "closed-loop execution",
                    "business impact"],
            skills=["tell diagnosable STAR stories", "quantify outcomes", "show personal ownership",
                    "demonstrate influence and disagreement handling"],
            exercises=["STAR Story Box diagnosis and strengthening loops"],
            assessments=["claim-vs-evidence checks", "per-component STAR diagnosis",
                         "story-to-competency evidence mapping"],
            completion_criteria=["6+ stories saved as evidence covering 8+ Phase-3 competencies",
                                 "no unquantified-claim flags on the final story set"],
        ),
        PhaseSpec(
            phase=4, title="Simulation / Hill-Climbing Exercise",
            objective="Apply everything in a stateful multi-objective simulation: hypothesis → intervention → measure → keep/reject.",
            topics=["multi-objective optimization under competing constraints", "hard safety floors",
                    "second-order consequences", "delayed effects"],
            skills=["form falsifiable hypotheses", "balance competing objectives",
                    "respond to incidents", "debrief decisions into evidence"],
            exercises=["seeded scenario: inherited perception team, rising regression rate, exec pressure"],
            assessments=["simulation debrief mapping decisions to competency evidence"],
            completion_criteria=["complete a simulation with balanced score above threshold and no unresolved safety incident"],
        ),
    ]

    return Blueprint(phases=phases, competencies=competencies)


# ------------------------------------------------------------------ storage


BLUEPRINT_KEY = "active"


def load_blueprint(store: Optional[Store] = None) -> Blueprint:
    store = store or get_store()
    raw = store.get("blueprint", BLUEPRINT_KEY)
    if raw:
        try:
            return Blueprint(**raw)
        except Exception:
            pass
    bp = seed_blueprint()
    store.put("blueprint", BLUEPRINT_KEY, bp)
    return bp


def save_blueprint(bp: Blueprint, store: Optional[Store] = None) -> None:
    (store or get_store()).put("blueprint", BLUEPRINT_KEY, bp)


# ------------------------------------------------------------------ graph ops


def competency_index(bp: Blueprint) -> Dict[str, Competency]:
    return {c.id: c for c in bp.competencies}


def validate_graph(bp: Blueprint) -> List[str]:
    """Return a list of integrity problems (empty = healthy)."""
    problems: List[str] = []
    idx = competency_index(bp)
    for c in bp.competencies:
        for p in c.prerequisites:
            if p not in idx:
                problems.append(f"{c.id}: unknown prerequisite '{p}'")
    # cycle detection (DFS)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {cid: WHITE for cid in idx}

    def dfs(cid: str, stack: List[str]) -> None:
        color[cid] = GRAY
        for p in idx[cid].prerequisites:
            if p not in idx:
                continue
            if color[p] == GRAY:
                problems.append("prerequisite cycle: " + " -> ".join(stack + [cid, p]))
            elif color[p] == WHITE:
                dfs(p, stack + [cid])
        color[cid] = BLACK

    for cid in idx:
        if color[cid] == WHITE:
            dfs(cid, [])
    return problems


def downstream_map(bp: Blueprint) -> Dict[str, Set[str]]:
    """competency_id -> set of competencies that (transitively) depend on it."""
    idx = competency_index(bp)
    direct: Dict[str, Set[str]] = {cid: set() for cid in idx}
    for c in bp.competencies:
        for p in c.prerequisites:
            if p in direct:
                direct[p].add(c.id)

    result: Dict[str, Set[str]] = {}

    def collect(cid: str) -> Set[str]:
        if cid in result:
            return result[cid]
        acc: Set[str] = set()
        for d in direct[cid]:
            acc.add(d)
            acc |= collect(d)
        result[cid] = acc
        return acc

    for cid in idx:
        collect(cid)
    return result


# Display dimensions used by the dashboard readiness bars. Distinct from the
# per-competency Dimension tag (which is never collapsed); this is a UX grouping.
DISPLAY_DIMENSIONS = ["Technical Depth", "System Design", "Execution",
                      "Leadership", "Communication", "Safety/Risk"]

SAFETY_COMPETENCIES = {"p1.safety_metrics", "p3.safety_culture"}


def display_dimensions_for(c: Competency) -> List[str]:
    dims: List[str] = []
    if c.id in SAFETY_COMPETENCIES:
        dims.append("Safety/Risk")
    if c.phase == 1:
        dims.append("Technical Depth")
    elif c.phase == 2:
        dims.append("System Design")
    elif c.phase == 3:
        dims.append("Leadership" if c.dimension == Dimension.LEADERSHIP else "Execution")
        if c.id in ("p3.cross_functional", "p3.org_influence"):
            dims.append("Communication")
    elif c.phase == 4:
        dims.append("Execution")
    return dims
