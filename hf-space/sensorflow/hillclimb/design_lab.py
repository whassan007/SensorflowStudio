"""System-design lab: typed component-graph challenges with a rule-based grader.

The user submits a graph of typed components (source/ingestion/stream/batch/
storage/feature/training/inference/eval/monitoring/feedback) plus a written
rationale per key decision. The grader checks: required stages present, no
orphan components, feedback-loop closure, single-point-of-failure detection,
stated capacity math, and rationale coverage of {scalability, reliability,
latency, cost, observability, failure handling, tradeoffs}.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from sensorflow.hillclimb.evaluate import QUANT_RE, TRADEOFF_MARKERS
from sensorflow.hillclimb.models import Evidence, Store, get_store, new_id

COMPONENT_TYPES = ["source", "ingestion", "stream", "batch", "storage", "feature",
                   "training", "inference", "eval", "monitoring", "feedback"]


class Component(BaseModel):
    id: str
    type: str
    name: str = ""
    note: str = ""  # capacity/annotation text, counts toward capacity math


class Edge(BaseModel):
    source: str
    target: str


class DesignSubmission(BaseModel):
    challenge_id: str
    user_id: str = "default"
    components: List[Component]
    edges: List[Edge]
    rationales: Dict[str, str] = Field(default_factory=dict)


class Challenge(BaseModel):
    challenge_id: str
    title: str
    brief: str
    # any-of groups: each inner list is satisfied by one present type
    required_stages: List[List[str]]
    key_decisions: List[str]
    competency_ids: List[str]
    requires_feedback_loop: bool = False


CHALLENGES: List[Challenge] = [
    Challenge(
        challenge_id="petabyte_ingestion",
        title="Petabyte-Scale Sensor Ingestion",
        brief=("Design ingestion for a fleet producing ~2 PB/week of camera+LiDAR data. "
               "Data must land queryable within 24h, survive a regional outage, and support "
               "backfill after schema changes. State your capacity math."),
        required_stages=[["source"], ["ingestion"], ["stream", "batch"], ["storage"], ["monitoring"]],
        key_decisions=["Partitioning & ordering strategy", "Hot/cold storage tiering",
                       "Backfill / schema-evolution handling", "Failure & retry semantics"],
        competency_ids=["p2.petabyte_ingestion", "p2.storage_design", "p2.streaming_batch"],
    ),
    Challenge(
        challenge_id="parallel_inference",
        title="Parallel Inference Serving",
        brief=("Serve a perception model at 20k inferences/sec with a 100ms p99 budget on a "
               "GPU fleet that must survive a zone outage. Show batching, sharding, autoscaling "
               "and the capacity math behind your replica count."),
        required_stages=[["source"], ["inference"], ["monitoring"], ["storage", "feature"]],
        key_decisions=["Batching vs latency budget", "Replica/shard topology",
                       "Autoscaling signal & headroom", "Degradation mode under overload"],
        competency_ids=["p2.parallel_inference", "p2.reliability_tradeoffs"],
    ),
    Challenge(
        challenge_id="eval_platform",
        title="Continuous Evaluation Platform",
        brief=("Design an evaluation platform that scores every model build against 500M frames, "
               "supports slice-level metrics, human-in-the-loop sampling, and blocks bad releases. "
               "Explain where metrics are aggregated and how results stay reproducible."),
        required_stages=[["source"], ["storage"], ["eval"], ["monitoring"], ["batch", "stream"]],
        key_decisions=["Metric aggregation & storage layout", "Sampling strategy for human review",
                       "Release-gate policy", "Reproducibility / lineage"],
        competency_ids=["p2.eval_platform", "p1.statistical_evaluation"],
    ),
    Challenge(
        challenge_id="feedback_loop",
        title="Monitoring & Data Feedback Loop",
        brief=("Close the loop: production monitoring detects failure patterns, mines the "
               "corresponding frames, routes them through labeling into the next training set, "
               "and verifies the fix shipped. The loop must be a real cycle, with monitoring on "
               "the loop itself."),
        required_stages=[["source"], ["inference"], ["monitoring"], ["feedback"], ["training"], ["eval"]],
        key_decisions=["Failure-mining trigger criteria", "Label routing & quality control",
                       "Loop latency (failure → fixed model)", "How you detect the loop itself breaking"],
        competency_ids=["p2.feedback_loops", "p2.monitoring_observability", "p1.model_monitoring"],
        requires_feedback_loop=True,
    ),
]

RATIONALE_DIMENSIONS: Dict[str, List[str]] = {
    "scalability": ["scal", "shard", "partition", "horizontal", "throughput", "parallel", "fan-out", "fan-in"],
    "reliability": ["replic", "redundan", "failover", "retry", "durab", "availab", "quorum", "multi-region", "zone"],
    "latency": ["latency", "p99", "p95", "real-time", "ms", "tail", "deadline", "sla"],
    "cost": ["cost", "$", "budget", "cheap", "spend", "tier", "spot", "utilization", "expensive"],
    "observability": ["observ", "monitor", "metric", "trace", "log", "alert", "dashboard", "slo"],
    "failure_handling": ["failure", "fallback", "degrad", "dead letter", "backpressure", "circuit",
                         "poison", "outage", "recover", "replay"],
    "tradeoffs": TRADEOFF_MARKERS,
}


class DimensionGrade(BaseModel):
    dimension: str
    score: int = Field(ge=0, le=5)
    gaps: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class StructuralChecks(BaseModel):
    missing_stages: List[str] = Field(default_factory=list)
    orphan_components: List[str] = Field(default_factory=list)
    feedback_loop_closed: bool = False
    feedback_loop_required: bool = False
    single_points_of_failure: List[str] = Field(default_factory=list)
    capacity_math_found: bool = False
    capacity_quotes: List[str] = Field(default_factory=list)


class DesignGrade(BaseModel):
    grade_id: str = Field(default_factory=lambda: new_id("dg"))
    challenge_id: str
    structural: StructuralChecks
    dimension_grades: List[DimensionGrade]
    overall_score: int = Field(ge=0, le=5)
    gaps: List[str] = Field(default_factory=list)
    evidence_id: Optional[str] = None


def get_challenge(challenge_id: str) -> Optional[Challenge]:
    return next((c for c in CHALLENGES if c.challenge_id == challenge_id), None)


# ------------------------------------------------------------------- checks


def _adjacency(sub: DesignSubmission) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {c.id: set() for c in sub.components}
    for e in sub.edges:
        if e.source in adj and e.target in adj:
            adj[e.source].add(e.target)
    return adj


def _undirected(adj: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    und: Dict[str, Set[str]] = {k: set() for k in adj}
    for u, vs in adj.items():
        for v in vs:
            und[u].add(v)
            und[v].add(u)
    return und


def _articulation_points(und: Dict[str, Set[str]]) -> Set[str]:
    """Classic Tarjan articulation points on the undirected component graph."""
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    points: Set[str] = set()
    timer = [0]

    def dfs(u: str) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v in und[u]:
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    points.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    points.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in und:
        if node not in disc:
            parent[node] = None
            dfs(node)
    return points


def _feedback_loop_closed(sub: DesignSubmission, adj: Dict[str, Set[str]]) -> bool:
    """A feedback/monitoring node participates in a directed cycle."""
    loop_nodes = [c.id for c in sub.components if c.type in ("feedback", "monitoring")]
    for start in loop_nodes:
        # BFS from successors of start; loop closed if we can reach start again
        seen: Set[str] = set()
        frontier = list(adj.get(start, ()))
        while frontier:
            node = frontier.pop()
            if node == start:
                return True
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(adj.get(node, ()))
    return False


def grade_submission(sub: DesignSubmission, store: Optional[Store] = None,
                     save_evidence: bool = True) -> DesignGrade:
    store = store or get_store()
    challenge = get_challenge(sub.challenge_id)
    if challenge is None:
        raise ValueError(f"unknown challenge '{sub.challenge_id}'")

    present_types = {c.type for c in sub.components}
    missing = ["/".join(group) for group in challenge.required_stages
               if not any(t in present_types for t in group)]

    adj = _adjacency(sub)
    connected: Set[str] = set()
    for e in sub.edges:
        connected.add(e.source)
        connected.add(e.target)
    orphans = [c.name or c.id for c in sub.components if c.id not in connected]

    loop_closed = _feedback_loop_closed(sub, adj)

    und = _undirected(adj)
    art = _articulation_points(und)
    # Only interior nodes count as SPOFs (sources/sinks trivially "cut" the path).
    degree = {k: len(v) for k, v in und.items()}
    spof_ids = [a for a in art if degree.get(a, 0) >= 2]
    comp_by_id = {c.id: c for c in sub.components}
    spofs = [comp_by_id[i].name or i for i in spof_ids if i in comp_by_id]

    all_text = " ".join(list(sub.rationales.values()) + [c.note for c in sub.components])
    capacity_quotes = [m.group(0) for m in QUANT_RE.finditer(all_text)][:6]
    capacity_found = len(capacity_quotes) > 0

    structural = StructuralChecks(
        missing_stages=missing,
        orphan_components=orphans,
        feedback_loop_closed=loop_closed,
        feedback_loop_required=challenge.requires_feedback_loop,
        single_points_of_failure=spofs,
        capacity_math_found=capacity_found,
        capacity_quotes=capacity_quotes,
    )

    # ----------------------------------------------- rationale dimension grades
    text_lower = all_text.lower()
    dimension_grades: List[DimensionGrade] = []
    for dim, cues in RATIONALE_DIMENSIONS.items():
        matched = [c for c in cues if c in text_lower]
        base = 0 if not matched else (2 if len(matched) == 1 else 4)
        bonus = 0
        gaps: List[str] = []
        if dim == "scalability" and capacity_found:
            bonus = 1
        elif dim == "scalability" and not capacity_found:
            gaps.append("No capacity math anywhere: state volumes, rates, and how they map to fleet size.")
        if dim == "reliability":
            if not spofs:
                bonus = 1
            else:
                gaps.append(f"Single point(s) of failure: {', '.join(spofs)} — removal disconnects the pipeline.")
        if dim == "observability" and "monitoring" in present_types:
            bonus = 1
        if dim == "failure_handling" and orphans:
            gaps.append(f"Orphan component(s) {', '.join(orphans)} — dead boxes suggest unfinished failure design.")
        if not matched:
            gaps.append(f"Rationales never address {dim.replace('_', ' ')}.")
        dimension_grades.append(DimensionGrade(
            dimension=dim, score=min(5, base + bonus), gaps=gaps,
            evidence=matched[:4]))

    # ------------------------------------------------------------- aggregate
    gaps: List[str] = []
    if missing:
        gaps.append(f"Missing required stage(s): {', '.join(missing)}.")
    if orphans:
        gaps.append(f"Orphan component(s): {', '.join(orphans)}.")
    if challenge.requires_feedback_loop and not loop_closed:
        gaps.append("The feedback loop is not closed: no directed cycle returns mined data to training.")
    if spofs:
        gaps.append(f"SPOF: {', '.join(spofs)}.")
    if not capacity_found:
        gaps.append("No stated capacity math.")
    for dg in dimension_grades:
        gaps.extend(dg.gaps)

    structural_penalty = (len(missing) * 1.5 + (1 if orphans else 0)
                          + (2 if challenge.requires_feedback_loop and not loop_closed else 0)
                          + (1 if spofs else 0) + (1 if not capacity_found else 0))
    dim_avg = sum(d.score for d in dimension_grades) / len(dimension_grades)
    overall = max(0, min(5, round(dim_avg - structural_penalty + 1)))
    if not missing and not orphans and loop_closed and capacity_found and dim_avg >= 3.5:
        overall = max(overall, 4)

    grade = DesignGrade(challenge_id=sub.challenge_id, structural=structural,
                        dimension_grades=dimension_grades, overall_score=overall,
                        gaps=gaps[:12])

    store.put("design_submissions", grade.grade_id,
              {"submission": sub.model_dump(), "grade": grade.model_dump()})

    if save_evidence:
        ev = Evidence(
            user_id=sub.user_id,
            competency_ids=challenge.competency_ids,
            artifact_type="design_submission",
            source=f"Design Lab: {challenge.title}",
            summary=f"Scored {overall}/5 on '{challenge.title}' ({len(sub.components)} components).",
            quotes=capacity_quotes[:3],
            score=float(overall),
            confidence=0.7,
            payload={"grade_id": grade.grade_id, "challenge_id": sub.challenge_id},
        )
        store.put("evidence", ev.evidence_id, ev)
        grade.evidence_id = ev.evidence_id

    return grade
