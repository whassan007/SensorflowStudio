"""Evidence hierarchy and the strictly-typed RetrospectiveScorecard.

Tier semantics (enforced in types, rendered distinctly):
    TIER1_OBSERVED      FACT — read directly from the failure artifact.
    TIER2_DERIVED       DERIVED FACT — deterministically computed from facts.
    TIER3_RETRIEVED     RETRIEVED REQUIREMENT — backed by a retrieval hit.
    TIER4_AI_HYPOTHESIS AI HYPOTHESIS — inference, never presented as fact.
The SAFETY DETERMINATION (severity + launch recommendation) is produced only
by the deterministic policy engine (sensorflow.retro.policy).

The no-citation-without-retrieval rule lives here: RetrievedStandard has NO
optional escape hatch — a citation cannot be constructed without the actual
retrieved text, its relevance score, and its source metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

UNKNOWN = "UNKNOWN"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceTier(str, Enum):
    OBSERVED = "TIER1_OBSERVED"
    DERIVED = "TIER2_DERIVED"
    RETRIEVED = "TIER3_RETRIEVED"
    AI_HYPOTHESIS = "TIER4_AI_HYPOTHESIS"


TIER_RENDER_NAMES = {
    EvidenceTier.OBSERVED: "FACT",
    EvidenceTier.DERIVED: "DERIVED FACT",
    EvidenceTier.RETRIEVED: "RETRIEVED REQUIREMENT",
    EvidenceTier.AI_HYPOTHESIS: "AI HYPOTHESIS",
}


class Severity(str, Enum):
    BENIGN = "BENIGN"
    DISRUPTIVE = "DISRUPTIVE"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


class LaunchRecommendation(str, Enum):
    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceItem(BaseModel):
    tier: EvidenceTier
    key: str                        # machine key, e.g. "ego_speed_mps"
    statement: str                  # human-readable statement
    value: Optional[str] = None     # string form; UNKNOWN if absent
    provenance: str                 # where this came from (field/tool/doc/agent)

    @field_validator("statement")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("evidence statement must be non-empty")
        return v


class RetrievedStandard(BaseModel):
    """A citation. Constructible ONLY with real retrieval output."""
    source: str
    document: str
    version: str
    section: str
    retrieved_text: str = Field(min_length=20)
    relevance_score: float = Field(ge=0.0, le=1.0)
    doc_id: str
    doc_type: str
    jurisdiction: str
    effective_date: str
    synthetic: bool
    label: str                      # SYNTHETIC_EXAMPLE... or PARAPHRASE...
    chunk_id: str


class RootCauseHypothesis(BaseModel):
    hypothesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence_keys: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    tier: EvidenceTier = EvidenceTier.AI_HYPOTHESIS

    @field_validator("tier")
    @classmethod
    def _hypothesis_is_never_fact(cls, v: EvidenceTier) -> EvidenceTier:
        if v != EvidenceTier.AI_HYPOTHESIS:
            raise ValueError("root-cause hypotheses are always TIER4_AI_HYPOTHESIS")
        return v


class StatSignificance(BaseModel):
    method: str                      # e.g. "seqeval.PairedSequentialTest"
    significant: Optional[bool]      # None => not evaluable from this artifact
    detail: str


class UncertaintyReport(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    unknown_metrics: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.missing_fields or self.unknown_metrics)


class RetrospectiveScorecard(BaseModel):
    # identity / provenance
    evaluation_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    policy_version: str
    agent_version: str
    backend_used: str

    # classification
    failure_type: str               # FALSE_POSITIVE | FALSE_NEGATIVE | ...
    severity: Severity              # POLICY-decided severity (authoritative)
    ai_proposed_severity: Optional[Severity] = None
    severity_divergence: bool = False
    safety_critical_recall_impact: Optional[float] = None
    scr_impact_detail: str = ""
    behavioral_consequence: str
    launch_recommendation: LaunchRecommendation
    launch_rationale: List[str] = Field(default_factory=list)

    # context
    baseline_model: Optional[str] = None
    candidate_model: Optional[str] = None
    scenario: Dict[str, Any] = Field(default_factory=dict)
    object_class: Optional[str] = None
    ground_truth: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

    # kinematics / behavior (None => UNKNOWN, listed in uncertainty)
    ego_speed_mps: Optional[float] = None
    distance_to_object_m: Optional[float] = None
    relative_velocity_mps: Optional[float] = None
    stopping_distance_m: Optional[float] = None
    ttc_s: Optional[float] = None
    ttc_validity: List[str] = Field(default_factory=list)
    planner_response: Optional[Dict[str, Any]] = None
    disengagement_probability: Optional[float] = None

    # statistics
    metric_delta: Optional[Dict[str, float]] = None
    statistical_significance: Optional[StatSignificance] = None
    distribution_shift: Optional[Dict[str, Any]] = None

    # evidence chain
    root_cause_hypotheses: List[RootCauseHypothesis] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(min_length=1)
    retrieved_standards: List[RetrievedStandard] = Field(default_factory=list)
    uncertainty: UncertaintyReport
    human_review_required: bool
    human_review_reasons: List[str] = Field(default_factory=list)

    def evidence_by_tier(self, tier: EvidenceTier) -> List[EvidenceItem]:
        return [e for e in self.evidence if e.tier == tier]

    # ------------------------------------------------------------- rendering

    def render_markdown(self) -> str:
        lines: List[str] = [
            f"# Retrospective Scorecard — {self.evaluation_id}",
            "",
            f"- **Failure type:** {self.failure_type}",
            f"- **Severity (policy determination):** {self.severity.value}"
            + (f"  _(AI proposed {self.ai_proposed_severity.value}; divergence "
               f"flagged for human review)_" if self.severity_divergence
               and self.ai_proposed_severity else ""),
            f"- **Launch recommendation:** {self.launch_recommendation.value}",
            f"- **Human review required:** "
            f"{'YES — ' + '; '.join(self.human_review_reasons) if self.human_review_required else 'no'}",
            f"- **Models:** baseline={self.baseline_model or UNKNOWN} "
            f"candidate={self.candidate_model or UNKNOWN}",
            f"- **Policy/agent:** {self.policy_version} / {self.agent_version} "
            f"(backend: {self.backend_used})",
            "",
            "## Evidence chain (tiered)",
        ]
        for tier in EvidenceTier:
            items = self.evidence_by_tier(tier)
            if not items:
                continue
            lines.append(f"### {TIER_RENDER_NAMES[tier]} ({tier.value})")
            for e in items:
                val = f" = `{e.value}`" if e.value is not None else ""
                lines.append(f"- **{e.key}**{val}: {e.statement}  \n"
                             f"  _provenance: {e.provenance}_")
            lines.append("")

        lines.append("## RETRIEVED REQUIREMENTS (citations)")
        if not self.retrieved_standards:
            lines.append("_none retrieved — no requirement is cited_")
        for s in self.retrieved_standards:
            lines.append(
                f"- **[{s.label}]** {s.document} v{s.version} — §{s.section} "
                f"(source: {s.source}, relevance {s.relevance_score:.2f})  \n"
                f"  > {s.retrieved_text[:280]}{'…' if len(s.retrieved_text) > 280 else ''}")
        lines.append("")

        lines.append("## AI HYPOTHESES (inference — never fact)")
        for h in self.root_cause_hypotheses:
            lines.append(f"- ({h.confidence:.0%} confidence) {h.hypothesis}  \n"
                         f"  _supported by: {', '.join(h.supporting_evidence_keys) or 'n/a'};"
                         f" missing: {', '.join(h.missing_evidence) or 'none'}_")
        lines.append("")

        lines.append("## SAFETY DETERMINATION (deterministic policy)")
        lines.append(f"- Severity: **{self.severity.value}**")
        lines.append(f"- Launch: **{self.launch_recommendation.value}**")
        for r in self.launch_rationale:
            lines.append(f"  - {r}")
        lines.append("")

        if self.uncertainty.has_gaps:
            lines.append("## UNCERTAINTY / MISSING EVIDENCE")
            for f in self.uncertainty.missing_fields:
                lines.append(f"- missing field: `{f}` -> {UNKNOWN}")
            for m in self.uncertainty.unknown_metrics:
                lines.append(f"- metric not computable: `{m}` -> {UNKNOWN}")
            for n in self.uncertainty.notes:
                lines.append(f"- {n}")

        return "\n".join(lines)
