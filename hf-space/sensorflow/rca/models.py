"""Core RCA entities: stages, findings, and the Investigation state machine."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------- stages

STAGES: List[Dict] = [
    {"index": 0, "key": "comparison_validity", "title": "Comparison Validity",
     "question": "Are the offline and shadow numbers even measuring the same thing?"},
    {"index": 1, "key": "offline_audit", "title": "Offline Audit",
     "question": "Can the offline +5% be reproduced, and is the eval set leak-free?"},
    {"index": 2, "key": "population_validation", "title": "Population Validation",
     "question": "Do the two evaluations see comparable populations and volumes?"},
    {"index": 3, "key": "distribution_shift", "title": "Distribution Shift",
     "question": "How different are the offline and shadow data distributions?"},
    {"index": 4, "key": "conditional_performance", "title": "Conditional Performance",
     "question": "Where (which segments) does the delta live in each environment?"},
    {"index": 5, "key": "paired_comparison", "title": "Paired Model Comparison",
     "question": "On the same units, which predictions flipped between A and B?"},
    {"index": 6, "key": "statistical_significance", "title": "Statistical Significance",
     "question": "Is the shadow regression distinguishable from noise, cluster-aware?"},
    {"index": 7, "key": "feature_parity", "title": "Feature Parity",
     "question": "Do the models see the same feature values online as offline?"},
    {"index": 8, "key": "serving_parity", "title": "Serving Parity",
     "question": "Is the serving configuration identical to the offline harness?"},
    {"index": 9, "key": "shadow_traffic", "title": "Shadow Traffic Audit",
     "question": "Is shadow traffic a fair sample, or biased by selection/drops?"},
    {"index": 10, "key": "label_integrity", "title": "Label / Ground-Truth Integrity",
     "question": "Are shadow labels mature, unbiased, and on the same policy?"},
    {"index": 11, "key": "root_cause_scoring", "title": "Root Cause Scoring + Decision",
     "question": "Which hypotheses does the accumulated evidence support?"},
    {"index": 12, "key": "recommendations_report", "title": "Experiments + Report",
     "question": "What is the minimum additional evidence needed, and the verdict?"},
]

STAGE_KEYS = [s["key"] for s in STAGES]

STAGE_STATUS = ("pending", "in_progress", "complete", "complete_with_unknowns")

# ---------------------------------------------------------------- root causes

ROOT_CAUSES = (
    "TRUE_MODEL_REGRESSION",
    "DISTRIBUTION_SHIFT",
    "FEATURE_SKEW",
    "SERVING_MISMATCH",
    "LABEL_LATENCY",
    "SAMPLING_BIAS",
    "STATISTICAL_NOISE",
    "OFFLINE_CONTAMINATION",
)

HYPOTHESIS_LABELS: Dict[str, str] = {
    "TRUE_MODEL_REGRESSION": "Candidate genuinely regresses on production data",
    "DISTRIBUTION_SHIFT": "Production population differs from the offline eval set",
    "FEATURE_SKEW": "Online/offline feature pipeline mismatch (training-serving skew)",
    "SERVING_MISMATCH": "Shadow serving config differs (threshold / quantization / runtime)",
    "LABEL_LATENCY": "Shadow labels immature / provisional and biased",
    "SAMPLING_BIAS": "Shadow traffic selection is not a fair sample",
    "STATISTICAL_NOISE": "Deltas are within noise; effective sample too small",
    "OFFLINE_CONTAMINATION": "Offline +5% inflated by train/eval leakage",
}

FINDING_STATUS = ("PASS", "MISMATCH", "UNKNOWN")
SEVERITIES = ("INFO", "WARN", "CRITICAL")


# ------------------------------------------------------------------- findings


@dataclass
class Finding:
    """One structured evidence record produced by a stage."""

    id: str
    stage: str                 # stage key
    code: str                  # stable machine code, e.g. FP_FEATURE_SKEW:obj_distance_m
    title: str
    status: str                # PASS / MISMATCH / UNKNOWN
    severity: str              # INFO / WARN / CRITICAL
    detail: str = ""
    source: str = "auto"       # auto / human
    created_at: float = field(default_factory=time.time)
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def make_finding(stage: str, code: str, title: str, status: str, severity: str,
                 detail: str = "", data: Optional[Dict] = None,
                 source: str = "auto") -> Finding:
    assert status in FINDING_STATUS, status
    assert severity in SEVERITIES, severity
    return Finding(id=f"fnd-{uuid.uuid4().hex[:10]}", stage=stage, code=code,
                   title=title, status=status, severity=severity, detail=detail,
                   data=data or {}, source=source)


# --------------------------------------------------------------- investigation


class StageOrderError(Exception):
    """Raised when the enforced stage ordering is violated."""


class UnknownsNotAcknowledgedError(Exception):
    """Raised when completing a stage with critical UNKNOWNs without an ack."""


@dataclass
class StageState:
    index: int
    key: str
    title: str
    status: str = "pending"
    completed_at: Optional[float] = None
    acknowledged_unknowns: bool = False
    ack_note: str = ""
    skip_acknowledged: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Investigation:
    id: str
    name: str
    baseline_model: str
    candidate_model: str
    scenario_cause: str            # the planted truth (hidden in training mode)
    seed: int
    training_mode: bool = False
    revealed: bool = False
    created_at: float = field(default_factory=time.time)
    claims: Dict = field(default_factory=dict)
    stages: List[StageState] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    # human overrides on the scoring board: {hypothesis: {confidence, note}}
    human_assessments: Dict[str, Dict] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)

    # ----------------------------------------------------------- construction

    @classmethod
    def new(cls, name: str, baseline_model: str, candidate_model: str,
            scenario_cause: str, seed: int, training_mode: bool = False,
            claims: Optional[Dict] = None) -> "Investigation":
        inv = cls(
            id=f"inv-{uuid.uuid4().hex[:8]}",
            name=name,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            scenario_cause=scenario_cause,
            seed=seed,
            training_mode=training_mode,
            claims=claims or {},
            stages=[StageState(index=s["index"], key=s["key"], title=s["title"])
                    for s in STAGES],
        )
        inv.stages[0].status = "in_progress"
        inv.log("created", f"Investigation created ({name})")
        return inv

    # ---------------------------------------------------------------- helpers

    def log(self, kind: str, message: str, data: Optional[Dict] = None) -> None:
        self.events.append({"ts": time.time(), "kind": kind, "message": message,
                            "data": data or {}})

    def stage_by_key(self, key: str) -> StageState:
        for st in self.stages:
            if st.key == key:
                return st
        raise KeyError(f"Unknown stage {key}")

    def findings_for(self, stage_key: str) -> List[Finding]:
        return [f for f in self.findings if f.stage == stage_key]

    def critical_unknowns(self, stage_key: str) -> List[Finding]:
        return [f for f in self.findings_for(stage_key)
                if f.status == "UNKNOWN" and f.severity == "CRITICAL"]

    def upsert_auto_findings(self, stage_key: str, findings: List[Finding]) -> None:
        """Replace prior auto findings for a stage (recorded human ones kept)."""
        self.findings = [f for f in self.findings
                         if not (f.stage == stage_key and f.source == "auto")]
        self.findings.extend(findings)

    def add_human_finding(self, finding: Finding) -> None:
        self.findings.append(finding)
        self.log("finding_recorded", f"Human finding recorded on {finding.stage}",
                 {"code": finding.code})

    # ------------------------------------------------------------ state machine

    def complete_stage(self, index: int, acknowledge_unknowns: bool = False,
                       ack_note: str = "") -> StageState:
        """Complete a stage. Ordering is enforced: every earlier stage must be
        complete (possibly with acknowledged unknowns). Critical UNKNOWN
        findings block completion unless explicitly acknowledged, and the
        acknowledgment is recorded on the stage and in the event log."""
        if not 0 <= index < len(self.stages):
            raise KeyError(f"No stage {index}")
        stage = self.stages[index]
        for earlier in self.stages[:index]:
            if earlier.status not in ("complete", "complete_with_unknowns"):
                raise StageOrderError(
                    f"Stage {index} ({stage.title}) cannot complete before stage "
                    f"{earlier.index} ({earlier.title}) is complete. The methodology "
                    "is ordered on purpose: early plausible explanations do not "
                    "excuse skipping measurement-validity checks.")

        unknowns = self.critical_unknowns(stage.key)
        if unknowns and not acknowledge_unknowns:
            raise UnknownsNotAcknowledgedError(
                f"Stage {stage.title} has {len(unknowns)} critical UNKNOWN "
                "finding(s). Resolve them or explicitly acknowledge proceeding "
                "with unknowns.")

        if unknowns:
            stage.status = "complete_with_unknowns"
            stage.acknowledged_unknowns = True
            stage.ack_note = ack_note or "proceeding with unknowns"
            self.log("unknowns_acknowledged",
                     f"Proceeding with {len(unknowns)} unknown(s) on {stage.title}",
                     {"codes": [f.code for f in unknowns], "note": stage.ack_note})
        else:
            stage.status = "complete"
        stage.completed_at = time.time()

        nxt = index + 1
        if nxt < len(self.stages) and self.stages[nxt].status == "pending":
            self.stages[nxt].status = "in_progress"
        self.log("stage_completed", f"Stage {stage.title} -> {stage.status}")
        return stage

    def reopen_stage(self, index: int) -> StageState:
        stage = self.stages[index]
        stage.status = "in_progress"
        stage.completed_at = None
        self.log("stage_reopened", f"Stage {stage.title} reopened")
        return stage

    # ------------------------------------------------------------- serialization

    def to_dict(self, include_cause: bool = False) -> Dict:
        d = {
            "id": self.id,
            "name": self.name,
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "seed": self.seed,
            "training_mode": self.training_mode,
            "revealed": self.revealed,
            "created_at": self.created_at,
            "claims": self.claims,
            "stages": [s.to_dict() for s in self.stages],
            "findings": [f.to_dict() for f in self.findings],
            "human_assessments": self.human_assessments,
            "events": self.events[-200:],
        }
        # The planted cause is only exposed for non-training investigations or
        # once the user reveals it.
        if include_cause or (not self.training_mode) or self.revealed:
            d["scenario_cause"] = self.scenario_cause
        return d

    def to_json_dict(self) -> Dict:
        """Full-fidelity dict for persistence (always includes the cause)."""
        d = self.to_dict(include_cause=True)
        d["events"] = self.events
        return d

    @classmethod
    def from_json_dict(cls, d: Dict) -> "Investigation":
        inv = cls(
            id=d["id"], name=d["name"], baseline_model=d["baseline_model"],
            candidate_model=d["candidate_model"],
            scenario_cause=d["scenario_cause"], seed=d["seed"],
            training_mode=d.get("training_mode", False),
            revealed=d.get("revealed", False),
            created_at=d.get("created_at", time.time()),
            claims=d.get("claims", {}),
            stages=[StageState(**s) for s in d.get("stages", [])],
            findings=[Finding(**f) for f in d.get("findings", [])],
            human_assessments=d.get("human_assessments", {}),
            events=d.get("events", []),
        )
        return inv
