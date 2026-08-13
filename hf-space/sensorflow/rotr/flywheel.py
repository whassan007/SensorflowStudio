"""Evaluation flywheel: confirmed violation -> failure signature -> cluster
-> HITL validation -> regression-test artifact, under dataset-role
governance with a contamination guard.

Roles are immutable once assigned; members of PROTECTED roles (REGRESSION,
LAUNCH) are NEVER training-eligible without a recorded governance override
— semantics mirrored from raremine.lineage (ContaminationError subclasses
raremine's LeakageError when importable so callers can catch either).
`sensorflow.studio2`'s registry is used when it lands (guarded import,
re-checked at integration time); until then the local implementation
persists to runs/rotr/.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from sensorflow.rotr import SOFTWARE_VERSION
from sensorflow.rotr import store
from sensorflow.rotr.models import (
    HITLReview, PROTECTED_ROLES, Provenance, TrainingCandidate, now_iso,
)

FLYWHEEL_VERSION = "rotr-flywheel-1.0.0"

# Mirror raremine's guard exception when importable (landed package,
# read-only reuse) so platform callers can catch one exception type.
try:
    from sensorflow.raremine.lineage import LeakageError as _LeakageBase
except Exception:                                     # pragma: no cover
    _LeakageBase = RuntimeError


class ContaminationError(_LeakageBase):
    """Raised when a protected evaluation member would leak into training."""


# Registry backend: studio2 when it lands, else local (recorded on artifacts).
# The local store is ALWAYS written (the guard never depends on the
# concurrent workstream); studio2 registration is a best-effort mirror.
try:
    from sensorflow.studio2.registry import get_registry as _studio2_get_registry
    REGISTRY_BACKEND = "studio2+local"
except Exception:
    _studio2_get_registry = None
    REGISTRY_BACKEND = "local"


def _mirror_to_studio2(candidate: "TrainingCandidate") -> Optional[str]:
    """Best-effort mirror of the regression artifact into the studio2
    entity registry (guarded: studio2 is a concurrent workstream)."""
    if _studio2_get_registry is None:
        return None
    try:
        entity = _studio2_get_registry().register_dataset(
            name=f"rotr-regression-{candidate.candidate_id}",
            role="REGRESSION",
            provenance=candidate.provenance.model_dump(),
            meta={"violation_id": candidate.violation_id,
                  "cluster_id": candidate.cluster_id,
                  "run_id": candidate.run_id, "source": "sensorflow.rotr"},
            actor="rotr-flywheel")
        return entity.get("entity_id")
    except Exception:
        return None


def _audit(event: str, subject: str, actor: str, detail: str = "") -> None:
    doc = store.read_json("flywheel", "audit.json") or {"events": []}
    doc["events"].append({"at": now_iso(), "event": event, "subject": subject,
                          "actor": actor, "detail": detail})
    store.write_json(doc, "flywheel", "audit.json")


# ------------------------------------------------------------ HITL queue


def build_queue(run_id: str, violations: List[Dict],
                clusters: List[Dict]) -> List[HITLReview]:
    vid_to_cluster = {}
    for c in clusters:
        for vid in c["member_violation_ids"]:
            vid_to_cluster[vid] = c["cluster_id"]
    reviews = []
    for v in violations:
        rid = f"rev-{v['violation_id']}"
        reviews.append(HITLReview(
            review_id=rid, run_id=run_id, violation_id=v["violation_id"],
            cluster_id=vid_to_cluster.get(v["violation_id"]),
            status="PENDING",
            provenance=Provenance(
                scenario_id=v.get("scenario_id"),
                software_version=f"{SOFTWARE_VERSION}/{FLYWHEEL_VERSION}",
                source="SYNTHETIC")))
    store.write_json({"run_id": run_id,
                      "reviews": [r.model_dump() for r in reviews]},
                     "flywheel", f"queue-{run_id}.json")
    return reviews


def get_queue(run_id: str) -> List[Dict]:
    doc = store.read_json("flywheel", f"queue-{run_id}.json")
    return doc["reviews"] if doc else []


def act(run_id: str, review_id: str, action: str, actor: str,
        notes: str = "") -> Dict:
    """VALIDATE turns a confirmed violation into a regression-test artifact
    (protected role, not training-eligible). REJECT closes the review."""
    doc = store.read_json("flywheel", f"queue-{run_id}.json")
    if not doc:
        raise KeyError(f"no HITL queue for run {run_id}")
    review = next((r for r in doc["reviews"] if r["review_id"] == review_id),
                  None)
    if review is None:
        raise KeyError(f"unknown review {review_id}")
    if review["status"] != "PENDING":
        raise ValueError(f"review {review_id} already {review['status']}")

    action = action.upper()
    if action not in ("VALIDATE", "REJECT"):
        raise ValueError("action must be VALIDATE or REJECT")
    review["status"] = "VALIDATED" if action == "VALIDATE" else "REJECTED"
    review["action"] = action
    review["actor"] = actor
    review["notes"] = notes
    store.write_json(doc, "flywheel", f"queue-{run_id}.json")
    _audit(f"hitl_{action.lower()}", review_id, actor, notes)

    candidate = None
    if action == "VALIDATE":
        candidate = _create_candidate(run_id, review, actor)
        _append_suite_member(run_id, review, candidate)
    return {"review": review,
            "candidate": candidate.model_dump() if candidate else None}


def _create_candidate(run_id: str, review: Dict, actor: str) -> TrainingCandidate:
    cid = f"tc-{uuid.uuid4().hex[:10]}"
    cand = TrainingCandidate(
        candidate_id=cid, run_id=run_id,
        violation_id=review["violation_id"],
        cluster_id=review.get("cluster_id"),
        dataset_role="REGRESSION", training_eligible=False,
        guard_state="PROTECTED",
        provenance=Provenance(
            software_version=f"{SOFTWARE_VERSION}/{FLYWHEEL_VERSION}",
            source="SYNTHETIC"))
    doc = cand.model_dump()
    doc["studio2_entity_id"] = _mirror_to_studio2(cand)
    doc["registry_backend"] = REGISTRY_BACKEND
    store.write_json(doc, "flywheel", "candidates", f"{cid}.json")
    _audit("candidate_created", cid, actor,
           f"role=REGRESSION (protected) violation={review['violation_id']} "
           f"registry_backend={REGISTRY_BACKEND}")
    return cand


def _append_suite_member(run_id: str, review: Dict,
                         candidate: TrainingCandidate) -> None:
    suite = store.read_json("flywheel", "regression-suite.json") or {
        "suite_id": "rotr-regression-suite", "members": [],
        "registry_backend": REGISTRY_BACKEND}
    suite["members"].append({
        "candidate_id": candidate.candidate_id,
        "violation_id": review["violation_id"],
        "cluster_id": review.get("cluster_id"),
        "run_id": run_id, "role": "REGRESSION",
        "added_at": now_iso(), "added_by": review.get("actor"),
        "immutable_role_note": "REGRESSION members re-run against every "
                               "candidate; never training-eligible without "
                               "a recorded governance override"})
    store.write_json(suite, "flywheel", "regression-suite.json")


def get_suite() -> Dict:
    return store.read_json("flywheel", "regression-suite.json") or {
        "suite_id": "rotr-regression-suite", "members": [],
        "registry_backend": REGISTRY_BACKEND}


# ------------------------------------------------------------ contamination guard


def list_candidates() -> List[Dict]:
    out = []
    for name in store.list_dir("flywheel", "candidates"):
        doc = store.read_json("flywheel", "candidates", name)
        if doc:
            out.append(doc)
    return out


def get_candidate(candidate_id: str) -> Optional[Dict]:
    return store.read_json("flywheel", "candidates", f"{candidate_id}.json")


def promote_to_training(candidate_id: str, actor: str) -> Dict:
    """Refused for protected roles unless a governance override is recorded
    (raremine guard semantics)."""
    doc = get_candidate(candidate_id)
    if doc is None:
        raise KeyError(f"unknown candidate {candidate_id}")
    if doc["dataset_role"] in PROTECTED_ROLES and not doc.get("override"):
        _audit("promotion_refused", candidate_id, actor,
               f"protected role {doc['dataset_role']} without override")
        raise ContaminationError(
            f"candidate {candidate_id} holds protected role "
            f"{doc['dataset_role']}: training promotion refused without a "
            "recorded governance override")
    doc["training_eligible"] = True
    doc["guard_state"] = "OVERRIDDEN" if doc.get("override") else "ELIGIBLE"
    store.write_json(doc, "flywheel", "candidates", f"{candidate_id}.json")
    _audit("promoted_to_training", candidate_id, actor,
           f"guard_state={doc['guard_state']}")
    return doc


def governance_override(candidate_id: str, actor: str, reason: str) -> Dict:
    if not reason.strip():
        raise ValueError("a governance override requires a recorded reason")
    doc = get_candidate(candidate_id)
    if doc is None:
        raise KeyError(f"unknown candidate {candidate_id}")
    doc["override"] = {"actor": actor, "reason": reason, "at": now_iso()}
    doc["guard_state"] = "OVERRIDDEN"
    store.write_json(doc, "flywheel", "candidates", f"{candidate_id}.json")
    _audit("governance_override", candidate_id, actor, reason)
    return doc
