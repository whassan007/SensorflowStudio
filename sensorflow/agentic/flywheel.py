"""Evaluation flywheel — deterministic service.

validated failure -> taxonomy -> cluster -> evaluation suite create/update,
with the governance fields required by the spec (suite_id, version,
creation_reason, source_failures, sampling_policy, coverage,
known_limitations, approval_status, ...), diversity-aware sampling (no
duplicate-frame stuffing) and a CONTAMINATION GUARD: suite members are never
training-eligible without an explicit recorded override.

The guard mirrors the raremine leakage-guard concept; raremine is an
in-progress module, so it is imported defensively and a local guard with the
same semantics is used when it is unavailable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from sensorflow.agentic import data as data_mod
from sensorflow.agentic import store as store_mod
from sensorflow.agentic.models import (EvaluationSuite, FailureEvent,
                                       SuiteMember, new_id, now_iso)
from sensorflow.seqeval.sequential import PairedSequentialTest

# ---- contamination guard: prefer raremine's LeakageError concept -----------
try:  # raremine is an in-progress fenced module: import only defensively
    from sensorflow.raremine.lineage import LeakageError as _RaremineLeakageError
    LeakageError = _RaremineLeakageError
    _LEAKAGE_SOURCE = "sensorflow.raremine.lineage.LeakageError"
except Exception:  # pragma: no cover - depends on concurrent work
    class LeakageError(RuntimeError):
        """Raised when a protected evaluation example would leak to training."""
    _LEAKAGE_SOURCE = "local fallback (raremine unavailable at import time)"

MAX_FRAMES_PER_OBJECT = 2  # diversity-aware sampling: no duplicate stuffing


def _diverse_members(failure: FailureEvent) -> List[SuiteMember]:
    per_object: Dict[str, int] = defaultdict(int)
    seen_frames = set()
    members: List[SuiteMember] = []
    for inst in sorted(failure.instances,
                       key=lambda i: (i.sequence_id, i.frame_index)):
        key = f"{inst.sequence_id}:{inst.object_instance_id}"
        frame_key = f"{inst.sequence_id}:{inst.frame_id}"
        if per_object[key] >= MAX_FRAMES_PER_OBJECT or frame_key in seen_frames:
            continue
        per_object[key] += 1
        seen_frames.add(frame_key)
        members.append(SuiteMember(
            member_id=new_id("mem"),
            sequence_id=inst.sequence_id,
            frame_id=inst.frame_id,
            object_instance_id=inst.object_instance_id,
            source_failure_id=failure.failure_id,
            training_eligible=False,  # contamination guard: unconditional default
        ))
    return members


def create_or_update_suite(failure: FailureEvent, proposal: Dict,
                           actor: str = "flywheel") -> EvaluationSuite:
    """Create (or version-bump) a suite from a VALIDATED failure."""
    if not failure.validated:
        raise LeakageError(
            f"failure {failure.failure_id} is not human-validated; the "
            "flywheel refuses to create evaluation suites from unconfirmed "
            "failures")

    name = proposal["suite_name"]
    existing_id = None
    for fname in store_mod.list_dir("suites"):
        doc = store_mod.read_json("suites", fname)
        if doc and doc.get("name") == name:
            existing_id = doc["suite_id"]
            break

    members = _diverse_members(failure)
    coverage = {
        "sequences": len({m.sequence_id for m in members}),
        "objects": len({m.object_instance_id for m in members}),
        "frames": len(members),
        "construction_zone_share": round(
            sum(1 for i in failure.instances if i.construction_zone)
            / max(len(failure.instances), 1), 3),
    }

    if existing_id:
        doc = store_mod.read_json("suites", f"{existing_id}.json")
        suite = EvaluationSuite.model_validate(doc)
        known = {(m.sequence_id, m.frame_id) for m in suite.members}
        added = [m for m in members if (m.sequence_id, m.frame_id) not in known]
        suite.members.extend(added)
        suite.version += 1
        if failure.failure_id not in suite.source_failures:
            suite.source_failures.append(failure.failure_id)
        suite.coverage = coverage
        suite.updated_at = now_iso()
    else:
        suite = EvaluationSuite(
            suite_id=new_id("suite"),
            name=name,
            creation_reason=proposal["creation_reason"],
            source_failures=[failure.failure_id],
            taxonomy_tags=proposal.get("taxonomy_tags", []),
            sampling_policy=proposal.get("sampling_policy", ""),
            coverage=coverage,
            known_limitations=proposal.get("known_limitations", []),
            approval_status="draft",
            members=members,
        )
    store_mod.write_json(suite, "suites", f"{suite.suite_id}.json")
    store_mod.audit("suite_created_or_updated", failure.failure_id, actor,
                    detail=f"suite {suite.suite_id} ({suite.name}) v{suite.version} "
                           f"members={len(suite.members)}",
                    payload={"suite_id": suite.suite_id})
    return suite


def get_suite(suite_id: str) -> Optional[EvaluationSuite]:
    doc = store_mod.read_json("suites", f"{suite_id}.json")
    return EvaluationSuite.model_validate(doc) if doc else None


def list_suites() -> List[Dict]:
    out = []
    for fname in store_mod.list_dir("suites"):
        doc = store_mod.read_json("suites", fname)
        if doc:
            out.append(doc)
    return sorted(out, key=lambda d: d.get("created_at", ""), reverse=True)


def promote_member_to_training(suite_id: str, member_id: str,
                               actor: Optional[str] = None,
                               override_reason: Optional[str] = None) -> Dict:
    """Contamination guard: raises LeakageError unless an explicit override
    (who + why) is provided; the override is recorded on the suite AND in the
    audit trail."""
    suite = get_suite(suite_id)
    if suite is None:
        raise KeyError(f"Unknown suite {suite_id}")
    member = next((m for m in suite.members if m.member_id == member_id), None)
    if member is None:
        raise KeyError(f"Unknown member {member_id}")
    if not actor or not override_reason:
        raise LeakageError(
            f"member {member_id} belongs to a protected evaluation suite "
            "(training_eligible=false); promotion to training requires an "
            f"explicit recorded override (who + why) [{_LEAKAGE_SOURCE}]")
    member.training_eligible = True
    suite.governance_overrides.append({
        "member_id": member_id, "actor": actor, "reason": override_reason,
        "timestamp": now_iso(),
    })
    suite.updated_at = now_iso()
    store_mod.write_json(suite, "suites", f"{suite.suite_id}.json")
    store_mod.audit("contamination_guard_override",
                    member.source_failure_id, actor,
                    detail=f"member {member_id} of suite {suite_id} made "
                           f"training-eligible: {override_reason}")
    return {"suite_id": suite_id, "member_id": member_id,
            "training_eligible": True, "override_recorded": True}


# ------------------------------------------------------------------ regression hook


STANDING_SUITES = ["general", "historical-regression", "rare-event",
                   "safety-critical"]


def regression_evaluate(seed: int = data_mod.DEFAULT_SEED,
                        include_created_suites: bool = True) -> Dict:
    """Evaluate the candidate against general + historical-regression +
    construction-zone + rare-event + safety-critical suites.

    Per suite: baseline/candidate success rates, delta, anytime-valid CI and
    the three-outcome decision — all statistics DELEGATED to seqeval's
    PairedSequentialTest (never reimplemented)."""
    campaign = data_mod.get_campaign(seed)
    obs = campaign.observations

    def subset(name: str):
        if name == "general":
            return obs
        if name == "historical-regression":
            return [o for o in obs if o.gt_class in ("vehicle", "truck")]
        if name == "rare-event":
            return [o for o in obs if o.occluded or o.distance_m > 45.0]
        if name == "safety-critical":
            return [o for o in obs if o.gt_class in ("pedestrian", "cyclist",
                                                     "motorcycle")]
        return []

    suites: List[Dict] = [{"suite": n, "members": subset(n)} for n in STANDING_SUITES]
    if include_created_suites:
        by_frame = {(o.sequence_id, o.frame_id, o.object_instance_id): o for o in obs}
        for doc in list_suites():
            members = [by_frame.get((m["sequence_id"], m["frame_id"],
                                     m["object_instance_id"]))
                       for m in doc.get("members", [])]
            suites.append({"suite": doc["name"],
                           "suite_id": doc["suite_id"],
                           "members": [m for m in members if m is not None]})

    results = []
    for entry in suites:
        members = entry["members"]
        if not members:
            results.append({"suite": entry["suite"], "n": 0,
                            "decision": "INSUFFICIENT_EVIDENCE",
                            "note": "no members resolvable in the campaign"})
            continue
        b = [o.baseline.predicted_class == o.gt_class for o in members]
        c = [o.candidate.predicted_class == o.gt_class for o in members]
        d = [float(ci) - float(bi) for bi, ci in zip(b, c)]
        # cluster by sequence for honest correlation handling
        by_seq: Dict[str, List[float]] = defaultdict(list)
        for o, di in zip(members, d):
            by_seq[o.sequence_id].append(di)
        cluster_means = [sum(v) / len(v) for v in by_seq.values()]

        test = PairedSequentialTest(delta=0.01, alpha=0.05)
        test.update_clusters(cluster_means)
        test.record_objects(b, c)
        decision = test.evaluate()
        snap = test.snapshot()
        results.append({
            "suite": entry["suite"],
            "suite_id": entry.get("suite_id"),
            "n": len(members),
            "n_clusters": len(cluster_means),
            "baseline_rate": round(sum(b) / len(b), 5),
            "candidate_rate": round(sum(c) / len(c), 5),
            "delta": round(sum(c) / len(c) - sum(b) / len(b), 5),
            "delta_ci": snap["delta_ci"],
            "decision": decision,
            "e_regression": snap["e_regression"],
            "stats_delegated_to": "sensorflow.seqeval.sequential.PairedSequentialTest",
        })

    out = {"evaluated_at": now_iso(), "seed": seed, "suites": results,
           "note": ("INSUFFICIENT_EVIDENCE is reported as-is; it is never a "
                    "pass")}
    store_mod.write_json(out, "regression", "latest.json")
    store_mod.audit("regression_evaluated", None, "flywheel",
                    detail=f"{len(results)} suites evaluated")
    return out
