"""Control-plane registry: versioned entities for the whole evaluation system.

Entities (all persisted as JSON under runs/studio2/registry/<kind>/):

    models        ModelVersion       name/version/checkpoint + provenance
    datasets      DatasetVersion     role (TRAINING/VALIDATION/TEST/REGRESSION/
                                     LAUNCH/MONITORING), immutable lineage,
                                     role-transition rules with the raremine
                                     contamination-guard pattern
    scenarios     ScenarioVersion    generator + seed + recipe hash
    policies      PolicyVersion      content-hashed policy documents
    experiments   Experiment         candidate-vs-baseline grouping
    runs          EvaluationRun      full reproducibility tuple; runs missing
                                     any component are NON_REPRODUCIBLE
    safety_cases  SafetyCase         pointers to safety evidence packages /
                                     agentic scorecards
    decisions     ReleaseDecision    written by release_gate.py
    approvals     HumanApproval      the separate recorded human action

Auto-ingest adapters scan the existing runs/ stores (megaeval, seqeval,
safety evidence, agentic scorecards, bevfusion) and register entities
retroactively with provenance where derivable. Ingest is idempotent: entity
ids are content-derived from the source references.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, List, Optional

from sensorflow.studio2 import store

# ------------------------------------------------------------------ constants

KINDS = ("models", "datasets", "scenarios", "policies", "experiments",
         "runs", "safety_cases", "decisions", "approvals")

DATASET_ROLES = ("TRAINING", "VALIDATION", "TEST", "REGRESSION",
                 "LAUNCH", "MONITORING")

# Same semantics as raremine.models.PROTECTED_EVAL_DESTINATIONS: examples in
# these roles decide launches, so silently moving them across the
# training/evaluation boundary would corrupt every future measurement.
PROTECTED_EVAL_ROLES = frozenset({"TEST", "REGRESSION", "LAUNCH"})

# The reproducibility tuple. A run missing any component cannot be replayed
# bit-for-bit and is marked NON_REPRODUCIBLE.
REPRO_COMPONENTS = ("model_version_id", "dataset_version_id",
                    "scenario_version_id", "config_hash",
                    "calibration_version", "seed", "policy_version_id")

REPRODUCIBLE = "REPRODUCIBLE"
NON_REPRODUCIBLE = "NON_REPRODUCIBLE"


class RoleTransitionError(RuntimeError):
    """Raised on any attempt to move a dataset across the training/evaluation
    contamination boundary without an explicit governance override."""


def new_id(prefix: str, *seed_parts: str) -> str:
    """Deterministic when seed_parts are given (used by ingest for
    idempotency), random-ish otherwise."""
    if seed_parts:
        h = hashlib.sha256("|".join(seed_parts).encode()).hexdigest()[:10]
    else:
        h = hashlib.sha256(os.urandom(16)).hexdigest()[:10]
    return f"{prefix}-{h}"


def content_hash(doc: Dict, exclude: tuple = ("policy_version", "created_at")) -> str:
    blob = json.dumps({k: v for k, v in doc.items() if k not in exclude},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ registry


class Registry:
    """Thin persisted entity store with the governance rules baked in."""

    def __init__(self):
        self._lock = threading.RLock()

    # -------------------------------------------------------------- storage

    def _path_parts(self, kind: str, entity_id: str):
        return ("registry", kind, f"{entity_id}.json")

    def put(self, kind: str, entity: Dict) -> Dict:
        if kind not in KINDS:
            raise KeyError(f"unknown entity kind {kind!r}; known: {KINDS}")
        entity = dict(entity)
        entity.setdefault("kind", kind)
        entity.setdefault("created_at", store.now_iso())
        store.write_json(entity, *self._path_parts(kind, entity["entity_id"]))
        return entity

    def get(self, kind: str, entity_id: str) -> Optional[Dict]:
        return store.read_json(*self._path_parts(kind, entity_id))

    def list(self, kind: str) -> List[Dict]:
        if kind not in KINDS:
            raise KeyError(f"unknown entity kind {kind!r}; known: {KINDS}")
        out = []
        for name in store.list_dir("registry", kind):
            if not name.endswith(".json"):
                continue
            doc = store.read_json("registry", kind, name)
            if doc:
                out.append(doc)
        out.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        return out

    def counts(self) -> Dict[str, int]:
        return {kind: len(store.list_dir("registry", kind)) for kind in KINDS}

    # -------------------------------------------------------------- models

    def register_model(self, name: str, version: str, checkpoint: str = "",
                       provenance: Optional[Dict] = None,
                       meta: Optional[Dict] = None) -> Dict:
        entity_id = new_id("mv", name, version, checkpoint)
        existing = self.get("models", entity_id)
        if existing:
            return existing
        return self.put("models", {
            "entity_id": entity_id, "name": name, "version": version,
            "checkpoint": checkpoint, "provenance": provenance or {},
            "meta": meta or {}})

    # -------------------------------------------------------------- datasets

    def register_dataset(self, name: str, role: str,
                         lineage_parents: Optional[List[str]] = None,
                         provenance: Optional[Dict] = None,
                         meta: Optional[Dict] = None,
                         actor: str = "studio2") -> Dict:
        if role not in DATASET_ROLES:
            raise ValueError(f"unknown dataset role {role!r}; known: {DATASET_ROLES}")
        entity_id = new_id("dv", name, role_seed := role)
        existing = self.get("datasets", entity_id)
        if existing:
            return existing
        entity = self.put("datasets", {
            "entity_id": entity_id, "name": name, "role": role,
            "protected_evaluation": role in PROTECTED_EVAL_ROLES,
            # lineage is immutable: parents are recorded once, at creation
            "lineage": {"parents": list(lineage_parents or []),
                        "recorded_at": store.now_iso()},
            "role_history": [{"from": None, "to": role, "actor": actor,
                              "override": None, "timestamp": store.now_iso()}],
            "governance_overrides": [],
            "provenance": provenance or {}, "meta": meta or {}})
        store.audit("dataset_registered", "datasets", entity_id, actor=actor,
                    detail=f"{name} role={role}")
        return entity

    def transition_role(self, dataset_id: str, new_role: str, actor: str,
                        override_reason: Optional[str] = None) -> Dict:
        """The ONLY path that changes a dataset's role.

        Contamination guard (raremine.lineage pattern):
          - protected eval role -> TRAINING requires an explicit override
            (who + why), otherwise RoleTransitionError;
          - TRAINING -> protected eval role is guarded the same way (training
            examples silently becoming launch evidence is the same corruption
            in the other direction).
        Every transition and every override is audited.
        """
        if new_role not in DATASET_ROLES:
            raise ValueError(f"unknown dataset role {new_role!r}")
        with self._lock:
            entity = self.get("datasets", dataset_id)
            if entity is None:
                raise KeyError(f"unknown dataset {dataset_id}")
            old_role = entity["role"]
            if old_role == new_role:
                return entity

            crossing = ((old_role in PROTECTED_EVAL_ROLES and new_role == "TRAINING")
                        or (old_role == "TRAINING" and new_role in PROTECTED_EVAL_ROLES))
            override = None
            if crossing:
                if not (override_reason and override_reason.strip() and actor.strip()):
                    raise RoleTransitionError(
                        f"dataset {dataset_id} ({entity['name']}): {old_role} -> "
                        f"{new_role} crosses the training/evaluation contamination "
                        "boundary; it requires an explicit governance override "
                        "recording who and why")
                override = {"actor": actor, "reason": override_reason,
                            "timestamp": store.now_iso(),
                            "transition": f"{old_role}->{new_role}"}
                entity["governance_overrides"].append(override)
                store.audit("governance_override", "datasets", dataset_id,
                            actor=actor, detail=override_reason)

            entity["role"] = new_role
            entity["protected_evaluation"] = new_role in PROTECTED_EVAL_ROLES
            entity["role_history"].append(
                {"from": old_role, "to": new_role, "actor": actor,
                 "override": override, "timestamp": store.now_iso()})
            self.put("datasets", entity)
            store.audit("role_transition", "datasets", dataset_id, actor=actor,
                        detail=f"{old_role} -> {new_role}")
            return entity

    # -------------------------------------------------------------- scenarios

    def register_scenario(self, name: str, generator: str, seed: Optional[int],
                          recipe: Optional[Dict] = None,
                          provenance: Optional[Dict] = None) -> Dict:
        recipe_hash = content_hash(recipe or {}, exclude=())
        entity_id = new_id("sv", name, generator, str(seed), recipe_hash)
        existing = self.get("scenarios", entity_id)
        if existing:
            return existing
        return self.put("scenarios", {
            "entity_id": entity_id, "name": name, "generator": generator,
            "seed": seed, "recipe": recipe or {}, "recipe_hash": recipe_hash,
            "provenance": provenance or {}})

    # -------------------------------------------------------------- policies

    def register_policy(self, name: str, doc: Dict,
                        provenance: Optional[Dict] = None) -> Dict:
        version = content_hash(doc)
        entity_id = f"pol-{version}"
        existing = self.get("policies", entity_id)
        if existing:
            return existing
        return self.put("policies", {
            "entity_id": entity_id, "name": name, "policy_version": version,
            "doc": doc, "provenance": provenance or {}})

    # -------------------------------------------------------------- experiments

    def register_experiment(self, name: str, candidate_model_id: str,
                            baseline_model_id: str,
                            meta: Optional[Dict] = None) -> Dict:
        entity_id = new_id("exp", name, candidate_model_id, baseline_model_id)
        existing = self.get("experiments", entity_id)
        if existing:
            return existing
        return self.put("experiments", {
            "entity_id": entity_id, "name": name,
            "candidate_model_id": candidate_model_id,
            "baseline_model_id": baseline_model_id, "meta": meta or {}})

    # -------------------------------------------------------------- runs

    def register_run(self, name: str, engine: str,
                     tuple_components: Dict, results: Optional[Dict] = None,
                     experiment_id: Optional[str] = None,
                     provenance: Optional[Dict] = None) -> Dict:
        """tuple_components: values for REPRO_COMPONENTS (None/'' = missing).
        The reproducibility verdict is computed here and cannot be supplied."""
        missing = [c for c in REPRO_COMPONENTS
                   if tuple_components.get(c) in (None, "")]
        entity_id = new_id("run", name, engine,
                           content_hash(tuple_components, exclude=()))
        existing = self.get("runs", entity_id)
        if existing:
            return existing
        return self.put("runs", {
            "entity_id": entity_id, "name": name, "engine": engine,
            "experiment_id": experiment_id,
            "reproducibility_tuple": {c: tuple_components.get(c)
                                      for c in REPRO_COMPONENTS},
            "missing_components": missing,
            "reproducibility": NON_REPRODUCIBLE if missing else REPRODUCIBLE,
            "results": results or {}, "provenance": provenance or {}})

    # -------------------------------------------------------------- safety cases

    def register_safety_case(self, name: str, case_kind: str, reference: Dict,
                             provenance: Optional[Dict] = None) -> Dict:
        entity_id = new_id("sc", name, case_kind,
                           content_hash(reference, exclude=()))
        existing = self.get("safety_cases", entity_id)
        if existing:
            return existing
        return self.put("safety_cases", {
            "entity_id": entity_id, "name": name, "case_kind": case_kind,
            "reference": reference, "provenance": provenance or {}})


# ------------------------------------------------------------------ ingest


def _safe_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def ingest_existing_stores(registry: Registry, repo_root: str = ".") -> Dict:
    """Scan the landed packages' stores and register entities retroactively.

    Best-effort and idempotent. Nothing is fabricated: components that cannot
    be derived from the source records stay missing, so most ingested runs are
    honestly NON_REPRODUCIBLE (megaeval lineage carries the most complete
    tuple; it still lacks scenario + calibration components).
    """
    counts = {"models": 0, "datasets": 0, "runs": 0, "safety_cases": 0,
              "scenarios": 0, "policies": 0}
    sources: Dict[str, str] = {}

    # ---- megaeval runs (runs/megaeval/runs/*/run.json -> {"state": {...}})
    mega_dir = os.path.join(repo_root, "runs", "megaeval", "runs")
    if os.path.isdir(mega_dir):
        sources["megaeval"] = mega_dir
        for rid in sorted(os.listdir(mega_dir)):
            state = (_safe_json(os.path.join(mega_dir, rid, "run.json")) or {}).get("state")
            if not state or state.get("status") != "published":
                continue
            lin = state.get("lineage") or {}
            model = registry.register_model(
                name=state.get("model_version") or "unknown",
                version=state.get("model_version") or "unknown",
                checkpoint=lin.get("model_checkpoint") or "",
                provenance={"source_package": "megaeval", "run_id": rid})
            counts["models"] += 1
            dataset = registry.register_dataset(
                name=lin.get("dataset_version") or state.get("population_id") or rid,
                role="TEST",
                provenance={"source_package": "megaeval",
                            "population_id": state.get("population_id")},
                meta={"label_version": lin.get("label_version")})
            counts["datasets"] += 1
            registry.register_run(
                name=rid, engine="megaeval",
                tuple_components={
                    "model_version_id": model["entity_id"],
                    "dataset_version_id": dataset["entity_id"],
                    "scenario_version_id": None,  # megaeval has no scenario axis
                    "config_hash": content_hash(
                        {"threshold": lin.get("threshold_config"),
                         "sampling": lin.get("sampling_config"),
                         "metric_version": lin.get("metric_version"),
                         "evaluator": lin.get("evaluator_code_version")},
                        exclude=()),
                    "calibration_version": None,  # not recorded in megaeval lineage
                    "seed": state.get("seed"),
                    "policy_version_id": None,
                },
                results={"headline": state.get("headline")},
                provenance={"source_package": "megaeval", "run_id": rid,
                            "lineage": lin})
            counts["runs"] += 1

    # ---- seqeval runs (runs/seqeval/runs/*/run.json)
    seq_dir = os.path.join(repo_root, "runs", "seqeval", "runs")
    if os.path.isdir(seq_dir):
        sources["seqeval"] = seq_dir
        for rid in sorted(os.listdir(seq_dir)):
            st = _safe_json(os.path.join(seq_dir, rid, "run.json"))
            if not st:
                continue
            policy = registry.register_policy(
                name="seqeval-policy", doc=st.get("policy") or {},
                provenance={"source_package": "seqeval", "run_id": rid})
            counts["policies"] += 1
            cand = (st.get("candidate") or {}).get("model_version") or "candidate"
            base = (st.get("baseline") or {}).get("model_version") or "baseline"
            model = registry.register_model(
                name=cand, version=cand,
                provenance={"source_package": "seqeval", "run_id": rid})
            counts["models"] += 1
            registry.register_run(
                name=rid, engine="seqeval",
                tuple_components={
                    "model_version_id": model["entity_id"],
                    "dataset_version_id": st.get("population_id"),
                    "scenario_version_id": None,
                    "config_hash": content_hash(st.get("policy") or {}, exclude=()),
                    "calibration_version": None,
                    "seed": st.get("seed"),
                    "policy_version_id": policy["entity_id"],
                },
                results={"decision": st.get("decision"), "gate": st.get("gate"),
                         "stopping_reason": st.get("stopping_reason"),
                         "baseline": base, "candidate": cand},
                provenance={"source_package": "seqeval", "run_id": rid})
            counts["runs"] += 1

    # ---- safety evidence packages (runs/safety/evidence/*.json)
    ev_dir = os.path.join(repo_root, "runs", "safety", "evidence")
    if os.path.isdir(ev_dir):
        sources["safety"] = ev_dir
        for name in sorted(os.listdir(ev_dir)):
            if not name.endswith(".json"):
                continue
            pkg = _safe_json(os.path.join(ev_dir, name))
            if not pkg:
                continue
            registry.register_safety_case(
                name=pkg.get("package_id") or name,
                case_kind="safety_evidence_package",
                reference={"package_id": pkg.get("package_id"),
                           "candidate_run_id": (pkg.get("candidate") or {}).get("run_id"),
                           "baseline_run_id": (pkg.get("baseline") or {}).get("run_id"),
                           "release_ready": (pkg.get("decision") or {}).get("release_ready"),
                           "path": os.path.join("runs", "safety", "evidence", name)},
                provenance={"source_package": "safety"})
            counts["safety_cases"] += 1

    # ---- agentic scorecards (runs/agentic/scorecards/*.json) — in-flight pkg,
    # but ingest reads plain JSON so it works regardless of importability
    sc_dir = os.path.join(repo_root, "runs", "agentic", "scorecards")
    if os.path.isdir(sc_dir):
        sources["agentic"] = sc_dir
        for name in sorted(os.listdir(sc_dir)):
            if not name.endswith(".json"):
                continue
            card = _safe_json(os.path.join(sc_dir, name))
            if not card:
                continue
            registry.register_safety_case(
                name=card.get("scorecard_id") or name,
                case_kind="agentic_scorecard",
                reference={"scorecard_id": card.get("scorecard_id"),
                           "failure_id": card.get("failure_id"),
                           "policy_version": card.get("policy_version"),
                           "recommended_option": card.get("recommended_option"),
                           "path": os.path.join("runs", "agentic", "scorecards", name)},
                provenance={"source_package": "agentic"})
            counts["safety_cases"] += 1

    # ---- bevfusion comparison runs (runs/bevfusion/bevrun-*.json)
    bev_dir = os.path.join(repo_root, "runs", "bevfusion")
    if os.path.isdir(bev_dir):
        sources["bevfusion"] = bev_dir
        for name in sorted(os.listdir(bev_dir)):
            if not (name.startswith("bevrun-") and name.endswith(".json")):
                continue
            rep = _safe_json(os.path.join(bev_dir, name))
            if not rep:
                continue
            params = rep.get("params") or {}
            scenario = registry.register_scenario(
                name=f"bevfusion-suite-{params.get('seed')}",
                generator="bevfusion.scenes.generate_sequences",
                seed=params.get("seed"), recipe=params,
                provenance={"source_package": "bevfusion", "run_id": rep.get("run_id")})
            counts["scenarios"] += 1
            engines = rep.get("engines") or {}
            model = registry.register_model(
                name=engines.get("candidate") or "bev-fused",
                version=engines.get("candidate") or "bev-fused",
                provenance={"source_package": "bevfusion", "run_id": rep.get("run_id")})
            counts["models"] += 1
            registry.register_run(
                name=rep.get("run_id") or name, engine="bevfusion",
                tuple_components={
                    "model_version_id": model["entity_id"],
                    "dataset_version_id": None,  # scenes are generated, not versioned data
                    "scenario_version_id": scenario["entity_id"],
                    "config_hash": content_hash(params, exclude=()),
                    "calibration_version": None,
                    "seed": params.get("seed"),
                    "policy_version_id": None,
                },
                results={"recommendation": rep.get("recommendation"),
                         "headline_deltas": rep.get("headline_deltas")},
                provenance={"source_package": "bevfusion", "run_id": rep.get("run_id")})
            counts["runs"] += 1

    store.audit("ingest", None, None, detail=f"scanned {sorted(sources)}",
                payload={"counts": counts})
    return {"sources": sources, "registered": counts,
            "totals": registry.counts()}


# ------------------------------------------------------------------ singleton

_REGISTRY: Optional[Registry] = None
_REG_LOCK = threading.RLock()


def get_registry() -> Registry:
    global _REGISTRY
    with _REG_LOCK:
        if _REGISTRY is None:
            _REGISTRY = Registry()
        return _REGISTRY
