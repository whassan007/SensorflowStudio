"""Regression-controller state machine: sanity -> screening -> sequential
confirmation -> targeted escalation, with CI/CD-style gate outcomes.

Outcomes (three-outcome by construction, never "no news is good news"):
    REGRESSION            -> gate "block": evidence + affected strata attached.
    PASS                  -> gate "allow": overall AND every pre-registered
                             safety primary reached the equivalence-style claim
                             (drop confidently smaller than the margin).
    INSUFFICIENT_EVIDENCE -> gate "expand_or_report": budget exhausted before
                             either claim; reported with explicit
                             "not proven equivalent" language.

`evaluate_regression(...)` at the bottom is the clean entry point the safety
Regression Gate (sensorflow/safety/) can call; the REST surface lives in
api.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from typing import Dict, List, Optional

import numpy as np

from sensorflow.megaeval import population as pop_mod
from sensorflow.seqeval import attribution as attribution_mod
from sensorflow.seqeval import hierarchy as hierarchy_mod
from sensorflow.seqeval import ledger as ledger_mod
from sensorflow.seqeval import paired as paired_mod
from sensorflow.seqeval import planner as planner_mod
from sensorflow.seqeval.sequential import (DECISION_INSUFFICIENT, DECISION_PASS,
                                           DECISION_REGRESSION, approx_mde)

DEFAULT_POLICY: Dict = {
    "metric": "recall",
    "delta_margin": 0.005,          # practical-significance margin, absolute points
    "alpha": 0.05,                  # family alpha (regression direction)
    "alpha_pass": 0.05,             # per-required-node equivalence alpha
    "alpha_shares": dict(hierarchy_mod.DEFAULT_ALPHA_SHARES),
    "condition_dim": "lighting",
    "safety_primaries": ["pedestrian|night", "cyclist|night"],
    "target_n": 24_000,
    "min_per_stratum": 200,
    "safety_floor": 2500,
    "stage_fractions": [0.15, 0.40, 1.0],   # screening, confirmation, full plan
    "batches_per_stage": 3,
    "screening_e_threshold": 1.5,
    "stop_on_regression": True,
    "escalation": {"enabled": True, "max_extra_per_stratum": 4000,
                   "batch_objects": 500},
    "plan_seed": 12345,
}

GATE_FOR_DECISION = {
    DECISION_REGRESSION: "block",
    DECISION_PASS: "allow",
    DECISION_INSUFFICIENT: "expand_or_report",
}

INSUFFICIENT_LANGUAGE = ("Budget exhausted before either claim could be made: the "
                         "candidate is NOT PROVEN EQUIVALENT to the baseline within "
                         "the practical margin, and no regression beyond the margin "
                         "was confirmed. Do not treat this as a pass.")


def merged_policy(overrides: Optional[Dict] = None) -> Dict:
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(policy.get(k), dict):
            policy[k].update(v)
        else:
            policy[k] = v
    return policy


class SequentialRegressionRun:
    def __init__(self, population_id: str, baseline: paired_mod.SimulatedModel,
                 candidate: paired_mod.SimulatedModel, policy: Optional[Dict] = None,
                 seed: Optional[int] = None):
        self.run_id = f"seq-{uuid.uuid4().hex[:10]}"
        self.population_id = population_id
        self.baseline = baseline
        self.candidate = candidate
        self.policy = merged_policy(policy)
        if seed is None:
            seed = int(hashlib.sha256(
                f"{population_id}|{baseline.fingerprint()}|{candidate.fingerprint()}"
                .encode()).hexdigest()[:8], 16)
        self.seed = int(seed)
        self.status = "created"           # created|running|done|failed
        self.stage = "pending"            # sanity|screening|sequential|escalation|done
        self.decision = DECISION_INSUFFICIENT
        self.stopping_reason: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = ledger_mod.now_iso()
        self.finished_at: Optional[str] = None
        self.samples_used = 0
        self.escalation_used = 0
        self.sanity: Dict = {}
        self.plan: Optional[Dict] = None
        self.hier: Optional[hierarchy_mod.HierarchyController] = None
        self.attribution: Optional[Dict] = None
        self.ledger = ledger_mod.EvidenceLedger(self.run_id)
        self.message: Optional[str] = None

    # ---------------------------------------------------------------- state

    def gate(self) -> str:
        return GATE_FOR_DECISION[self.decision]

    def to_dict(self, include_trajectories: bool = True) -> Dict:
        nodes = []
        trajectories = {}
        if self.hier is not None:
            for key, node in self.hier.nodes.items():
                if node.test.n_objects == 0:
                    continue
                snap = node.test.snapshot()
                snap.update({"node": key, "level": node.level,
                             "safety_primary": node.safety_primary,
                             "suspect": node.suspect,
                             "alpha_allocated": round(self.hier.alpha_for(node), 6),
                             "e_threshold_regression": round(1.0 / self.hier.alpha_for(node), 2),
                             "e_threshold_pass": round(1.0 / self.hier.alpha_pass, 2)})
                nodes.append(snap)
                if include_trajectories:
                    trajectories[key] = {
                        "points": node.test.trajectory,
                        "boundaries": {
                            "log_e_regression": round(float(np.log(1.0 / self.hier.alpha_for(node))), 4),
                            "log_e_pass": round(float(np.log(1.0 / self.hier.alpha_pass)), 4),
                            "delta_margin": -self.policy["delta_margin"],
                        },
                    }
        plan_summary = None
        if self.plan is not None:
            plan_summary = {k: v for k, v in self.plan.items() if k != "_arrays"}
        out = {
            "run_id": self.run_id,
            "population_id": self.population_id,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "stage": self.stage,
            "decision": self.decision,
            "gate": self.gate(),
            "message": self.message,
            "stopping_reason": self.stopping_reason,
            "error": self.error,
            "seed": self.seed,
            "policy": self.policy,
            "sanity": self.sanity,
            "budget": {
                "planned_total": plan_summary["total_allocated"] if plan_summary else None,
                "samples_used": self.samples_used,
                "escalation_used": self.escalation_used,
                "full_population": plan_summary["population_objects"] if plan_summary else None,
                "fraction_of_population": (round(self.samples_used / plan_summary["population_objects"], 5)
                                           if plan_summary else None),
            },
            "plan": plan_summary,
            "nodes": sorted(nodes, key=lambda r: (r["level"], r["node"])),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }
        if include_trajectories:
            out["trajectories"] = trajectories
        return out


class SeqevalStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.runs: Dict[str, SequentialRegressionRun] = {}
        self._persisted: Dict[str, Dict] = {}
        self._load_persisted()

    def _load_persisted(self) -> None:
        base = os.path.join(ledger_mod.seq_root(), "runs")
        if not os.path.isdir(base):
            return
        for rid in sorted(os.listdir(base)):
            path = os.path.join(base, rid, "run.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    self._persisted[rid] = json.load(f)
            except Exception:
                continue

    def create_run(self, population_id: str, baseline: Dict, candidate: Dict,
                   policy: Optional[Dict] = None, seed: Optional[int] = None
                   ) -> SequentialRegressionRun:
        if pop_mod.load_meta(population_id) is None:
            raise KeyError(f"Unknown population {population_id}")
        run = SequentialRegressionRun(
            population_id,
            paired_mod.SimulatedModel(baseline.get("model_version", "baseline"),
                                      baseline.get("effects")),
            paired_mod.SimulatedModel(candidate.get("model_version", "candidate"),
                                      candidate.get("effects")),
            policy=policy, seed=seed)
        with self.lock:
            self.runs[run.run_id] = run
        return run

    def get_state(self, run_id: str, include_trajectories: bool = True) -> Optional[Dict]:
        with self.lock:
            run = self.runs.get(run_id)
        if run is not None:
            return run.to_dict(include_trajectories=include_trajectories)
        state = self._persisted.get(run_id)
        if state is not None and not include_trajectories:
            state = {k: v for k, v in state.items() if k != "trajectories"}
        return state

    def list_states(self) -> List[Dict]:
        seen = set()
        out = []
        with self.lock:
            for run in self.runs.values():
                seen.add(run.run_id)
                out.append(run.to_dict(include_trajectories=False))
        for rid, state in self._persisted.items():
            if rid not in seen:
                out.append({k: v for k, v in state.items() if k != "trajectories"})
        out.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return out

    def save_run(self, run: SequentialRegressionRun) -> None:
        d = ledger_mod.run_dir(run.run_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump(run.to_dict(include_trajectories=True), f)

    def start_async(self, run: SequentialRegressionRun) -> None:
        threading.Thread(target=self.execute_sync, args=(run,), daemon=True).start()

    # ---------------------------------------------------------------- engine

    def execute_sync(self, run: SequentialRegressionRun) -> None:
        try:
            run.status = "running"
            meta = pop_mod.load_meta(run.population_id)
            policy = run.policy

            # ---- stage: sanity + frozen plan ----------------------------
            run.stage = "sanity"
            plan = planner_mod.build_plan(
                meta, target_n=policy["target_n"], seed=policy["plan_seed"],
                condition_dim=policy["condition_dim"],
                safety_primaries=policy["safety_primaries"],
                min_per_stratum=policy["min_per_stratum"],
                safety_floor=policy["safety_floor"])
            run.plan = plan
            frame = paired_mod.load_frame(meta)

            all_ids = np.concatenate([plan["_arrays"][f"s{s}_ids"]
                                      for s in sorted(int(k) for k in plan["strata"])])
            smoke = all_ids[:min(500, all_ids.size)]
            outcomes = paired_mod.paired_outcomes(meta, run.baseline, run.candidate,
                                                  smoke, policy["condition_dim"])
            b_rate = float(outcomes["baseline"].mean())
            c_rate = float(outcomes["candidate"].mean())
            discordance = float(np.mean(outcomes["baseline"] != outcomes["candidate"]))
            run.sanity = {
                "plan_hash": plan["plan_hash"],
                "dataset_fingerprint": plan["dataset_fingerprint"],
                "baseline_fingerprint": run.baseline.fingerprint(),
                "candidate_fingerprint": run.candidate.fingerprint(),
                "smoke_n": int(smoke.size),
                "smoke_baseline_rate": round(b_rate, 4),
                "smoke_candidate_rate": round(c_rate, 4),
                "smoke_discordance": round(discordance, 4),
                "ok": bool(0.2 < b_rate <= 1.0 and 0.2 < c_rate <= 1.0
                           and discordance < 0.5),
            }
            if not run.sanity["ok"]:
                run.decision = DECISION_INSUFFICIENT
                run.stopping_reason = "sanity_failed"
                run.message = ("Sanity stage failed (degenerate outcome rates or "
                               "excessive discordance); evaluation aborted. "
                               + INSUFFICIENT_LANGUAGE)
                self._finalize(run)
                return

            # ---- build hierarchy + per-stratum cluster iterators ---------
            n_cond = len(pop_mod.DIMENSIONS[policy["condition_dim"]])
            hier = hierarchy_mod.HierarchyController(
                plan["strata"], n_cond=n_cond, delta=policy["delta_margin"],
                alpha=policy["alpha"], alpha_pass=policy["alpha_pass"],
                alpha_shares=policy["alpha_shares"],
                band_names=planner_mod.BAND_NAMES)
            run.hier = hier

            clusters = {int(s): planner_mod.stratum_clusters(plan, int(s))
                        for s in plan["strata"]}
            reserves = {int(s): planner_mod.stratum_clusters(plan, int(s), reserve=True)
                        for s in plan["strata"]}
            pointers = {s: 0 for s in clusters}
            taken_objects = {s: 0 for s in clusters}
            weights = {int(s): float(v["weight"]) for s, v in plan["strata"].items()}
            key_to_sid = {f"stratum:{v['key']}": int(s) for s, v in plan["strata"].items()}
            bands = np.digitize(frame["difficulty"],
                                np.asarray(planner_mod.DIFFICULTY_BANDS)).astype(np.int8)

            def make_batch(chunks: List) -> Optional[Dict[str, np.ndarray]]:
                """chunks: list of (stratum_id, [cluster object_id arrays])."""
                sids, ids = [], []
                for s, cl in chunks:
                    for arr in cl:
                        ids.append(arr)
                        sids.append(np.full(arr.size, s, dtype=np.int64))
                if not ids:
                    return None
                oid = np.concatenate(ids)
                sid = np.concatenate(sids)
                oc = paired_mod.paired_outcomes(meta, run.baseline, run.candidate,
                                                oid, policy["condition_dim"])
                return {
                    "object_id": oid,
                    "container_id": frame["container_id"][oid],
                    "class": frame["class"][oid].astype(np.int64),
                    "cond": frame[policy["condition_dim"]][oid].astype(np.int64),
                    "band": bands[oid].astype(np.int64),
                    "weight": np.asarray([weights[s] for s in sid], dtype=np.float64),
                    "d": oc["d"], "b": oc["baseline"], "c": oc["candidate"],
                }

            def process(chunks: List) -> None:
                batch = make_batch(chunks)
                if batch is None:
                    return
                hier.update_with_batch(batch)
                hier.evaluate(policy["screening_e_threshold"])
                hier.record_trajectories()
                run.samples_used += int(batch["d"].size)

            def take_until(s: int, target: int) -> List[np.ndarray]:
                out = []
                while pointers[s] < len(clusters[s]) and taken_objects[s] < target:
                    arr = clusters[s][pointers[s]]
                    out.append(arr)
                    taken_objects[s] += arr.size
                    pointers[s] += 1
                return out

            stopped = False

            # ---- stages: screening then sequential confirmation ----------
            for stage_idx, frac in enumerate(policy["stage_fractions"]):
                run.stage = "screening" if stage_idx == 0 else "sequential"
                for sub in range(policy["batches_per_stage"]):
                    prev_frac = (policy["stage_fractions"][stage_idx - 1]
                                 if stage_idx > 0 else 0.0)
                    eff_frac = prev_frac + (frac - prev_frac) * (sub + 1) / policy["batches_per_stage"]
                    chunks = []
                    for s in clusters:
                        target = int(np.ceil(eff_frac * plan["strata"][str(s)]["allocated"]))
                        cl = take_until(s, target)
                        if cl:
                            chunks.append((s, cl))
                    process(chunks)
                    if policy["stop_on_regression"] and hier.any_regression():
                        run.stopping_reason = "regression_confirmed"
                        stopped = True
                        break
                    if hier.required_pass() and not hier.undecided_keys():
                        run.stopping_reason = "all_passed"
                        stopped = True
                        break
                if stopped:
                    break

            # ---- targeted escalation -------------------------------------
            esc = policy["escalation"]
            if not stopped and esc.get("enabled", True):
                run.stage = "escalation"
                for key in list(hier.undecided_keys()):
                    node = hier.nodes.get(key)
                    if node is None or node.level != 3:
                        continue
                    if not (node.suspect or node.safety_primary):
                        continue
                    s = key_to_sid.get(key)
                    if s is None:
                        continue
                    pool = reserves.get(s, [])
                    ptr, extra = 0, 0
                    while (ptr < len(pool) and extra < esc["max_extra_per_stratum"]
                           and node.test.decision == DECISION_INSUFFICIENT):
                        chunk, size = [], 0
                        while ptr < len(pool) and size < esc["batch_objects"]:
                            chunk.append(pool[ptr])
                            size += pool[ptr].size
                            ptr += 1
                        if not chunk:
                            break
                        process([(s, chunk)])
                        extra += size
                        run.escalation_used += size
                        if policy["stop_on_regression"] and hier.any_regression():
                            run.stopping_reason = "regression_confirmed"
                            stopped = True
                            break
                    if stopped:
                        break

            # ---- finalize -------------------------------------------------
            if run.stopping_reason is None:
                run.stopping_reason = ("escalation_exhausted"
                                       if esc.get("enabled", True) else "budget_exhausted")
            if hier.any_regression():
                run.decision = DECISION_REGRESSION
                run.message = "Anytime-valid regression confirmed; see attribution."
            elif hier.required_pass():
                run.decision = DECISION_PASS
                run.message = ("Overall and all pre-registered safety primaries proved "
                               "equivalent within the practical margin.")
            else:
                run.decision = DECISION_INSUFFICIENT
                run.message = INSUFFICIENT_LANGUAGE
            run.attribution = attribution_mod.build_attribution(hier)
            self._write_ledger(run)
            self._finalize(run)
        except Exception as e:  # pragma: no cover - defensive
            run.status = "failed"
            run.error = str(e)
            run.finished_at = ledger_mod.now_iso()
            try:
                self.save_run(run)
            except Exception:
                pass

    # ---------------------------------------------------------------- output

    def _write_ledger(self, run: SequentialRegressionRun) -> None:
        hier = run.hier
        for node in hier.nodes.values():
            t = node.test
            if t.n_objects == 0:
                continue
            clustering = node.clustering()
            nu = (t.n01 + t.n10) / t.n_objects if t.n_objects else 0.0
            est = t.delta_estimate() or 0.0
            var_d = max(nu - est ** 2, 1e-6)
            base_rate = t.sum_baseline / t.n_objects
            cand_rate = t.sum_candidate / t.n_objects
            lo, hi = t.delta_interval()
            alpha_alloc = hier.alpha_for(node)
            run.ledger.append({
                "run_id": run.run_id,
                "baseline_model": run.baseline.to_dict(),
                "candidate_model": run.candidate.to_dict(),
                "dataset_version": run.population_id,
                "seed": run.seed,
                "stratum": node.key,
                "level": node.level,
                "metric": run.policy["metric"],
                "baseline_value": round(base_rate, 6),
                "candidate_value": round(cand_rate, 6),
                "abs_delta": round(cand_rate - base_rate, 6),
                "rel_delta": (round((cand_rate - base_rate) / base_rate, 6)
                              if base_rate else None),
                "delta_ci_low": round(lo, 6),
                "delta_ci_high": round(hi, 6),
                "mde_abs": approx_mde(var_d, clustering["n_effective"],
                                      run.policy["delta_margin"], alpha_alloc),
                "n": t.n_objects,
                "n_effective": clustering["n_effective"],
                "n_clusters": clustering["n_clusters"],
                "design_effect": clustering["design_effect"],
                "confidence_level": 1.0 - run.policy["alpha"],
                "e_value": round(t.e_reg.e_value, 6),
                "p_analogue": round(t.e_reg.p_analogue(), 8),
                "bayes_p_regression": t.bayes_p_regression(),
                "decision": t.decision,
                "test_method": ledger_mod.TEST_METHOD,
                "multiple_testing_method": hierarchy_mod.MULTIPLE_TESTING_METHOD,
                "practical_margin": run.policy["delta_margin"],
                "alpha_allocated": round(alpha_alloc, 6),
                "stopping_reason": run.stopping_reason,
                "timestamp": ledger_mod.now_iso(),
            })
        run.ledger.write_lineage({
            "run_id": run.run_id,
            "dataset_version": run.population_id,
            "dataset_fingerprint": run.sanity.get("dataset_fingerprint"),
            "plan_id": run.plan["plan_id"],
            "plan_hash": run.plan["plan_hash"],
            "seed": run.seed,
            "statistical_config": run.policy,
            "baseline_model": run.baseline.to_dict(),
            "candidate_model": run.candidate.to_dict(),
        })

    def _finalize(self, run: SequentialRegressionRun) -> None:
        run.status = "done"
        run.stage = "done"
        run.finished_at = ledger_mod.now_iso()
        self.save_run(run)
        # best-effort integration with platform accounting + audit trail
        try:
            from sensorflow.evaluation.process_units import ProcessMeter
            from sensorflow.evaluation.records import get_store as get_legacy_store
            legacy = get_legacy_store()
            ProcessMeter(legacy, run.run_id).record("regression_tracking",
                                                    run.samples_used)
            legacy.audit("seqeval_decision", "seqeval_run", run.run_id,
                         detail=f"{run.decision} ({run.stopping_reason}); "
                                f"samples={run.samples_used}")
            legacy.save()
        except Exception:
            pass


_STORE: Optional[SeqevalStore] = None
_STORE_LOCK = threading.Lock()


def get_seqeval_store() -> SeqevalStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = SeqevalStore()
        return _STORE


def reset_seqeval_store() -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = None


def evaluate_regression(population_id: str, baseline: Dict, candidate: Dict,
                        policy: Optional[Dict] = None, seed: Optional[int] = None
                        ) -> Dict:
    """Clean entry point for the safety Regression Gate.

    baseline/candidate: {"model_version": str, "effects": {stratum: pp_delta}}.
    Returns a gate verdict with the evidence needed to block or allow:
    {gate, decision, stopping_reason, samples_used, affected_strata,
     regression_map, run_id, message}.
    """
    store = get_seqeval_store()
    run = store.create_run(population_id, baseline, candidate, policy=policy, seed=seed)
    store.execute_sync(run)
    att = run.attribution or {"regression_map": [], "affected_strata": []}
    return {
        "run_id": run.run_id,
        "gate": run.gate(),
        "decision": run.decision,
        "message": run.message,
        "stopping_reason": run.stopping_reason,
        "samples_used": run.samples_used,
        "planned_total": run.plan["total_allocated"] if run.plan else None,
        "full_population": run.plan["population_objects"] if run.plan else None,
        "affected_strata": att["affected_strata"],
        "regression_map": att["regression_map"],
    }
