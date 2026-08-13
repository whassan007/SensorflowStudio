"""Machine-readable evidence ledger + seqeval storage root.

Every stratum/node decision is persisted as one append-only JSONL record with
every field an auditor needs to reproduce the decision: models, dataset
version, seed, stratum, metric, baseline/candidate values, absolute+relative
deltas, MDE at the achieved n, raw and effective sample sizes, confidence
level, e-value and its p-analogue, the decision, the test and multiple-testing
methods, the stopping reason and a timestamp. A lineage record (evaluator
version + full statistical config) accompanies each run.

Layout:
    runs/seqeval/
        plans/{plan_id}.json|.npz     frozen sampling plans (planner.py)
        cache/{dataset_fp}-{model_fp}.npz   prediction cache (paired.py)
        runs/{run_id}/run.json        controller state
        runs/{run_id}/ledger.jsonl    evidence records
        runs/{run_id}/lineage.json    evaluator + statistical config lineage
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

SEQEVAL_VERSION = "seqeval-1.0.0"

_SEQ_ROOT = os.path.join("runs", "seqeval")
_ROOT_LOCK = threading.Lock()


def seq_root() -> str:
    return _SEQ_ROOT


def set_seqeval_root(path: str) -> None:
    """Test hook: relocate all seqeval storage."""
    global _SEQ_ROOT
    with _ROOT_LOCK:
        _SEQ_ROOT = str(path)


def run_dir(run_id: str) -> str:
    return os.path.join(seq_root(), "runs", run_id)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


REQUIRED_FIELDS = [
    "baseline_model", "candidate_model", "dataset_version", "seed",
    "stratum", "level", "metric",
    "baseline_value", "candidate_value", "abs_delta", "rel_delta",
    "delta_ci_low", "delta_ci_high", "mde_abs",
    "n", "n_effective", "n_clusters", "design_effect",
    "confidence_level", "e_value", "p_analogue", "bayes_p_regression",
    "decision", "test_method", "multiple_testing_method",
    "practical_margin", "alpha_allocated",
    "stopping_reason", "timestamp",
]

TEST_METHOD = ("paired cluster-mean empirical-Bernstein confidence sequence + "
               "one-sided betting e-process (anytime-valid)")


class EvidenceLedger:
    """Append-only JSONL evidence store for one controller run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = run_dir(run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "ledger.jsonl")
        self._lock = threading.Lock()

    def append(self, record: Dict) -> Dict:
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            raise ValueError(f"Evidence record missing fields: {missing}")
        record = {**record, "record_type": record.get("record_type", "node_decision")}
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def records(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def write_lineage(self, lineage: Dict) -> Dict:
        lineage = {"evaluator_version": SEQEVAL_VERSION,
                   "timestamp": now_iso(), **lineage}
        with open(os.path.join(self.dir, "lineage.json"), "w") as f:
            json.dump(lineage, f, indent=1)
        return lineage

    def lineage(self) -> Optional[Dict]:
        path = os.path.join(self.dir, "lineage.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
