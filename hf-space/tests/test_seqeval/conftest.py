"""Shared fixtures: isolated megaeval + seqeval roots and one mid-size
population reused by every seqeval test module."""

from __future__ import annotations

import pytest

from sensorflow.megaeval import population as pop_mod
from sensorflow.seqeval import ledger as ledger_mod
from sensorflow.seqeval import paired as paired_mod
from sensorflow.seqeval.controller import reset_seqeval_store


@pytest.fixture(scope="session")
def seq_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("seqeval_env")
    pop_mod.set_mega_root(str(root / "mega"))
    ledger_mod.set_seqeval_root(str(root / "seq"))
    reset_seqeval_store()
    paired_mod.reset_prediction_cache()
    meta = pop_mod.generate_population("seq-pop", num_objects=90_000, seed=11)
    yield {"root": root, "meta": meta}
    pop_mod.set_mega_root("runs/megaeval")
    ledger_mod.set_seqeval_root("runs/seqeval")
    reset_seqeval_store()
    paired_mod.reset_prediction_cache()


# A compact policy for fast Monte-Carlo style tests.
FAST_POLICY = {
    "target_n": 6000,
    "safety_floor": 800,
    "min_per_stratum": 120,
    "escalation": {"enabled": True, "max_extra_per_stratum": 600,
                   "batch_objects": 300},
}
