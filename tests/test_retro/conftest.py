"""Shared fixtures for the retro (Agentic Retrospective Safety Analyzer) suite.

- retro_root: function-scoped isolated runs/retro replacement (RETRO_RUNS_DIR).
- fallback_index: forces the deterministic numpy/hashed-TF store so RAG tests
  never depend on chromadb's model download or ANN nondeterminism.
- registry: audited tool registry bound to the isolated root.
"""

from __future__ import annotations

import pytest

import sensorflow.retro.rag.retriever as retriever_mod
from sensorflow.retro.rag.retriever import SafetyCaseIndex
from sensorflow.retro.rag.store import NumpyCosineStore
from sensorflow.retro.tools.builtin import build_registry


@pytest.fixture()
def retro_root(tmp_path, monkeypatch):
    root = tmp_path / "retro"
    monkeypatch.setenv("RETRO_RUNS_DIR", str(root))
    return root


@pytest.fixture(scope="session")
def fallback_index():
    return SafetyCaseIndex(store=NumpyCosineStore())


@pytest.fixture(autouse=True)
def _pin_fallback_index(fallback_index, monkeypatch):
    """All retro tests retrieve through the deterministic fallback index."""
    monkeypatch.setattr(retriever_mod, "_INDEX", fallback_index)


@pytest.fixture()
def registry(retro_root):
    return build_registry(analysis_id="test-analysis")
