"""Optional-dependency integrations: chromadb store and the MCP server bonus.

These skip honestly when the optional packages are absent — the fallback
paths are covered by the rest of the suite either way.
"""

from __future__ import annotations

import pytest

from sensorflow.retro.rag.embedder import chromadb_available


@pytest.mark.skipif(not chromadb_available(), reason="chromadb not installed")
def test_chroma_store_same_interface():
    from sensorflow.retro.rag.store import ChromaStore
    store = ChromaStore(collection_name="retro_test_iface")
    store.add(["c1", "c2"],
              ["pedestrian crossing in heavy rain at night",
               "plastic bag drifting across a highway lane"],
              [{"doc_id": "D1", "synthetic": "true"},
               {"doc_id": "D2", "synthetic": "true"}])
    hits = store.query("missed pedestrian in rain", k=1)
    assert hits and hits[0].chunk_id == "c1"
    assert 0.0 <= hits[0].relevance_score <= 1.0
    assert hits[0].metadata["doc_id"] == "D1"


@pytest.mark.skipif(not chromadb_available(), reason="chromadb not installed")
def test_full_index_on_chroma_backend():
    from sensorflow.retro.rag.retriever import SafetyCaseIndex
    from sensorflow.retro.rag.store import ChromaStore
    index = SafetyCaseIndex(store=ChromaStore(collection_name="retro_test_full"))
    hits = index.search("phantom braking caused by plastic bag", k=3)
    assert hits and any(h.doc_id in ("SFS-SAFE-001", "SFS-RETRO-HIST-01")
                        for h in hits)
    assert all(h.label for h in hits)  # synthetic labels survive chroma round-trip


def test_mcp_server_bonus_builds_if_installed(retro_root):
    try:
        import mcp  # noqa: F401
    except ImportError:
        pytest.skip("mcp not installed; registry remains the MCP-style boundary")
    from sensorflow.retro.tools.mcp_server import build_mcp_server
    server, registry = build_mcp_server()
    assert {s.name for s in registry.specs()} >= {
        "log_reader", "safety_standard_rag", "metric_calculator",
        "historical_failure_search", "distribution_analysis",
        "create_evaluation_case"}
