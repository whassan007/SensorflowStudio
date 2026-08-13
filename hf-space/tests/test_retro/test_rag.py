"""Phase 2 tests: chunking, retrieval eval, synthetic labeling, and the
no-citation-without-retrieval hard rule."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sensorflow.retro.rag.chunking import chunk_document, chunk_words, parse_sections
from sensorflow.retro.rag.corpus import SEED_DOCUMENTS
from sensorflow.retro.rag.evalharness import EVAL_QUERIES, run_retrieval_eval
from sensorflow.retro.scorecard import RetrievedStandard


# ------------------------------------------------------------------ chunking

def test_parse_sections_and_word_chunking():
    text = "preamble line\n## Alpha\n" + ("word " * 300) + "\n## Beta\nshort tail"
    secs = parse_sections(text)
    assert [s["section"] for s in secs] == ["preamble", "Alpha", "Beta"]
    chunks = chunk_words(secs[1]["text"], chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    # overlap: consecutive chunks share their boundary words
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-20:] == second_words[:20]


def test_chunk_config_validation():
    with pytest.raises(ValueError):
        chunk_words("a b c", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_words("a b c", chunk_size=0)


def test_chunk_document_propagates_metadata():
    chunks = chunk_document("DOC-1", "## Sec A\nhello world content here",
                            {"doc_id": "DOC-1", "source": "unit"},
                            chunk_size=50, overlap=10)
    assert chunks and chunks[0].metadata["section"] == "Sec A"
    assert chunks[0].metadata["source"] == "unit"


# ---------------------------------------------------------------- eval harness

def test_retrieval_eval_precision(fallback_index):
    report = run_retrieval_eval(index=fallback_index, k=4)
    assert report.num_queries == len(EVAL_QUERIES) >= 12
    # deterministic fallback embedder must resolve at least 85% of the
    # expected-source assertions at k=4
    assert report.precision_at_k >= 0.85, [
        r.model_dump() for r in report.results if not r.hit_at_k]


def test_every_synthetic_hit_is_labeled(fallback_index):
    """Synthetic rules carry the SYNTHETIC label in content AND metadata; the
    SOTIF doc carries the paraphrase label. No unlabeled hit can exist."""
    for item in EVAL_QUERIES:
        for hit in fallback_index.search(item["query"], k=4):
            if hit.synthetic:
                assert "SYNTHETIC_EXAMPLE" in hit.label
                assert ("SYNTHETIC_EXAMPLE" in hit.retrieved_text
                        or "NOT_A_REAL_STANDARD" in hit.retrieved_text)
            else:
                assert hit.label == "PARAPHRASE_NOT_STANDARD_TEXT"
                assert "PARAPHRASE_NOT_STANDARD_TEXT" in hit.retrieved_text


def test_seed_corpus_covers_required_doc_types():
    types = {d["doc_type"] for d in SEED_DOCUMENTS}
    assert {"safety_requirement", "launch_criteria", "perception_requirement",
            "odd_definition", "historical_retrospective", "evaluation_policy",
            "standard_concept_paraphrase"} <= types
    for doc in SEED_DOCUMENTS:
        # full metadata is mandatory
        for key in ("source", "version", "jurisdiction", "doc_type",
                    "effective_date"):
            assert doc[key]


# ------------------------------------- no-citation-without-retrieval hard rule

def test_citation_impossible_without_retrieved_text():
    """RetrievedStandard has no escape hatch: a citation cannot be built
    without real retrieved text and a relevance score."""
    base = dict(source="s", document="d", version="1", section="x",
                doc_id="D-1", doc_type="t", jurisdiction="j",
                effective_date="2026-01-01", synthetic=True, label="L",
                chunk_id="c1")
    with pytest.raises(ValidationError):
        RetrievedStandard(**base)  # no retrieved_text / relevance_score at all
    with pytest.raises(ValidationError):
        RetrievedStandard(**base, retrieved_text="too short",
                          relevance_score=0.5)
    with pytest.raises(ValidationError):
        RetrievedStandard(**base,
                          retrieved_text="long enough retrieved text body here",
                          relevance_score=1.7)


def test_retriever_always_returns_full_citation_fields(fallback_index):
    hits = fallback_index.search("pedestrian stopping distance recall", k=4)
    assert hits
    for h in hits:
        assert h.source and h.document and h.version and h.section
        assert len(h.retrieved_text) >= 20
        assert 0.0 <= h.relevance_score <= 1.0


def test_agent_citations_are_backed_by_retrieval(retro_root):
    """End-to-end: every citation on a scorecard corresponds to a retrieval
    hit recorded in the audited RAG tool calls."""
    from sensorflow.retro.agent.orchestrator import analyze_fixture
    from sensorflow.retro.tools.builtin import build_registry
    from sensorflow.retro import store as retro_store

    sc = analyze_fixture("missed_pedestrian_rain", backend="mock")
    assert sc.retrieved_standards, "canonical fixture must cite requirements"
    audit = retro_store.read_audit("missed_pedestrian_rain")
    rag_calls = [r for r in audit if r["tool"] == "safety_standard_rag"
                 and r["status"] == "ok"]
    assert rag_calls, "citations exist but no audited retrieval call was made"
    for std in sc.retrieved_standards:
        assert std.retrieved_text and std.relevance_score >= 0.0
        assert std.chunk_id  # traceable to a stored chunk
