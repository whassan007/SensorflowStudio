"""Retrieval evaluation harness: query set with expected-source assertions.

precision@k here means: fraction of queries whose expected doc_id appears in
the top-k results (a.k.a. hit-rate@k / recall@k over one relevant document
per query — reported explicitly so nobody mistakes it for a graded metric).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from sensorflow.retro.rag.retriever import SafetyCaseIndex, get_index

# Each query names the doc that a correct retriever must surface.
EVAL_QUERIES: List[Dict[str, str]] = [
    {"query": "pedestrian missed inside stopping distance recall requirement",
     "expected_doc_id": "SFS-SAFE-001"},
    {"query": "phantom braking limits for plastic bag road debris misclassification",
     "expected_doc_id": "SFS-SAFE-001"},
    {"query": "launch gate CRITICAL severity unresolved scorecard PASS blocked",
     "expected_doc_id": "SFS-LAUNCH-004"},
    {"query": "INSUFFICIENT_EVIDENCE can never become PASS",
     "expected_doc_id": "SFS-LAUNCH-004"},
    {"query": "rain degradation pedestrian recall confidence threshold raising prohibited",
     "expected_doc_id": "SFS-PERC-002"},
    {"query": "confidence calibration empirical precision rolling window",
     "expected_doc_id": "SFS-PERC-002"},
    {"query": "urban ODD posted speed crosswalk envelope",
     "expected_doc_id": "SFS-ODD-003"},
    {"query": "heavy rain outside ODD minimal-risk maneuver",
     "expected_doc_id": "SFS-ODD-003"},
    {"query": "historical retrospective phantom brake mylar balloon deformable clutter",
     "expected_doc_id": "SFS-RETRO-HIST-01"},
    {"query": "pedestrian dark clothing rain night fusion confidence below tracking threshold",
     "expected_doc_id": "SFS-RETRO-HIST-01"},
    {"query": "safety-critical recall SCR denominator criticality context",
     "expected_doc_id": "SFS-EVAL-005"},
    {"query": "asymmetric false negative false positive cost context dependent",
     "expected_doc_id": "SFS-EVAL-005"},
    {"query": "SOTIF functional insufficiency performance limitation intended functionality",
     "expected_doc_id": "SOTIF-CONCEPTS-21448"},
    {"query": "unknown hazardous scenarios triggering conditions discovery",
     "expected_doc_id": "SOTIF-CONCEPTS-21448"},
]


class QueryResult(BaseModel):
    query: str
    expected_doc_id: str
    hit_at_k: bool
    top_doc_ids: List[str]
    top_score: float


class RetrievalEvalReport(BaseModel):
    k: int
    num_queries: int
    num_hits: int
    precision_at_k: float  # hit-rate@k with one relevant doc per query
    store_backend: str
    results: List[QueryResult]


def run_retrieval_eval(index: Optional[SafetyCaseIndex] = None,
                       k: int = 4) -> RetrievalEvalReport:
    index = index or get_index()
    results: List[QueryResult] = []
    hits = 0
    for item in EVAL_QUERIES:
        found = index.search(item["query"], k=k)
        doc_ids = [f.doc_id for f in found]
        hit = item["expected_doc_id"] in doc_ids
        hits += int(hit)
        results.append(QueryResult(
            query=item["query"], expected_doc_id=item["expected_doc_id"],
            hit_at_k=hit, top_doc_ids=doc_ids,
            top_score=found[0].relevance_score if found else 0.0))
    return RetrievalEvalReport(
        k=k, num_queries=len(EVAL_QUERIES), num_hits=hits,
        precision_at_k=round(hits / len(EVAL_QUERIES), 4),
        store_backend=index.backend_name, results=results)
