# Sensorflow Studio — Aggregate-First Evaluation Architecture (megaeval)

The `sensorflow/megaeval/` layer replaces the "evaluation-record browser" model
(annotation as the primary unit) with an **aggregate-first, risk-first,
cohort-first, drill-down-on-demand** design.

## Mental model → code

| Concept | Meaning | Module / storage |
| --- | --- | --- |
| Dataset | immutable source population | `population.py` → `runs/megaeval/populations/{pop}/part-*.npz` |
| Evaluation Run | model+config being evaluated (async job, first-class) | `runs.py` → `runs/megaeval/runs/{run}/run.json` |
| Population | all objects evaluated in a run | headline row of the metric cube |
| Cohort | slice by class/weather/lighting/road/scenario/sensor/distance/speed/occlusion | cube cells + `POST /api/evaluations/query` |
| Container | scene/frame/segment holding related objects | `containers.npz` aggregate table |
| Annotation | individual object — forensic view only | per-object `objects-part-*.npz`, loaded one partition at a time |

## Data flow

```
generate population (vectorized numpy, partitioned npz)
        │
create run (queued) ──► worker pool (4 threads) processes partitions:
        │                 model simulation → PARTIAL sufficient statistics
        │                 (n, tp, fp, fn, sum_iou, conf sums, histograms …)
        │                 grouped by the 9 cube dimensions, per partition
        ▼
reduce (concat + groupby-sum) ──► materialize METRIC CUBE (cube.npz)
        │                          + error index, container table, sketches,
        │                          container embeddings
        ▼
publish ──► queries served by QueryRouter: cache → cube → (rare) record scan
```

Aggregation **never** scans raw records at query time; queries like
"pedestrian recall at night in rain for model v42" are cube lookups.

## Single-node stand-ins for distributed infrastructure

This repo runs on one laptop; the seams are shaped so each piece maps to the
production-scale equivalent:

| Here (single node) | Production analogue |
| --- | --- |
| npz column partitions (`part-0000.npz…`, hash-partitioned by container) | Parquet files in an Iceberg/Delta table, partitioned by scene |
| `ThreadPoolExecutor` workers over partitions | Spark executors / Flink task slots |
| partial stats per partition + `reduce_partials` | map-side combine + shuffle reduce |
| `QueryRouter` (cache → cube → scan) with latency+source reporting | OLAP engine (Druid/Pinot/ClickHouse) over the lakehouse |
| `QueryCache` keyed by hash(dataset+model+filters+group_by+metrics) | result cache / materialized views |
| in-process SSE progress events | job-status topic (Kafka) + progress service |
| numpy cosine similarity over container embeddings | vector DB (hybrid retrieval) |

## Honesty rules encoded in the API

- Cube counts and derived ratios are **exact** (`meta.exact = true`).
- Percentiles/distributions come from fixed-bin quantile histograms;
  cardinalities from HyperLogLog — always labeled approximate.
- Precision/recall from human review are **sampling estimates with 95% CIs**
  (stratified risk-weighted design: Wilson per stratum, weighted normal-approx
  combination), reported with n reviewed / frame size / method.
- Every query response reports `source` (cache/cube/scan), `cache_hit`,
  and `latency_ms` so aggregate-first behavior is observable in the UI.
- Evaluation runs are reproducible: lineage (dataset, model+checkpoint, label
  version, evaluator code version, metric version, thresholds, sampling config,
  seed, hardware) is stored on the run and derives the deterministic seed.

## Performance SLOs (measured locally, 320k-object population, 67k cube cells)

| Query | SLO | Measured |
| --- | --- | --- |
| aggregate dashboard (group by class) | < 2 s | ~75 ms |
| filtered cohort query | < 3 s | ~350 ms first / sub-ms cached |
| cached query | < 100 ms | ~0.1 ms |
| object investigation (one container, forensic) | < 1 s | ~50–200 ms |

The legacy `sensorflow/evaluation/` layer (label-evaluation pipeline, triage,
HITL, training flywheel) remains unchanged; megaeval integrates with its
regression tracker and alerting rather than duplicating them.
