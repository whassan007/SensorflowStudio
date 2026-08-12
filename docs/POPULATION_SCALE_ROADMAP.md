# Population-Scale Evaluation — Roadmap

Trustworthy foundation for population-level intelligence on Sensorflow Studio.  
**Rule:** extend MegaEval + LabelEval; no parallel stores, auth, or browser ML.

Canonical code: `hf-space/`. Shared perception modules also live under root `sensorflow/`.

---

## Non-goals (all phases)

- Rebuild from scratch or replace MegaEval cube/runs
- New microservice fleet or second database
- Second auth system
- Sync browser-side ML inference
- Hard-coded safety thresholds (use config / policy JSON)
- Silent overwrite of evaluation / regression history
- Full ODD combinatorics, VLM mining, or distributed executors before their phase

---

## Phase 1 — Foundation (this session)

**Extend:**
- `sensorflow/evaluation/records.py` concepts (do not fork store)
- `sensorflow/megaeval/{runs,analysis,api,cube}.py`
- `quality_gate.py`, `launch_gate_evaluator.py`
- React MegaEval / Evaluation / Quality pages

**Create:**
| Artifact | Path |
|----------|------|
| Inventory | `docs/PLATFORM_INVENTORY.md` |
| This roadmap | `docs/POPULATION_SCALE_ROADMAP.md` |
| Platform package | `hf-space/sensorflow/platform/` |
| UI hooks | `hf-space/src/components/platform/`, service `platform.ts` |
| Tests | `hf-space/tests/test_platform/` |

**Deliverables:**
1. Schema entities with versioning/provenance
2. `AggregateLevel` evaluation abstraction (frame→…→population)
3. Metric engine: container PRF/IoU + verification rates
4. Container quality profile API
5. Model compare API (A/B/C) wrapping MegaEval compare
6. Routes: `/api/evaluations`, `/api/containers`, `/api/models/compare`, `/api/gates`
7. UI: container quality table, model compare panel, multi-gate skeleton
8. Evidence package JSON export stub
9. Tests run green

**Hooks left for later (TODO markers):** multi-grader vectors, embedding/VLM mining, SSAM conflict engine, ODD coverage, distributed reduce.

---

## Phase 2 — Multi-grader consensus & label trust

**Extend:** `evaluation/graders.py`, `triage.py`, `reporting.py`  
**Add:** consensus vectors, disputed/auto-accept rates into container profiles  
**UI:** GraderDisagreement + Command Center quality tab  
**Do not:** replace triage policy store

---

## Phase 3 — Rare-event & embedding mining

**Extend:** `evaluation/rare_events.py`, MegaEval `embeddings.npz` / `sketches.py`  
**Add:** VLM/embedding similarity search APIs (async jobs)  
**UI:** RareEventDashboard + InvestigationTab  
**Do not:** sync browser embeddings

---

## Phase 4 — Safety / SSAM conflict engine

**Extend:** SSAM routes in `app_backend.py`, `SSAMSafetyDashboard`  
**Add:** DRAC / DeltaS conflict entities (`Conflict`), EvidencePackage links  
**Config-driven thresholds only**  
**Do not:** hard-code severity cutoffs in UI

---

## Phase 5 — ODD combinatorial coverage

**Extend:** `ODDDefinition` / `ODDObservation` stubs from Phase 1  
**Add:** coverage matrix over dims (weather×lighting×road×scenario…) via MegaEval cube  
**UI:** Coverage gate fills multi-gate skeleton  
**Do not:** new OLAP store — use cube queries

---

## Phase 6 — Release / regression / HITL orchestration

**Extend:** `evaluation/regression.py`, review queue, launch gate  
**Wire:** Regression + Safety + Release gates to real metrics  
**Preserve:** full regression history (append-only)

---

## Phase 7 — Distributed scale

**Replace stand-ins** documented in `hf-space/ARCHITECTURE.md`:
| Today | Target |
|-------|--------|
| npz partitions | Parquet / Iceberg |
| ThreadPoolExecutor | Spark / Flink / Ray |
| QueryRouter cache | Pinot / Druid / ClickHouse |
| JSON EvalStore | Keep for control-plane metadata; lakehouse for facts |

Keep API contracts from Phases 1–6 stable.

---

## Aggregate ladder (canonical)

```
frame → clip/sequence → scene → drive → container → dataset/population → cohort → fleet/population aggregate
```

Implemented in Phase 1 as `AggregateLevel` + `EvaluationScope`; MegaEval already materializes population/cohort/container; LabelEval owns frame/annotation.

---

## Observability checklist (carry forward)

- Every query: `source`, `cache_hit`, `latency_ms` (MegaEval pattern)
- Every gate: config path + threshold values in result
- Every evidence package: model/dataset versions + compute usage + provenance
- No silent history overwrite
