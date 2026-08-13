# Sensorflow Studio — Platform Inventory

**Canonical stack:** `hf-space/` (FastAPI + React + `sensorflow/evaluation` + `sensorflow/megaeval`).  
**Root twin:** older subset (`sensorflow/` without evaluation/megaeval; `app_backend.py` without those routers; thin SSAM React in `src/`). Prefer upgrading `hf-space/` and keep shared root APIs working where duplicated.

**Phase 1 foundation module:** `hf-space/sensorflow/platform/` (extends MegaEval + LabelEval; does not replace them).

---

## 1. Frontend pages / components

| Area | Path | Status | Goal mapping |
|------|------|--------|--------------|
| Hash router / nav | `hf-space/src/App.tsx` | EXISTS | Command Center, Evaluation, Quality, Models, SSAM |
| Command Center (MegaEval) | `pages/CommandCenterPage.tsx` + `components/megaeval/*` | EXISTS | Population → cohort → container → object |
| LabelEval pages | Overview, Datasets, LabelGen, RareEvents, Quality, Regression, Triage, Review, Training, Models, Evaluation, Audit, Pipeline | EXISTS | Label QA flywheel |
| SSAM / legacy | `SSAMSafetyDashboard`, `LegacyStudioPage`, MapCanvas, DataGrid | EXISTS | Safety viz (Phase 4 hooks) |
| Platform Phase 1 hooks | `components/platform/*` | **Phase 1** | Container quality cards, model compare panel, multi-gate skeleton |
| Services | `services/{labeleval,megaeval,api,platform}.ts` | EXISTS + Phase 1 | Prefer existing clients |
| Types | `types/{labeleval,megaeval}.ts` | EXISTS | Extend via platform types |

**Gap (later):** Drive/Scene first-class UI; deep ODD/safety panels; distributed job UI.

---

## 2. Backend APIs

| Stack | Mount | Prefixes |
|-------|-------|----------|
| LabelEval | `evaluation/api.py` → `app.include_router` | `/api/datasets`, `/api/labeleval/*`, `/api/quality/*`, `/api/review/*`, `/api/models`, `/api/events/stream` |
| MegaEval | `megaeval/api.py` | `/api/megaeval/*`, `POST /api/evaluations/query` |
| Legacy studio / SSAM / perception | `app_backend.py` | `/api/ssam/*`, `/api/perception/*`, `/api/gates/quality`, `/api/gates/launch`, train/infer/MITL |
| **Platform Phase 1** | `platform/api.py` | `/api/evaluations/*`, `/api/containers/*`, `/api/models/compare`, `/api/gates/*` (multi-gate) |
| Entry | `space_app.py` | mounts `/dashboard` → Vite `dist/` |

Root `app_backend.py`: legacy routes only (no LabelEval/MegaEval/platform).

---

## 3. Schemas / data models

| Layer | Path | Entities |
|-------|------|----------|
| Perception | `schemas/unified_frame.py`, `taxonomy_axes.py` | FusedFrame, Object3D, UnifiedSequence |
| LabelEval store | `evaluation/records.py` | Dataset, Scene, Sequence, Frame, Annotation, Track, Scenario, RareEvent, AnomalyDetection, Validation, Graders, Triage, Review, Model, ProcessUsage, Audit… |
| MegaEval runtime | `megaeval/runs.py`, cube/population | EvaluationRun, container table (npz), metric cube |
| Comparative | `comparative_analytics.py` | Configuration, PerformanceMetrics |
| **Platform Phase 1** | `platform/entities.py` | Unified provenance models: Container, Drive, Group, Sensor, Object, Trajectory, Label, Evaluation, Metric, Cohort, Embedding, Conflict, ODD*, QualityGate, GateResult, EvidencePackage, ComputeUsage, ModelVersion |

---

## 4. Container / group / frame / label / model / evaluation concepts

| Concept | LabelEval | MegaEval | Platform Phase 1 |
|---------|-----------|----------|------------------|
| Dataset | Pydantic Dataset | Population (immutable npz) | Links both via EvaluationScope |
| Container | GAP | Integer + dims in containers.npz | `Container` + quality profile API |
| Group | Verification buckets | Cohort = cube slice | `Group` + `Cohort` aliases documented |
| Drive / Scene | Thin Scene | ≈ Container | Drive/Scene stubs with provenance |
| Frame / Label | Frame, Annotation | Forensic objects | Re-exports + Label alias |
| Model | Model + training jobs | `model_version` on run | ModelVersion + multi-compare |
| Evaluation | Per-annotation evidence | Async EvaluationRun | AggregateLevel ladder |

---

## 5. ML pipelines

| Module | Role | Status |
|--------|------|--------|
| `evaluation/pipeline.py` | Label QA flywheel | EXISTS |
| detectors / graders / validation / rare_events / regression / triage | Label quality | EXISTS |
| `perception_automator`, `temporal_tracker`, fusion, adapters | 3D perception | EXISTS |
| `quality_gate`, `launch_gate_evaluator` | Sequence gates | EXISTS → wrapped by multi-gate |
| MegaEval `runs.evaluate_partition` | Simulated population eval | EXISTS |
| Multi-grader consensus vectors | — | **DEFER** Phase 2 |
| Embedding / VLM rare-event mining | sketches + embeddings stub | **DEFER** Phase 3 |
| SSAM DRAC/DeltaS upgrades | SSAM APIs exist | **DEFER** Phase 4 |
| ODD combinatorial coverage | ODD entity stubs | **DEFER** Phase 5 |
| Distributed processing | ThreadPool / npz stand-ins | **DEFER** Phase 7 |

---

## 6. Async / jobs

| Pattern | Location | Notes |
|---------|----------|-------|
| MegaEval runs | `megaeval/runs.py` MegaStore | queued → workers → reduce → published |
| LabelEval pipeline | `pipeline.run(background=True)` + `queue.py` | Threaded stages + SSE |
| Training jobs | EvalStore TrainingJob | |
| Process units | `process_units.py` | Cost accounting |
| Platform expensive ops | Prefer MegaEval job pattern; sync for small profiles | Phase 1 |

---

## 7. Viz libraries

`@deck.gl/*`, `react-map-gl`, MUI, lucide-react. **No** recharts/plotly — MegaEval uses hand-rolled SVG sparklines. Phase 1 UI matches that pattern.

---

## 8. Storage / indexing

| Store | Format | Path |
|-------|--------|------|
| LabelEval | JSON | `runs/labeleval/store.json` |
| MegaEval populations/runs | npz + JSON | `runs/megaeval/` |
| Pipeline artifacts | JSON | `runs/pipeline/` |
| Platform evidence exports | JSON | `runs/platform/evidence/` |
| Future lakehouse | Parquet/Iceberg | Documented in ARCHITECTURE.md — **not** introduced in Phase 1 |

---

## 9. Auth

**GAP:** no runtime API auth. HF deploy tokens only (`deploy/huggingface`). Do **not** add a second auth system in Phase 1.

---

## 10. Tests

| Dir | Coverage |
|-----|----------|
| `hf-space/tests/test_labeleval/` | detectors, graders, validation, queue, e2e |
| `hf-space/tests/test_megaeval/` | cube, sketches, sampling, analysis, API lifecycle |
| `hf-space/tests/test_pipeline/` | quality/launch gates, perception, tracker, schema |
| `hf-space/tests/test_platform/` | **Phase 1** metrics, container quality, model compare, gates, evidence |
| Root `tests/test_pipeline/` | Mirror of pipeline only |

---

## Exists vs gap (user goals)

| Goal | Exists | Gap / Phase 1 action |
|------|--------|----------------------|
| Unified entity schema | Partial dual stacks | Platform entities + provenance |
| Eval across aggregate levels | MegaEval + LabelEval separate | `AggregateLevel` + EvaluationScope |
| Container quality API | MegaEval containers list | `/api/containers/.../quality` profile |
| Model compare A/B/C | MegaEval A vs B | `/api/models/compare` multi-run |
| Multi-gate | Quality + Launch only | Config-driven Scenario/Coverage/Regression/Safety/Release skeleton |
| Evidence package | LabelEval evaluation_record | Exportable EvidencePackage JSON |
| Population scale (billions) | npz stand-ins | Phase 7 distributed — hooks only |
