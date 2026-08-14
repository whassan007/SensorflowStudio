# Sensorflow Studio — Execution Integrity & Observability Audit

**Date:** 2026-08-14  
**Scope:** Legacy Studio (`static/`, `app_backend.py`, `sensorflow/`, `train.py`, `infer.py`, `autograder.py`)  
**Mirror:** `hf-space/` copies are effectively identical for these surfaces — findings apply to both.  
**Prime directive:** A UI state is NOT evidence. PASS / Done / 100% / Found / `success:true` / toasts / YAML alone / file existence alone are not proof of execution.

---

## Stage integrity table

| COMPONENT | CURRENT BEHAVIOR | ACTUAL EXECUTION? | EVIDENCE | RISK | REQUIRED FIX |
|-----------|------------------|-------------------|----------|------|--------------|
| 1 Dataset Configuration / Load & Preprocess | Save config writes JSON; Pre-Check = script file existence; “Load Catalog Metadata” returns hardcoded KPIs (`ingestion_pct: "100.0%"`) after `sleep(0.3)`; Validate & Browse is real disk scan. Toast “saved & verified” without path checks. | **Partial** — browse real; catalog preprocess is not load | `app_backend.py` precheck/details/preprocess; `static/app.js` save/browse handlers | **High** | Real Load & Preprocess with discovery counts + ledger; never show 100% Loaded from catalog |
| 2 3D Ingest & Fusion | Calls fusion engine; vendor adapters default `demo_stub`; local path can be real. UI PASS when `manifest.json` exists. | **Yes local**; **stub vendors** | ingest API + adapters; gate badges | **Medium** | PASS/DONE only with provenance; surface `demo_stub` + execution_id |
| 3 3D Perception / Auto-label (SAM) | Frontend hardcodes `no_sam: true`. Automator falls back to GT/synthetic masks; still `status: ok` → PASS. | **No real SAM** on Studio UI path | `static/app.js` perception handler; `perception_automator.py` | **Critical** | Record model/checkpoint/counts; NOT_EXECUTED if model not run; refuse fake SUCCESS |
| 4 Temporal Tracking | Real Kalman+Hungarian over proposals. | **Yes** (inputs may be fake) | track API | **Medium** | Propagate contamination; ledger entry |
| 5 Quality Gate | Real metric math vs sequence GT; contaminated proposals inflate pass. | **Yes math**; honesty depends on inputs | quality gate API | **High** | Fail closed on stub/GT-fallback; evidence card |
| 6 Launch Gate | Threshold check on metric card. | **Yes** (inherited) | launch gate evaluator | **Medium** | Same provenance gate |
| 7 Model Setup (2D) | UI model card only; no weight verification. | **No** | model cards in `app.js` | **Low–Medium** | Verify weights exist before train/infer |
| 8 Training Execution | Spawns real `train.py`. Status fabricates losses; forces progress=1.0; ignores exit_code. UI “Process finished” with no verdict. | **Partial** | train start/status; `train.py` | **Critical** | PID, command, exit_code, checkpoint hash, ledger |
| 9 Auto-Labeler Inference | Runs `infer.py` with check=True. Empty source exits 0 → ok + empty images. UI always success toast. | **Partial** | infer API; `infer.py` | **High** | Counts + checkpoint + exit_code; NOT_EXECUTED/FAILED if 0 images / missing weights |
| 10 Auto-Grader Quality | Real autograder when predictions exist; empty issues → “Perfect score!”. | **Yes** (heuristic) | grade API; `autograder.py` | **Medium** | Metrics only from grader output; ledger |
| 11 MITL Review | Hardcoded Unsplash samples; claims HF ingest; Mock Triage Auditor. | **No** | nvidia/load; mitl/evaluate | **Critical** | Label fixtures; never SUCCEEDED for mock ingest |
| 12 Model Benchmarking | `fallback_mock` hardcoded metrics with `status: ok`. | **No** (unless live card) | `/api/benchmark/compare` | **Critical** | Refuse OK without live metrics; stub = NOT_EXECUTED |
| 13 Export & Deploy | May run Ultralytics export; returns path without exists(); UI “✓ complete” without `res.ok`. | **Partial** | export API | **High** | Verify artifact; ledger |
| 14 MCP Settings | Real R/W when file exists; invented servers + “changes simulated”. | **Partial** | mcp APIs | **Medium** | Honest status; evidence stub |
| 15 SSAM Safety | Hardcoded CA intersections presented as statewide FHWA dataset. | **No real lake** | ssam endpoints + catalog | **High** | Label synthetic demo; stub evidence |

---

## High-risk fake-success occurrences

### Backend (`app_backend.py`)

| Location | Pattern |
|----------|---------|
| `/api/precheck` | Success = `train.py` / `infer.py` / `autograder.py` **file existence only** |
| `/api/train/status` | Fabricates decreasing losses; forces `progress=1.0`; ignores process `returncode` |
| `/api/infer/run` | Empty run still `status: ok` when script exits 0 with 0 images |
| `/api/export` | Returns export path without verifying file exists |
| `/api/nvidia/load` | Hardcoded samples; claims HF ingest success |
| MITL evaluate | Mock Triage Auditor returned as success-shaped response |
| `/api/benchmark/compare` | `fallback_mock` hardcoded metrics, `status: ok` |
| `DATASET_METADATA_STORE` | Hardcoded `ingestion_pct: "100.0%"` catalog KPIs |
| `/api/dataset/preprocess` | `time.sleep(0.3)` + catalog echo — no disk load |
| MCP get/toggle | Invented config / “changes simulated” with ok/warning |
| SSAM store | Hardcoded intersections as statewide dataset |
| `/api/pipeline/status` | File existence ⇒ `*_complete: true` |

### Frontend (`static/app.js`)

| Location | Pattern |
|----------|---------|
| Gate badges | Completion stages → **PASS** from file-complete; never FAIL |
| Config save | “verified successfully” without `res.ok` / path validation |
| Training poll | “Process finished” with no exit_code / success check |
| Infer | Always success toast (even empty / `!res.ok`) |
| Grader | Empty issues → “Perfect score!” |
| Export | “✓ Export complete” without `res.ok` |
| Perception | **`no_sam: true` hardcoded** |

### Scripts / sensorflow

| Location | Pattern |
|----------|---------|
| `train.py` dgx-spark | Prints remote offload; uses local device |
| `infer.py` | No images → print and **exit 0** |
| `perception_automator.py` | Silent SAM off; synthetic masks/LiDAR; GT class copy |
| Vendor adapters | Built-in ~3-frame `demo_stub` lakes |

---

## Cross-cutting patterns

1. **`status: "ok"` / success toast without evidence**
2. **File existence = done/PASS**
3. **Hardcoded metrics / “100%” catalog KPIs**
4. **Simulated delays / stubs claiming complete**
5. **Subprocess integrity gaps** (exit ignored, empty exit 0)
6. **GT contamination loop** (`no_sam` → synthetic/GT proposals → quality/launch PASS)
7. **Frontend rarely checks `res.ok`**

---

## What is relatively honest today

- Validate & Browse (`/api/dataset/browse`) — real filesystem scan
- Local ingest (non-stub) — real frames → manifest
- Tracking / quality math / launch thresholds / autograder heuristics / YOLO train+infer — real when inputs and deps exist
- Catalog preprocess messaging partially hardened (`catalog_only`, `browse_hint`) but KPIs still read as Loaded

---

## Remediation shipped in `feat/execution-integrity`

**Implemented:** Execution ledger + APIs; real Dataset Load & Preprocess; Auto-Label / Infer / Train / Grader evidence wiring; YAML semantic validation; script verification; backend health; Strict Mode; Evidence Cards + Global Execution Console; failure + deterministic tests; remaining stages show honest `NOT_EXECUTED` / unverified stubs.

**Deferred (honest stubs only):** Full MITL/HF ingest, real SSAM lake, real MCP process probes, SAM GPU path when checkpoint absent, full 15-stage process verification beyond Dataset/Auto-Label/Train/Grader.
