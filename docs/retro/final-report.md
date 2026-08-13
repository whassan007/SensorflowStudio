# Agentic Retrospective Safety Analyzer — Final Report

Delivered in one run, structured phase-by-phase per the source spec. All numbers
below are real measurements taken on this machine; anything not measurable here
is explicitly labeled as such.

**Verification totals: 74/74 retro tests green; full repo suite 598/598 green;
tsc clean; production build green; browser walkthrough completed (screenshots
in `docs/retro/screenshots/`).**

---

## Hardware findings on THIS machine (real, measured)

| Property | Value |
|---|---|
| OS | macOS 26.6.1 (Darwin), arm64 — Apple Silicon |
| GPU | Apple M4 (unified memory, Metal/MPS only) |
| Python | 3.11.9 (`.venv`) |
| CUDA / ROCm | **None — vLLM UNSUPPORTED on this machine** |
| Ollama | **Not reachable** at `localhost:11434` at verification time (connection refused) |
| Compat chain | `gpu` = FAIL (clear message); `driver/torch/vllm/model/quantization` = SKIPPED (blocked by failed `gpu` link) |

Measured latencies (labeled honestly):

- **Mock backend, end-to-end analysis via HTTP API**: 0.26–0.30 s per analysis
  (3 runs: 0.298 s, 0.261 s, 0.275 s) on the warm server. This is deterministic
  scripted analysis — no LLM.
- **Ollama**: no numbers reported because no server was running. The client
  supports real inference + latency measurement and labels results
  "Ollama-on-CPU/Metal" when a server is present.
- **vLLM**: NO benchmark numbers exist anywhere in this delivery. The benchmark
  script (`sensorflow/retro/inference/vllm_server/benchmark.py`) is runnable
  only on CUDA/ROCm hosts and says so; it was not executed here.

## Dependency outcomes

| Dependency | Outcome |
|---|---|
| `chromadb` | **Installed cleanly, used live.** The RAG index reports `store_backend: chromadb` at runtime. Deterministic hashed-TF + numpy cosine store remains as the always-available fallback (same interface), and tests force the fallback path so they never depend on chroma. |
| `mcp` | **Installed cleanly.** Bonus MCP server wrapper (`sensorflow/retro/tools/mcp_server.py`) adapts the tool registry; the installed package exposes `MCPServer` (not `FastMCP`), handled dynamically. The registry itself remains the MCP-style boundary. |
| `pydantic-ai-slim` | **Installed**, but orchestration deliberately uses the internal typed loop (typed pydantic contracts + explicit tool loop) for full determinism and auditability; PydanticAI adds no value on the mock path and would obscure the audit trail. |

## Reuse map (read-only imports)

- `sensorflow/safety/ssam_ext.py` → `projected_ttc` math for TTC (with validity flags added on top).
- `sensorflow/megaeval` stats → `DistributionAnalysisTool` delegates distribution-shift summaries.
- `sensorflow/seqeval/paired.py` → `PairedSequentialTest` for statistical significance in the scorecard.
- `sensorflow/evaluation/copilot.py` pattern → Ollama backend client shape (endpoint, model discovery, timeouts).

## Test counts

74 retro tests (all passing), plus full-repo suite 598/598:

| File | Tests | Covers |
|---|---|---|
| `test_inference.py` | 8 | env detection on this machine, vLLM-unsupported honesty, compat chain clarity, mock client contract |
| `test_rag.py` | 9 | chunking, retrieval eval precision, synthetic labels always present, no-citation-without-retrieval |
| `test_tools.py` | 8 | read-only default, write authorization, audit completeness, path allowlist, schemas |
| `test_metrics.py` | 14 | stopping-distance parameterization, TTC validity flags, SCR, behavioral impact |
| `test_policy.py` | 13 | asymmetric costs (benign-FN < phantom-FP), gate determinism, AI-severity override flow, INSUFFICIENT_EVIDENCE never PASS |
| `test_agent.py` | 10 | both canonical fixtures, fact/inference separation, stripped telemetry → UNKNOWN, persistence |
| `test_api.py` | 9 | full API lifecycle via TestClient, vLLM local-unavailability |
| `test_optional_deps.py` | 3 | chromadb store, MCP server bonus (skip if not installed) |

---

## Phase 1 — Inference foundation (`sensorflow/retro/inference/`)

**What was built**: `env_detect.py` (real OS/Python/Torch/GPU/CUDA/ROCm/MPS
detection; detected Apple M4 on this machine), `compat.py` (validation chain
GPU→driver→torch→vLLM→model→quantization; failed links block downstream checks
with clear reasons), `client.py` (pluggable mock/ollama/vllm client with health
check, deterministic test-prompt validation, token counts, latency),
`vllm_server/` (config split into `server.env` / `model.env` / `runtime.env`,
`start_vllm.sh` that refuses to run on macOS, `benchmark.py` with TTFT
P50/P95/P99, tokens/sec, warm-vs-cold), `requirements-retro.txt`, package README.

**How to run**:

```bash
.venv/bin/python -m sensorflow.retro.inference.env_detect     # env report
.venv/bin/python -m sensorflow.retro.inference.compat          # compat chain
# vLLM host (CUDA/ROCm only): edit *.env, then ./start_vllm.sh; benchmark.py
```

**Validation checklist**: env detection real on this machine PASS · vLLM
honestly reported unsupported PASS · mock client contract PASS · vLLM
configs/scripts provided-but-not-executed, stated explicitly PASS.

**Known limitations**: vLLM artifacts unverified here by definition; Ollama
path exercised only via health-check code (no server running); compatibility
matrix is general-knowledge with "verify against current vLLM release" caveats.

## Phase 2 — Safety-case RAG (`sensorflow/retro/rag/`)

**What was built**: parser + configurable chunking, embedder (chromadb default
embedder when installed; deterministic hashed-TF fallback with documented
tradeoff), store (`ChromaStore` / `NumpyCosineStore` behind one protocol),
retriever that ALWAYS returns `{source, document, version, section,
retrieved_text, relevance_score}` + full metadata, seed corpus of 7 SYNTHETIC
documents (safety requirements, launch criteria, perception requirements, ODD,
historical retrospectives, evaluation policy, SOTIF concept paraphrase — every
one tagged `SYNTHETIC_EXAMPLE / NOT_A_REAL_STANDARD` in content AND metadata),
retrieval eval harness.

**Real numbers**: precision@4 = **1.0** over 14 expected-source queries,
`store_backend: chromadb` (live server) — fallback path separately tested.

**Hard rule (tested)**: no safety-requirement citation is ever emitted without
a retrieval hit backing it; every scorecard `retrieved_standards` entry maps to
an audited `safety_standard_rag` call.

**How to run**: `GET /api/retro/rag/search?q=...&k=4`, `GET /api/retro/rag/eval`.

**Known limitations**: corpus is small and synthetic by design; lexical
fallback scores are not comparable to chroma's embedding distances (documented).

## Phase 3 — Agent + MCP-style tools (`sensorflow/retro/agent/`, `tools/`)

**What was built**: `ToolRegistry` (input/output schema, `read_only` default
true, timeout, error behavior, audit log with args/result-hash/timestamp for
every call including denials); tools: `LogReaderTool` (allowlisted to `runs/`
+ fixtures dir), `SafetyStandardRAGTool`, `MetricCalculatorTool` (stopping
distance with reaction/velocity/grade/friction/decel/system latency reusing
ssam_ext math; TTC with validity flags; SCR vs policy criticality; behavioral
impact observed-vs-corrected), `HistoricalFailureSearchTool`,
`DistributionAnalysisTool` (megaeval delegation), `CreateEvaluationCaseTool`
(the ONE write tool, requires explicit authorization, audited). Bonus: real MCP
server wrapper since `mcp` installed. Orchestrator: internal typed loop
(Log Agent → Evidence Analyzer → RAG Agent → Safety Synthesizer); mock backend
= deterministic scripted analysis, Ollama backend = real LLM synthesis with
follow-up RAG queries. Five fixtures including both canonical examples and the
missing-telemetry / benign-FP / non-critical-FN variants.

**Tested guarantees**: stripped telemetry → UNKNOWN, never guessed; fact and
inference never mixed (hypotheses are Tier 4 only); write tool denied without
authorization; every tool call audited.

**How to run**: `POST /api/retro/analyze?fixture_id=missed_pedestrian_rain&backend=mock`
(or upload a JSON log; `backend=ollama` when a server is running).

**Known limitations**: Ollama synthesis path implemented and unit-tested at the
prompt/parsing level but not exercised end-to-end here (no server); MCP server
wrapper is smoke-tested, not exercised by a real MCP client.

## Phase 4 — Scorecard + policy gate (`scorecard.py`, `policy.py`)

**What was built**: evidence hierarchy enforced in types (Tier1 OBSERVED /
Tier2 DERIVED / Tier3 RETRIEVED / Tier4 AI_HYPOTHESIS); strictly-typed
`RetrospectiveScorecard` with every spec field + markdown renderer showing the
FACT / DERIVED FACT / RETRIEVED REQUIREMENT / AI HYPOTHESIS / SAFETY
DETERMINATION distinction; deterministic severity BENIGN/DISRUPTIVE/CRITICAL/
FATAL from contextual evidence with the asymmetric FN/FP cost model (VRU ×
distance-vs-stopping × relative motion × remaining reaction time for FN;
intervention magnitude × disruption for FP) — tested that a benign distant FN
ranks BELOW a hard-braking phantom FP; LLM severity proposal adjudicated by the
policy engine with divergence recorded + flagged for human review; policy-
versioned launch gate (`retro-policy/1.0.0`) where INSUFFICIENT_EVIDENCE can
never become PASS.

**Fixture B verified result** (mock, end-to-end): FALSE_NEGATIVE, severity
CRITICAL (AI proposal agreed, no divergence), launch FAIL (GATE-01), human
review required, 38 evidence items, 7 retrieved standards, 2 hypotheses,
stopping distance 26.18 m vs 26.0 m range, TTC 1.7 s with validity flags.

**Known limitations**: severity weights are demonstration policy values
(versioned, documented), not calibrated to a real fleet's risk model.

## API + persistence

All endpoints live under `/api/retro` (env, compat, backends, rag/search,
rag/eval, fixtures, analyze, analyses, analyses/{id}, analyses/{id}/audit,
tools) — all verified 200 via curl sweep and the pytest TestClient suite.
Artifacts persist under `runs/retro/{analyses,audit,uploads}`.

## Frontend

"Retrospective Analyzer" page under SAFETY & COMPLIANCE. All new files
(`src/pages/retro/`, `src/components/retro/`, `src/services/retro.ts`,
`src/types/retro.ts`) + minimal additive insertions (App.tsx nav/import/render,
LabelEvalContext page id, HelpMenu page name, pageHelp + glossary entries —
required because those maps are typed `Record<PageId, …>`). Delivers: fixture
picker + backend selector with honest availability badges, vertical evidence-
chain view color-coded by tier with legend, scorecard with severity/launch
banner + human-review flag, retrieved-standards panel with SYNTHETIC_EXAMPLE
badges, audit-trail tab, hardware/compat card. `tsc` clean; `vite build` green.

## Verification & servers

- Full pytest: **598 passed, 0 failed** (includes 74 retro tests).
- Browser walkthrough of fixture B end-to-end on `http://localhost:5199/#/retro`
  (screenshots 01–06 in `docs/retro/screenshots/`).
- The pre-existing servers on :8000 and :5173 were left untouched; because the
  :8000 backend predates this router (started without `--reload`), verification
  runs on a fresh backend at :8100 paired with `vite.retro-verify.config.ts` on
  :5199 — both left running. Nothing was committed.

## Honest gaps

1. No vLLM execution or benchmark numbers — impossible on macOS; scripts ship unverified-by-design.
2. No real-LLM end-to-end run — Ollama was not running at verification time; the mock path is the verified path.
3. The safety-case corpus is synthetic; SOTIF content is concept paraphrase only.
4. The long-running :8000/:5173 servers won't show the retro API/page until restarted (concurrent-work constraint, not a defect).
5. PydanticAI installed but intentionally unused for orchestration (typed internal loop chosen for auditability).
