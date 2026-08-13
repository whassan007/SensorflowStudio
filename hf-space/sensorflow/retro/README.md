# Retro — Agentic Retrospective Safety Analyzer

Local-first, evidence-driven analysis of perception failures. Traceable chain:

```
RAW FAILURE -> OBSERVED EVIDENCE -> DERIVED METRICS -> RETRIEVED ENGINEERING
EVIDENCE -> AGENT HYPOTHESIS -> BEHAVIORAL ANALYSIS -> SAFETY POLICY ->
RETROSPECTIVE SCORECARD -> HUMAN DECISION
```

Core rule: the LLM interprets/correlates/hypothesizes; deterministic code
computes metrics and owns the safety boundary. Launch determinations only
pass through the deterministic policy gate (`policy.py`).

## Backends and reproducible startup

### mock (default — no model, fully deterministic)
Nothing to start. All tests run against this backend.

```bash
.venv/bin/python -c "
from sensorflow.retro.inference.client import get_backend
b = get_backend('mock'); print(b.health()); print(b.validate())"
```

### ollama (real local inference on this macOS machine — CPU/Metal)
```bash
# 1. Install & start Ollama (https://ollama.com), then pull a model:
ollama pull gemma3:latest        # or any chat model
# 2. Optionally pin the model:
export RETRO_OLLAMA_MODEL=gemma3:latest
# 3. Validate:
.venv/bin/python -c "
from sensorflow.retro.inference.client import get_backend
b = get_backend('ollama'); print(b.health()); print(b.validate())"
```
Numbers from this backend are **Ollama-on-CPU/Metal**, never vLLM numbers.

### vllm (CUDA/ROCm Linux hosts ONLY — not runnable on this machine)
```bash
# On the GPU host:
pip install vllm
cd sensorflow/retro/inference/vllm_server
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct PORT=8001 ./start_vllm.sh
# Benchmark (TTFT p50/p95/p99, tokens/sec, warm-vs-cold):
python benchmark.py --base-url http://localhost:8001/v1 --label "vLLM <GPU model>"

# Back on this machine, point the client at the host:
export RETRO_VLLM_URL=http://gpu-host:8001/v1
```
`start_vllm.sh` refuses to run on macOS with a clear error. `env_detect.py`
and `compat.py` report this machine as unsupported for vLLM (verified by
tests). See `docs/retro/compatibility-matrix.md` for CUDA vs ROCm.

## Quick tour

```bash
# Environment + compat report
.venv/bin/python -m sensorflow.retro.inference.compat

# Analyze a canonical fixture end to end (mock backend)
.venv/bin/python -c "
from sensorflow.retro.agent.orchestrator import analyze_fixture
sc = analyze_fixture('missed_pedestrian_rain', backend='mock')
print(sc.render_markdown())"

# API (mounted on the studio backend at :8000)
curl -s localhost:8000/api/retro/env | python -m json.tool
curl -s -X POST 'localhost:8000/api/retro/analyze?fixture_id=phantom_brake_plastic_bag&backend=mock'
```

## Layout

- `inference/` — env detection, vLLM compat chain, pluggable client,
  vLLM server configs + startup + benchmark (GPU hosts only).
- `rag/` — safety-case retrieval. Seed corpus is **SYNTHETIC** demonstration
  material; every synthetic rule is tagged `SYNTHETIC_EXAMPLE` /
  `NOT_A_REAL_STANDARD` in content and metadata. SOTIF/ISO 21448 entries are
  paraphrase-level concept summaries, not standard text.
- `tools/` — MCP-style audited tool registry (read-only default, path
  allowlists, audit trail with args/result-hash/timestamp).
- `agent/` — orchestrator: Log Agent -> Evidence Analyzer -> RAG Agent ->
  Safety Synthesizer. Mock backend = deterministic scripted analysis.
- `scorecard.py` — tiered evidence (OBSERVED/DERIVED/RETRIEVED/AI_HYPOTHESIS)
  and the strictly-typed `RetrospectiveScorecard`.
- `policy.py` — deterministic severity framework (asymmetric FN/FP cost) and
  the versioned launch-recommendation gate.
- `api.py` — FastAPI router at `/api/retro`; artifacts under `runs/retro/`.

## Tests

```bash
.venv/bin/python -m pytest tests/test_retro -q
```
