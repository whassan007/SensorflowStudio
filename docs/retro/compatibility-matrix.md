# CUDA vs ROCm compatibility matrix for the retro inference layer

Scope: running the vLLM server (`sensorflow/retro/inference/vllm_server/`)
that backs the Retrospective Analyzer's `vllm` backend.

**Honesty statement.** This matrix was written on a macOS (Apple Silicon)
machine where **none of the GPU rows below can be executed or verified**. It
is compiled from general knowledge of the CUDA/ROCm/vLLM ecosystems as of the
author's knowledge cutoff. vLLM moves fast: **verify every row against the
release notes of the vLLM version you actually install.** Rows marked
UNCERTAIN are flagged rather than guessed.

## Platform support at a glance

| Platform | vLLM | Notes |
|---|---|---|
| Linux + NVIDIA (CUDA 12.x) | Supported | Primary, best-tested target |
| Linux + AMD (ROCm 6.x, MI200/MI300 class) | Supported | Official ROCm builds; narrower feature set |
| Linux + AMD consumer (RDNA) | UNCERTAIN | Community/experimental; verify per release |
| macOS (Apple Silicon or Intel) | **NOT SUPPORTED** | No CUDA, no ROCm; this dev machine. Use Ollama (Metal) locally |
| Windows native | NOT SUPPORTED | WSL2 + CUDA is the workaround |
| CPU-only Linux | Limited/experimental | A CPU backend exists but is not a serving-grade path; verify |

## Package installation

| Aspect | CUDA (NVIDIA) | ROCm (AMD) |
|---|---|---|
| Install | `pip install vllm` (prebuilt wheels, CUDA 12.x) | No universal PyPI wheel historically; use AMD's ROCm docker images or build from source with ROCm 6.x. Verify current release — ROCm wheels have been appearing |
| Driver floor | Driver >= 535-series for CUDA 12.x wheels (verify) | ROCm >= 6.0 for current vLLM; MI300 needs recent ROCm |
| Attention kernels | FlashAttention-2 / FlashInfer prebuilt | ROCm FlashAttention port (Composable Kernel / Triton based); fewer head-dim/feature combos supported |

## PyTorch builds

| Aspect | CUDA | ROCm |
|---|---|---|
| Wheel source | `pip install torch` (cu12x wheels on PyPI) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x` |
| Detection | `torch.version.cuda` set, `torch.cuda.is_available()` | `torch.version.hip` set; ROCm *masquerades as* `torch.cuda` (HIP) — `torch.cuda.is_available()` is True on ROCm builds |
| Pitfall | CPU-only wheel silently installed if index not pinned | Mixing a CUDA wheel onto a ROCm host fails at runtime, not install time |

## Quantization support (verify against your vLLM release)

| Method | CUDA | ROCm |
|---|---|---|
| AWQ | Supported (Ampere+; kernels tuned for 4-bit) | UNCERTAIN/limited — AWQ kernels historically CUDA-only; do not assume |
| GPTQ | Supported (incl. Marlin kernels on newer GPUs) | Partial — basic GPTQ paths have worked via Triton; Marlin is CUDA-specific |
| FP8 (W8A8) | Supported on Hopper/Ada (H100, L40S); emulated/absent on older | Supported on MI300 (native FP8); not on older CDNA |
| INT8 (SmoothQuant-style) | Supported | UNCERTAIN — verify |
| bitsandbytes | Supported (load-time) | NOT SUPPORTED historically |

`compat.py` encodes a conservative subset of this table
(`CUDA_QUANTIZATIONS` / `ROCM_QUANTIZATIONS`) and says "verify against the
installed vLLM release" in its PASS messages.

## Memory management

| Aspect | CUDA | ROCm |
|---|---|---|
| PagedAttention KV cache | Core feature, mature | Same architecture, HIP-compiled kernels |
| `GPU_MEMORY_UTILIZATION` | Fraction of VRAM pre-allocated (default 0.9) | Same flag; MI300's 192 GB HBM changes sizing math |
| Unified/managed memory | Not used by vLLM | Not used by vLLM |
| Multi-GPU (TENSOR_PARALLEL_SIZE) | NCCL | RCCL (NCCL API-compatible); verify topology support (xGMI vs NVLink) |

## Kernel compatibility

| Aspect | CUDA | ROCm |
|---|---|---|
| Core kernels | CUDA C++/CUTLASS + Triton | HIPified C++ + Composable Kernel + Triton-on-ROCm |
| Triton | Mature | Works on ROCm but kernel autotuning/coverage lags; some fused ops fall back to slower paths |
| CUDA Graphs | Used for decode speedups | HIP graph support has lagged; verify (`--enforce-eager` is the fallback) |
| Custom all-reduce | CUDA-specific fast path | Typically disabled on ROCm; RCCL path used |

## What this repo can and cannot claim

- CAN claim (verified here): vLLM is unsupported on this macOS machine —
  `env_detect.py` + `compat.py` prove it at runtime, and the test suite
  asserts it.
- CANNOT claim: any vLLM latency/throughput number. `benchmark.py` produces
  them only when pointed at a real CUDA/ROCm host; no such run happened here.
- Local real-inference numbers, if any appear in reports, come from an
  **Ollama server on Apple CPU/Metal** and are labeled as such.
