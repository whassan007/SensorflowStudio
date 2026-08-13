#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server.
#
# RUNNABLE ONLY ON CUDA/ROCm LINUX HOSTS. This script was authored on a macOS
# development machine where vLLM cannot run; it refuses to start on Darwin
# with an honest error instead of failing cryptically.
#
# Config is separated into three env files (server/model/runtime); every
# variable can be overridden from the environment:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct PORT=8010 ./start_vllm.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -s)" = "Darwin" ]; then
  echo "ERROR: vLLM does not run on macOS (no CUDA, no ROCm)." >&2
  echo "Run this script on a Linux host with an NVIDIA or AMD GPU." >&2
  echo "On this machine use the 'ollama' or 'mock' backend instead." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v rocm-smi >/dev/null 2>&1; then
  echo "ERROR: neither nvidia-smi nor rocm-smi found — no supported GPU stack." >&2
  exit 2
fi

# shellcheck disable=SC1091
source ./server.env
source ./model.env
source ./runtime.env

echo "Starting vLLM: model=${MODEL_NAME} host=${HOST}:${PORT}" \
     "tp=${TENSOR_PARALLEL_SIZE} gpu_mem=${GPU_MEMORY_UTILIZATION}" \
     "max_len=${MAX_MODEL_LEN} quant=${QUANTIZATION}"

ARGS=(
  --model "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --dtype "${DTYPE}"
)
[ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${API_KEY}" ] && ARGS+=(--api-key "${API_KEY}")

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
