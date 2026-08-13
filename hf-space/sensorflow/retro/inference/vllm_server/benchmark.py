#!/usr/bin/env python3
"""Latency/throughput benchmark for an OpenAI-compatible LLM endpoint.

Measures, over N requests against a live endpoint:
  * TTFT (time to first streamed token): P50 / P95 / P99
  * end-to-end latency percentiles
  * completion tokens/sec (per request and aggregate)
  * warm vs cold: the first request after startup is reported separately

HONESTY NOTE: this script was written on a macOS machine where vLLM cannot
run. It has NOT been executed against a vLLM server here and this repository
contains NO vLLM benchmark numbers. Run it yourself on a CUDA/ROCm host:

    python benchmark.py --base-url http://gpu-host:8001/v1 --n 32

It also works against Ollama's OpenAI-compatible endpoint
(http://localhost:11434/v1) — numbers from that path are Ollama-on-CPU/Metal
numbers and must be labeled as such, never as vLLM results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Dict, List, Optional

import httpx

PROMPTS = [
    "Summarize why time-to-collision alone is an insufficient safety metric.",
    "List three causes of phantom braking in camera-based perception stacks.",
    "Explain the difference between a false negative and a missed detection track.",
    "Describe how rain degrades LiDAR point cloud density at range.",
]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def stream_once(client: httpx.Client, base_url: str, model: str, prompt: str,
                max_tokens: int) -> Dict:
    """One streaming chat completion; returns ttft/e2e/token counts."""
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    completion_tokens = 0
    with client.stream(
        "POST", f"{base_url}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.0, "stream": True},
        timeout=180.0,
    ) as res:
        res.raise_for_status()
        for line in res.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                completion_tokens += 1  # one content delta ~= one token for vLLM/Ollama
    e2e = time.perf_counter() - t0
    return {"ttft_s": ttft if ttft is not None else e2e, "e2e_s": e2e,
            "completion_tokens": completion_tokens,
            "tokens_per_s": completion_tokens / e2e if e2e > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default=None, help="default: first served model")
    ap.add_argument("--n", type=int, default=16, help="number of measured (warm) requests")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", default=None,
                    help="required honesty label for the report, e.g. "
                         "'vLLM A100-80GB' or 'Ollama on Apple M-series (Metal)'")
    args = ap.parse_args()

    client = httpx.Client()
    try:
        res = client.get(f"{args.base_url}/models", timeout=5.0)
        res.raise_for_status()
        served = [m["id"] for m in res.json().get("data", [])]
    except Exception as exc:
        print(f"ERROR: endpoint {args.base_url} unreachable: {exc}", file=sys.stderr)
        print("No benchmark numbers produced. (This is expected on the macOS "
              "dev machine: vLLM cannot run here.)", file=sys.stderr)
        return 2

    model = args.model or (served[0] if served else None)
    if not model:
        print("ERROR: endpoint serves no models", file=sys.stderr)
        return 2

    label = args.label or f"UNLABELED endpoint at {args.base_url}"
    print(f"# Benchmark: {model} @ {args.base_url}")
    print(f"# Hardware label: {label}")

    print("## Cold request (first after client startup)")
    cold = stream_once(client, args.base_url, model, PROMPTS[0], args.max_tokens)
    print(f"  ttft={cold['ttft_s'] * 1000:.0f}ms e2e={cold['e2e_s']:.2f}s "
          f"tokens/s={cold['tokens_per_s']:.1f}")

    print(f"## Warm requests (n={args.n})")
    rows = []
    for i in range(args.n):
        r = stream_once(client, args.base_url, model,
                        PROMPTS[i % len(PROMPTS)], args.max_tokens)
        rows.append(r)
        print(f"  [{i + 1:>3}/{args.n}] ttft={r['ttft_s'] * 1000:.0f}ms "
              f"e2e={r['e2e_s']:.2f}s tok/s={r['tokens_per_s']:.1f}")

    ttfts = [r["ttft_s"] * 1000 for r in rows]
    e2es = [r["e2e_s"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows]
    report = {
        "hardware_label": label,
        "model": model,
        "base_url": args.base_url,
        "n_warm": args.n,
        "cold": {k: round(v, 4) for k, v in cold.items()},
        "ttft_ms": {"p50": round(percentile(ttfts, 50), 1),
                    "p95": round(percentile(ttfts, 95), 1),
                    "p99": round(percentile(ttfts, 99), 1)},
        "e2e_s": {"p50": round(percentile(e2es, 50), 3),
                  "p95": round(percentile(e2es, 95), 3),
                  "p99": round(percentile(e2es, 99), 3)},
        "tokens_per_s": {"mean": round(statistics.mean(tps), 2),
                         "min": round(min(tps), 2), "max": round(max(tps), 2)},
        "warm_vs_cold_ttft_ratio": round(
            cold["ttft_s"] * 1000 / max(percentile(ttfts, 50), 1e-9), 2),
    }
    print("\n## Summary (JSON)")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
