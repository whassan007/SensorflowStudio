"""vLLM compatibility validation chain.

Chain order (each link depends on the previous):
    GPU -> driver -> torch -> vLLM -> model -> quantization

Every link produces a CompatCheck with PASS / FAIL / SKIPPED (blocked by an
earlier failure) plus a human-readable reason and remediation. On this
project's development machine (macOS, no CUDA/ROCm) the chain fails honestly
at the GPU link and reports vLLM as unsupported.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from sensorflow.retro.inference.env_detect import EnvironmentReport, detect_environment

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"

# Quantization methods vLLM supports on CUDA; ROCm support is narrower.
# See docs/retro/compatibility-matrix.md for the substantiated matrix.
CUDA_QUANTIZATIONS = {"awq", "gptq", "fp8", "int8", "none"}
ROCM_QUANTIZATIONS = {"fp8", "gptq", "none"}  # verify against current vLLM release

# Very rough VRAM floor (GB) per model family for a usable context window.
MODEL_VRAM_FLOOR_GB = {
    "7b": 16, "8b": 18, "13b": 28, "14b": 30, "32b": 68, "70b": 140,
}


class CompatCheck(BaseModel):
    link: str            # gpu | driver | torch | vllm | model | quantization
    status: str          # PASS | FAIL | SKIPPED
    reason: str
    remediation: Optional[str] = None


class CompatReport(BaseModel):
    vllm_supported: bool
    platform_summary: str
    checks: List[CompatCheck]
    failed_link: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


def _estimate_vram_floor(model_name: str) -> Optional[int]:
    lname = model_name.lower()
    for size, gb in MODEL_VRAM_FLOOR_GB.items():
        if size in lname:
            return gb
    return None


def check_vllm_compatibility(env: Optional[EnvironmentReport] = None,
                             model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                             quantization: str = "none") -> CompatReport:
    """Run the full GPU->driver->torch->vLLM->model->quantization chain."""
    env = env or detect_environment(probe_ollama=False)
    checks: List[CompatCheck] = []
    failed: Optional[str] = None

    def blocked(link: str, by: str) -> None:
        checks.append(CompatCheck(
            link=link, status=SKIPPED,
            reason=f"not evaluated: blocked by failed '{by}' check"))

    # --- 1. GPU -------------------------------------------------------------
    cuda_gpus = [g for g in env.gpus if g.vendor == "nvidia"]
    rocm_gpus = [g for g in env.gpus if g.vendor == "amd"]
    gpu_vendor = "nvidia" if cuda_gpus else ("amd" if rocm_gpus else None)

    if gpu_vendor:
        gpu = (cuda_gpus or rocm_gpus)[0]
        checks.append(CompatCheck(
            link="gpu", status=PASS,
            reason=f"{gpu.vendor.upper()} GPU detected: {gpu.model}"
                   + (f" ({gpu.memory_mb} MB VRAM)" if gpu.memory_mb else "")))
    else:
        failed = "gpu"
        apple = next((g for g in env.gpus if g.vendor == "apple"), None)
        if env.is_macos:
            reason = (f"No CUDA or ROCm capable GPU. This is macOS "
                      f"({env.machine_arch})"
                      + (f" with an Apple GPU ({apple.model}), which exposes "
                         f"Metal/MPS only" if apple else "")
                      + ". vLLM requires NVIDIA CUDA or AMD ROCm and does not "
                        "run on macOS.")
            remediation = ("Run the vLLM server on a Linux host with an NVIDIA "
                           "or AMD data-center/consumer GPU; use the 'ollama' "
                           "backend for local inference on this machine.")
        else:
            reason = "No NVIDIA or AMD GPU detected (nvidia-smi / rocm-smi absent or empty)."
            remediation = "Install GPU drivers or move to a GPU host."
        checks.append(CompatCheck(link="gpu", status=FAIL, reason=reason,
                                  remediation=remediation))

    # --- 2. Driver ----------------------------------------------------------
    if failed:
        blocked("driver", failed)
    elif gpu_vendor == "nvidia":
        drv = cuda_gpus[0].driver_version
        if drv:
            checks.append(CompatCheck(link="driver", status=PASS,
                                      reason=f"NVIDIA driver {drv} reported by nvidia-smi"))
        else:
            failed = "driver"
            checks.append(CompatCheck(
                link="driver", status=FAIL,
                reason="nvidia-smi present but did not report a driver version",
                remediation="Reinstall the NVIDIA driver (>=535 recommended for "
                            "current vLLM/CUDA 12 wheels)."))
    else:  # amd
        if env.rocm_smi_present:
            checks.append(CompatCheck(link="driver", status=PASS,
                                      reason="ROCm stack present (rocm-smi responds)"))
        else:
            failed = "driver"
            checks.append(CompatCheck(link="driver", status=FAIL,
                                      reason="AMD GPU without a working ROCm install",
                                      remediation="Install ROCm >= 6.x for the target GPU."))

    # --- 3. torch -----------------------------------------------------------
    if failed:
        blocked("torch", failed)
    elif not env.torch.installed:
        failed = "torch"
        checks.append(CompatCheck(
            link="torch", status=FAIL,
            reason=f"PyTorch is not installed in this environment ({env.torch.detail})",
            remediation="pip install a torch build matching the GPU stack "
                        "(cu12x wheels for CUDA; rocm wheels for AMD)."))
    elif gpu_vendor == "nvidia" and not env.torch.cuda_available:
        failed = "torch"
        checks.append(CompatCheck(
            link="torch", status=FAIL,
            reason=f"torch {env.torch.version} installed but torch.cuda.is_available() "
                   "is False (CPU-only build or driver mismatch)",
            remediation="Install a CUDA-enabled torch wheel matching the driver."))
    elif gpu_vendor == "amd" and not env.torch.rocm_version:
        failed = "torch"
        checks.append(CompatCheck(
            link="torch", status=FAIL,
            reason=f"torch {env.torch.version} is not a ROCm build",
            remediation="Install the ROCm torch wheel (torch.version.hip must be set)."))
    else:
        checks.append(CompatCheck(
            link="torch", status=PASS,
            reason=f"torch {env.torch.version} with a matching accelerator build"))

    # --- 4. vLLM ------------------------------------------------------------
    if failed:
        blocked("vllm", failed)
    elif not env.vllm_installed:
        failed = "vllm"
        checks.append(CompatCheck(
            link="vllm", status=FAIL,
            reason="vllm package is not installed",
            remediation="pip install vllm (CUDA) or build the ROCm image; see "
                        "docs/retro/compatibility-matrix.md."))
    else:
        checks.append(CompatCheck(link="vllm", status=PASS,
                                  reason=f"vllm {env.vllm_version} importable"))

    # --- 5. Model fits ------------------------------------------------------
    if failed:
        blocked("model", failed)
    else:
        floor = _estimate_vram_floor(model_name)
        vram_mb = (cuda_gpus or rocm_gpus)[0].memory_mb
        if floor is None or vram_mb is None:
            checks.append(CompatCheck(
                link="model", status=PASS,
                reason=f"cannot pre-validate VRAM for '{model_name}' "
                       "(unknown size class or unreported VRAM); vLLM will "
                       "validate at load time"))
        elif vram_mb / 1024 < floor:
            failed = "model"
            checks.append(CompatCheck(
                link="model", status=FAIL,
                reason=f"'{model_name}' needs roughly >={floor} GB VRAM unquantized; "
                       f"GPU reports {vram_mb / 1024:.0f} GB",
                remediation="Use quantization (AWQ/GPTQ/FP8), a smaller model, or "
                            "tensor parallelism across more GPUs."))
        else:
            checks.append(CompatCheck(
                link="model", status=PASS,
                reason=f"'{model_name}' plausibly fits in {vram_mb / 1024:.0f} GB VRAM"))

    # --- 6. Quantization ----------------------------------------------------
    q = (quantization or "none").lower()
    if failed:
        blocked("quantization", failed)
    else:
        supported = CUDA_QUANTIZATIONS if gpu_vendor == "nvidia" else ROCM_QUANTIZATIONS
        if q in supported:
            checks.append(CompatCheck(
                link="quantization", status=PASS,
                reason=f"quantization '{q}' is supported on {gpu_vendor} "
                       "(verify against the installed vLLM release)"))
        else:
            failed = "quantization"
            checks.append(CompatCheck(
                link="quantization", status=FAIL,
                reason=f"quantization '{q}' is not supported on {gpu_vendor} "
                       f"(supported: {sorted(supported)})",
                remediation="Pick a supported method or run unquantized."))

    if env.is_macos:
        summary = (f"macOS {env.os_version} ({env.machine_arch}): vLLM UNSUPPORTED "
                   "on this machine. Use the ollama or mock backend locally; run "
                   "vLLM on a CUDA/ROCm Linux host.")
    elif failed:
        summary = f"vLLM unsupported here: chain failed at '{failed}'."
    else:
        summary = "All compatibility checks passed; vLLM should run on this host."

    return CompatReport(
        vllm_supported=failed is None,
        platform_summary=summary,
        checks=checks,
        failed_link=failed,
        notes=list(env.notes),
    )


def format_report(report: CompatReport) -> str:
    lines = [report.platform_summary, ""]
    for c in report.checks:
        mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIPPED": "[SKIP]"}[c.status]
        lines.append(f"{mark} {c.link:<13} {c.reason}")
        if c.remediation:
            lines.append(f"       remediation: {c.remediation}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(check_vllm_compatibility()))
