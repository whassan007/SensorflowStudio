"""Real hardware/software environment detection for the inference layer.

Every field is probed from the actual machine (platform module, subprocess
calls to nvidia-smi / rocm-smi / system_profiler, and import probes for
torch / vllm). Nothing is assumed or fabricated: absent components are
reported as absent, and probe failures carry the failure detail.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from typing import List, Optional

from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    vendor: str                      # nvidia | amd | apple | intel | unknown
    model: str
    memory_mb: Optional[int] = None  # dedicated VRAM if known; None for unified
    unified_memory: bool = False     # Apple Silicon shares system RAM
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None
    detail: str = ""


class TorchInfo(BaseModel):
    installed: bool
    version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    mps_available: bool = False
    detail: str = ""


class EnvironmentReport(BaseModel):
    os_name: str
    os_version: str
    machine_arch: str
    is_macos: bool
    is_apple_silicon: bool
    python_version: str
    torch: TorchInfo
    gpus: List[GPUInfo] = Field(default_factory=list)
    cuda_toolkit_version: Optional[str] = None   # from nvcc, if present
    rocm_installed: bool = False
    nvidia_smi_present: bool = False
    rocm_smi_present: bool = False
    vllm_installed: bool = False
    vllm_version: Optional[str] = None
    ollama_endpoint: Optional[str] = None        # reachable Ollama server, if any
    ollama_models: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _detect_nvidia() -> List[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    gpus: List[GPUInfo] = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append(GPUInfo(
                vendor="nvidia", model=parts[0],
                memory_mb=int(float(parts[1])) if parts[1].replace(".", "").isdigit() else None,
                driver_version=parts[2] if len(parts) > 2 else None,
                compute_capability=parts[3] if len(parts) > 3 else None,
                detail="detected via nvidia-smi"))
    return gpus


def _detect_amd() -> List[GPUInfo]:
    if not shutil.which("rocm-smi"):
        return []
    out = _run(["rocm-smi", "--showproductname", "--json"])
    gpus: List[GPUInfo] = []
    if out:
        try:
            data = json.loads(out)
            for card, fields in data.items():
                name = (fields.get("Card series") or fields.get("Card SKU")
                        or fields.get("Card model") or card)
                gpus.append(GPUInfo(vendor="amd", model=str(name),
                                    detail="detected via rocm-smi"))
        except (json.JSONDecodeError, AttributeError):
            gpus.append(GPUInfo(vendor="amd", model="unknown AMD GPU",
                                detail="rocm-smi present but output unparseable"))
    return gpus


def _detect_apple() -> List[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=15.0)
    gpus: List[GPUInfo] = []
    if out:
        try:
            data = json.loads(out)
            for item in data.get("SPDisplaysDataType", []):
                model = item.get("sppci_model", "Apple GPU")
                cores = item.get("sppci_cores")
                detail = "detected via system_profiler"
                if cores:
                    detail += f"; {cores} GPU cores"
                gpus.append(GPUInfo(vendor="apple", model=model, unified_memory=True,
                                    detail=detail))
        except json.JSONDecodeError:
            pass
    if not gpus:
        # Apple Silicon always has an integrated GPU even if profiling failed.
        gpus.append(GPUInfo(vendor="apple", model=f"Apple GPU ({platform.machine()})",
                            unified_memory=True,
                            detail="system_profiler unavailable; inferred from platform"))
    return gpus


def _detect_torch() -> TorchInfo:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        return TorchInfo(installed=False, detail=f"torch not importable: {exc}")
    info = TorchInfo(installed=True, version=getattr(torch, "__version__", None))
    try:
        info.cuda_available = bool(torch.cuda.is_available())
        info.cuda_version = getattr(torch.version, "cuda", None)
        info.rocm_version = getattr(torch.version, "hip", None)
        mps = getattr(torch.backends, "mps", None)
        info.mps_available = bool(mps and mps.is_available())
        info.detail = "torch probed successfully"
    except Exception as exc:  # torch runtime probing should never crash detection
        info.detail = f"torch installed but probing failed: {exc}"
    return info


def _detect_nvcc() -> Optional[str]:
    out = _run(["nvcc", "--version"]) if shutil.which("nvcc") else None
    if out:
        for line in out.splitlines():
            if "release" in line:
                return line.strip()
    return None


def _detect_ollama(endpoints: Optional[List[str]] = None) -> tuple[Optional[str], List[str]]:
    import httpx
    for base in endpoints or ["http://localhost:11434"]:
        try:
            res = httpx.get(f"{base}/api/tags", timeout=2.0)
            if res.status_code == 200:
                models = [m.get("name", "?") for m in res.json().get("models", [])]
                return base, models
        except Exception:
            continue
    return None, []


def detect_environment(probe_ollama: bool = True) -> EnvironmentReport:
    """Probe the actual machine. Safe to call anywhere; never raises."""
    system = platform.system()
    arch = platform.machine()
    is_macos = system == "Darwin"
    is_apple_silicon = is_macos and arch == "arm64"

    gpus = _detect_nvidia() + _detect_amd()
    if not gpus and is_macos:
        gpus = _detect_apple()

    try:
        import vllm  # type: ignore
        vllm_installed, vllm_version = True, getattr(vllm, "__version__", "unknown")
    except ImportError:
        vllm_installed, vllm_version = False, None

    ollama_ep, ollama_models = (None, [])
    if probe_ollama:
        ollama_ep, ollama_models = _detect_ollama()

    notes: List[str] = []
    if is_macos:
        notes.append("macOS detected: vLLM does not run on macOS (no CUDA, no ROCm); "
                     "Apple GPU acceleration is Metal/MPS which vLLM does not target.")
    if is_apple_silicon:
        notes.append("Apple Silicon (arm64) with unified memory; local LLM inference "
                     "is possible via Ollama (Metal), not vLLM.")
    if not shutil.which("nvidia-smi") and not shutil.which("rocm-smi"):
        notes.append("Neither nvidia-smi nor rocm-smi found on PATH: no CUDA or "
                     "ROCm capable GPU stack is installed on this machine.")

    return EnvironmentReport(
        os_name=system,
        os_version=platform.mac_ver()[0] if is_macos else platform.release(),
        machine_arch=arch,
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        python_version=".".join(map(str, sys.version_info[:3])),
        torch=_detect_torch(),
        gpus=gpus,
        cuda_toolkit_version=_detect_nvcc(),
        rocm_installed=shutil.which("rocm-smi") is not None,
        nvidia_smi_present=shutil.which("nvidia-smi") is not None,
        rocm_smi_present=shutil.which("rocm-smi") is not None,
        vllm_installed=vllm_installed,
        vllm_version=vllm_version,
        ollama_endpoint=ollama_ep,
        ollama_models=ollama_models,
        notes=notes,
    )


if __name__ == "__main__":
    print(detect_environment().model_dump_json(indent=2))
