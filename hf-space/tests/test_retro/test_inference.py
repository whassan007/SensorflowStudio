"""Phase 1 tests: environment detection, compat chain, backend client."""

from __future__ import annotations

import platform

import pytest

from sensorflow.retro.inference.client import (MockBackend, VALIDATION_EXPECT,
                                               all_backend_statuses,
                                               get_backend)
from sensorflow.retro.inference.compat import (FAIL, PASS, SKIPPED,
                                               check_vllm_compatibility,
                                               format_report)
from sensorflow.retro.inference.env_detect import detect_environment

IS_MACOS = platform.system() == "Darwin"


# ------------------------------------------------------------ env detection

def test_env_detection_reports_this_machine_honestly():
    env = detect_environment(probe_ollama=False)
    assert env.os_name == platform.system()
    assert env.machine_arch == platform.machine()
    assert env.python_version.count(".") == 2
    if IS_MACOS:
        assert env.is_macos
        # No CUDA/ROCm stack exists on macOS — detection must say so.
        assert not env.nvidia_smi_present
        assert not env.rocm_smi_present
        assert not any(g.vendor in ("nvidia", "amd") for g in env.gpus)
        assert any("vLLM does not run on macOS" in n for n in env.notes)


def test_env_detection_never_fabricates_torch():
    env = detect_environment(probe_ollama=False)
    try:
        import torch  # noqa: F401
        assert env.torch.installed
    except ImportError:
        assert not env.torch.installed
        assert "not importable" in env.torch.detail


# --------------------------------------------------------------- compat chain

@pytest.mark.skipif(not IS_MACOS, reason="assertion set is for the macOS dev host")
def test_vllm_reported_unsupported_on_this_machine():
    report = check_vllm_compatibility()
    assert report.vllm_supported is False
    assert report.failed_link == "gpu"
    assert "UNSUPPORTED" in report.platform_summary
    gpu = next(c for c in report.checks if c.link == "gpu")
    assert gpu.status == FAIL
    assert "vLLM requires NVIDIA CUDA or AMD ROCm" in gpu.reason
    assert gpu.remediation and "ollama" in gpu.remediation


def test_compat_chain_failure_clarity_blocked_links():
    """Every link after the first failure is SKIPPED with a clear blocked-by
    message — the chain never pretends to evaluate unreachable links."""
    report = check_vllm_compatibility()
    if report.vllm_supported:  # only on a real GPU host
        assert all(c.status == PASS for c in report.checks)
        return
    seen_fail = False
    for check in report.checks:
        if check.status == FAIL and not seen_fail:
            seen_fail = True
            assert check.reason  # human-readable
            continue
        if seen_fail:
            assert check.status == SKIPPED
            assert "blocked by failed" in check.reason
    text = format_report(report)
    assert "[FAIL]" in text and "remediation:" in text


def test_compat_chain_covers_all_links_in_order():
    report = check_vllm_compatibility()
    assert [c.link for c in report.checks] == [
        "gpu", "driver", "torch", "vllm", "model", "quantization"]


# ------------------------------------------------------------- client contract

def test_mock_backend_contract():
    b = get_backend("mock")
    assert isinstance(b, MockBackend)
    st = b.health()
    assert st.available and st.backend == "mock"

    v = b.validate()
    assert v.passed and v.got == VALIDATION_EXPECT
    assert v.prompt_tokens > 0 and v.completion_tokens > 0
    assert v.latency_ms >= 0

    # deterministic: identical prompt -> identical reply and token counts
    r1 = b.generate("what happened in scenario X?")
    r2 = b.generate("what happened in scenario X?")
    assert r1.text == r2.text
    assert r1.prompt_tokens == r2.prompt_tokens
    assert r1.completion_tokens == r2.completion_tokens


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("gpt-in-the-sky")


def test_all_backend_statuses_never_raise():
    statuses = all_backend_statuses()
    assert {s.backend for s in statuses} == {"mock", "ollama", "vllm"}
    mock = next(s for s in statuses if s.backend == "mock")
    assert mock.available
    # ollama/vllm may or may not be reachable; whatever the answer, it must
    # carry a human-readable detail rather than an exception.
    for s in statuses:
        assert isinstance(s.detail, str)
