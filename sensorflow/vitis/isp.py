"""Composable ISP pipeline over both acceleration backends.

Pipeline (stage order is configurable, this is the default):

    bad_pixel_correction -> demosaic -> hdr_tone_map -> gain -> denoise -> resize

Each frame is degraded into a defective RAW Bayer mosaic (bad pixels, noise,
simulated over/under-exposure), then reconstructed by the ISP running on the
reference backend and on the Vitis emulated backend. Per-stage PSNR/SSIM of
the emulated output against the reference output localizes exactly where
fixed-point/streaming/LUT constraints bite.

Throughput report: measured CPU wall time per stage (reference backend) vs
the emulated backend's MODELED FPGA latency per stage. FPGA numbers are
analytical (pixels/cycle x clock), clearly labeled modeled-not-measured.
On real hardware the dataflow pipeline overlaps stages, so modeled
end-to-end fps = 1 / max(stage latency), also reported.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import ndimage

from sensorflow.bevfusion.scenes import generate_sequences
from sensorflow.vitis.backend import (
    PipelineConfig, VisionBackend, get_backend, make_bayer_mosaic,
)
from sensorflow.vitis.png import png_data_uri
from sensorflow.vitis.render import render_bev_rgb
from sensorflow.vitis.store import new_id, save_run

DEFAULT_STAGES = ["bad_pixel_correction", "demosaic", "hdr_tone_map",
                  "gain", "denoise", "resize"]


# --------------------------------------------------------------------------
# Quality metrics
# --------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(min(99.0, 10.0 * np.log10(1.0 / mse)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM with gaussian windows (sigma=1.5), luminance-only."""
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = ndimage.gaussian_filter(a, 1.5)
    mu_b = ndimage.gaussian_filter(b, 1.5)
    va = ndimage.gaussian_filter(a * a, 1.5) - mu_a ** 2
    vb = ndimage.gaussian_filter(b * b, 1.5) - mu_b ** 2
    cov = ndimage.gaussian_filter(a * b, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2) /
         ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))
    return float(np.clip(s.mean(), -1.0, 1.0))


# --------------------------------------------------------------------------
# RAW degradation (shared source for both backends)
# --------------------------------------------------------------------------

def make_defective_raw(rgb: np.ndarray, seed: int,
                       bad_pixel_fraction: float = 0.003,
                       underexposure: float = 0.55) -> np.ndarray:
    """RGB -> underexposed noisy RGGB mosaic with stuck/dead pixels."""
    rng = np.random.default_rng(seed)
    raw = make_bayer_mosaic(np.clip(rgb * underexposure, 0.0, 1.0))
    raw = raw + rng.normal(0.0, 0.01, raw.shape).astype(np.float32)
    n_bad = int(raw.size * bad_pixel_fraction)
    ys = rng.integers(0, raw.shape[0], n_bad)
    xs = rng.integers(0, raw.shape[1], n_bad)
    raw[ys, xs] = rng.choice([0.0, 1.0], n_bad).astype(np.float32)
    return np.clip(raw, 0.0, 1.0)


# --------------------------------------------------------------------------
# Stage implementations (backend-parameterized)
# --------------------------------------------------------------------------

def _stage_fns(params: Dict) -> Dict[str, Callable[[VisionBackend, np.ndarray], np.ndarray]]:
    gain = float(params.get("gain", 1.8))
    denoise = params.get("denoise", "gaussian")
    sigma = float(params.get("denoise_sigma", 0.8))
    out_h = int(params.get("out_h", 96))
    out_w = int(params.get("out_w", 128))
    white_point = float(params.get("white_point", 4.0))
    return {
        "bad_pixel_correction": lambda be, x: be.bad_pixel_correction(x, threshold=0.18),
        "demosaic": lambda be, x: be.demosaic(x),
        "hdr_tone_map": lambda be, x: be.hdr_tone_map(x, white_point=white_point),
        "gain": lambda be, x: be.gain_exposure(x, gain=gain, offset=0.0),
        "denoise": (lambda be, x: be.median_filter(x, ksize=3)) if denoise == "median"
                   else (lambda be, x: be.gaussian_filter(x, sigma=sigma)),
        "resize": lambda be, x: be.resize(x, out_h, out_w, method="area"),
    }


def run_isp(n_frames: int = 4, seed: int = 11,
            stages: Optional[List[str]] = None,
            stage_params: Optional[Dict] = None,
            width_bits: int = 12, int_bits: int = 4,
            max_line_buffer_depth: int = 2048, use_lut_approx: bool = True,
            lut_bits: int = 8, device: str = "versal-ai-edge",
            include_previews: bool = True, persist: bool = True) -> Dict:
    """Run the ISP over both backends; per-stage quality + throughput."""
    from sensorflow.vitis.backend import DeviceConfig
    t0 = time.perf_counter()
    stages = [s for s in (stages or DEFAULT_STAGES) if s in DEFAULT_STAGES]
    if not stages:
        raise ValueError("No valid ISP stages requested")
    fns = _stage_fns(stage_params or {})
    config = PipelineConfig(precision={"default": (width_bits, int_bits)},
                            max_line_buffer_depth=max_line_buffer_depth,
                            use_lut_approx=use_lut_approx, lut_bits=lut_bits,
                            device=DeviceConfig(name=device))
    ref = get_backend("reference")
    vit = get_backend("vitis_emulated", config)

    sequences = generate_sequences(n_sequences=max(1, (n_frames + 3) // 4),
                                   frames_per_sequence=4, seed=seed)
    frames = [(seq, fr) for seq in sequences for fr in seq.frames][:n_frames]

    stage_quality: Dict[str, Dict[str, List[float]]] = {
        s: {"psnr": [], "ssim": []} for s in stages}
    stage_cpu_ms: Dict[str, List[float]] = {s: [] for s in stages}
    stage_modeled: Dict[str, Dict] = {}
    previews: List[Dict] = []

    for fi, (seq, frame) in enumerate(frames):
        rgb = render_bev_rgb(frame, seq, seed)
        raw = make_defective_raw(rgb, seed * 7919 + fi)
        x_ref: np.ndarray = raw
        x_vit: np.ndarray = raw
        frame_preview = {"frame_id": frame.frame_id,
                         "cohort": f"{seq.time_of_day}/{seq.weather}",
                         "stages": []} if (include_previews and fi == 0) else None
        for s in stages:
            ref.reset_profile()
            vit.reset_profile()
            x_ref = fns[s](ref, x_ref)
            x_vit = fns[s](vit, x_vit)
            stage_quality[s]["psnr"].append(psnr(x_ref, x_vit))
            stage_quality[s]["ssim"].append(ssim(x_ref, x_vit))
            stage_cpu_ms[s].append(sum(r.measured_cpu_ms for r in ref.profile))
            modeled = [r.modeled for r in vit.profile if r.modeled]
            if modeled:
                stage_modeled[s] = {
                    "modeled_not_measured": True,
                    "latency_ms": round(sum(m["latency_ms"] for m in modeled), 4),
                    "placement": modeled[-1]["placement"],
                    "clock_mhz": modeled[-1]["clock_mhz"],
                    "throughput_mpix_per_s": modeled[-1]["throughput_mpix_per_s"],
                }
            if frame_preview is not None:
                frame_preview["stages"].append({
                    "stage": s,
                    "reference_png": png_data_uri(x_ref),
                    "vitis_png": png_data_uri(x_vit),
                    "diff_png": png_data_uri(
                        np.clip(np.abs(x_ref.astype(np.float32) -
                                       x_vit.astype(np.float32)) * 8.0, 0, 1)),
                })
        if frame_preview is not None:
            frame_preview["input_png"] = png_data_uri(raw)
            previews.append(frame_preview)

    stage_report = []
    for s in stages:
        cpu_ms = float(np.mean(stage_cpu_ms[s]))
        entry = {
            "stage": s,
            "psnr_db": round(float(np.mean(stage_quality[s]["psnr"])), 2),
            "ssim": round(float(np.mean(stage_quality[s]["ssim"])), 4),
            "measured_cpu_ms": round(cpu_ms, 3),
        }
        if s in stage_modeled:
            m = stage_modeled[s]
            entry["modeled_fpga_ms"] = m["latency_ms"]
            entry["modeled_placement"] = m["placement"]
            entry["modeled_speedup_x"] = round(cpu_ms / max(m["latency_ms"], 1e-9), 1)
            entry["modeled_not_measured"] = True
        stage_report.append(entry)

    total_cpu_ms = sum(e["measured_cpu_ms"] for e in stage_report)
    total_fpga_ms = sum(e.get("modeled_fpga_ms", 0.0) for e in stage_report)
    max_stage_ms = max((e.get("modeled_fpga_ms", 0.0) for e in stage_report),
                       default=0.0)
    throughput = {
        "measured_cpu_ms_per_frame": round(total_cpu_ms, 3),
        "measured_cpu_fps": round(1000.0 / max(total_cpu_ms, 1e-9), 1),
        "modeled_fpga_ms_per_frame_serial": round(total_fpga_ms, 4),
        "modeled_fpga_fps_pipelined": round(1000.0 / max(max_stage_ms, 1e-9), 1),
        "modeled_speedup_x_serial": round(total_cpu_ms / max(total_fpga_ms, 1e-9), 1),
        "modeled_not_measured": True,
        "note": ("FPGA numbers are analytically modeled from pixels/cycle "
                 "and clock frequency, NOT measured on hardware. Pipelined "
                 "fps assumes HLS dataflow overlap (1/max stage latency)."),
    }

    run_id = new_id("isprun")
    payload = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "isp",
        "params": {"n_frames": len(frames), "seed": seed, "stages": stages,
                   "stage_params": stage_params or {},
                   "width_bits": width_bits, "int_bits": int_bits,
                   "max_line_buffer_depth": max_line_buffer_depth,
                   "use_lut_approx": use_lut_approx, "lut_bits": lut_bits,
                   "device": device},
        "config": config.to_dict(),
        "stage_report": stage_report,
        "throughput": throughput,
        "previews": previews,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": {
            "min_stage_psnr_db": min(e["psnr_db"] for e in stage_report),
            "modeled_speedup_x_serial": throughput["modeled_speedup_x_serial"],
            "stages": stages,
        },
        "emulation_note": ("vitis_emulated is a constraint-faithful CPU "
                           "emulator; FPGA throughput is modeled, not measured."),
    }
    if persist:
        save_run("isp", run_id, payload)
    return payload
