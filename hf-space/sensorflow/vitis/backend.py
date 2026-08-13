"""Pluggable vision-acceleration backend abstraction.

Two working implementations of the `VisionBackend` interface:

ReferenceCPUBackend
    Plain float32 numpy reference implementations. This is the ground truth
    every other backend is compared against.

VitisEmulatedBackend
    The SAME operators, but executed under faithful models of the constraints
    a Vitis Vision / FPGA implementation imposes:

    * Fixed-point quantization — every op's intermediates and outputs are
      quantized to a configurable ap_fixed<W,I>-style format (signed, W total
      bits, I integer bits) with SATURATION on overflow (AP_SAT) and
      TRUNCATION toward negative infinity on rounding (AP_TRN), matching the
      HLS ap_fixed defaults.
    * Line-buffer / streaming depth — Vitis Vision kernels stream rows
      through line buffers whose width is bounded by XFCVDEPTH. When the
      image width exceeds the configured depth, the emulator processes the
      image in independent vertical strips with NO halo exchange, which
      produces localized seam artifacts at strip boundaries — exactly the
      failure mode of an under-provisioned line buffer.
    * HLS-style approximations — divide and sqrt are implemented as
      2^lut_bits-entry lookup tables on the mantissa (bounded relative
      error ~= 2^-lut_bits), the way HLS designs avoid full dividers.
    * Latency/throughput model — a deterministic per-op model
      (pixels/cycle, clock MHz, PL vs AIE placement) produces simulated
      speedup numbers. These are MODELED, NOT MEASURED, and every report
      carries `"modeled_not_measured": True`.

"vitis_hw" stub
    A third backend name is registered but not constructible. A real
    hardware backend would subclass `VisionBackend`, drive xfOpenCV/Vitis
    Vision kernels through PYNQ overlays or XRT buffers, and return measured
    latencies instead of modeled ones. Nothing else in this package would
    change: callers only ever see the `VisionBackend` interface.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

KNOWN_DEVICES = {
    "versal-ai-edge": {
        "clock_mhz": 400.0,
        "has_aie": True,
        "description": "AMD Versal AI Edge (VE2302-class): PL + AI Engine array.",
    },
    "zynq-ultrascale": {
        "clock_mhz": 300.0,
        "has_aie": False,
        "description": "AMD Zynq UltraScale+ MPSoC (ZCU104-class): PL only.",
    },
}

# Per-op throughput model: pixels/cycle in PL, preferred placement, and a
# fixed pipeline-fill overhead in cycles. AIE placement (when the device has
# an AI Engine array) gets a vectorization multiplier on pixels/cycle.
OP_MODEL: Dict[str, Dict] = {
    "resize":               {"ppc": 2.0,  "placement": "PL",  "overhead": 4096},
    "crop":                 {"ppc": 4.0,  "placement": "PL",  "overhead": 1024},
    "demosaic":             {"ppc": 1.0,  "placement": "PL",  "overhead": 8192},
    "rgb_to_yuv":           {"ppc": 2.0,  "placement": "PL",  "overhead": 2048},
    "yuv_to_rgb":           {"ppc": 2.0,  "placement": "PL",  "overhead": 2048},
    "bad_pixel_correction": {"ppc": 1.0,  "placement": "PL",  "overhead": 8192},
    "hdr_merge":            {"ppc": 1.0,  "placement": "PL",  "overhead": 8192},
    "hdr_tone_map":         {"ppc": 1.0,  "placement": "PL",  "overhead": 4096},
    "gain_exposure":        {"ppc": 4.0,  "placement": "PL",  "overhead": 1024},
    "lens_distortion":      {"ppc": 0.5,  "placement": "PL",  "overhead": 16384},
    "gaussian_filter":      {"ppc": 1.0,  "placement": "AIE", "overhead": 8192},
    "median_filter":        {"ppc": 0.5,  "placement": "PL",  "overhead": 8192},
    "optical_flow":         {"ppc": 0.25, "placement": "AIE", "overhead": 65536},
    "stereo_block_match":   {"ppc": 0.125, "placement": "PL", "overhead": 65536},
}
AIE_VECTOR_SPEEDUP = 4.0  # AIE tiles process wider SIMD vectors per cycle.


@dataclass
class DeviceConfig:
    """Target device for the latency/throughput model."""

    name: str = "versal-ai-edge"
    clock_mhz: float = 0.0   # 0 -> take the device default
    has_aie: bool = False

    def __post_init__(self):
        if self.name not in KNOWN_DEVICES:
            raise ValueError(
                f"Unknown device {self.name!r}; known: {sorted(KNOWN_DEVICES)}")
        spec = KNOWN_DEVICES[self.name]
        if self.clock_mhz <= 0:
            self.clock_mhz = spec["clock_mhz"]
        self.has_aie = bool(spec["has_aie"])

    def to_dict(self) -> Dict:
        return {"name": self.name, "clock_mhz": self.clock_mhz,
                "has_aie": self.has_aie,
                "description": KNOWN_DEVICES[self.name]["description"]}


@dataclass
class PipelineConfig:
    """Constraint configuration shared by every op in one pipeline instance.

    precision maps op name -> (W, I) total/integer bits; the "default" entry
    applies to any op without its own entry. max_line_buffer_depth models
    XFCVDEPTH: the widest image a streaming kernel can hold rows of.
    """

    precision: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"default": (16, 6)})
    max_line_buffer_depth: int = 2048
    lut_bits: int = 8            # 2^lut_bits entries in divide/sqrt LUTs
    use_lut_approx: bool = True  # False -> exact divide/sqrt (still quantized)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    def precision_for(self, op: str) -> Tuple[int, int]:
        w, i = self.precision.get(op, self.precision.get("default", (16, 6)))
        if w <= i or w < 2:
            raise ValueError(f"Invalid ap_fixed<{w},{i}> for op {op!r}")
        return int(w), int(i)

    def to_dict(self) -> Dict:
        return {
            "precision": {k: list(v) for k, v in self.precision.items()},
            "max_line_buffer_depth": self.max_line_buffer_depth,
            "lut_bits": self.lut_bits,
            "use_lut_approx": self.use_lut_approx,
            "device": self.device.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "PipelineConfig":
        d = dict(d or {})
        precision = {k: tuple(v) for k, v in (d.get("precision") or
                                              {"default": (16, 6)}).items()}
        dev = d.get("device") or {}
        device = DeviceConfig(name=dev.get("name", "versal-ai-edge"),
                              clock_mhz=float(dev.get("clock_mhz", 0.0)))
        return cls(precision=precision,
                   max_line_buffer_depth=int(d.get("max_line_buffer_depth", 2048)),
                   lut_bits=int(d.get("lut_bits", 8)),
                   use_lut_approx=bool(d.get("use_lut_approx", True)),
                   device=device)


# --------------------------------------------------------------------------
# Numeric models: exact float vs fixed-point + LUT approximations
# --------------------------------------------------------------------------

class _FloatNumerics:
    """Exact float32 numerics for the reference backend."""

    def q(self, x: np.ndarray, op: str) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def recip(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        with np.errstate(divide="ignore"):
            return np.where(x != 0, 1.0 / np.where(x == 0, 1.0, x), 0.0)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(np.asarray(x, dtype=np.float32), 0.0))


class _FixedNumerics:
    """ap_fixed<W,I> quantization (AP_TRN + AP_SAT) plus LUT divide/sqrt."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        n = 1 << config.lut_bits
        # Mantissa m in [0.5, 1): LUTs indexed by the top lut_bits of m.
        m = 0.5 + (np.arange(n, dtype=np.float64) + 0.5) / (2.0 * n)
        self._recip_lut = (1.0 / m).astype(np.float32)
        self._sqrt_lut = np.sqrt(m).astype(np.float32)
        self._sqrt2 = np.float32(math.sqrt(2.0))

    def q(self, x: np.ndarray, op: str) -> np.ndarray:
        w, i = self.config.precision_for(op)
        frac = w - i
        scale = float(2 ** frac)
        lo = -float(2 ** (i - 1))
        hi = float(2 ** (i - 1)) - 1.0 / scale
        x = np.asarray(x, dtype=np.float64)
        # AP_SAT then AP_TRN (floor toward -inf at the fractional resolution).
        return (np.floor(np.clip(x, lo, hi) * scale) / scale).astype(np.float32)

    def _mantissa_index(self, m: np.ndarray) -> np.ndarray:
        n = 1 << self.config.lut_bits
        idx = ((m - 0.5) * 2.0 * n).astype(np.int64)
        return np.clip(idx, 0, n - 1)

    def recip(self, x: np.ndarray) -> np.ndarray:
        if not self.config.use_lut_approx:
            return _FloatNumerics().recip(x)
        x = np.asarray(x, dtype=np.float32)
        sign = np.sign(x)
        ax = np.abs(x)
        m, e = np.frexp(np.where(ax > 0, ax, 1.0))  # ax = m * 2^e, m in [0.5,1)
        r = self._recip_lut[self._mantissa_index(m)] * np.exp2(-e.astype(np.float32))
        return np.where(ax > 0, sign * r, 0.0).astype(np.float32)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        if not self.config.use_lut_approx:
            return _FloatNumerics().sqrt(x)
        x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
        m, e = np.frexp(np.where(x > 0, x, 1.0))
        root = self._sqrt_lut[self._mantissa_index(m)]
        half_e = e // 2
        odd = (e - 2 * half_e) != 0
        r = root * np.exp2(half_e.astype(np.float32)) * np.where(odd, self._sqrt2, 1.0)
        return np.where(x > 0, r, 0.0).astype(np.float32)


# --------------------------------------------------------------------------
# Op profiling records
# --------------------------------------------------------------------------

@dataclass
class OpReport:
    op: str
    pixels: int
    measured_cpu_ms: float
    modeled: Optional[Dict] = None  # only for the emulated backend

    def to_dict(self) -> Dict:
        d = {"op": self.op, "pixels": self.pixels,
             "measured_cpu_ms": round(self.measured_cpu_ms, 4)}
        if self.modeled is not None:
            d["modeled"] = self.modeled
        return d


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------

class VisionBackend:
    """Vision op interface every acceleration backend implements.

    All images are float32 numpy arrays scaled to [0, 1] unless noted:
    grayscale (H, W), color (H, W, 3), Bayer RAW (H, W) RGGB.
    """

    name = "abstract"
    is_emulated = False

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.profile: List[OpReport] = []
        self._numerics = _FloatNumerics()

    # -- profiling ---------------------------------------------------------

    def reset_profile(self) -> None:
        self.profile = []

    def profile_report(self) -> List[Dict]:
        return [r.to_dict() for r in self.profile]

    def _record(self, op: str, pixels: int, measured_ms: float) -> None:
        self.profile.append(OpReport(op=op, pixels=pixels,
                                     measured_cpu_ms=measured_ms))

    # -- helpers shared by both implementations -----------------------------

    def _run(self, op: str, img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray],
             stripable: bool = True) -> np.ndarray:
        """Execute `fn` under this backend's numeric + streaming model."""
        t0 = time.perf_counter()
        out = self._execute(op, img, fn, stripable)
        out = self._numerics.q(out, op)
        self._record(op, int(np.prod(img.shape[:2])),
                     (time.perf_counter() - t0) * 1000.0)
        return out

    def _execute(self, op: str, img: np.ndarray,
                 fn: Callable[[np.ndarray], np.ndarray],
                 stripable: bool) -> np.ndarray:
        return fn(np.asarray(img, dtype=np.float32))

    # -- ops -----------------------------------------------------------------

    def resize(self, img: np.ndarray, out_h: int, out_w: int,
               method: str = "bilinear") -> np.ndarray:
        if method not in ("bilinear", "bicubic", "area"):
            raise ValueError(f"Unknown resize method {method!r}")

        def _fn(x: np.ndarray) -> np.ndarray:
            h, w = x.shape[:2]
            if method == "area" and out_h <= h and out_w <= w:
                return _area_resize(x, out_h, out_w)
            order = 1 if method == "bilinear" else 3
            yy = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
            xx = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
            gy, gx = np.meshgrid(yy, xx, indexing="ij")
            if x.ndim == 2:
                return ndimage.map_coordinates(x, [gy, gx], order=order,
                                               mode="nearest").astype(np.float32)
            chans = [ndimage.map_coordinates(x[..., c], [gy, gx], order=order,
                                             mode="nearest")
                     for c in range(x.shape[2])]
            return np.stack(chans, axis=-1).astype(np.float32)

        return self._run("resize", img, _fn, stripable=False)

    def crop(self, img: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            return a[y:y + h, x:x + w].copy()
        return self._run("crop", img, _fn, stripable=False)

    def demosaic(self, raw: np.ndarray) -> np.ndarray:
        """Bilinear demosaic of an RGGB Bayer mosaic (H, W) -> (H, W, 3)."""
        def _fn(a: np.ndarray) -> np.ndarray:
            return _demosaic_bilinear(a)
        return self._run("demosaic", raw, _fn)

    def rgb_to_yuv(self, img: np.ndarray) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            m = np.array([[0.299, 0.587, 0.114],
                          [-0.14713, -0.28886, 0.436],
                          [0.615, -0.51499, -0.10001]], dtype=np.float32)
            return a @ m.T
        return self._run("rgb_to_yuv", img, _fn, stripable=False)

    def yuv_to_rgb(self, img: np.ndarray) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            m = np.array([[1.0, 0.0, 1.13983],
                          [1.0, -0.39465, -0.58060],
                          [1.0, 2.03211, 0.0]], dtype=np.float32)
            return np.clip(a @ m.T, 0.0, 1.0)
        return self._run("yuv_to_rgb", img, _fn, stripable=False)

    def bad_pixel_correction(self, img: np.ndarray,
                             threshold: float = 0.2) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            med = _median3(a)
            bad = np.abs(a - med) > threshold
            return np.where(bad, med, a)
        return self._run("bad_pixel_correction", img, _fn)

    def hdr_merge(self, exposures: List[np.ndarray],
                  exposure_times: List[float]) -> np.ndarray:
        """Weighted radiance merge of bracketed exposures (hat weighting)."""
        base = exposures[0]

        def _fn(_: np.ndarray) -> np.ndarray:
            num = np.zeros_like(np.asarray(base, dtype=np.float32))
            den = np.zeros_like(num)
            for e, t in zip(exposures, exposure_times):
                e = np.asarray(e, dtype=np.float32)
                wgt = 1.0 - np.abs(2.0 * e - 1.0)  # hat weight, peak mid-tone
                num += wgt * e / max(float(t), 1e-6)
                den += wgt
            den_r = self._numerics.recip(np.maximum(den, 1e-3))
            return num * den_r
        return self._run("hdr_merge", base, _fn)

    def hdr_tone_map(self, img: np.ndarray, white_point: float = 4.0) -> np.ndarray:
        """Extended Reinhard operator: x(1 + x/wp^2)/(1 + x)."""
        def _fn(a: np.ndarray) -> np.ndarray:
            wp2 = float(white_point) ** 2
            denom_r = self._numerics.recip(1.0 + a)
            return np.clip(a * (1.0 + a / wp2) * denom_r, 0.0, 1.0)
        return self._run("hdr_tone_map", img, _fn)

    def gain_exposure(self, img: np.ndarray, gain: float = 1.0,
                      offset: float = 0.0) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            return np.clip(a * float(gain) + float(offset), 0.0, 1.0)
        return self._run("gain_exposure", img, _fn)

    def lens_distortion(self, img: np.ndarray, k1: float, k2: float = 0.0,
                        mode: str = "apply") -> np.ndarray:
        """Radial (Brown) distortion. mode='apply' distorts, 'correct' undoes."""
        if mode not in ("apply", "correct"):
            raise ValueError(f"mode must be apply|correct, got {mode!r}")

        def _fn(a: np.ndarray) -> np.ndarray:
            return _radial_remap(a, k1, k2, invert=(mode == "correct"),
                                 numerics=self._numerics)
        return self._run("lens_distortion", img, _fn, stripable=False)

    def gaussian_filter(self, img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            axes = (0, 1)
            return ndimage.gaussian_filter(a, sigma=sigma, axes=axes,
                                           mode="nearest").astype(np.float32)
        return self._run("gaussian_filter", img, _fn)

    def median_filter(self, img: np.ndarray, ksize: int = 3) -> np.ndarray:
        def _fn(a: np.ndarray) -> np.ndarray:
            if a.ndim == 2:
                return ndimage.median_filter(a, size=ksize,
                                             mode="nearest").astype(np.float32)
            return ndimage.median_filter(a, size=(ksize, ksize, 1),
                                         mode="nearest").astype(np.float32)
        return self._run("median_filter", img, _fn)

    def optical_flow(self, prev: np.ndarray, nxt: np.ndarray, levels: int = 3,
                     window: int = 9, iterations: int = 3) -> np.ndarray:
        """Pyramidal iterative dense Lucas-Kanade flow (H, W, 2), (dx, dy)."""
        def _fn(a: np.ndarray) -> np.ndarray:
            return _dense_lk_flow(a, np.asarray(nxt, dtype=np.float32),
                                  levels=levels, window=window,
                                  iterations=iterations,
                                  numerics=self._numerics)
        return self._run("optical_flow", prev, _fn)

    def stereo_block_match(self, left: np.ndarray, right: np.ndarray,
                           max_disparity: int = 48,
                           block: int = 9) -> np.ndarray:
        """SAD block-matching disparity of grayscale pair -> (H, W) pixels."""
        def _fn(a: np.ndarray) -> np.ndarray:
            return _stereo_sad(a, np.asarray(right, dtype=np.float32),
                               max_disparity=max_disparity, block=block,
                               numerics=self._numerics)
        return self._run("stereo_block_match", left, _fn)


# --------------------------------------------------------------------------
# Reference backend
# --------------------------------------------------------------------------

class ReferenceCPUBackend(VisionBackend):
    """Float32 numpy reference. Ground truth for every parity comparison."""

    name = "reference"
    is_emulated = False


# --------------------------------------------------------------------------
# Emulated Vitis backend
# --------------------------------------------------------------------------

class VitisEmulatedBackend(VisionBackend):
    """CPU emulation of a Vitis Vision FPGA pipeline. See module docstring.

    HONESTY: this backend does NOT run on FPGA hardware. Its outputs model
    the numerics and streaming behavior of a Vitis implementation; its
    latency numbers are analytically modeled, never measured on silicon.
    """

    name = "vitis_emulated"
    is_emulated = True

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self._numerics = _FixedNumerics(self.config)

    def _execute(self, op: str, img: np.ndarray,
                 fn: Callable[[np.ndarray], np.ndarray],
                 stripable: bool) -> np.ndarray:
        img = self._numerics.q(np.asarray(img, dtype=np.float32), op)
        depth = self.config.max_line_buffer_depth
        width = img.shape[1] if img.ndim >= 2 else 0
        if stripable and width > depth:
            # Line buffer too narrow: process independent vertical strips
            # with no halo -> seam artifacts at strip boundaries.
            strips = [fn(img[:, x0:x0 + depth])
                      for x0 in range(0, width, depth)]
            return np.concatenate(strips, axis=1)
        return fn(img)

    def _record(self, op: str, pixels: int, measured_ms: float) -> None:
        model = OP_MODEL.get(op, {"ppc": 1.0, "placement": "PL", "overhead": 4096})
        placement = model["placement"]
        ppc = model["ppc"]
        if placement == "AIE" and not self.config.device.has_aie:
            placement = "PL"
        elif placement == "AIE":
            ppc *= AIE_VECTOR_SPEEDUP
        cycles = pixels / ppc + model["overhead"]
        modeled_ms = cycles / (self.config.device.clock_mhz * 1e3)
        self.profile.append(OpReport(
            op=op, pixels=pixels, measured_cpu_ms=measured_ms,
            modeled={
                "modeled_not_measured": True,
                "placement": placement,
                "pixels_per_cycle": ppc,
                "clock_mhz": self.config.device.clock_mhz,
                "cycles": int(cycles),
                "latency_ms": round(modeled_ms, 4),
                "throughput_mpix_per_s": round(pixels / max(modeled_ms, 1e-9) / 1e3, 2),
            }))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_VITIS_HW_DOC = (
    "Stub for a real Vitis/PYNQ/XRT hardware backend. To slot in real "
    "hardware: subclass VisionBackend, load the Vitis Vision overlay "
    "(pynq.Overlay / xrt.xclbin), implement each op by dispatching to the "
    "corresponding xf::cv kernel via XRT buffers, and report MEASURED "
    "latencies in _record. No caller-facing code changes are required."
)

BACKENDS: Dict[str, Dict] = {
    "reference": {
        "cls": ReferenceCPUBackend,
        "available": True,
        "emulated": False,
        "description": "Float32 numpy reference implementation (CPU).",
    },
    "vitis_emulated": {
        "cls": VitisEmulatedBackend,
        "available": True,
        "emulated": True,
        "description": (
            "CPU emulator of a Vitis Vision FPGA pipeline. Faithfully models "
            "ap_fixed<W,I> quantization (AP_TRN/AP_SAT), XFCVDEPTH line-buffer "
            "limits, LUT divide/sqrt, and a per-op latency model. NOT real "
            "hardware; all speedups are modeled, not measured."),
    },
    "vitis_hw": {
        "cls": None,
        "available": False,
        "emulated": False,
        "description": _VITIS_HW_DOC,
    },
}


def get_backend(name: str, config: Optional[PipelineConfig] = None) -> VisionBackend:
    entry = BACKENDS.get(name)
    if entry is None:
        raise ValueError(f"Unknown backend {name!r}; known: {sorted(BACKENDS)}")
    if not entry["available"]:
        raise NotImplementedError(
            f"Backend {name!r} is not available on this machine. {entry['description']}")
    return entry["cls"](config)


def list_backends() -> List[str]:
    return sorted(BACKENDS)


def backend_status() -> Dict:
    return {
        "backends": [
            {"name": name, "available": e["available"], "emulated": e["emulated"],
             "description": e["description"]}
            for name, e in sorted(BACKENDS.items())
        ],
        "devices": {k: dict(v) for k, v in KNOWN_DEVICES.items()},
        "hardware_present": False,
        "note": ("No FPGA hardware is attached. 'vitis_emulated' is a "
                 "constraint-faithful CPU emulator; all its speedup numbers "
                 "are modeled, not measured."),
    }


# --------------------------------------------------------------------------
# Shared op kernels (pure functions; numerics injected where divides occur)
# --------------------------------------------------------------------------

def _area_resize(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = x.shape[:2]
    if h % out_h == 0 and w % out_w == 0:
        fy, fx = h // out_h, w // out_w
        if x.ndim == 2:
            return x[:out_h * fy, :out_w * fx].reshape(
                out_h, fy, out_w, fx).mean(axis=(1, 3)).astype(np.float32)
        return x[:out_h * fy, :out_w * fx].reshape(
            out_h, fy, out_w, fx, -1).mean(axis=(1, 3)).astype(np.float32)
    # Non-integer factor: smooth then bilinear sample.
    sig = (max(h / out_h, 1.0) / 2.0, max(w / out_w, 1.0) / 2.0)
    sm = ndimage.gaussian_filter(x, sigma=sig if x.ndim == 2 else sig + (0,),
                                 mode="nearest")
    yy = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
    xx = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    if x.ndim == 2:
        return ndimage.map_coordinates(sm, [gy, gx], order=1,
                                       mode="nearest").astype(np.float32)
    return np.stack([ndimage.map_coordinates(sm[..., c], [gy, gx], order=1,
                                             mode="nearest")
                     for c in range(x.shape[2])], axis=-1).astype(np.float32)


def _median3(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return ndimage.median_filter(a, size=3, mode="nearest").astype(np.float32)
    return ndimage.median_filter(a, size=(3, 3, 1),
                                 mode="nearest").astype(np.float32)


_BAYER_R = "R"


def make_bayer_mosaic(rgb: np.ndarray) -> np.ndarray:
    """Sample an RGB image (H, W, 3) into an RGGB Bayer mosaic (H, W)."""
    h, w = rgb.shape[:2]
    raw = np.zeros((h, w), dtype=np.float32)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R
    raw[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G
    raw[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B
    return raw


def _demosaic_bilinear(raw: np.ndarray) -> np.ndarray:
    """Bilinear demosaic of RGGB via mask-normalized convolution."""
    h, w = raw.shape
    rm = np.zeros((h, w), dtype=np.float32)
    gm = np.zeros((h, w), dtype=np.float32)
    bm = np.zeros((h, w), dtype=np.float32)
    rm[0::2, 0::2] = 1.0
    gm[0::2, 1::2] = 1.0
    gm[1::2, 0::2] = 1.0
    bm[1::2, 1::2] = 1.0
    k_rb = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]],
                    dtype=np.float32)
    k_g = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]],
                   dtype=np.float32)

    def interp(mask, kernel):
        num = ndimage.convolve(raw * mask, kernel, mode="nearest")
        den = ndimage.convolve(mask, kernel, mode="nearest")
        return num / np.maximum(den, 1e-6)

    return np.clip(np.stack([interp(rm, k_rb), interp(gm, k_g),
                             interp(bm, k_rb)], axis=-1), 0.0, 1.0
                   ).astype(np.float32)


def _radial_remap(a: np.ndarray, k1: float, k2: float, invert: bool,
                  numerics) -> np.ndarray:
    h, w = a.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    norm = max(cy, cx)
    yy, xx = np.meshgrid((np.arange(h) - cy) / norm,
                         (np.arange(w) - cx) / norm, indexing="ij")
    r2 = yy * yy + xx * xx
    if invert:
        # Undoing distortion needs the inverse radial map; approximate with
        # fixed-point Newton iterations (each divide goes through numerics).
        scale = np.ones_like(r2, dtype=np.float32)
        for _ in range(3):
            rs2 = r2 * scale * scale
            f = 1.0 + k1 * rs2 + k2 * rs2 * rs2
            scale = numerics.recip(np.maximum(f, 1e-3)).astype(np.float32)
        factor = 1.0 / np.maximum(scale, 1e-6)
    else:
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
    src_y = yy * factor * norm + cy
    src_x = xx * factor * norm + cx
    if a.ndim == 2:
        return ndimage.map_coordinates(a, [src_y, src_x], order=1,
                                       mode="nearest").astype(np.float32)
    return np.stack([ndimage.map_coordinates(a[..., c], [src_y, src_x],
                                             order=1, mode="nearest")
                     for c in range(a.shape[2])], axis=-1).astype(np.float32)


def _to_gray(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        return (0.299 * a[..., 0] + 0.587 * a[..., 1] +
                0.114 * a[..., 2]).astype(np.float32)
    return a.astype(np.float32)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = img.shape
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    return ndimage.map_coordinates(img, [gy + flow[..., 1], gx + flow[..., 0]],
                                   order=1, mode="nearest").astype(np.float32)


def _dense_lk_flow(prev: np.ndarray, nxt: np.ndarray, levels: int, window: int,
                   iterations: int, numerics) -> np.ndarray:
    """Coarse-to-fine iterative dense Lucas-Kanade. Returns (H, W, 2) (dx, dy).

    The 2x2 normal-equation solve per pixel requires a divide by the
    determinant; that divide goes through the backend numerics (exact float
    for reference, LUT reciprocal for the Vitis emulator).
    """
    p = _to_gray(prev)
    n = _to_gray(nxt)
    pyr_p, pyr_n = [p], [n]
    for _ in range(levels - 1):
        p = ndimage.gaussian_filter(p, 1.0, mode="nearest")[::2, ::2]
        n = ndimage.gaussian_filter(n, 1.0, mode="nearest")[::2, ::2]
        pyr_p.append(p)
        pyr_n.append(n)

    flow = np.zeros(pyr_p[-1].shape + (2,), dtype=np.float32)
    for lvl in range(levels - 1, -1, -1):
        ip, inx = pyr_p[lvl], pyr_n[lvl]
        if flow.shape[:2] != ip.shape:
            up = np.zeros(ip.shape + (2,), dtype=np.float32)
            for c in range(2):
                up[..., c] = ndimage.zoom(flow[..., c],
                                          (ip.shape[0] / flow.shape[0],
                                           ip.shape[1] / flow.shape[1]),
                                          order=1) * 2.0
            flow = up
        ix = ndimage.sobel(ip, axis=1, mode="nearest") / 8.0
        iy = ndimage.sobel(ip, axis=0, mode="nearest") / 8.0
        for _ in range(iterations):
            warped = _warp(inx, flow)
            it = warped - ip
            sxx = ndimage.uniform_filter(ix * ix, window, mode="nearest")
            syy = ndimage.uniform_filter(iy * iy, window, mode="nearest")
            sxy = ndimage.uniform_filter(ix * iy, window, mode="nearest")
            sxt = ndimage.uniform_filter(ix * it, window, mode="nearest")
            syt = ndimage.uniform_filter(iy * it, window, mode="nearest")
            det = sxx * syy - sxy * sxy
            inv_det = numerics.recip(np.maximum(det, 1e-9))
            valid = det > 1e-7
            du = np.where(valid, -(syy * sxt - sxy * syt) * inv_det, 0.0)
            dv = np.where(valid, -(sxx * syt - sxy * sxt) * inv_det, 0.0)
            flow = flow + np.stack([np.clip(du, -4, 4),
                                    np.clip(dv, -4, 4)], axis=-1)
            flow = numerics.q(flow, "optical_flow")
    return flow.astype(np.float32)


def _stereo_sad(left: np.ndarray, right: np.ndarray, max_disparity: int,
                block: int, numerics) -> np.ndarray:
    l = _to_gray(left)
    r = _to_gray(right)
    h, w = l.shape
    best_cost = np.full((h, w), np.inf, dtype=np.float32)
    best_disp = np.zeros((h, w), dtype=np.float32)
    for d in range(max_disparity + 1):
        shifted = np.empty_like(r)
        if d == 0:
            shifted[:] = r
        else:
            shifted[:, d:] = r[:, :-d]
            shifted[:, :d] = r[:, :1]
        cost = ndimage.uniform_filter(np.abs(l - shifted), block, mode="nearest")
        cost = numerics.q(cost, "stereo_block_match")
        better = cost < best_cost
        best_cost = np.where(better, cost, best_cost)
        best_disp = np.where(better, float(d), best_disp)
    return best_disp
