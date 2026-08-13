"""AMD/Xilinx Vitis Vision acceleration layer (optional backend).

This package adds a pluggable hardware-acceleration backend abstraction to
Sensorflow Studio, plus three features built on top of it:

1. HIL quantization-gap regression detection (hil.py)
2. Accelerated ISP + synthetic edge-case generation (isp.py, augment.py)
3. Temporal & stereo stability profiling (temporal.py)

HONESTY NOTE — no FPGA hardware is attached to this machine. The
"vitis_emulated" backend runs on CPU but faithfully models Vitis Vision /
FPGA constraints: ap_fixed<W,I>-style fixed-point quantization with
saturation and truncation rounding, XFCVDEPTH-style line-buffer depth limits
(windowed processing with boundary artifacts when exceeded), LUT-based
divide/sqrt approximations, and a deterministic per-op latency/throughput
model. All speedup numbers it produces are MODELED, NOT MEASURED, and are
labeled as such everywhere they appear. A real Vitis/PYNQ/XRT backend can be
slotted in later behind the same `VisionBackend` interface (see the
documented "vitis_hw" stub in backend.py).
"""

from sensorflow.vitis.backend import (  # noqa: F401
    DeviceConfig,
    PipelineConfig,
    ReferenceCPUBackend,
    VisionBackend,
    VitisEmulatedBackend,
    backend_status,
    get_backend,
    list_backends,
)
