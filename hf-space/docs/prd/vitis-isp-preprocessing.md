# PRD: Accelerated ISP + Synthetic Edge-Case Generation

**Feature Name:** Accelerated ISP Preprocessing & Synthetic Edge-Case Generation
**Package:** `sensorflow/vitis` (`isp.py`, `augment.py`) — API under
`/api/vitis/isp/*` and `/api/vitis/augment/*`
**Status:** Implemented on the emulated backend. No FPGA hardware is attached;
all FPGA throughput figures are analytically modeled and labeled
`modeled_not_measured: true`.

## Value Proposition

Two problems, one accelerator. First, ISP pipelines (bad-pixel correction →
demosaic → HDR tone-map → gain → denoise → resize) are the earliest place a
hardware port silently degrades a perception stack — this feature runs the
composable ISP on the reference and Vitis-constrained paths simultaneously
and reports PSNR/SSIM *per stage*, so the exact stage where fixed-point or
streaming constraints bite is visible, alongside a measured-CPU vs
modeled-FPGA throughput comparison that quantifies the acceleration case.
Second, the same accelerated op library mints rare-edge-case evaluation data
at scale: parameterized sensor-noise/low-light/HDR/distortion/blur/glare
augmentations generate stress variants of evaluation frames with complete
lineage, guarded as evaluation-only so synthetic stress data can never leak
into training.

## Architecture & Workflow

```
   +-----------------------------------------------------------------+
   |                     Cloud evaluation platform                    |
   |   evaluation/megaeval record store        raremine scene bank    |
   |   (eval-set supplements, lineage,         (candidate flow for    |
   |    leakage guard destinations)             costume-ped variants) |
   +--------------------^--------------------------------^-----------+
                        | eval-only variant records      | candidates
                        |  (full lineage, protected)     | (try/except)
   +--------------------+--------------------------------+-----------+
   |               sensorflow/vitis: ISP + augment                    |
   |                                                                  |
   |  RAW Bayer -> bad_pixel -> demosaic -> tone_map -> gain          |
   |            -> denoise -> resize        (per-stage PSNR/SSIM)     |
   |                                                                  |
   |  +----------------+        +-------------------------------+     |
   |  | reference CPU  |  vs    | vitis_emulated (ap_fixed<W,I>,|     |
   |  | float32        |        | XFCVDEPTH, LUT div/sqrt,      |     |
   |  +----------------+        | modeled pixels/cycle @ MHz)   |     |
   |                            +---------------+---------------+     |
   |                                            |  << EMULATION      |
   |                            +---------------+---------------+     |
   |                            | (future) vitis_hw: xf::cv ISP |     |
   |                            | kernels on real PL, measured  |     |
   |                            | throughput via XRT profiling  |     |
   |                            +-------------------------------+     |
   +------------------------------------------------------------------+
                        | runs/vitis/{isp,augment}/*.json
                        v
                 dashboard (stage chain, throughput, gallery)
```

## Key Metrics Tracked

From `POST /api/vitis/isp/run`:

- `stage_report[].psnr_db`, `stage_report[].ssim` — emulated vs reference,
  per stage
- `stage_report[].measured_cpu_ms`, `stage_report[].modeled_fpga_ms`,
  `stage_report[].modeled_speedup_x`, `stage_report[].modeled_placement`
  (PL/AIE)
- `throughput.measured_cpu_fps`, `throughput.modeled_fpga_fps_pipelined`
  (dataflow overlap: 1 / max stage latency),
  `throughput.modeled_speedup_x_serial` — all modeled fields flagged
  `modeled_not_measured`
- `summary.min_stage_psnr_db` — the weakest stage at the configured bit-width

From `POST /api/vitis/augment/generate` and `GET /api/vitis/augment/variants`:

- per-variant `lineage` (recipe with resolved params, seed, source frame,
  backend config), `evaluation_only: true`, `training_eligible: false`,
  `recommended_dataset_destination: REGRESSION_EVALUATION_SET`
- `raremine_hook.routed_candidates` (or the documented fallback note when
  `sensorflow.raremine` is not importable)

## Rollout

1. **Emulated (now):** tune stage order, bit-widths, and augmentation
   recipes against per-stage PSNR; mint evaluation-only stress sets.
2. **HIL bench:** synthesize the ISP as an xf::cv dataflow pipeline, replace
   modeled latencies with XRT-profiled measurements, and validate that
   per-stage PSNR on silicon matches the emulator within its documented
   error bounds.
3. **Fleet:** run augmentation generation as a batch service against real
   fleet evaluation frames; generated variants flow into the protected
   evaluation destinations with the same lineage records.

## Risks

- **Modeled throughput is not a benchmark.** pixels/cycle × clock is an
  upper bound that ignores memory bandwidth, AXI stalls, and resource
  contention. Every figure carries `modeled_not_measured`; stage-2 exists
  precisely to replace them.
- **Leakage.** Synthetic stress variants in training would corrupt the
  regression signal the variants exist to protect. Records default to
  `training_eligible: false` and a protected destination; consumers must
  respect the flag (enforced convention, mirrored from raremine's
  `PROTECTED_EVAL_DESTINATIONS`).
- **Schematic realism.** Variants derive from schematic BEV renders, not
  camera frames. The recipes and lineage plumbing are production-shaped; the
  pixels are placeholders until real frames are wired in.
- **In-progress raremine API.** The candidate-routing hook degrades to a
  no-op with a note if the raremine package changes shape mid-flight.
