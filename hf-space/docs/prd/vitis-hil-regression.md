# PRD: HIL Quantization-Gap Regression Detection

**Feature Name:** HIL Quantization-Gap Regression Detection
**Package:** `sensorflow/vitis` (`hil.py`) — API under `/api/vitis/hil/*`
**Status:** Implemented on the emulated backend. No FPGA hardware is attached
to this environment; the `vitis_emulated` backend is a constraint-faithful CPU
emulator and every number it produces is labeled accordingly.

## Value Proposition

Perception teams that deploy camera pipelines on FPGA/ACAP silicon routinely
discover — too late — that the fixed-point, streaming, LUT-approximated
hardware path does not produce the same detections as the float32 path the
model was validated on. This feature turns that discovery into a continuous,
statistically disciplined regression gate: the SAME frames run through both
paths, per-object deltas are paired, degradation is *attributed to its
hardware cause* (bit-width, line-buffer depth, or HLS approximation), and an
anytime-valid sequential test issues a three-outcome verdict plus the minimal
bit-width configuration that still passes. Teams get a principled answer to
"how few bits can we ship?" before committing RTL.

## Architecture & Workflow

```
              +--------------------------------------------------+
              |            Cloud evaluation platform             |
              |  bevfusion scene DB      seqeval sequential gate |
              |  (synthetic frames,      (PairedSequentialTest,  |
              |   GT tracks, cohorts)     anytime-valid CS)      |
              +------------+------------------------+------------+
                           | frames                 ^ paired deltas
                           v                        |
              +------------+------------------------+------------+
              |         HIL harness (sensorflow/vitis/hil.py)    |
              |                                                  |
              |   +-----------------+    +--------------------+  |
              |   | reference       |    | vitis_emulated     |  |
              |   | float32 CPU     |    | ap_fixed<W,I>,     |  |
              |   | backend         |    | XFCVDEPTH strips,  |  |
              |   +-----------------+    | LUT div/sqrt       |  |
              |            |             +--------------------+  |
              |            |                   ^                 |
              |            |                   |  << EMULATION   |
              |            |     +-------------+---------------+ |
              |            |     | (future) vitis_hw:          | |
              |            |     | PYNQ/XRT overlay on real    | |
              |            |     | FPGA — same VisionBackend   | |
              |            |     | interface, measured numbers | |
              |            |     +-----------------------------+ |
              |            v                                     |
              |   paired per-object comparison + OFAT ablation   |
              +--------------------------------------------------+
                           | verdict + minimal passing W
                           v
                    runs/vitis/hil/*.json  ->  dashboard
```

The emulated backend stands in for the FPGA leg of a hardware-in-the-loop
bench. When real silicon is attached, `vitis_hw` replaces `vitis_emulated`
behind the identical `VisionBackend` interface and the harness, statistics,
API, and dashboard are unchanged.

## Key Metrics Tracked

Exactly the numbers the implementation outputs (`POST /api/vitis/hil/run`,
`POST /api/vitis/hil/sweep`):

- `totals.dropped_by_vitis`, `totals.spurious_in_vitis`, `totals.class_flips`
- `drift.mean_confidence_drift`, `drift.mean_abs_confidence_drift`,
  `drift.mean_position_drift_px`, `drift.mean_pair_iou`
- `gap_score` — scalar severity (0.45·drop rate + 0.2·class-flip rate +
  0.2·|confidence drift| + 0.15·IoU loss)
- `ablation.attribution.{precision_only, streaming_only, hls_approx_only}` —
  one-factor-at-a-time share of the gap
- `verdict.decision` ∈ {REGRESSION, PASS, INSUFFICIENT_EVIDENCE} with
  `verdict.method` = `seqeval_anytime_valid` (or `paired_t_fallback` when
  seqeval is unavailable) and the confidence interval `verdict.ci`
- `minimal_passing_config.width_bits` from the bit-width sweep
- `cohort_delta` per day/night/rain cohort

## Rollout

1. **Emulated (now):** every run executes on `vitis_emulated`; verdicts
   quantify the *modeled* quantization gap and gate bit-width choices early.
2. **HIL bench:** attach a Versal AI Edge / Zynq UltraScale+ board via
   Vitis/XRT; implement `vitis_hw` behind `VisionBackend`; re-run the same
   sweeps to validate the emulator's error model against silicon, then gate
   releases on measured deltas.
3. **Fleet:** promote the sequential gate into the release pipeline —
   candidate bitstreams must reach PASS at their shipped bit-width on the
   protected evaluation sets before OTA rollout.

## Risks

- **Emulator fidelity.** The emulator models truncation, saturation, strip
  seams, and LUT error, but not DSP-block rounding modes, AXI backpressure,
  or clock-domain effects. Mitigation: stage-2 silicon correlation before
  trusting absolute numbers; the emulator is honest about being a model.
- **Statistical misuse.** Frame-level clustering is required (objects within
  a frame are correlated); feeding per-object deltas as independent samples
  would inflate confidence. The harness only feeds per-frame cluster means.
- **Attribution confounds.** One-factor-at-a-time ablation misses factor
  interactions (e.g. precision × streaming). Sum-normalized attribution is
  reported with its method note; a full factorial mode is a follow-up.
- **Synthetic-scene bias.** Scenes are schematic BEV rasters; absolute drop
  rates will differ on camera imagery. Relative bit-width trends transfer
  better than absolute rates.
