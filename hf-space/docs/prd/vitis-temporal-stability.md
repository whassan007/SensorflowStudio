# PRD: Temporal & Stereo Stability Profiling

**Feature Name:** Temporal & Stereo Stability Profiling
**Package:** `sensorflow/vitis` (`temporal.py`) — API under
`/api/vitis/temporal/*`
**Status:** Implemented on the emulated backend. No FPGA hardware is attached;
the fixed-point optical-flow/stereo path runs on the constraint-faithful CPU
emulator.

## Value Proposition

Frame-level mAP hides the failures that hurt downstream planners: detections
that flicker, boxes that jitter against real motion, tracks that fragment and
swap IDs. This feature builds a MODEL-INDEPENDENT motion baseline — pyramidal
dense Lucas-Kanade optical flow over the scene, an op FPGAs accelerate well —
and scores every perception engine against it. Because ground truth for
motion comes from the flow field rather than from any engine, the metrics
expose instability without circularity, penalize engines that drop tracking
where the flow proves the object stayed observable, and forgive ID switches
that coincide with genuine flow discontinuities. A synthetic stereo pair
derived from the exact scene geometry closes the loop on 3D: block-matching
disparity is converted to metric depth and checked against scene geometry.
Finally, the whole profile runs twice — float32 flow vs fixed-point flow — so
the report certifies whether the accelerated metric itself can be trusted.

## Architecture & Workflow

```
  +--------------------------------------------------------------------+
  |                      Cloud evaluation platform                      |
  |  bevfusion scene DB + engines            engine leaderboards       |
  |  (GT tracks, planted occlusions,         (stability score joins    |
  |   perception-v1 / perception-v3)          accuracy metrics)        |
  +-----------+-----------------------------------------^--------------+
              | sequences + engine detections           | stability report
              v                                         |
  +-----------+-----------------------------------------+--------------+
  |            sensorflow/vitis/temporal.py profiler                    |
  |                                                                     |
  |  BEV rasters --> dense pyramidal LK flow --> flow-predicted         |
  |                  (per backend)               object positions       |
  |                                                                     |
  |  scene geometry --> synthetic stereo pair --> SAD disparity         |
  |                                               --> metric depth      |
  |                                                                     |
  |  +----------------+          +------------------------------+       |
  |  | reference flow |  meta-   | vitis_emulated flow          |       |
  |  | (float32)      |  check   | (ap_fixed, LUT recip)        |       |
  |  +----------------+  <-----> +--------------+---------------+       |
  |                                             |  << EMULATION        |
  |                              +--------------+---------------+       |
  |                              | (future) vitis_hw: xf::cv    |       |
  |                              | pyrLK + stereoBM on PL/AIE   |       |
  |                              +------------------------------+       |
  +---------------------------------------------------------------------+
              | runs/vitis/temporal/*.json
              v
      dashboard (engine comparison, cohorts, timeline strip)
```

## Key Metrics Tracked

From `POST /api/vitis/temporal/run` (per engine, per backend, with
`cohorts` breakdown over day/clear, night/clear, day/rain, occluded):

- `flicker_rate` — present→absent→present events on flow-continuous tracks,
  per opportunity
- `mean_jitter` — |engine displacement − flow-predicted displacement| /
  (|flow displacement| + 0.5 m)
- `fragmentation_per_track` — distinct engine track IDs per GT instance over
  flow-continuous spans, minus one
- `id_switches`, `id_switches_at_flow_break`,
  `unexcused_id_switch_fraction` — ID switches not explained by a flow
  discontinuity
- `stability_score` — 100 × (0.35·(1−4·flicker) + 0.25·e^(−1.5·jitter) +
  0.25·(1−0.6·fragmentation) + 0.15·(1−unexcused))
- `stereo.median_abs_disparity_error_px`, `stereo.median_rel_depth_error`,
  `stereo.median_abs_depth_error_near_m` (near field < 30 m)
- `backend_agreement.ranking_agrees`,
  `backend_agreement.stability_score_delta_by_engine`,
  `backend_agreement.max_abs_score_delta` — the trust meta-check
- `timeline_sample` — per-frame detected/flow-continuous/occluded strip for
  the most gap-prone track (drives the dashboard timeline)

## Rollout

1. **Emulated (now):** rank engines (e.g. perception-v1-camera vs
   perception-v3-bevfusion) by stability; wire the score next to accuracy
   metrics in engine comparisons.
2. **HIL bench:** move flow + stereo to xf::cv pyrLK / stereoBM kernels on
   real PL/AIE via the `vitis_hw` backend; confirm `backend_agreement`
   holds on silicon (the emulator predicts it should — verify).
3. **Fleet:** run the profiler over recorded fleet sequences as a nightly
   job; alert when an engine's stability score drops or when the accelerated
   and reference metrics diverge (hardware fault canary).

## Risks

- **Flow quality bounds metric quality.** Where flow fails (textureless
  regions, huge displacements), continuity is under-detected and instability
  is under-penalized. The residual gate (`FLOW_RESIDUAL_OK_M`) trades recall
  of opportunities for precision of blame.
- **Schematic scenes flatter flow.** High-contrast BEV rasters are easy for
  LK; real imagery will lower flow coverage. Rollout stage 3 must re-tune
  the residual gate on real sequences.
- **Stereo far-field error.** Depth error grows quadratically with range at
  fixed disparity error; the near-field figure is the meaningful one, and
  the report says so explicitly.
- **Score weights are opinionated.** The 0.35/0.25/0.25/0.15 weighting is a
  product decision, documented in the formula above and trivially tunable;
  rankings should be checked for weight sensitivity before gating releases.
