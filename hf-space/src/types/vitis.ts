/** Types mirroring the /api/vitis JSON payloads (snake_case from FastAPI). */

// ------------------------------------------------------------- backends

export interface BackendInfo {
  name: string;
  available: boolean;
  emulated: boolean;
  description: string;
}

export interface BackendStatus {
  backends: BackendInfo[];
  devices: Record<string, { clock_mhz: number; has_aie: boolean; description: string }>;
  hardware_present: boolean;
  note: string;
}

// ------------------------------------------------------------- HIL

export interface HilVerdict {
  decision: 'REGRESSION' | 'PASS' | 'INSUFFICIENT_EVIDENCE';
  mean_delta: number;
  ci: [number, number];
  n: number;
  method: string;
}

export interface HilComparison {
  config: Record<string, unknown>;
  totals: {
    gt_objects: number;
    ref_detected: number;
    vitis_detected: number;
    paired_detections: number;
    dropped_by_vitis: number;
    spurious_in_vitis: number;
    class_flips: number;
  };
  drift: {
    mean_confidence_drift: number;
    mean_abs_confidence_drift: number;
    mean_position_drift_px: number;
    mean_pair_iou: number;
  };
  gap_score: number;
  cohort_delta: Record<string, number>;
}

export interface HilAblation {
  legs: Record<string, { gap_score: number }>;
  attribution: Record<string, number>;
  note: string;
}

export interface HilRun {
  run_id: string;
  created_at: string;
  params: Record<string, unknown>;
  comparison: HilComparison;
  verdict: HilVerdict;
  ablation: HilAblation | null;
  elapsed_s: number;
  emulation_note: string;
}

export interface HilSweepPoint {
  width_bits: number;
  int_bits: number;
  gap_score: number;
  dropped: number;
  class_flips: number;
  mean_abs_confidence_drift: number;
  mean_position_drift_px: number;
  mean_pair_iou: number;
  decision: HilVerdict['decision'];
  mean_delta: number;
}

export interface HilSweep {
  run_id: string;
  created_at: string;
  params: Record<string, unknown>;
  points: HilSweepPoint[];
  minimal_passing_config: {
    width_bits: number;
    int_bits: number;
    decision: string;
    note?: string;
  } | null;
  elapsed_s: number;
  emulation_note: string;
}

// ------------------------------------------------------------- ISP

export interface IspStageEntry {
  stage: string;
  psnr_db: number;
  ssim: number;
  measured_cpu_ms: number;
  modeled_fpga_ms?: number;
  modeled_placement?: string;
  modeled_speedup_x?: number;
  modeled_not_measured?: boolean;
}

export interface IspThroughput {
  measured_cpu_ms_per_frame: number;
  measured_cpu_fps: number;
  modeled_fpga_ms_per_frame_serial: number;
  modeled_fpga_fps_pipelined: number;
  modeled_speedup_x_serial: number;
  modeled_not_measured: boolean;
  note: string;
}

export interface IspPreviewStage {
  stage: string;
  reference_png: string;
  vitis_png: string;
  diff_png: string;
}

export interface IspRun {
  run_id: string;
  created_at: string;
  params: Record<string, unknown>;
  stage_report: IspStageEntry[];
  throughput: IspThroughput;
  previews: { frame_id: string; cohort: string; input_png?: string; stages: IspPreviewStage[] }[];
  elapsed_s: number;
  emulation_note: string;
}

// ------------------------------------------------------------- augment

export interface AugmentationSpec {
  name: string;
  description: string;
  defaults: Record<string, number>;
}

export interface AugmentVariant {
  variant_id: string;
  frame_id: string;
  dataset_id: string;
  scene_id: string;
  sequence_id: string;
  weather: string;
  time_of_day: string;
  scenario_tags: string[];
  evaluation_only: boolean;
  training_eligible: boolean;
  protected_evaluation: boolean;
  recommended_dataset_destination: string;
  lineage: {
    batch_id: string;
    source_frame_id: string;
    source_sequence_id: string;
    recipe: { aug: string; params: Record<string, number> }[];
    seed: number;
    backend: string;
    backend_emulated: boolean;
  };
  recipe: { aug: string; params: Record<string, number> }[];
  thumbnail_png?: string;
}

export interface AugmentBatch {
  run_id: string;
  created_at: string;
  params: Record<string, unknown>;
  variants: AugmentVariant[];
  raremine_hook: { available: boolean; routed_candidates: number; note?: string };
  elapsed_s: number;
  emulation_note: string;
}

// ------------------------------------------------------------- temporal

export interface EngineStability {
  flicker_rate: number;
  mean_jitter: number;
  fragmentation_per_track: number;
  id_switches: number;
  id_switches_at_flow_break: number;
  unexcused_id_switch_fraction: number;
  stability_score: number;
  cohorts?: Record<string, EngineStability>;
}

export interface StereoReport {
  objects_checked: number;
  median_abs_depth_error_m: number | null;
  median_rel_depth_error: number | null;
  median_abs_depth_error_near_m: number | null;
  p90_abs_depth_error_m: number | null;
  median_abs_disparity_error_px: number | null;
  note?: string;
}

export interface TimelineFrame {
  frame_index: number;
  detected: boolean;
  track_id: string | number | null;
  flow_continuous: boolean;
  flow_residual_m: number | null;
  occluded: boolean;
}

export interface TemporalRun {
  run_id: string;
  created_at: string;
  params: Record<string, unknown>;
  results: Record<string, { engines: Record<string, EngineStability>; stereo: StereoReport }>;
  backend_agreement: {
    ranking_agrees: boolean;
    ranking_reference: string[];
    ranking_vitis_emulated: string[];
    stability_score_delta_by_engine: Record<string, number>;
    max_abs_score_delta: number;
    note: string;
  };
  timeline_sample: {
    instance_id: string;
    engine: string;
    sequence_id: string;
    frames: TimelineFrame[];
  } | null;
  elapsed_s: number;
  emulation_note: string;
}

// ------------------------------------------------------------- PRD / listings

export interface PrdListEntry {
  id: string;
  file: string;
  available: boolean;
}

export interface PrdDoc {
  id: string;
  file: string;
  markdown: string;
}

export interface RunListEntry {
  run_id: string;
  created_at: string | null;
  summary: Record<string, unknown>;
}
