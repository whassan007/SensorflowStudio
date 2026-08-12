/**
 * Shared API contract for the L4 Perception Label Evaluation platform.
 * These types mirror the FastAPI responses served by sensorflow/evaluation/api.py.
 * Keep field names snake_case to match the JSON wire format exactly.
 */

import type { MegaevalStatus } from './megaeval';

// ---------------------------------------------------------------- enums

export type GroundTruthType =
  | 'PSEUDO_GROUND_TRUTH'
  | 'VENDOR_GROUND_TRUTH'
  | 'HUMAN_VERIFIED_GROUND_TRUTH'
  | 'GOLD_STANDARD';

export type TriageStatus = 'AUTO_GRADED' | 'FLAGGED' | 'VERIFIED' | 'REJECTED' | 'PENDING';

export type FailureReason =
  | 'LOW_IOU'
  | 'POSITION_ERROR'
  | 'ORIENTATION_ERROR'
  | 'INSUFFICIENT_POINT_SUPPORT'
  | 'ANOMALY'
  | 'GRADER_DISAGREEMENT'
  | 'TRACK_FRAGMENTATION'
  | 'ID_SWITCH'
  | 'MODEL_REGRESSION'
  | 'LOW_CONFIDENCE'
  | 'SENSOR_DISAGREEMENT';

export type ServiceState = 'HEALTHY' | 'RUNNING' | 'DEGRADED' | 'BLOCKED' | 'FAILED' | 'IDLE';

export type ScenarioType =
  | 'near_collision'
  | 'extreme_ttc'
  | 'extreme_pet'
  | 'vru_interaction'
  | 'unusual_object_behavior'
  | 'severe_occlusion'
  | 'sensor_failure'
  | 'nighttime_glare'
  | 'adverse_weather'
  | 'unusual_trajectory'
  | 'unexpected_road_geometry';

export type DetectorName =
  | 'knn'
  | 'lof'
  | 'isolation_forest'
  | 'ocsvm'
  | 'dbscan'
  | 'autoencoder'
  | 'vae'
  | 'gan'
  | 'few_shot';

export type EnsembleStrategy = 'majority_vote' | 'weighted_average' | 'meta_classifier';

export type ReviewAction =
  | 'approve'
  | 'reject'
  | 'correct'
  | 'merge_tracks'
  | 'split_track'
  | 'mark_ignore';

export type HaystackCategory =
  | 'normal'
  | 'anomaly'
  | 'rare_event'
  | 'false_positive'
  | 'false_negative'
  | 'verified';

// ---------------------------------------------------------------- core records

export interface GateLine {
  gate: string;
  actual: number | string | boolean;
  threshold: number | string | boolean;
  passed: boolean;
  applicable: boolean;
}

export interface TriageDecision {
  decision_id: string;
  annotation_id: string;
  status: TriageStatus;
  failure_reasons: FailureReason[];
  primary_failure_reason: FailureReason | null;
  gate_lines: GateLine[];
  policy_id: string;
  policy_values: Record<string, number>;
  decided_at: string;
}

export interface EvaluationRecord {
  annotation_id: string;
  dataset_id: string;
  frame_id: string;
  object_class: string;
  model_version: string;
  ground_truth_id: string | null;
  ground_truth_type: GroundTruthType | null;
  detection: { confidence: number };
  geometry: {
    iou_3d: number | null;
    position_error: number | null;
    orientation_error_deg: number | null;
    dimension_error: number | null;
    point_density: number | null;
    point_in_box_ratio: number | null;
    ground_contact_error: number | null;
  };
  tracking: { id_switch: boolean; fragmentation: boolean; track_quality: number | null };
  anomaly: {
    score: number;
    is_anomaly: boolean;
    detector_scores: Record<string, number>;
    normalized_scores: Record<string, number>;
    ensemble_strategy: EnsembleStrategy;
    decision_threshold: number;
  };
  grading: {
    consensus: number | null;
    class_agreement: number | null;
    spatial_agreement: number | null;
    temporal_agreement: number | null;
    grader_count: number;
    disagreement_types: string[];
  };
  validation: {
    passed: boolean;
    checks: GateLine[];
  };
  decision: TriageDecision | null;
  injected_errors: string[]; // synthetic ground-truth of injected defects (for demo transparency)
}

export interface Annotation {
  annotation_id: string;
  frame_id: string;
  object_id: string;
  class_name: string;
  confidence: number;
  bbox_2d: [number, number, number, number] | null; // x,y,w,h in image px
  bbox_3d: [number, number, number, number, number, number, number] | null; // x,y,z,l,w,h,yaw
  track_id: string | null;
  model: string;
  model_version: string;
  source: string;
  status: TriageStatus;
}

export interface FrameSummary {
  frame_id: string;
  sequence_id: string;
  scene_id: string;
  timestamp_us: number;
  ego_pose: { x: number; y: number; z: number; yaw: number; speed_mps: number };
  num_lidar_points: number;
  camera: { width: number; height: number };
  annotations: Annotation[];
  gt_boxes: {
    gt_id: string;
    class_name: string;
    bbox_3d: [number, number, number, number, number, number, number];
    bbox_2d: [number, number, number, number] | null;
    gt_type: GroundTruthType;
  }[];
  lidar_points_bev: [number, number, number][]; // downsampled x,y,z for display
}

export interface RareEvent {
  event_id: string;
  scenario_type: ScenarioType;
  severity: 'low' | 'medium' | 'high' | 'critical';
  rarity_score: number;
  anomaly_score: number;
  confidence: number;
  evidence_frames: string[];
  sensor_evidence: Record<string, string>;
  dataset_id: string;
  description: string;
  verified: boolean;
}

export interface HaystackPoint {
  id: string; // annotation_id or event_id
  x: number;
  y: number;
  category: HaystackCategory;
  anomaly_score: number;
  class_name: string;
  frame_id: string;
  kind: 'annotation' | 'rare_event';
}

// ---------------------------------------------------------------- datasets / groups

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  version: string;
  created_at: string;
  num_scenes: number;
  num_sequences: number;
  num_frames: number;
  num_annotations: number;
  gt_availability: {
    has_reference: boolean;
    gt_type: GroundTruthType | null;
    coverage: number; // fraction of annotations with a reference
    evaluation_confidence: 'high' | 'medium' | 'low' | 'none';
  };
  lineage: {
    generated_from_model: string | null;
    corrected_by_review_batch: string | null;
    validated_by_policy: string | null;
    parent_dataset: string | null;
  };
  status: string;
}

export interface QualityGroupSummary {
  group_id: string;
  dataset_id: string;
  name: 'verified' | 'non_verified' | 'hitl' | 'rejected' | string;
  count: number;
  pct: number;
}

export interface QualityGroupDetail extends QualityGroupSummary {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  mean_iou_3d: number | null;
  mean_consensus: number | null;
  mean_anomaly_score: number | null;
  tracking_quality: number | null;
  failure_reason_counts: Record<string, number>;
  annotation_ids: string[];
}

export interface QualityGroupsResponse {
  dataset_id: string;
  total: number;
  verification_rate: number;
  groups: QualityGroupSummary[];
}

// ---------------------------------------------------------------- metrics / overview

export interface ClassMetrics {
  class_name: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
}

export interface ScenarioMetrics {
  scenario: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
}

export interface QualityMetrics {
  dataset_id: string | null;
  gt_available: boolean;
  gt_type: GroundTruthType | null;
  gt_coverage: number;
  global: {
    precision: number | null;
    recall: number | null;
    f1: number | null;
    map_3d: number | null;
    safety_critical_recall: number | null;
    mean_iou_3d: number | null;
    mean_position_error: number | null;
    mean_orientation_error_deg: number | null;
    idf1: number | null;
    id_swap_rate: number | null;
    fragmentation_rate: number | null;
    anomaly_rate: number;
    grader_consensus: number | null;
  };
  per_class: ClassMetrics[];
  per_scenario: ScenarioMetrics[];
}

export interface FunnelStage {
  stage: string;
  count: number;
  pct_of_input: number;
}

export interface FunnelResponse {
  main_path: FunnelStage[]; // raw -> auto labels -> evaluated -> validated -> auto graded -> verified -> training
  side_path: FunnelStage[]; // flagged -> hitl -> relabeled -> revalidated -> verified
}

export interface OverviewResponse {
  counters: {
    frames_processed: number;
    auto_labeled: number;
    evaluated: number;
    auto_graded: number;
    flagged: number;
    in_hitl: number;
    verified: number;
    rejected: number;
    rare_events: number;
  };
  metrics: QualityMetrics['global'];
  verification_rate: number;
  automation_rate: number;
  process_units_total: number;
  active_dataset: string | null;
  active_model: string | null;
}

// ---------------------------------------------------------------- queue / services / pipeline

export interface QueueStatus {
  backend: string; // in_memory | redis | kafka
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  throughput_per_s: number;
  depth_by_topic: Record<string, number>;
}

export interface ServiceStatus {
  service: string;
  state: ServiceState;
  processed: number;
  total: number;
  detail: string;
  process_units: number;
}

export interface PipelineStateResponse {
  running: boolean;
  stage: string;
  services: ServiceStatus[];
  queue: QueueStatus;
  counters: OverviewResponse['counters'];
  review_queue_count: number;
  regression_alert: boolean;
  last_run_id: string | null;
}

// ---------------------------------------------------------------- anomaly config / benchmark

export interface AnomalyConfig {
  imbalance: {
    method: 'none' | 'smote' | 'oversample' | 'undersample' | 'class_weights';
    minority_boost: number;
  };
  detectors: {
    knn: { enabled: boolean; k: number };
    lof: { enabled: boolean; n_neighbors: number };
    isolation_forest: { enabled: boolean; n_estimators: number };
    ocsvm: { enabled: boolean; nu: number };
    dbscan: { enabled: boolean; eps: number; min_samples: number };
  };
  deep: {
    autoencoder: { enabled: boolean; latent_dim: number; epochs: number };
    vae: { enabled: boolean; latent_dim: number };
    gan: { enabled: boolean };
    reconstruction_threshold: number;
  };
  advanced: {
    few_shot: { enabled: boolean; support_per_class: number };
    ensemble_strategy: EnsembleStrategy;
    decision_threshold: number;
  };
}

export interface BenchmarkRow {
  technique: string;
  precision: number;
  recall: number;
  rare_recall: number;
  f1: number;
  box_error_3d: number;
  id_swap_rate: number;
  consensus: number;
  process_units: number;
  fp_rate: number;
}

export interface BenchmarkResponse {
  benchmark_id: string;
  rows: BenchmarkRow[];
  highlights: {
    best_rare_recall: string;
    best_safety_recall: string;
    lowest_fp_rate: string;
    lowest_process_units: string;
    lowest_tracking_error: string;
  };
  created_at: string;
}

// ---------------------------------------------------------------- regression

export interface RegressionMetricDelta {
  metric: string;
  baseline: number;
  current: number;
  delta: number;
  tolerance: number;
  regressed: boolean;
}

export interface RegressionEntry {
  model_version: string;
  baseline_version: string | null;
  dataset_version: string;
  run_id: string;
  date: string;
  regression_detected: boolean;
  affected_classes: string[];
  affected_scenarios: string[];
  deltas: RegressionMetricDelta[];
  kinds: string[]; // performance | tracking | annotation
}

export interface RegressionResponse {
  entries: RegressionEntry[];
  current_alert: boolean;
}

// ---------------------------------------------------------------- review / HITL

export interface ReviewTask {
  task_id: string;
  annotation_id: string;
  frame_id: string;
  dataset_id: string;
  failure_reasons: FailureReason[];
  primary_failure_reason: FailureReason | null;
  status: 'open' | 'in_review' | 'resolved';
  created_at: string;
  resolution: {
    action: ReviewAction;
    corrected_bbox_3d: number[] | null;
    corrected_class: string | null;
    revalidation_passed: boolean | null;
    final_status: TriageStatus | null;
    reviewed_at: string;
  } | null;
  evidence: EvaluationRecord | null;
}

export interface ReviewActionRequest {
  action: ReviewAction;
  corrected_bbox_3d?: number[];
  corrected_class?: string;
  merge_with_track_id?: string;
  note?: string;
}

export interface ReviewActionResponse {
  task: ReviewTask;
  revalidation: EvaluationRecord;
  message: string;
}

// ---------------------------------------------------------------- training / models

export interface TrainRequest {
  dataset_version: string;
  model_version?: string;
  configuration?: Record<string, unknown>;
  quality_policy?: string;
  training_parameters?: { epochs?: number; batch_size?: number; lr?: number };
}

export interface TrainResponse {
  job_id: string;
  model_id: string;
  model_version: string;
  status: string;
}

export interface TrainingJobStatus {
  job_id: string;
  model_id: string;
  model_version: string;
  dataset_version: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  epoch: number;
  total_epochs: number;
  loss: number;
  rare_recall: number;
  safety_recall: number;
  process_units: number;
  logs: string[];
  started_at: string;
  lineage: DatasetSummary['lineage'];
}

export interface ModelSummary {
  model_id: string;
  model_version: string;
  name: string;
  trained_on_dataset: string | null;
  created_at: string;
  status: string;
  metrics: {
    precision: number | null;
    recall: number | null;
    f1: number | null;
    map_3d: number | null;
    safety_critical_recall: number | null;
    rare_recall: number | null;
  };
  regression_status: 'baseline' | 'improved' | 'regressed' | 'unknown';
}

// ---------------------------------------------------------------- process units / alerts / audit

export interface ProcessUnitsResponse {
  total: number;
  by_stage: Record<string, number>;
  unit_economics: {
    per_verified_event: number | null;
    per_million_frames: number | null;
    per_training_dataset: number | null;
  };
}

export interface Alert {
  alert_id: string;
  kind: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  evidence_link: { page: string; id: string | null };
  created_at: string;
}

export interface AuditEvent {
  event_id: string;
  timestamp: string;
  actor: string;
  action: string;
  entity_type: string;
  entity_id: string;
  detail: string;
}

// ---------------------------------------------------------------- copilot

export interface CopilotExplainRequest {
  context_type: 'false_positive' | 'false_negative' | 'anomaly' | 'regression' | 'disagreement' | 'general';
  annotation_id?: string;
  event_id?: string;
  model_version?: string;
  extra?: Record<string, unknown>;
}

export interface CopilotExplainResponse {
  status: string;
  provider: string; // ollama URL or 'offline_deterministic'
  analysis: string;
  structured: {
    failure_classification: string;
    observed_evidence: string[];
    likely_cause: string;
    contributing_factors: string[];
    hypothesis: string;
    recommended_investigation: string[];
    confidence: number;
  } | null;
}

// ---------------------------------------------------------------- SSE payload

export interface StreamEvent {
  ts: string;
  pipeline: PipelineStateResponse;
  training: TrainingJobStatus | null;
  alerts_count: number;
  /** Mega-scale evaluation layer status; absent on older backends and in the polling fallback. */
  megaeval?: MegaevalStatus;
}
