/** Types for the rare-event miner API (/api/raremine). */

export type Priority = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type Difficulty = 'EASY' | 'MODERATE' | 'HARD' | 'EXTREME';
export type EvidenceQuality = 'LOW' | 'MEDIUM' | 'HIGH';

export interface Evidence {
  modality: string;
  description: string;
}

export interface AlternativeHypothesis {
  hypothesis: string;
  confidence: number;
  status: 'RETAINED' | 'REJECTED';
  reason: string;
}

export interface TemporalValidation {
  available: boolean;
  status: 'VALIDATED' | 'NOT_AVAILABLE' | 'INCONCLUSIVE';
  evidence: string[];
}

export interface ObservedModelBehavior {
  failure_observed: boolean;
  failure_modes: string[];
  details: string[];
}

export interface Candidate {
  candidate_id: string;
  scene_id: string;
  sequence_id: string;
  object_id: string;
  track_id: string;
  frame_index: number;
  edge_case_detected: boolean;
  insufficient_visual_evidence: boolean;
  event_type: string;
  costume_type: string[];
  confidence_human_identity: number;
  confidence_costume: number;
  confidence_rare_event: number;
  visual_evidence: Evidence[];
  human_identity_evidence: Evidence[];
  alternative_hypotheses: AlternativeHypothesis[];
  silhouette_deviation: string;
  occlusion_level: string;
  occlusion_source: string;
  visible_human_features: string[];
  occluded_human_features: string[];
  unusual_geometry: string[];
  temporal_validation: TemporalValidation;
  location: { context?: string; distance_m?: number; position?: number[]; lighting?: string; weather?: string };
  perception_difficulty: Difficulty;
  difficulty_evidence: string[];
  observed_model_behavior: ObservedModelBehavior | null;
  predicted_failure_mode: string | null;
  curation_priority: Priority;
  priority_reason: string;
  recommended_dataset_destination: string;
  requires_human_validation: boolean;
  human_validation_reason: string;
  evidence_quality: EvidenceQuality;
  rank_in_scene: number;
}

export interface RepresentativeFrames {
  best_evidence: string | null;
  worst_case: string | null;
  model_failure: string | null;
  minimal_set: string[];
}

export interface LineageRecord {
  lineage_id: string;
  track_candidate_id: string;
  source_frame_id: string;
  source_sequence_id: string;
  dataset_version: string;
  curation_timestamp: string;
  curator: string;
  validation_status: string;
  training_eligible: boolean;
  evaluation_eligible: boolean;
  protected_evaluation: boolean;
  destination: string;
  governance_overrides: { actor: string; reason: string; timestamp: string }[];
}

export interface TrackCandidateSummary {
  track_candidate_id: string;
  sequence_id: string;
  track_id: string;
  frame_count: number;
  duration_frames: number;
  stage: string;
  destination: string;
  diversity_selected: boolean;
  duplicate_of: string | null;
  max_difficulty: Difficulty;
  max_visibility: EvidenceQuality;
  representative_frames: RepresentativeFrames;
  auto_validation: { status: string; detail?: string } | null;
  human_validation: { action: string; note: string; reviewer: string } | null;
  candidate: Candidate;
}

export interface TrackFrame {
  candidate_id: string;
  scene_id: string;
  frame_index: number;
  edge_case_detected: boolean;
  confidence_rare_event: number;
  evidence_quality: EvidenceQuality;
  perception_difficulty: Difficulty;
  failure_observed: boolean;
}

export interface TrackView extends TrackCandidateSummary {
  frames: TrackFrame[];
}

export interface SceneObjectView {
  object_id: string;
  is_candidate: boolean;
  position: number[];
  distance_m: number;
  bbox_2d: number[];
  context: string;
  gt?: { class_name: string; is_costumed: boolean; costume_type: string | null } | null;
}

export interface SceneView {
  scene_id: string;
  sequence_id: string;
  frame_index: number;
  lighting: string;
  weather: string;
  modalities: Record<string, boolean>;
  objects: SceneObjectView[];
  baseline_predictions:
    | { prediction_id: string; object_id: string | null; predicted_class: string; confidence: number }[]
    | null;
}

export interface BankInfo {
  bank_id: string;
  seed: number;
  num_scenes: number;
  num_sequences: number;
  num_planted_rare: number;
  num_confounders: number;
}

export interface MiningRunInfo {
  run_id: string;
  bank_id: string;
  num_frame_candidates: number;
  num_track_candidates: number;
  num_duplicates: number;
  num_diversity_selected: number;
}

export interface DedupReport {
  groups: number;
  kept: number;
  duplicates_archived: number;
  dedup_savings: number;
  signatures: { signature: Record<string, string>; members: number; kept: string }[];
}

export interface DiversityReport {
  budget: number;
  pool_size: number;
  selected_ids: string[];
  coverage_selected: number;
  coverage_naive_topk: number;
  coverage_matrix: Record<string, Record<string, number>>;
  note: string;
}

export interface StatusResponse {
  bank: BankInfo | null;
  last_run: MiningRunInfo | null;
  num_track_candidates: number;
  num_detected: number;
  priority_histogram: Record<string, number>;
  dedup_report: DedupReport | null;
  diversity_report: DiversityReport | null;
}

export interface CandidatesResponse {
  count: number;
  candidates: TrackCandidateSummary[];
}

export interface CandidateDetail extends TrackCandidateSummary {
  frame_candidates: Candidate[];
  lineage: LineageRecord | null;
}

export interface CalibrationBin {
  bin: string;
  n: number;
  true: number;
  observed_rate: number | null;
}

export interface CuratorReport {
  confusion: { tp: number; fp: number; fn: number; tn: number };
  planted_positives: number;
  mining_precision: number | null;
  mining_recall: number | null;
  false_discovery_rate: number | null;
  calibration: CalibrationBin[];
  curation_yield: { reviewed: number; approved: number; yield: number | null };
  model_value: { curated: number; expose_model_failure: number; fraction: number | null };
}

export interface QuantvalReport {
  candidates_evaluated: number;
  with_model_outputs: number;
  agreement_matrix: Record<string, Record<string, number>>;
  within_one_level_agreement: number | null;
  rows: {
    track_candidate_id: string;
    costume_type: string[];
    predicted_difficulty: Difficulty;
    observed: {
      frames_with_predictions: number;
      mean_baseline_confidence: number;
      mean_iou: number | null;
      miss_rate: number;
      class_confusion: Record<string, number>;
      track_stability: number;
      observed_difficulty: Difficulty;
      exposes_model_failure: boolean;
    } | null;
  }[];
  note: string;
}

export interface ImprovementReport {
  recurring_misses: {
    total: number;
    by_costume_type: Record<string, number>;
    by_occlusion: Record<string, number>;
    by_context: Record<string, number>;
    examples: Record<string, unknown>[];
  };
  over_fires: {
    total: number;
    by_confounder_type: Record<string, number>;
    examples: Record<string, unknown>[];
  };
  next_run_config: { sensitivity_boost: Record<string, number>; rare_event_threshold?: number };
  note: string;
}

export interface DestinationsResponse {
  destinations: Record<
    string,
    { track_candidate_id: string; stage: string; priority: Priority; costume_type: string[]; lineage: LineageRecord | null }[]
  >;
  counts: Record<string, number>;
}
