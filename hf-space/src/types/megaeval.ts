/**
 * API contract for the aggregate-first mega-scale evaluation layer (megaeval).
 *
 * Mental model: Dataset (population) -> Evaluation Run -> Population -> Cohort
 *               -> Container -> Annotation (deepest, forensic drill-down only).
 *
 * Endpoints (all JSON):
 *   POST /api/megaeval/population/generate     GeneratePopulationRequest -> PopulationMeta
 *   GET  /api/megaeval/populations             -> { populations: PopulationMeta[] }
 *   GET  /api/megaeval/dimensions              -> { dimensions: Record<DimName, string[]> }
 *   POST /api/megaeval/runs                    CreateRunRequest -> EvaluationRunInfo (async job!)
 *   GET  /api/megaeval/runs                    -> { runs: EvaluationRunInfo[] }
 *   GET  /api/megaeval/runs/{run_id}           -> EvaluationRunInfo
 *   GET  /api/megaeval/status                  -> MegaevalStatus (also inside SSE stream)
 *   POST /api/evaluations/query                EvaluationQueryRequest -> EvaluationQueryResponse
 *   GET  /api/megaeval/cache                   -> CacheStats
 *   GET  /api/megaeval/runs/{id}/funnel        -> QualityFunnelResponse
 *   GET  /api/megaeval/runs/{id}/containers?sort=&limit=&offset= -> ContainersResponse
 *   GET  /api/megaeval/runs/{id}/containers/{cid}/objects -> ContainerObjectsResponse
 *   POST /api/megaeval/errors/search           ErrorSearchRequest -> ErrorSearchResponse
 *   POST /api/megaeval/compare                 CompareRequest -> CompareResponse
 *   GET  /api/megaeval/runs/{id}/shift         -> ShiftResponse
 *   POST /api/megaeval/why                     WhyRequest -> WhyResponse
 *   POST /api/megaeval/similarity              SimilarityRequest -> SimilarityResponse
 *   GET  /api/megaeval/runs/{id}/review        -> ReviewState
 *   POST /api/megaeval/runs/{id}/review/plan   { target_n?: number } -> ReviewState
 *   POST /api/megaeval/runs/{id}/review/execute -> ReviewState
 *   GET  /api/megaeval/runs/{id}/distributions -> DistributionsResponse
 *
 * Notes:
 * - Runs are async: POST /runs returns immediately with status "queued"; poll
 *   GET /runs/{id} or use the SSE stream's `megaeval.active_runs` for progress.
 * - Results endpoints return HTTP 409 until the run is "published".
 * - Query responses carry meta.source ("cache"|"cube"|"scan"), meta.cache_hit and
 *   meta.latency_ms — the UI should surface these as badges.
 * - meta.exact=false means sketch-derived (approximate) numbers are included.
 */

export type DimName =
  | 'class' | 'weather' | 'lighting' | 'road_type' | 'scenario'
  | 'sensor' | 'distance_band' | 'speed_band' | 'occlusion';

export const DIM_NAMES: DimName[] = [
  'class', 'weather', 'lighting', 'road_type', 'scenario',
  'sensor', 'distance_band', 'speed_band', 'occlusion',
];

export type RunStatus =
  | 'created' | 'queued' | 'running' | 'reducing' | 'materializing'
  | 'published' | 'failed';

// ---------------------------------------------------------------- populations

export interface GeneratePopulationRequest {
  name?: string;
  num_objects?: number; // 1_000 .. 1_200_000, default 320_000
  seed?: number;
}

export interface TrainMixRow {
  class: string;
  weather: string;
  lighting: string;
  train_share: number;
  eval_share: number;
  eval_count: number;
}

export interface PopulationMeta {
  population_id: string;
  name: string;
  created_at: string;
  seed: number;
  num_objects: number;
  num_containers: number;
  num_partitions: number;
  avg_objects_per_container: number;
  dimensions: Record<DimName, string[]>;
  dim_counts: Record<DimName, Record<string, number>>;
  safety_critical_count: number;
  train_mix: TrainMixRow[];
}

// ---------------------------------------------------------------- runs

export interface CreateRunRequest {
  population_id: string;
  model_version: string;
  overrides?: Record<string, number>; // e.g. { night_penalty: 0.3 } to demo a regression
  worker_delay_s?: number;            // simulated distributed I/O per partition (default 0.5)
  workers?: number;                   // default 4
  seed?: number | null;
  label_version?: string;
}

export interface RunProgress {
  run_id: string;
  population_id: string;
  model_version: string;
  status: RunStatus;
  percent: number;
  partitions_done: number;
  partitions_total: number;
  objects_processed: number;
  objects_total: number;
  workers: number;
  throughput_objs_per_s: number;
  eta_s: number | null;
  error: string | null;
}

export interface RunLineage {
  evaluation_id: string;
  dataset_version: string;
  model_version: string;
  model_checkpoint: string;
  label_version: string;
  evaluator_code_version: string;
  metric_version: string;
  threshold_config: Record<string, number>;
  sampling_config: Record<string, unknown>;
  seed: number;
  hardware: string;
  timestamp: string;
}

export interface RunHeadline {
  n: number;
  tp?: number;
  fp?: number;
  fn?: number;
  loc_err?: number;
  anomalies?: number;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  mean_iou?: number | null;
  anomaly_rate?: number | null;
  safety_recall?: number | null;
  conf_mean?: number | null;
  error_rate?: number | null;
  containers?: number;
  containers_hll_estimate?: number;
}

export interface EvaluationRunInfo extends RunProgress {
  created_at: string;
  started_at: string | null;
  published_at: string | null;
  lineage: RunLineage;
  headline: RunHeadline;                       // {} until published
  per_class: Record<string, QueryRow>;         // {} until published
  overrides: Record<string, number>;
}

export interface CacheStats {
  entries: number;
  hits: number;
  misses: number;
  hit_rate: number | null;
}

export interface MegaevalStatus {
  active_runs: RunProgress[];
  cache: CacheStats | null;
  total_runs: number;
}

// ---------------------------------------------------------------- query API

export type MetricName =
  | 'n' | 'tp' | 'fp' | 'fn' | 'loc_err' | 'anomalies' | 'reviewed' | 'verified'
  | 'safety_n' | 'safety_tp'
  | 'precision' | 'recall' | 'f1' | 'mean_iou' | 'anomaly_rate'
  | 'conf_mean' | 'conf_std' | 'safety_recall' | 'review_coverage'
  | 'verified_rate' | 'error_rate'
  | 'confidence_p50' | 'confidence_p10' | 'confidence_p90'
  | 'iou_p50' | 'iou_p10' | 'iou_p90';

export interface EvaluationQueryRequest {
  evaluation_id: string;                    // run_id
  filters?: Partial<Record<DimName, string[]>>;
  metrics?: MetricName[];
  group_by?: DimName[];
  limit?: number;
}

/** One aggregate row: group_by dims as strings + requested metrics. n always present. */
export type QueryRow = { n: number } & Record<string, string | number | null>;

export interface QueryMeta {
  source: 'cache' | 'cube' | 'scan';
  cache_hit: boolean;
  latency_ms: number;
  cells_touched: number;
  exact: boolean;
  approximate_fields: string[];
}

export interface EvaluationQueryResponse {
  rows: QueryRow[];
  meta: QueryMeta;
}

// ---------------------------------------------------------------- funnel

export interface FunnelStage {
  stage: string;
  count: number;
  pct_of_population: number;
}

export interface QualityFunnelResponse {
  stages: FunnelStage[];
  estimated_precision: {
    estimate: number | null;
    ci_low: number | null;
    ci_high: number | null;
    n_reviewed: number | null;
  } | null;
}

// ---------------------------------------------------------------- containers

export type ContainerSortPreset =
  | 'worst_recall' | 'worst_precision' | 'worst_iou'
  | 'most_anomalies' | 'least_verified' | 'highest_risk';

export interface ContainerRow {
  container_id: number;
  weather: string;
  lighting: string;
  road_type: string;
  scenario: string;
  n_objects: number;
  tp: number;
  fp: number;
  fn: number;
  anomalies: number;
  reviewed: number;
  verified: number;
  recall: number | null;
  precision: number | null;
  mean_iou: number | null;
  risk_score: number;
  status: 'ok' | 'warn' | 'critical';
}

export interface ContainersResponse {
  total: number;
  sort: string;
  rows: ContainerRow[];
}

/** Forensic object row — the deepest drill-down (old annotation-table level). */
export interface ContainerObject {
  annotation_id: string;
  container_id: number;
  class: string;
  weather: string;
  lighting: string;
  road_type: string;
  scenario: string;
  sensor: string;
  distance_band: string;
  speed_band: string;
  occlusion: string;
  safety_critical: boolean;
  difficulty: number | null;
  detected: boolean;
  outcome: 'TP' | 'FN' | 'FP' | 'LOCALIZATION';
  iou: number | null;
  confidence: number | null;
  anomaly: boolean;
  sensor_disagree: boolean;
}

export interface ContainerObjectsResponse {
  container_id: number;
  objects: ContainerObject[];
}

// ---------------------------------------------------------------- error index

export type ErrorType = 'FN' | 'FP' | 'LOCALIZATION' | 'ANOMALY' | 'LOW_CONF';

export interface ErrorSearchRequest {
  run_id: string;
  error_types?: ErrorType[];
  filters?: Partial<Record<DimName, string[]>>;
  confidence_max?: number | null;
  confidence_min?: number | null;
  risk_min?: number | null;
  severity_min?: number | null;
  safety_only?: boolean;
  limit_containers?: number;
}

export interface WorstContainer {
  container_id: number;
  error_count: number;
  mean_risk: number;
  max_severity: number;
  safety_hits: number;
  weather?: string;
  lighting?: string;
  road_type?: string;
  scenario?: string;
  n_objects?: number;
  risk_score?: number;
}

export interface ErrorExample {
  error_id: number;
  annotation_id: string;
  container_id: number;
  error_type: ErrorType;
  class: string;
  severity: number;
  confidence: number;
  risk_score: number;
  safety_critical: boolean;
  scenario: string;
  lighting: string;
  weather: string;
}

export interface ErrorSearchResponse {
  matched_errors: number;
  by_type: Partial<Record<ErrorType, number>>;
  worst_containers: WorstContainer[];
  examples: ErrorExample[];
}

// ---------------------------------------------------------------- compare

export interface HeadlineDelta {
  metric: string;
  baseline: number;
  candidate: number;
  delta: number;
}

export interface PerClassDelta {
  class: string;
  n: number | null;
  recall_baseline: number | null;
  recall_candidate: number | null;
  recall_delta: number;
  precision_baseline: number | null;
  precision_candidate: number | null;
  precision_delta: number;
}

export interface CohortDelta {
  cohort: string; // e.g. "highway/night/cyclist"
  road_type?: string;
  lighting?: string;
  class?: string;
  n: number;
  recall_baseline: number;
  recall_candidate: number;
  recall_delta: number;
}

export interface CompareRequest {
  candidate_run_id: string;
  baseline_run_id: string;
  policy?: Record<string, unknown>;
}

export interface CompareResponse {
  candidate: { run_id: string; model_version: string };
  baseline: { run_id: string; model_version: string };
  headline_deltas: HeadlineDelta[];
  per_class: PerClassDelta[];
  worst_cohorts: CohortDelta[];
  regressions: CohortDelta[];
  policy: Record<string, unknown>;
  recommendation: 'PROMOTE' | 'DO_NOT_PROMOTE';
  blockers: string[];
}

// ---------------------------------------------------------------- review sampling

export interface StratumResult {
  stratum: string;
  N: number;
  n: number;
  correct: number;
  p: number;
  wilson_ci: [number, number];
  weight: number;
}

export interface MetricEstimate {
  estimate: number;
  ci_low: number;
  ci_high: number;
  n_reviewed: number;
  frame_size: number;
  method: string;
  strata: StratumResult[];
}

export interface SamplingFunnel {
  population_objects: number;
  containers: number;
  suspicious_containers: number;
  candidate_pool: number;
  statistically_selected: number;
  reviewed: number;
}

export interface StratumPlan {
  label: string;
  N: number;
  allocated: number;
}

export interface ReviewState {
  run_id: string;
  planned: boolean;
  executed: boolean;
  method?: string;
  target_n?: number;
  funnel?: SamplingFunnel;
  strata?: {
    precision: Record<string, StratumPlan>;
    recall: Record<string, StratumPlan>;
  };
  results?: {
    precision: MetricEstimate;
    recall: MetricEstimate;
  } | null;
}

// ---------------------------------------------------------------- shift / why / similarity

export interface ShiftRow {
  cohort: string;
  class: string;
  weather: string;
  lighting: string;
  train_share: number;
  eval_share: number;
  relative_change: number; // e.g. +0.90 == +90%
  eval_count: number;
  cohort_recall: number | null;
  overall_recall: number | null;
  recall_gap: number | null;
}

export interface ShiftResponse {
  run_id: string;
  shifts: ShiftRow[];
  method: string;
  thresholds: { min_eval_count: number; rel_threshold: number };
}

export interface WhyRequest {
  run_id: string;
  filters?: Partial<Record<DimName, string[]>>;
  metric?: 'recall' | 'precision';
}

export interface WhyFactor {
  factor: 'occlusion' | 'low_illumination' | 'long_range' | 'sensor_disagreement' | 'other';
  count: number;
  share: number;
}

export interface WhyCohort {
  cohort: string;
  n: number;
  fn?: number | null;
  fp?: number | null;
  [metric: string]: string | number | null | undefined;
}

export interface WhyResponse {
  metric: string;
  filters: Partial<Record<DimName, string[]>>;
  failure_count: number;
  factors: WhyFactor[];
  top_cohorts: WhyCohort[];
  method: string;
}

export interface SimilarityRequest {
  run_id: string;
  container_id: number;
  filters?: Partial<Record<DimName, string[]>>;
  k?: number;
}

export interface SimilarContainer {
  container_id: number;
  similarity: number;
  weather: string;
  lighting: string;
  road_type: string;
  scenario: string;
  n_objects: number;
  fn: number;
  fp: number;
  anomalies: number;
  risk_score: number;
}

export interface SimilarityResponse {
  query_container: number;
  retrieval?: string;
  results: SimilarContainer[];
}

// ---------------------------------------------------------------- distributions

export interface HistogramSketch {
  bins: number;
  lo: number;
  hi: number;
  counts: number[];
  percentiles: Record<string, number>;
}

export interface DistributionsResponse {
  run_id: string;
  exact: false;
  note: string;
  confidence?: HistogramSketch;
  iou?: HistogramSketch;
  containers_hll_estimate: number | null;
  containers_exact: number | null;
  available_sketch_metrics: string[];
}
