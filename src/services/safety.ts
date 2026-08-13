/**
 * Typed fetch wrappers + contracts for the Safety & Compliance layer
 * (/api/safety/*). Mirrors the style of services/megaeval.ts.
 *
 * Sub-systems: ODD coverage, release gates + evidence packages, extended
 * SSAM surrogate safety, calibration validation, discrepancy mining,
 * scenario database, semantic search, consensus evidence.
 */

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

function get<T>(url: string): Promise<T> {
  return fetch(url).then((res) => handle<T>(res));
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return fetch(url, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((res) => handle<T>(res));
}

// ---------------------------------------------------------------- ODD coverage

export interface OddDimensionInfo {
  values: string[];
  iso34503_category: string;
  instrumented: boolean;
  source: string;
}

export interface OddTaxonomy {
  standard_basis: string;
  dimensions: Record<string, OddDimensionInfo>;
  [k: string]: unknown;
}

export interface OddCell {
  cell_id: string;
  cell: Record<string, string>;
  n: number;
  tp: number;
  fn: number;
  recall: number | null;
  mean_iou: number | null;
  wilson_ci: [number, number] | null;
  ci_width: number | null;
  production_share: number;
  adequate: boolean;
  is_gap: boolean;
  gap_reasons: string[];
  performance_deficit: number;
}

export interface OddFillRequest {
  cell_id: string;
  cell: Record<string, string>;
  needed_samples: number;
  risk: number;
  generator: string;
  request: Record<string, unknown>;
}

export interface OddCoverageResponse {
  dims: string[];
  thresholds: { min_samples: number; max_ci_width: number; target_recall: number };
  summary: {
    total_cells: number;
    populated_cells: number;
    empty_cells: number;
    adequate_cells: number;
    gap_cells: number;
    coverage_rate: number;
    production_weighted_coverage: number;
    overall_recall: number;
  };
  gaps: OddCell[];
  fill_requests: OddFillRequest[];
  cells?: OddCell[];
  method: string;
  run_id: string;
  population_id: string;
  model_version: string;
}

export const getOddTaxonomy = () => get<OddTaxonomy>('/api/safety/odd/taxonomy');

export const getOddCoverage = (run: string, dims: string[] | null, includeCells: boolean) => {
  const p = new URLSearchParams({ run, include_cells: String(includeCells), max_gaps: '100' });
  if (dims && dims.length) p.set('dims', dims.join(','));
  return get<OddCoverageResponse>(`/api/safety/odd/coverage?${p.toString()}`);
};

export interface FillGapResult {
  [k: string]: unknown;
}

export const fillOddGap = (run: string, cell: Record<string, string>, numSequences = 2) =>
  post<FillGapResult>('/api/safety/odd/fill-gap', { run, cell, num_sequences: numSequences });

// ---------------------------------------------------------------- release gates

export interface GateCheck {
  check: string;
  actual: number | string | boolean | null;
  threshold: number | string | boolean | null;
  direction: string;
  passed: boolean;
}

export interface GateResult {
  gate: string;
  name: string;
  status: 'PASS' | 'BLOCK' | 'SKIPPED' | string;
  standard_refs: string[];
  checks: GateCheck[];
  evidence: Record<string, unknown>;
  notes: string;
  evaluated_at: string;
}

export interface GateEvaluation {
  candidate_run_id: string;
  baseline_run_id: string;
  decision: 'RELEASE_READY' | 'BLOCKED' | string;
  blocking_gates: string[];
  gates: GateResult[];
  policy: Record<string, Record<string, unknown>>;
  evidence_package_id: string;
  evaluated_at: string;
}

export const evaluateGates = (candidateRun: string, baselineRun: string, policyOverrides?: Record<string, unknown>) =>
  post<GateEvaluation>('/api/safety/gates/evaluate', {
    candidate_run: candidateRun,
    baseline_run: baselineRun,
    policy_overrides: policyOverrides,
  });

export const getGatePolicy = () =>
  get<{ policy: Record<string, Record<string, unknown>>; defaults: Record<string, Record<string, unknown>> }>(
    '/api/safety/gates/policy'
  );

export const setGatePolicy = (overrides: Record<string, unknown>) =>
  post<{ policy: Record<string, Record<string, unknown>> }>('/api/safety/gates/policy', { overrides });

export const getGateResult = (runId: string) =>
  get<GateEvaluation>(`/api/safety/gates/result/${encodeURIComponent(runId)}`);

export interface EvidenceMarkdown {
  run_id: string;
  format: 'markdown';
  markdown: string;
}

export const getEvidenceMarkdown = (runId: string) =>
  get<EvidenceMarkdown>(`/api/safety/evidence/${encodeURIComponent(runId)}?format=markdown`);

export const getEvidenceJson = (runId: string) =>
  get<Record<string, unknown>>(`/api/safety/evidence/${encodeURIComponent(runId)}?format=json`);

// ---------------------------------------------------------------- SSAM

export interface SsamConflict {
  vehicle_a: string;
  vehicle_b: string;
  conflict_type: string;
  t_start_s: number;
  t_end_s: number;
  duration_s: number;
  min_ttc_s: number | null;
  pet_s: number | null;
  max_drac_mps2: number | null;
  delta_s_mps: number | null;
  max_s_mps: number | null;
  collision_probability: number;
  probability_exposure_s: number;
  csi: number;
  mass_proxy_reduced: number;
  conflict_point: [number, number] | null;
}

export interface SsamAnalysis {
  params: Record<string, number>;
  conflicts: SsamConflict[];
  aggregate: {
    num_conflicts: number;
    by_type: Record<string, number>;
    min_ttc_s: number | null;
    min_pet_s: number | null;
    max_drac_mps2: number | null;
    max_delta_s_mps: number | null;
    max_s_mps: number | null;
    max_collision_probability: number | null;
    aggregate_csi: number;
    mean_csi_per_conflict: number;
    observed_duration_s: number;
  };
  measures_glossary: Record<string, string>;
  generated?: { scenario: string; seed: number; reaction_delay_s: number; simulated: boolean };
}

export interface SsamTrajectoryState {
  t: number;
  x: number;
  y: number;
  speed: number;
  heading: number;
}

export interface SsamTrajectory {
  vehicle_id: string;
  vehicle_type?: string;
  length?: number;
  width?: number;
  states: SsamTrajectoryState[];
}

export const analyzeSsam = (body: {
  scenario?: string;
  seed?: number;
  reaction_delay_s?: number;
  trajectories?: SsamTrajectory[];
  params?: Record<string, number>;
}) => post<SsamAnalysis>('/api/safety/ssam/analyze', body);

export interface SsamRunSummary {
  [k: string]: unknown;
}

export const getSsamSummary = (run: string) =>
  get<SsamRunSummary>(`/api/safety/ssam/summary?run=${encodeURIComponent(run)}`);

// ---------------------------------------------------------------- calibration

export type CalibrationMode = 'clean' | 'miscalibrated' | 'perception_failure';

export interface CalibrationCheck {
  name: string;
  actual: number;
  threshold: number;
  passed: boolean;
}

export interface CalibrationObject {
  object_id: string;
  class_name: string;
  residual_px: [number, number];
  residual_magnitude_px: number;
  inlier_ratio: number;
  flagged: boolean;
  failure_reason: string | null;
}

export interface CalibrationResult {
  status: 'CALIBRATED' | 'MISCALIBRATED' | 'PERCEPTION_FAILURE' | 'NEVER_RUN' | string;
  passed?: boolean;
  sensor_pair?: string;
  checks?: CalibrationCheck[];
  metrics?: {
    bias_px: number;
    bias_vector_px: [number, number];
    scatter_px: number;
    mean_inlier_ratio: number;
    outlier_fraction: number;
    estimated_rotation_offset_deg: number;
    num_objects: number;
  };
  per_object?: CalibrationObject[];
  thresholds?: Record<string, number>;
  diagnosis?: string;
  simulated?: boolean;
  mode?: string;
  injected?: Record<string, number>;
  seed?: number;
  created_at?: string;
  note?: string;
}

export const validateCalibration = (body: {
  mode: CalibrationMode;
  rotation_offset_deg?: number;
  translation_offset_m?: number;
  tamper_fraction?: number;
  num_objects?: number;
  seed?: number;
}) => post<CalibrationResult>('/api/safety/calibration/validate', body);

export const getCalibrationStatus = () => get<CalibrationResult>('/api/safety/calibration/status');

// ---------------------------------------------------------------- discrepancy mining

export interface DiscrepancyItem {
  [k: string]: unknown;
}

export interface DiscrepancyReport {
  dataset_id: string;
  total_labels?: number;
  num_discrepancies?: number;
  discrepancy_rate?: number;
  by_type?: Record<string, number>;
  by_class?: Record<string, number>;
  by_cohort?: Array<Record<string, unknown>>;
  cohorts?: Array<Record<string, unknown>>;
  discrepancies?: DiscrepancyItem[];
  discrepancies_truncated?: boolean;
  [k: string]: unknown;
}

export const mineDiscrepancies = (datasetId?: string) =>
  post<DiscrepancyReport>('/api/safety/discrepancy/mine', datasetId ? { dataset_id: datasetId } : {});

export const getDiscrepancySummary = (datasetId?: string) =>
  get<DiscrepancyReport | { datasets: DiscrepancyReport[] }>(
    datasetId
      ? `/api/safety/discrepancy/summary?dataset_id=${encodeURIComponent(datasetId)}`
      : '/api/safety/discrepancy/summary'
  );

// ---------------------------------------------------------------- scenario DB

export interface ScenarioRecord {
  scenario_id: string;
  source: string;
  scenario_type: string;
  severity: string;
  description: string;
  odd_tags: Record<string, string>;
  tags?: string[];
  created_at?: string;
  [k: string]: unknown;
}

export interface ScenarioCounts {
  total: number;
  by_source: Record<string, number>;
  by_severity: Record<string, number>;
  by_type: Record<string, number>;
}

export interface ScenariosResponse {
  counts: ScenarioCounts;
  scenarios: ScenarioRecord[];
}

export const searchScenarios = (filters: {
  scenario_type?: string;
  source?: string;
  severity?: string;
  weather?: string;
  lighting?: string;
  text?: string;
  limit?: number;
}) => {
  const p = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') p.set(k, String(v));
  });
  return get<ScenariosResponse>(`/api/safety/scenarios?${p.toString()}`);
};

export const populateScenarios = () =>
  post<{ imported_rare_events: number; counts: ScenarioCounts }>('/api/safety/scenarios/populate');

export const exportScenarios = (filters: {
  scenario_type?: string;
  source?: string;
  severity?: string;
  text?: string;
}) => post<Record<string, unknown>>('/api/safety/scenarios/export', filters);

// ---------------------------------------------------------------- semantic search

export interface SemanticStage {
  stage?: string;
  [k: string]: unknown;
}

export interface SemanticResult {
  [k: string]: unknown;
}

export interface SemanticSearchResponse {
  concept?: string;
  provider?: string;
  llm_used?: boolean;
  interpretation?: Record<string, unknown>;
  results: SemanticResult[];
  stages?: SemanticStage[];
  [k: string]: unknown;
}

export const semanticSearch = (body: {
  concept: string;
  target: 'containers' | 'scenarios';
  run?: string;
  k?: number;
  use_llm?: boolean | null;
  filters?: Record<string, unknown>;
}) => post<SemanticSearchResponse>('/api/safety/semantic-search', body);
