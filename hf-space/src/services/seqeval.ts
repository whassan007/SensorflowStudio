/**
 * Typed fetch wrappers + contracts for the anytime-valid sequential
 * regression-detection engine (/api/seqeval/*).
 *
 * Decision semantics (from GET /policy):
 *   REGRESSION            → block (evidence of a drop beyond the margin)
 *   PASS                  → allow (equivalence proven within the margin)
 *   INSUFFICIENT_EVIDENCE → not proven equivalent; never treated as a pass
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

// ---------------------------------------------------------------- contracts

export type SeqDecision = 'REGRESSION' | 'PASS' | 'INSUFFICIENT_EVIDENCE';

export interface SeqModelSpec {
  model_version: string;
  effects: Record<string, number>;
  fingerprint?: string;
}

export interface SeqNode {
  node: string; // "overall" | "class:vehicle" | "stratum:pedestrian|night" | ...
  level: number;
  n: number;
  n_clusters: number;
  delta_estimate: number;
  delta_ci: [number, number];
  e_regression: number;
  e_pass: number;
  p_analogue: number | null;
  bayes_p_regression: number | null;
  decision: SeqDecision;
  safety_primary: boolean;
  suspect: boolean;
  alpha_allocated: number;
  e_threshold_regression: number;
  e_threshold_pass: number;
}

export interface TrajectoryPoint {
  n: number;
  n_clusters: number;
  delta_estimate: number;
  delta_lower: number;
  delta_upper: number;
  log_e_regression: number;
  log_e_pass: number;
  decision: SeqDecision;
}

export interface NodeTrajectory {
  points: TrajectoryPoint[];
  boundaries: { log_e_regression: number; log_e_pass: number; delta_margin: number };
}

export interface SeqBudget {
  planned_total: number;
  samples_used: number;
  escalation_used: number;
  full_population: number;
  fraction_of_population: number;
}

export interface SeqStratumPlan {
  key: string;
  class: string;
  condition: string;
  N: number;
  allocated: number;
  clusters: number;
  reserve: number;
  weight: number;
  safety_primary: boolean;
}

export interface SeqPlan {
  plan_id: string;
  plan_hash: string;
  population_id: string;
  dataset_fingerprint: string;
  seed: number;
  created_at: string;
  config: Record<string, unknown>;
  population_objects: number;
  total_allocated: number;
  strata: Record<string, SeqStratumPlan>;
  frozen_before_candidate_outcomes: boolean;
}

export interface SeqRunState {
  run_id: string;
  population_id: string;
  baseline: SeqModelSpec;
  candidate: SeqModelSpec;
  status: 'running' | 'done' | 'failed' | string;
  stage: string;
  decision: SeqDecision | null;
  gate: 'block' | 'allow' | 'expand_budget_or_report' | string | null;
  message: string | null;
  stopping_reason: string | null;
  error: string | null;
  seed: number;
  policy: {
    metric: string;
    delta_margin: number;
    alpha: number;
    alpha_pass: number;
    alpha_shares: Record<string, number>;
    condition_dim: string;
    safety_primaries: string[];
    target_n: number;
    [k: string]: unknown;
  };
  sanity: Record<string, unknown> | null;
  budget: SeqBudget | null;
  plan: SeqPlan | null;
  nodes: SeqNode[];
  attribution?: SeqAttribution | null;
  trajectories?: Record<string, NodeTrajectory>;
  created_at: string;
  finished_at: string | null;
}

export interface SeqAttribution {
  run_id?: string;
  decision?: SeqDecision | null;
  gate?: string | null;
  classes?: string[];
  conditions?: string[];
  cells?: Array<{
    class: string;
    condition: string;
    delta: number | null;
    n?: number;
    flagged?: boolean;
    [k: string]: unknown;
  }>;
  [k: string]: unknown;
}

export interface SeqPolicyInfo {
  default_policy: Record<string, unknown>;
  decision_semantics: Record<string, string>;
  test_method: string;
  multiple_testing_method: string;
  notes: string[];
}

// ---------------------------------------------------------------- endpoints

export const listSeqRuns = () => get<{ runs: SeqRunState[] }>('/api/seqeval/runs');

export const getSeqRun = (runId: string) => get<SeqRunState>(`/api/seqeval/runs/${encodeURIComponent(runId)}`);

export const getSeqAttribution = (runId: string) =>
  get<SeqAttribution>(`/api/seqeval/runs/${encodeURIComponent(runId)}/attribution`);

export const getSeqEvidence = (runId: string) =>
  get<{ run_id: string; records: Array<Record<string, unknown>>; lineage: Record<string, unknown>; required_fields: string[] }>(
    `/api/seqeval/runs/${encodeURIComponent(runId)}/evidence`
  );

export const getSeqPolicy = () => get<SeqPolicyInfo>('/api/seqeval/policy');

export const startSeqRun = (body: {
  population_id: string;
  baseline: { model_version: string; effects: Record<string, number> };
  candidate: { model_version: string; effects: Record<string, number> };
  policy?: Record<string, unknown>;
  seed?: number;
  sync?: boolean;
}) => post<SeqRunState>('/api/seqeval/runs', body);
