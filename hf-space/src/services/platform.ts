/**
 * Platform Phase 1 API client — container quality, multi-model compare, gates, evidence.
 */
import { usePoll } from './labeleval';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

function get<T>(url: string): Promise<T> {
  return fetch(url).then((res) => handle<T>(res));
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((res) => handle<T>(res));
}

export type VerificationRates = {
  n_objects: number;
  verified: number;
  unverified: number;
  disputed: number;
  auto_accepted: number;
  hitl: number;
  reviewed: number;
  verified_rate: number | null;
  unverified_rate: number | null;
  disputed_rate: number | null;
  auto_accept_rate: number | null;
  hitl_rate: number | null;
  review_coverage: number | null;
};

export type ContainerQualityRow = {
  container_id: number;
  status: string | null;
  risk_score: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  mean_iou: number | null;
  verification: VerificationRates;
  dims: Record<string, string>;
};

export type ContainerQualityProfile = {
  run_id: string;
  model_version: string;
  total: number;
  summary: {
    precision: number | null;
    recall: number | null;
    f1: number | null;
    mean_iou: number | null;
    verification: VerificationRates;
  };
  containers: ContainerQualityRow[];
  verification_headline: VerificationRates;
};

export type GateResultRow = {
  result_id: string;
  gate_id: string;
  gate_type: string;
  passed: boolean | null;
  ready: boolean;
  message: string;
  failures: Array<Record<string, unknown>>;
  thresholds: Record<string, number>;
};

export type GateStatusResponse = {
  gates: GateResultRow[];
  summary: {
    wired_count: number;
    unwired_count: number;
    wired_all_passed: boolean;
    release_ready: boolean;
    unwired_gate_types: string[];
  };
};

export type ModelCompareResponse = {
  baseline_run_id: string;
  run_ids: string[];
  models: Array<{
    run_id: string;
    model_version: string;
    role: string;
    headline: Record<string, number | null>;
  }>;
  metric_matrix: Array<{
    metric: string;
    deltas_vs_baseline: Record<string, number | null>;
    [model: string]: unknown;
  }>;
  pairwise: Array<{
    candidate_model: string;
    recommendation: string;
    blockers: string[];
  }>;
  recommendations: Array<{ candidate: string; recommendation: string; blocker_count: number }>;
};

export const getContainerQuality = (runId: string, limit = 25) =>
  get<ContainerQualityProfile>(
    `/api/containers/quality?run_id=${encodeURIComponent(runId)}&limit=${limit}`
  );

export const compareModels = (runIds: string[], baselineRunId?: string | null) =>
  post<ModelCompareResponse>('/api/models/compare', {
    run_ids: runIds,
    baseline_run_id: baselineRunId ?? undefined,
  });

export const getGateStatus = (params?: {
  sequence_id?: string;
  candidate_run_id?: string;
  baseline_run_id?: string;
}) => {
  const q = new URLSearchParams();
  if (params?.sequence_id) q.set('sequence_id', params.sequence_id);
  if (params?.candidate_run_id) q.set('candidate_run_id', params.candidate_run_id);
  if (params?.baseline_run_id) q.set('baseline_run_id', params.baseline_run_id);
  const qs = q.toString();
  return get<GateStatusResponse>(`/api/gates/status${qs ? `?${qs}` : ''}`);
};

export const getEvaluationLevels = () =>
  get<{ levels: string[]; backends: Record<string, string> }>('/api/evaluations/levels');

export const exportEvidence = (body: Record<string, unknown>) =>
  post<{ package: Record<string, unknown>; path: string | null }>('/api/evaluations/evidence', body);

export function useContainerQuality(runId: string | null, refreshKey = 0) {
  return usePoll(
    () => (runId ? getContainerQuality(runId) : Promise.resolve(null)),
    null,
    [runId, refreshKey]
  );
}

export function useGateStatus(
  candidateRunId: string | null,
  baselineRunId: string | null,
  refreshKey = 0
) {
  return usePoll(
    () =>
      getGateStatus({
        candidate_run_id: candidateRunId ?? undefined,
        baseline_run_id: baselineRunId ?? undefined,
      }),
    8000,
    [candidateRunId, baselineRunId, refreshKey]
  );
}
