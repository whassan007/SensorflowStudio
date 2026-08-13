/** Typed fetch wrappers for the ROTR capability (/api/rotr/*). */

import type {
  RotrConsequenceDetail,
  RotrQueryResponse,
  RotrRegressionResult,
  RotrRunListItem,
  RotrRunSummary,
  RotrHITLReview,
  RotrStopshipPolicy,
  RotrSuite,
  RotrTrainingCandidate,
  RotrViolationDetail,
  RotrAttributionMatrix,
} from '../types/rotr';

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

export function createRun(params: {
  n_scenarios: number;
  seed: number;
  model_version: string;
}): Promise<RotrRunSummary> {
  return post('/api/rotr/runs', params);
}

export function listRuns(): Promise<RotrRunListItem[]> {
  return get('/api/rotr/runs');
}

export function getRun(runId: string): Promise<RotrRunSummary> {
  return get(`/api/rotr/runs/${encodeURIComponent(runId)}`);
}

export function getAttributionMatrix(runId: string): Promise<RotrAttributionMatrix> {
  return get(`/api/rotr/runs/${encodeURIComponent(runId)}/attribution`);
}

export function getViolation(runId: string, violationId: string): Promise<RotrViolationDetail> {
  return get(
    `/api/rotr/runs/${encodeURIComponent(runId)}/violations/${encodeURIComponent(violationId)}`
  );
}

export function getConsequence(
  runId: string,
  violationId: string
): Promise<RotrConsequenceDetail> {
  return get(
    `/api/rotr/runs/${encodeURIComponent(runId)}/violations/${encodeURIComponent(
      violationId
    )}/consequence`
  );
}

export function structuredQuery(params: {
  run_id: string;
  text?: string;
  filters?: Record<string, string>;
}): Promise<RotrQueryResponse> {
  return post('/api/rotr/query', params);
}

export function runRegression(params: {
  baseline_run_id: string;
  candidate_run_id: string;
}): Promise<RotrRegressionResult> {
  return post('/api/rotr/regression', params);
}

export function listRegressions(): Promise<RotrRegressionResult[]> {
  return get('/api/rotr/regressions');
}

export function getHitlQueue(runId: string): Promise<RotrHITLReview[]> {
  return get(`/api/rotr/runs/${encodeURIComponent(runId)}/hitl`);
}

export function hitlAction(params: {
  run_id: string;
  review_id: string;
  action: 'VALIDATE' | 'REJECT';
  actor: string;
  notes?: string;
}): Promise<{ review: RotrHITLReview; candidate: RotrTrainingCandidate | null }> {
  return post('/api/rotr/hitl/action', params);
}

export function getSuite(): Promise<RotrSuite> {
  return get('/api/rotr/flywheel/suite');
}

export function listCandidates(): Promise<RotrTrainingCandidate[]> {
  return get('/api/rotr/flywheel/candidates');
}

export function promoteCandidate(params: {
  candidate_id: string;
  actor: string;
}): Promise<RotrTrainingCandidate> {
  return post('/api/rotr/flywheel/promote', params);
}

export function overrideCandidate(params: {
  candidate_id: string;
  actor: string;
  reason: string;
}): Promise<RotrTrainingCandidate> {
  return post('/api/rotr/flywheel/override', params);
}

export function getStopshipPolicy(): Promise<RotrStopshipPolicy> {
  return get('/api/rotr/stopship/policy');
}
