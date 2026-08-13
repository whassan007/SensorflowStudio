/** Typed fetch wrappers for the Agentic Launch Readiness API (/api/agentic). */
import type {
  AuditRecord,
  EvaluationSuite,
  EvidenceGraph,
  FailureEvent,
  HumanReviewDecision,
  PipelineState,
  PolicyEvaluation,
  RegressionSuiteResult,
  Scorecard,
  Walkthrough,
} from '../types/agentic';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body);
    } catch {
      detail = await res.text().catch(() => '');
    }
    const err = new Error(detail || res.statusText) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  return res.json() as Promise<T>;
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };
const get = <T,>(url: string): Promise<T> => fetch(url).then((r) => handle<T>(r));
const post = <T,>(url: string, body?: unknown): Promise<T> =>
  fetch(url, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body ?? {}),
  }).then((r) => handle<T>(r));

const BASE = '/api/agentic';

// ------------------------------------------------------------- failures

export const detectFailures = () =>
  post<{ detected: number; failures: { failure_id: string; title: string }[] }>(
    `${BASE}/failures/detect`
  );

export const getFailureQueue = () =>
  get<{ failures: FailureEvent[]; stages: string[] }>(`${BASE}/failures`);

export const getFailure = (id: string) =>
  get<{ failure: FailureEvent }>(`${BASE}/failures/${id}`);

export const getState = (id: string) =>
  get<{ state: PipelineState }>(`${BASE}/failures/${id}/state`);

export const runStage = (id: string, stage?: string) =>
  post<{ state: PipelineState }>(
    `${BASE}/failures/${id}/analyze${stage ? `?stage=${encodeURIComponent(stage)}` : ''}`
  );

export const getEvidence = (id: string) =>
  get<{ graph: EvidenceGraph }>(`${BASE}/failures/${id}/evidence`);

export const getSnippets = (id: string) =>
  get<{ snippets: Record<string, unknown>[] }>(`${BASE}/failures/${id}/snippets`);

export const runLaunchAssessment = (id: string) =>
  post<{ policy_evaluation: PolicyEvaluation; scorecard_id: string | null }>(
    `${BASE}/failures/${id}/launch-assessment`
  );

// ------------------------------------------------------------- human review

export interface HumanReviewRequest {
  reviewer: string;
  decision: string;
  rationale: string;
  approved_option?: string | null;
  evidence_reviewed?: string[];
  override_reason?: string | null;
}

export const submitHumanReview = (id: string, req: HumanReviewRequest) =>
  post<{ review: HumanReviewDecision; decisions: HumanReviewDecision[] }>(
    `${BASE}/failures/${id}/human-review`,
    req
  );

export const getHumanReviews = (id: string) =>
  get<{ decisions: HumanReviewDecision[] }>(`${BASE}/failures/${id}/human-review`);

// ------------------------------------------------------------- policy / audit

export const getPolicy = () => get<{ policy: Record<string, unknown> }>(`${BASE}/policy`);

export const getAudit = (id: string) =>
  get<{ records: AuditRecord[]; chain: { valid: boolean; records: number } }>(
    `${BASE}/audit/${id}`
  );

// ------------------------------------------------------------- scorecards

export const getScorecard = (id: string) =>
  get<{ scorecard: Scorecard }>(`${BASE}/scorecards/${id}`);

// ------------------------------------------------------------- flywheel

export const listSuites = () => get<{ suites: EvaluationSuite[] }>(`${BASE}/evaluation-suites`);

export const runRegression = () =>
  post<{ suites: RegressionSuiteResult[]; note: string }>(`${BASE}/regression/evaluate`);

// ------------------------------------------------------------- worked example

export const getWorkedExample = (refresh = false) =>
  get<{ walkthrough: Walkthrough; cached: boolean }>(
    `${BASE}/worked-example${refresh ? '?refresh=true' : ''}`
  );
