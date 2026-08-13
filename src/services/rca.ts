/**
 * Typed fetch wrappers for the Regression RCA workbench API (/api/rca).
 * Mirrors the style of services/megaeval.ts.
 */
import type {
  Confidence,
  DecisionTree,
  DiagnosticsResponse,
  Experiments,
  Finding,
  FindingStatus,
  Investigation,
  InvestigationSummary,
  RcaReport,
  RevealResponse,
  RootCause,
  Scoreboard,
  Severity,
  StageMeta,
  StageState,
} from '../types/rca';

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
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => handle<T>(r));

export const getCauses = () =>
  get<{ causes: { id: RootCause; label: string }[] }>('/api/rca/causes');

export const getStages = () => get<{ stages: StageMeta[] }>('/api/rca/stages');

export const createInvestigation = (body: {
  name?: string;
  baseline_model?: string;
  candidate_model?: string;
  cause?: RootCause | null;
  seed?: number | null;
  training_mode?: boolean;
}) => post<Investigation>('/api/rca/investigations', body);

export const listInvestigations = () =>
  get<{ investigations: InvestigationSummary[] }>('/api/rca/investigations');

export const getInvestigation = (id: string) =>
  get<Investigation>(`/api/rca/investigations/${encodeURIComponent(id)}`);

export const completeStage = (id: string, index: number, body: {
  acknowledge_unknowns?: boolean;
  note?: string;
}) =>
  post<{ stage: StageState; investigation: Investigation }>(
    `/api/rca/investigations/${encodeURIComponent(id)}/stages/${index}/complete`,
    body
  );

export const reopenStage = (id: string, index: number) =>
  post<{ stage: StageState; investigation: Investigation }>(
    `/api/rca/investigations/${encodeURIComponent(id)}/stages/${index}/reopen`
  );

export const recordFinding = (id: string, body: {
  stage: string;
  title: string;
  status: FindingStatus;
  severity: Severity;
  detail?: string;
  code?: string;
}) =>
  post<{ finding: Finding }>(
    `/api/rca/investigations/${encodeURIComponent(id)}/findings`,
    body
  );

export const getDiagnostics = (id: string, stageKey: string) =>
  get<DiagnosticsResponse>(
    `/api/rca/investigations/${encodeURIComponent(id)}/diagnostics/${encodeURIComponent(stageKey)}`
  );

export const getScoreboard = (id: string, recordedOnly = false) =>
  get<Scoreboard>(
    `/api/rca/investigations/${encodeURIComponent(id)}/scoreboard${recordedOnly ? '?recorded_only=true' : ''}`
  );

export const assessHypothesis = (id: string, body: {
  hypothesis: RootCause;
  confidence: Confidence;
  note?: string;
}) =>
  post<Scoreboard>(
    `/api/rca/investigations/${encodeURIComponent(id)}/scoreboard/assess`,
    body
  );

export const getDecisionTree = (id: string) =>
  get<DecisionTree>(`/api/rca/investigations/${encodeURIComponent(id)}/decision-tree`);

export const getExperiments = (id: string) =>
  get<Experiments>(`/api/rca/investigations/${encodeURIComponent(id)}/experiments`);

export const getReport = (id: string) =>
  get<RcaReport>(`/api/rca/investigations/${encodeURIComponent(id)}/report`);

export const revealCause = (id: string) =>
  post<RevealResponse>(`/api/rca/investigations/${encodeURIComponent(id)}/reveal`);
