/** Typed fetch wrappers for the Retrospective Safety Analyzer API (/api/retro). */
import type {
  AnalysisSummary,
  AnalyzeResponse,
  AuditRecord,
  BackendStatus,
  CompatReport,
  EnvironmentReport,
  FixtureInfo,
  RetrospectiveScorecard,
  ToolSpec,
} from '../types/retro';

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

const get = <T,>(url: string): Promise<T> => fetch(url).then((r) => handle<T>(r));

export const getRetroEnv = () =>
  get<{ environment: EnvironmentReport }>('/api/retro/env');

export const getRetroCompat = () =>
  get<{ report: CompatReport; formatted: string }>('/api/retro/compat');

export const getRetroBackends = () =>
  get<{ backends: BackendStatus[] }>('/api/retro/backends');

export const getRetroFixtures = () =>
  get<{ fixtures: FixtureInfo[] }>('/api/retro/fixtures');

export const analyzeFixture = (fixtureId: string, backend: string) =>
  fetch(`/api/retro/analyze?fixture_id=${encodeURIComponent(fixtureId)}&backend=${backend}`, {
    method: 'POST',
  }).then((r) => handle<AnalyzeResponse>(r));

export const listRetroAnalyses = () =>
  get<{ analyses: AnalysisSummary[] }>('/api/retro/analyses');

export const getRetroAnalysis = (evaluationId: string) =>
  get<{ scorecard: RetrospectiveScorecard; markdown: string; audit_analysis_id: string }>(
    `/api/retro/analyses/${encodeURIComponent(evaluationId)}`
  );

export const getRetroAudit = (evaluationId: string) =>
  get<{ audit_analysis_id: string; records: AuditRecord[] }>(
    `/api/retro/analyses/${encodeURIComponent(evaluationId)}/audit`
  );

export const getRetroTools = () =>
  get<{ tools: ToolSpec[]; note: string }>('/api/retro/tools');

export const searchRetroRag = (q: string, k = 4) =>
  get<{ store_backend: string; hits: import('../types/retro').RetrievedStandard[] }>(
    `/api/retro/rag/search?q=${encodeURIComponent(q)}&k=${k}`
  );
