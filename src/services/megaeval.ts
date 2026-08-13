/**
 * Typed fetch wrappers for the aggregate-first mega-scale evaluation API
 * ("megaeval"). Mirrors the style of services/labeleval.ts.
 *
 * Async run semantics: POST /runs returns immediately with status "queued";
 * results endpoints return HTTP 409 until the run is "published".
 */
import type {
  CacheStats,
  CompareRequest,
  CompareResponse,
  ContainerObjectsResponse,
  ContainerSortPreset,
  ContainersResponse,
  CreateRunRequest,
  DimName,
  DistributionsResponse,
  ErrorSearchRequest,
  ErrorSearchResponse,
  EvaluationQueryRequest,
  EvaluationQueryResponse,
  EvaluationRunInfo,
  GeneratePopulationRequest,
  MegaevalStatus,
  PopulationMeta,
  QualityFunnelResponse,
  ReviewState,
  ShiftResponse,
  SimilarityRequest,
  SimilarityResponse,
  WhyRequest,
  WhyResponse,
} from '../types/megaeval';

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

// ---------------------------------------------------------------- populations

export const generatePopulation = (body: GeneratePopulationRequest) =>
  post<PopulationMeta>('/api/megaeval/population/generate', body);

export const getPopulations = () =>
  get<{ populations: PopulationMeta[] }>('/api/megaeval/populations');

export const getDimensions = () =>
  get<{ dimensions: Record<DimName, string[]> }>('/api/megaeval/dimensions');

// ---------------------------------------------------------------- runs

export const createRun = (body: CreateRunRequest) =>
  post<EvaluationRunInfo>('/api/megaeval/runs', body);

export const getRuns = () => get<{ runs: EvaluationRunInfo[] }>('/api/megaeval/runs');

export const getRun = (runId: string) =>
  get<EvaluationRunInfo>(`/api/megaeval/runs/${encodeURIComponent(runId)}`);

export const getMegaevalStatus = () => get<MegaevalStatus>('/api/megaeval/status');

// ---------------------------------------------------------------- query / cache

export const queryEvaluations = (body: EvaluationQueryRequest) =>
  post<EvaluationQueryResponse>('/api/evaluations/query', body);

export const getCacheStats = () => get<CacheStats>('/api/megaeval/cache');

// ---------------------------------------------------------------- run results

export const getRunFunnel = (runId: string) =>
  get<QualityFunnelResponse>(`/api/megaeval/runs/${encodeURIComponent(runId)}/funnel`);

export const getRunContainers = (
  runId: string,
  sort: ContainerSortPreset = 'highest_risk',
  limit = 50,
  offset = 0
) =>
  get<ContainersResponse>(
    `/api/megaeval/runs/${encodeURIComponent(runId)}/containers?sort=${encodeURIComponent(
      sort
    )}&limit=${limit}&offset=${offset}`
  );

export const getContainerObjects = (runId: string, containerId: number) =>
  get<ContainerObjectsResponse>(
    `/api/megaeval/runs/${encodeURIComponent(runId)}/containers/${containerId}/objects`
  );

export const searchErrors = (body: ErrorSearchRequest) =>
  post<ErrorSearchResponse>('/api/megaeval/errors/search', body);

export const compareRuns = (body: CompareRequest) =>
  post<CompareResponse>('/api/megaeval/compare', body);

export const getRunShift = (runId: string) =>
  get<ShiftResponse>(`/api/megaeval/runs/${encodeURIComponent(runId)}/shift`);

export const whyAnalysis = (body: WhyRequest) => post<WhyResponse>('/api/megaeval/why', body);

export const findSimilarContainers = (body: SimilarityRequest) =>
  post<SimilarityResponse>('/api/megaeval/similarity', body);

// ---------------------------------------------------------------- review sampling

export const getRunReview = (runId: string) =>
  get<ReviewState>(`/api/megaeval/runs/${encodeURIComponent(runId)}/review`);

export const planReview = (runId: string, targetN?: number) =>
  post<ReviewState>(
    `/api/megaeval/runs/${encodeURIComponent(runId)}/review/plan`,
    targetN !== undefined ? { target_n: targetN } : {}
  );

export const executeReview = (runId: string) =>
  post<ReviewState>(`/api/megaeval/runs/${encodeURIComponent(runId)}/review/execute`);

// ---------------------------------------------------------------- distributions

export const getRunDistributions = (runId: string) =>
  get<DistributionsResponse>(`/api/megaeval/runs/${encodeURIComponent(runId)}/distributions`);

// ---------------------------------------------------------------- formatting

/** Compact count formatting for large aggregates: 950 -> "950", 214_700 -> "214.7K", 1_300_000 -> "1.3M". */
export function fmtCompact(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return Math.round(n).toLocaleString();
}
