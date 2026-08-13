/**
 * Typed fetch wrappers for the rare-event miner API (/api/raremine).
 * Mirrors the style of services/megaeval.ts.
 */
import type {
  CandidateDetail,
  CandidatesResponse,
  CuratorReport,
  DedupReport,
  DestinationsResponse,
  DiversityReport,
  ImprovementReport,
  LineageRecord,
  QuantvalReport,
  SceneView,
  StatusResponse,
  TrackView,
} from '../types/raremine';

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

export const getStatus = () => get<StatusResponse>('/api/raremine/status');

export const generateScenes = (n: number, seed: number) =>
  post<{ status: string; bank: StatusResponse['bank'] }>('/api/raremine/scenes/generate', { n, seed });

export const runMining = (diversityBudget = 12, config: Record<string, unknown> = {}) =>
  post<{ status: string; run: StatusResponse['last_run']; dedup_report: DedupReport; diversity_report: DiversityReport }>(
    '/api/raremine/mine',
    { diversity_budget: diversityBudget, config }
  );

export const listCandidates = (filters: {
  priority?: string;
  status?: string;
  costume?: string;
  difficulty?: string;
  detected_only?: boolean;
} = {}) => {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== '') params.set(k, String(v));
  }
  const qs = params.toString();
  return get<CandidatesResponse>(`/api/raremine/candidates${qs ? `?${qs}` : ''}`);
};

export const getCandidate = (id: string) =>
  get<CandidateDetail>(`/api/raremine/candidates/${encodeURIComponent(id)}`);

export const getCandidateScene = (id: string, sceneId?: string) =>
  get<SceneView>(
    `/api/raremine/candidates/${encodeURIComponent(id)}/scene${sceneId ? `?scene_id=${encodeURIComponent(sceneId)}` : ''}`
  );

export const getTracks = () => get<{ count: number; tracks: TrackView[] }>('/api/raremine/tracks');

export const reviewCandidate = (id: string, action: 'approve' | 'reject', note: string, destination?: string) =>
  post<CandidateDetail & { lineage: LineageRecord | null }>(`/api/raremine/review/${encodeURIComponent(id)}`, {
    action,
    note,
    destination: destination ?? null,
  });

export const promoteToTraining = (trackCandidateId: string, curator = 'ui-curator') =>
  post<{ status: string; lineage: LineageRecord }>('/api/raremine/governance/promote-training', {
    track_candidate_id: trackCandidateId,
    curator,
  });

export const governanceOverride = (trackCandidateId: string, actor: string, reason: string) =>
  post<{ status: string; lineage: LineageRecord }>('/api/raremine/governance/override', {
    track_candidate_id: trackCandidateId,
    actor,
    reason,
  });

export const getDestinations = () => get<DestinationsResponse>('/api/raremine/destinations');
export const getDedupReport = () => get<DedupReport>('/api/raremine/dedup/report');
export const getDiversityReport = () => get<DiversityReport>('/api/raremine/diversity/report');
export const getQuantvalReport = () => get<QuantvalReport>('/api/raremine/reports/quantval');
export const getCuratorReport = () => get<CuratorReport>('/api/raremine/reports/curator');
export const getImprovementReport = () => get<ImprovementReport>('/api/raremine/reports/improvement');

export const explainCandidate = (id: string) =>
  post<{ status: string; provider: string; analysis: string }>(
    `/api/raremine/candidates/${encodeURIComponent(id)}/explain`
  );
