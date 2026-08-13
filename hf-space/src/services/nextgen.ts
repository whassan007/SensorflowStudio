/**
 * Typed fetch wrappers for the Next-Gen evaluation API (/api/nextgen).
 * Mirrors the style of services/rca.ts.
 */
import type {
  ArchitectureDocs,
  BehavioralAssessment,
  CatalogueEntry,
  CausalReplayResult,
  ComputeReport,
  CounterfactualScenario,
  DivergenceDemo,
  GauntletResult,
  NextgenStatus,
  SuiteWeights,
  TransformationStep,
  ValidityReport,
} from '../types/nextgen';

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

export const getStatus = () => get<NextgenStatus>('/api/nextgen/status');

export const getCatalogue = () =>
  get<{ transformations: CatalogueEntry[] }>('/api/nextgen/counterfactuals/catalogue');

export const generateCounterfactuals = (body: {
  recipe: TransformationStep[];
  seed?: number;
  n_scenarios?: number;
  frames_per_sequence?: number;
}) => post<{ scenarios: CounterfactualScenario[] }>('/api/nextgen/counterfactuals/generate', body);

export const listCounterfactuals = () =>
  get<{ scenarios: CounterfactualScenario[] }>('/api/nextgen/counterfactuals');

export const validateCounterfactual = (scenarioId: string) =>
  post<ValidityReport>(`/api/nextgen/counterfactuals/${encodeURIComponent(scenarioId)}/validate`);

export const getSuiteWeights = () =>
  get<SuiteWeights>('/api/nextgen/counterfactuals/suite-weights');

export const runReplay = (body: {
  scenario_id: string;
  engine?: string;
  seed?: number;
  corrected?: boolean;
  faults?: Record<string, unknown>[];
}) => post<BehavioralAssessment>('/api/nextgen/simulation/replay', body);

export const runCausalReplay = (body: {
  scenario_id: string;
  engine?: string;
  seed?: number;
  faults?: Record<string, unknown>[];
}) => post<CausalReplayResult>('/api/nextgen/causal/replay', body);

export const getDivergenceDemo = () =>
  get<DivergenceDemo>('/api/nextgen/metrics/divergence-demo');

export const runGauntlet = (body: {
  candidate_version?: string;
  baseline_version?: string;
  effects?: Record<string, number>;
  budget_units?: number;
  batch_units?: number;
  seed?: number;
}) => post<GauntletResult>('/api/nextgen/gauntlet/run', body);

export const getGauntletResults = (runId: string) =>
  get<GauntletResult>(`/api/nextgen/gauntlet/${encodeURIComponent(runId)}/results`);

export const runComputeBenchmark = (body: { n_scenarios?: number; frames_per_sequence?: number }) =>
  post<ComputeReport>('/api/nextgen/compute/benchmark', body);

export const getComputeReport = () => get<ComputeReport>('/api/nextgen/compute/report');

export const getArchitectureDocs = () =>
  get<ArchitectureDocs>('/api/nextgen/architecture/docs');
