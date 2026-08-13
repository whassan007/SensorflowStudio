/** API client for the Studio 2.0 control plane (/api/studio2). */

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const get = <T,>(url: string) => fetch(url).then((r) => handle<T>(r));
const post = <T,>(url: string, body?: unknown) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  }).then((r) => handle<T>(r));

// ------------------------------------------------------------------ types

export interface Studio2Status {
  registry_counts: Record<string, number>;
  dependencies: Record<string, string>;
  soft_dependencies: string[];
}

export interface RegistryEntity {
  entity_id: string;
  kind: string;
  name?: string;
  created_at?: string;
  provenance?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DatasetEntity extends RegistryEntity {
  role: string;
  protected_evaluation: boolean;
  governance_overrides: { actor: string; reason: string; timestamp: string }[];
  role_history: { from: string | null; to: string; actor: string }[];
  lineage: { parents: string[] };
}

export interface RunEntity extends RegistryEntity {
  engine: string;
  reproducibility: 'REPRODUCIBLE' | 'NON_REPRODUCIBLE';
  missing_components: string[];
  reproducibility_tuple: Record<string, unknown>;
  results?: Record<string, unknown>;
}

export interface ReleaseDecision {
  entity_id: string;
  status: 'GO' | 'REVIEW' | 'NO_GO';
  confidence: number;
  evidence_completeness: number;
  blocking_conditions: string[];
  unresolved_questions: string[];
  degraded_inputs: string[];
  human_approval_required: boolean;
  deployment_authorized: boolean;
  approval: { approver: string; rationale: string; approved_at: string } | null;
  policy_version: string;
  evidence_tuple: Record<string, unknown>;
  evaluated_at: string;
}

export interface HardwareRow {
  combination: Record<string, string>;
  combination_label: string;
  metrics: Record<string, number> | null;
  n: number;
  status: 'PASS' | 'FAIL' | 'INSUFFICIENT';
  critical: boolean;
  failed_checks?: string[];
  reason?: string;
  derivation?: Record<string, string>;
}

export interface HardwareMatrix {
  matrix_id: string;
  status: string;
  global_pass: boolean | null;
  global_metrics: Record<string, number> | null;
  global_vs_matrix_note: string | null;
  n_combinations: number;
  n_pass: number;
  n_fail: number;
  n_insufficient: number;
  min_support: { n: number; method: string };
  rows: HardwareRow[];
  insufficient: { combination_label: string; reason: string; critical: boolean }[];
  source_run_id?: string | null;
}

export interface FunnelStage {
  stage: string;
  label: string;
  available: boolean;
  source: string;
  data?: Record<string, unknown>;
  status?: string;
  reason?: string;
}

export interface FunnelPanel {
  available: boolean;
  source: string;
  data?: Record<string, unknown>;
  status?: string;
  reason?: string;
}

export interface Funnel {
  generated_at: string;
  stages: FunnelStage[];
  safety: FunnelPanel;
  model_comparison: FunnelPanel;
  drift: FunnelPanel;
  compute: FunnelPanel;
  availability: Record<string, boolean>;
}

export interface DemoStep {
  step: string;
  available: boolean;
  [key: string]: unknown;
}

export interface DemoResult {
  demo_id: string;
  seed: number;
  steps: DemoStep[];
  decision: ReleaseDecision;
  regression_dataset: DatasetEntity | null;
  generated_at: string;
}

// ------------------------------------------------------------------ calls

export const getStatus = () => get<Studio2Status>('/api/studio2/status');

export const listEntities = (kind: string) =>
  get<{ kind: string; entities: RegistryEntity[] }>(`/api/studio2/registry/${kind}`);

export const runIngest = () =>
  post<{ registered: Record<string, number>; totals: Record<string, number> }>(
    '/api/studio2/registry/ingest'
  );

export const listDecisions = () =>
  get<{ decisions: ReleaseDecision[] }>('/api/studio2/release/decisions');

export const evaluateRelease = () =>
  post<ReleaseDecision>('/api/studio2/release/evaluate', { use_live_sources: true });

export const approveDecision = (decisionId: string, approver: string, rationale: string) =>
  post<ReleaseDecision>(`/api/studio2/release/decisions/${decisionId}/approve`, {
    approver,
    rationale,
  });

export const getHardwareMatrix = (refresh = false) =>
  get<HardwareMatrix>(`/api/studio2/hardware/matrix${refresh ? '?refresh=true' : ''}`);

export const getFunnel = () => get<Funnel>('/api/studio2/funnel');

export const runDemo = (seed?: number) =>
  post<DemoResult>('/api/studio2/demo/run', seed !== undefined ? { seed } : {});

export const getLatestDemo = () => get<DemoResult>('/api/studio2/demo/latest');

export const listDocs = () => get<{ docs: string[] }>('/api/studio2/docs');

export const getDoc = (name: string) =>
  get<{ name: string; content: string }>(`/api/studio2/docs/${name}`);
