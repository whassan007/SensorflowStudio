/** Typed fetch wrappers for the Vitis acceleration layer API (/api/vitis). */
import type {
  AugmentBatch,
  AugmentationSpec,
  BackendStatus,
  HilRun,
  HilSweep,
  IspRun,
  PrdDoc,
  PrdListEntry,
  RunListEntry,
  TemporalRun,
} from '../types/vitis';

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

// ------------------------------------------------------------- backends

export const getBackendStatus = () => get<BackendStatus>('/api/vitis/backends/status');

// ------------------------------------------------------------- HIL

export interface HilRunRequest {
  n_sequences?: number;
  frames_per_sequence?: number;
  seed?: number;
  width_bits?: number;
  int_bits?: number;
  max_line_buffer_depth?: number;
  use_lut_approx?: boolean;
  lut_bits?: number;
  device?: string;
  regression_delta?: number;
  alpha?: number;
  run_ablation?: boolean;
}

export const runHil = (body: HilRunRequest) => post<HilRun>('/api/vitis/hil/run', body);

export const runHilSweep = (body: HilRunRequest & { widths?: number[] }) =>
  post<HilSweep>('/api/vitis/hil/sweep', body);

export const listHilRuns = () => get<{ runs: RunListEntry[] }>('/api/vitis/hil/runs');

// ------------------------------------------------------------- ISP

export interface IspRunRequest {
  n_frames?: number;
  seed?: number;
  stages?: string[];
  stage_params?: Record<string, unknown>;
  width_bits?: number;
  int_bits?: number;
  max_line_buffer_depth?: number;
  use_lut_approx?: boolean;
  device?: string;
  include_previews?: boolean;
}

export const runIsp = (body: IspRunRequest) => post<IspRun>('/api/vitis/isp/run', body);

export const getIspStages = () => get<{ stages: string[] }>('/api/vitis/isp/stages');

// ------------------------------------------------------------- augment

export const getAugmentRecipes = () =>
  get<{ augmentations: AugmentationSpec[] }>('/api/vitis/augment/recipes');

export const generateAugmentBatch = (body: {
  recipes?: { aug: string; params?: Record<string, number> }[];
  n_variants?: number;
  seed?: number;
  backend?: string;
  width_bits?: number;
  device?: string;
  include_thumbnails?: boolean;
}) => post<AugmentBatch>('/api/vitis/augment/generate', body);

// ------------------------------------------------------------- temporal

export const getTemporalEngines = () =>
  get<{ engines: string[] }>('/api/vitis/temporal/engines');

export const runTemporal = (body: {
  engines?: string[];
  n_sequences?: number;
  frames_per_sequence?: number;
  seed?: number;
  width_bits?: number;
  device?: string;
}) => post<TemporalRun>('/api/vitis/temporal/run', body);

// ------------------------------------------------------------- PRD

export const listPrds = () => get<{ prds: PrdListEntry[] }>('/api/vitis/prd');

export const getPrd = (id: string) =>
  get<PrdDoc>(`/api/vitis/prd/${encodeURIComponent(id)}`);
