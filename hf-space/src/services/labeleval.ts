/**
 * Typed fetch wrappers for the L4 Perception Label Evaluation API,
 * plus a useStream hook (SSE with polling fallback) and a usePoll helper.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type {
  AnomalyConfig,
  Alert,
  AuditEvent,
  BenchmarkResponse,
  CopilotExplainRequest,
  CopilotExplainResponse,
  DatasetSummary,
  EvaluationRecord,
  FrameSummary,
  FunnelResponse,
  GateLine,
  HaystackPoint,
  ModelSummary,
  OverviewResponse,
  PipelineStateResponse,
  ProcessUnitsResponse,
  QualityGroupDetail,
  QualityGroupsResponse,
  QualityMetrics,
  QueueStatus,
  RareEvent,
  RegressionResponse,
  ReviewActionRequest,
  ReviewActionResponse,
  ReviewTask,
  StreamEvent,
  TrainingJobStatus,
  TrainRequest,
  TrainResponse,
} from '../types/labeleval';
import type { MegaevalStatus } from '../types/megaeval';
import { getMegaevalStatus } from './megaeval';

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

// ---------------------------------------------------------------- responses not in the shared contract

export interface PrecheckResponse {
  status: string;
  message: string;
  checks: GateLine[];
}

export interface GenerateDatasetRequest {
  name?: string;
  num_sequences?: number;
  frames_per_sequence?: number;
  seed?: number;
}

export interface RunResponse {
  run_id: string;
  status: string;
}

export interface FrameResponse {
  frame: FrameSummary;
  prev: string | null;
  next: string | null;
}

// ---------------------------------------------------------------- endpoints

export const getOverview = () => get<OverviewResponse>('/api/labeleval/overview');
export const getPipeline = () => get<PipelineStateResponse>('/api/labeleval/pipeline');
export const getFunnel = () => get<FunnelResponse>('/api/labeleval/funnel');
export const getQueueStatus = () => get<QueueStatus>('/api/queue/status');

export const getDatasets = () => get<{ datasets: DatasetSummary[] }>('/api/datasets');
export const getDataset = (id: string) => get<DatasetSummary>(`/api/datasets/${encodeURIComponent(id)}`);
export const precheckDataset = (datasetId?: string) =>
  post<PrecheckResponse>('/api/dataset/precheck', datasetId ? { dataset_id: datasetId } : {});
export const generateDataset = (body: GenerateDatasetRequest) =>
  post<DatasetSummary>('/api/labeleval/datasets/generate', body);
export const runPipeline = (datasetId: string, policyId?: string) =>
  post<RunResponse>('/api/labeleval/run', { dataset_id: datasetId, ...(policyId ? { policy_id: policyId } : {}) });

export const getFrameIds = (datasetId: string) =>
  get<{ frame_ids: string[] }>(`/api/labeleval/frames?dataset_id=${encodeURIComponent(datasetId)}`);
export const getFrame = (frameId: string) =>
  get<FrameResponse>(`/api/labeleval/frames/${encodeURIComponent(frameId)}`);
export const getEvaluation = (annotationId: string) =>
  get<EvaluationRecord>(`/api/labeleval/evaluations/${encodeURIComponent(annotationId)}`);
export const getHaystack = (datasetId?: string | null) =>
  get<{ points: HaystackPoint[] }>(
    datasetId ? `/api/labeleval/haystack?dataset_id=${encodeURIComponent(datasetId)}` : '/api/labeleval/haystack'
  );
export const getRareEvents = () => get<{ events: RareEvent[] }>('/api/labeleval/rare-events');

export const getQualityMetrics = (datasetId?: string | null) =>
  get<QualityMetrics>(
    datasetId ? `/api/quality/metrics?dataset_id=${encodeURIComponent(datasetId)}` : '/api/quality/metrics'
  );
export const getQualityGroups = (datasetId?: string | null) =>
  get<QualityGroupsResponse>(
    datasetId ? `/api/quality/groups?dataset_id=${encodeURIComponent(datasetId)}` : '/api/quality/groups'
  );
export const getQualityGroupDetail = (groupId: string) =>
  get<QualityGroupDetail>(`/api/quality/groups/${encodeURIComponent(groupId)}`);

export const getRegression = () => get<RegressionResponse>('/api/regression');

export const getReviewTasks = () => get<{ tasks: ReviewTask[] }>('/api/review/tasks');
export const getReviewTask = (id: string) => get<ReviewTask>(`/api/review/tasks/${encodeURIComponent(id)}`);
export const postReviewAction = (id: string, body: ReviewActionRequest) =>
  post<ReviewActionResponse>(`/api/review/tasks/${encodeURIComponent(id)}`, body);

export const getModels = () => get<{ models: ModelSummary[] }>('/api/models');
export const postTrain = (body: TrainRequest) => post<TrainResponse>('/api/train', body);
export const getTrainJobs = () => get<{ jobs: TrainingJobStatus[] }>('/api/train/jobs');
export const getTrainJob = (jobId: string) =>
  get<TrainingJobStatus>(`/api/train/jobs/${encodeURIComponent(jobId)}`);

export const getAnomalyConfig = () => get<AnomalyConfig>('/api/labeleval/config');
export const postAnomalyConfig = (body: AnomalyConfig) => post<AnomalyConfig>('/api/labeleval/config', body);

export const runBenchmark = (datasetId?: string | null) =>
  post<BenchmarkResponse>('/api/benchmark/techniques', datasetId ? { dataset_id: datasetId } : {});
export const getBenchmark = () => get<BenchmarkResponse>('/api/benchmark/techniques');

export const getProcessUnits = () => get<ProcessUnitsResponse>('/api/labeleval/process-units');
export const getAlerts = () => get<{ alerts: Alert[] }>('/api/labeleval/alerts');
export const getAudit = (limit = 200) => get<{ events: AuditEvent[] }>(`/api/labeleval/audit?limit=${limit}`);

export const copilotExplain = (body: CopilotExplainRequest) =>
  post<CopilotExplainResponse>('/api/copilot/explain', body);

// ---------------------------------------------------------------- polling hook

export interface PollState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/**
 * Simple poll-based data fetching. Pass intervalMs = null to fetch once.
 * `deps` re-triggers the initial fetch (with loading state) when changed.
 */
export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs: number | null,
  deps: ReadonlyArray<unknown> = []
): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    setLoading(true);
    const run = async () => {
      try {
        const result = await fetcherRef.current();
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) {
          setLoading(false);
          if (intervalMs !== null) timer = setTimeout(run, intervalMs);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, tick, ...deps]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);
  return { data, loading, error, refresh };
}

// ---------------------------------------------------------------- SSE stream hook

/**
 * Subscribes to /api/events/stream via EventSource. If the stream errors,
 * falls back to polling GET /api/labeleval/pipeline every 2s (synthesizing a
 * StreamEvent so consumers see a single shape).
 */
export function useStream(): StreamEvent | null {
  const [event, setEvent] = useState<StreamEvent | null>(null);

  useEffect(() => {
    let disposed = false;
    let es: EventSource | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let lastAlerts = 0;

    const startPolling = () => {
      if (pollTimer || disposed) return;
      const poll = async () => {
        try {
          const pipeline = await getPipeline();
          try {
            lastAlerts = (await getAlerts()).alerts.length;
          } catch {
            /* keep previous alert count */
          }
          let megaeval: MegaevalStatus | undefined;
          try {
            megaeval = await getMegaevalStatus();
          } catch {
            /* megaeval layer unavailable — leave the field absent */
          }
          if (!disposed) {
            setEvent({
              ts: new Date().toISOString(),
              pipeline,
              training: null,
              alerts_count: lastAlerts,
              ...(megaeval ? { megaeval } : {}),
            });
          }
        } catch {
          /* backend offline — keep last known state */
        }
      };
      void poll();
      pollTimer = setInterval(() => void poll(), 2000);
    };

    try {
      es = new EventSource('/api/events/stream');
      es.onmessage = (msg: MessageEvent<string>) => {
        try {
          setEvent(JSON.parse(msg.data) as StreamEvent);
        } catch {
          /* ignore malformed event */
        }
      };
      es.onerror = () => {
        if (es) {
          es.close();
          es = null;
        }
        startPolling();
      };
    } catch {
      startPolling();
    }

    return () => {
      disposed = true;
      if (es) es.close();
      if (pollTimer) clearInterval(pollTimer);
    };
  }, []);

  return event;
}
