/**
 * Typed fetch wrappers + contracts for the BEV-Fusion perception engine
 * comparison (/api/bevfusion/*) and the studio-ux frame replay used by the
 * interactive top-down BEV canvas.
 */

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

// ---------------------------------------------------------------- contracts

export interface BevHeadline {
  precision: number;
  recall: number;
  f1: number;
  mean_iou: number;
  position_error_m: number;
  class_accuracy: number;
  safety_recall: number;
  id_switch_rate: number;
  fragmentation_rate: number;
  idf1: number;
}

export interface BevCohortStats {
  n_gt: number;
  recall: number;
  position_error_m: number;
  mean_iou: number;
  false_positives: number;
}

export interface BevEngineEval {
  headline: BevHeadline;
  counts: { tp: number; fp: number; fn: number; pred_tracks: number; gt_tracks: number };
  cohorts: Record<string, BevCohortStats>;
  per_class: Record<string, { n_gt: number; recall: number }>;
}

export interface BevHeadlineDelta {
  metric: string;
  baseline: number;
  candidate: number;
  delta: number;
  improved: boolean;
}

export interface BevCohortDelta {
  cohort: string;
  n_gt: number;
  recall_baseline: number;
  recall_candidate: number;
  recall_delta: number;
  position_error_baseline_m: number;
  position_error_candidate_m: number;
  mean_iou_baseline: number;
  mean_iou_candidate: number;
  explanation: string;
}

export interface BevClassDelta {
  class: string;
  n_gt: number;
  recall_baseline: number;
  recall_candidate: number;
  recall_delta: number;
}

export interface BevReport {
  run_id: string;
  created_at: string;
  params: { n_sequences: number; frames_per_sequence: number; seed: number };
  engines: { baseline: string; candidate: string };
  scale: { frames: number; gt_boxes: number; camera_detections: number; lidar_detections: number; fused_boxes: number };
  baseline: BevEngineEval;
  candidate: BevEngineEval;
  headline_deltas: BevHeadlineDelta[];
  per_cohort: BevCohortDelta[];
  per_class: BevClassDelta[];
  policy: Record<string, number>;
  blockers: string[];
  improvements: string[];
  recommendation: 'PROMOTE' | 'DO_NOT_PROMOTE';
  process_units: Record<string, unknown>;
  notes: string;
}

export interface BevStatus {
  engines: { baseline: string; candidate: string };
  n_runs: number;
  runs: Array<{ run_id: string; created_at: string; params: BevReport['params']; recommendation: string | null }>;
  latest: { run_id: string; created_at: string; params: BevReport['params']; recommendation: string | null } | null;
  ready: boolean;
}

export const getBevStatus = () => get<BevStatus>('/api/bevfusion/status');

export const getBevReport = () => get<BevReport>('/api/bevfusion/report');

export const runBevComparison = (body: { n_sequences?: number; frames_per_sequence?: number; seed?: number }) =>
  post<BevReport>('/api/bevfusion/run', body);

// ---------------------------------------------------------------- frame replay (studio-ux)

/** [x, y, z, l, w, h, yaw] in ego frame. */
export type Bbox3D = [number, number, number, number, number, number, number];

export interface BevGtBox {
  instance_id: string;
  class_name: string;
  bbox_3d: Bbox3D;
  occluded: boolean;
  distance: number;
}

export interface BevDetection {
  modality: 'camera' | 'lidar';
  x: number;
  y: number;
  cov: [[number, number], [number, number]];
  dims: [number, number, number];
  yaw: number;
  class_name: string;
  confidence: number;
}

export interface BevFusedBox {
  bbox_3d: Bbox3D;
  class_name: string;
  confidence: number;
  track_id: number | string;
  propagated: boolean;
}

export interface BevReplayFrame {
  frame_id: string;
  index: number;
  gt: BevGtBox[];
  camera: BevDetection[];
  lidar: BevDetection[];
  fused: BevFusedBox[];
  baseline: BevFusedBox[];
}

export interface BevReplayResponse {
  sequence_id: string;
  sequence_index: number;
  n_sequences: number;
  time_of_day: string;
  weather: string;
  params: BevReport['params'];
  frames: BevReplayFrame[];
}

export const getBevReplay = (params: { seed: number; n_sequences: number; frames_per_sequence: number }, sequence: number) =>
  get<BevReplayResponse>(
    `/api/studio-ux/bev/replay?seed=${params.seed}&n_sequences=${params.n_sequences}` +
      `&frames_per_sequence=${params.frames_per_sequence}&sequence=${sequence}`
  );
