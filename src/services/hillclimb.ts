/** Typed fetch wrappers for the Hill Climbing EM API (/api/hillclimb). */

import type {
  Blueprint,
  DesignChallenge,
  DesignComponent,
  DesignEdge,
  DesignGrade,
  DiagnosticSession,
  EvidenceListResponse,
  Exercise,
  GraphResponse,
  InterviewSession,
  Journey,
  NextBestAction,
  ReadinessResponse,
  SimulationCatalog,
  SimulationState,
  StarDiagnosis,
  SubmitExerciseResponse,
  UserProfile,
} from '../types/hillclimb';

const BASE = '/api/hillclimb';
const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

function get<T>(url: string): Promise<T> {
  return fetch(`${BASE}${url}`).then((res) => handle<T>(res));
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return fetch(`${BASE}${url}`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((res) => handle<T>(res));
}

function put<T>(url: string, body: unknown): Promise<T> {
  return fetch(`${BASE}${url}`, {
    method: 'PUT',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  }).then((res) => handle<T>(res));
}

// ------------------------------------------------------------------- profile

export const getProfile = () => get<UserProfile>('/profile');
export const saveProfile = (p: UserProfile) => put<UserProfile>('/profile', p);

// ----------------------------------------------------------------- blueprint

export const getBlueprint = () => get<Blueprint>('/blueprint');
export const getGraph = () => get<GraphResponse>('/graph');

// ---------------------------------------------------------------- diagnostic

export const startDiagnostic = (seed = 11) =>
  post<DiagnosticSession>('/diagnostic/start', { seed });
export const answerDiagnostic = (diagnosticId: string, answer: string) =>
  post<DiagnosticSession>(`/diagnostic/${encodeURIComponent(diagnosticId)}/answer`, { answer });

// ----------------------------------------------------------------- exercises

export const generateExercise = (competencyId: string, difficulty = 2, seed?: number) =>
  post<Exercise>('/exercise/generate', { competency_id: competencyId, difficulty, seed });
export const submitExercise = (exerciseId: string, answer: string, asAssessment = false) =>
  post<SubmitExerciseResponse>('/exercise/submit', {
    exercise_id: exerciseId,
    answer,
    as_assessment: asAssessment,
  });

// ---------------------------------------------------------------------- STAR

export const diagnoseStar = (text: string, saveEvidence: boolean) =>
  post<StarDiagnosis>('/star/diagnose', { text, save_evidence: saveEvidence });

// ----------------------------------------------------------------- design lab

export const getDesignChallenges = () =>
  get<{ challenges: DesignChallenge[]; component_types: string[] }>('/design/challenges');
export const submitDesign = (
  challengeId: string,
  components: DesignComponent[],
  edges: DesignEdge[],
  rationales: Record<string, string>
) =>
  post<DesignGrade>('/design/submit', {
    challenge_id: challengeId,
    components,
    edges,
    rationales,
  });

// ----------------------------------------------------------------- simulation

export const getSimulationCatalog = () => get<SimulationCatalog>('/simulation/catalog');
export const startSimulation = (seed = 42, maxTurns = 8) =>
  post<SimulationState>('/simulation/start', { seed, max_turns: maxTurns });
export const stepSimulation = (
  simId: string,
  hypothesis: string,
  interventionId: string,
  revertPrevious = false
) =>
  post<SimulationState>(`/simulation/${encodeURIComponent(simId)}/step`, {
    hypothesis,
    intervention_id: interventionId,
    revert_previous: revertPrevious,
  });

// ------------------------------------------------------------------ interview

export const startInterview = (mode: string, seed?: number) =>
  post<InterviewSession>('/interview/start', seed === undefined ? { mode } : { mode, seed });
export const answerInterview = (sessionId: string, answer: string) =>
  post<InterviewSession>(`/interview/${encodeURIComponent(sessionId)}/answer`, { answer });
export const endInterview = (sessionId: string) =>
  post<InterviewSession>(`/interview/${encodeURIComponent(sessionId)}/end`);

// ----------------------------------------------------- evidence / readiness

export const getEvidence = (competencyId?: string) =>
  get<EvidenceListResponse>(
    competencyId ? `/evidence?competency_id=${encodeURIComponent(competencyId)}` : '/evidence'
  );
export const getReadiness = () => get<ReadinessResponse>('/readiness');
export const getNextAction = () => get<NextBestAction>('/next-action');
export const getJourney = () => get<Journey>('/journey');
