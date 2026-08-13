/** Types for the Hill Climbing EM API (/api/hillclimb). */

export type Dimension = 'Knowledge' | 'Technical Reasoning' | 'Leadership' | 'Execution';

export type ReadinessStateId =
  | 'NOT_STARTED'
  | 'LEARNING'
  | 'PRACTICING'
  | 'NEEDS_REVIEW'
  | 'COMPETENT'
  | 'STRONG'
  | 'INTERVIEW_READY';

export type JourneyStateId =
  | 'NOT_STARTED'
  | 'DIAGNOSTIC'
  | 'LEARNING'
  | 'PRACTICE'
  | 'ASSESSMENT'
  | 'REMEDIATION'
  | 'REASSESS';

export interface UserProfile {
  user_id: string;
  name: string;
  target_role: string;
  experience_years: number;
  focus_note: string;
  created_at: string;
  updated_at: string;
}

// ------------------------------------------------------------------ blueprint

export interface Competency {
  id: string;
  name: string;
  phase: number;
  dimension: Dimension;
  description: string;
  prerequisites: string[];
  topics: string[];
}

export interface PhaseSpec {
  phase: number;
  title: string;
  objective: string;
  topics: string[];
  skills: string[];
  exercises: string[];
  assessments: string[];
  completion_criteria: string[];
}

export interface Blueprint {
  version: number;
  source: string;
  note: string;
  phases: PhaseSpec[];
  competencies: Competency[];
}

export interface GraphResponse {
  nodes: Competency[];
  edges: { source: string; target: string }[];
  problems: string[];
}

// ----------------------------------------------------------------- evaluation

export interface EvaluationResult {
  competency: string;
  score: number;
  confidence: number;
  evidence: string[];
  strengths: string[];
  weaknesses: string[];
  missing_evidence: string[];
  misconceptions: string[];
  recommended_action: string;
  follow_up_question: string;
  evaluator: string;
}

export interface RubricItem {
  criterion: string;
  check: string;
  keywords: string[];
  weight: number;
}

export interface LinkedTool {
  label: string;
  page: string;
  api: string;
}

export interface Exercise {
  exercise_id: string;
  competency_id: string;
  difficulty: number;
  prerequisites: string[];
  scenario: string;
  expected_reasoning: string[];
  evaluation_rubric: RubricItem[];
  common_failure_modes: string[];
  follow_up_questions: string[];
  template_id: string;
  seed: number;
  family: string;
  linked_tool: LinkedTool | null;
}

export interface CompetencyReadiness {
  competency_id: string;
  knowledge_score: number;
  application_score: number;
  evidence_score: number;
  readiness_state: ReadinessStateId;
  evidence_ids: string[];
  last_updated: string;
}

export interface SubmitExerciseResponse {
  evaluation: EvaluationResult;
  coaching: string;
  attempt_id: string;
  evidence_id: string;
  linked_tool: LinkedTool | null;
  readiness_for_competency: CompetencyReadiness | null;
  journey: Journey | null;
}

// ------------------------------------------------------------------ diagnostic

export interface DiagnosticQuestion {
  exercise_id: string;
  competency_id: string;
  scenario: string;
}

export interface DiagnosticSession {
  diagnostic_id: string;
  user_id: string;
  seed: number;
  total_questions: number;
  answered: number;
  status: 'active' | 'complete';
  current_exercise_id: string | null;
  current_competency: string | null;
  asked: string[];
  results: { competency_id: string; score: number; evaluation: EvaluationResult }[];
  current_question?: DiagnosticQuestion;
}

// ------------------------------------------------------------------------ STAR

export interface StarComponent {
  component: 'S' | 'T' | 'A' | 'R';
  label: string;
  sentences: string[];
  present: boolean;
  issues: string[];
  strengths: string[];
}

export interface ClaimFlag {
  sentence: string;
  kind: 'unquantified_claim' | 'measurable_evidence';
  detail: string;
}

export interface StarCheck {
  check: string;
  label: string;
  passed: boolean;
  detail: string;
  evidence: string | null;
}

export interface StarDiagnosis {
  components: StarComponent[];
  checks: StarCheck[];
  claim_flags: ClaimFlag[];
  competencies: { competency_id: string; matched_cues: string[]; reason: string }[];
  overall_score: number;
  coaching: string[];
  evidence_id: string | null;
}

// ------------------------------------------------------------------ design lab

export interface DesignChallenge {
  challenge_id: string;
  title: string;
  brief: string;
  required_stages: string[][];
  key_decisions: string[];
  competency_ids: string[];
  requires_feedback_loop: boolean;
}

export interface DesignComponent {
  id: string;
  type: string;
  name: string;
  note: string;
}

export interface DesignEdge {
  source: string;
  target: string;
}

export interface StructuralChecks {
  missing_stages: string[];
  orphan_components: string[];
  feedback_loop_closed: boolean;
  feedback_loop_required: boolean;
  single_points_of_failure: string[];
  capacity_math_found: boolean;
  capacity_quotes: string[];
}

export interface DimensionGrade {
  dimension: string;
  score: number;
  gaps: string[];
  evidence: string[];
}

export interface DesignGrade {
  grade_id: string;
  challenge_id: string;
  structural: StructuralChecks;
  dimension_grades: DimensionGrade[];
  overall_score: number;
  gaps: string[];
  evidence_id: string | null;
}

// ------------------------------------------------------------------ simulation

export interface InterventionSpec {
  id: string;
  label: string;
  description: string;
  immediate: Record<string, number>;
  delayed: { delay: number; effects: Record<string, number>; note?: string }[];
  second_order: string;
}

export interface SimulationCatalog {
  scenarios: { scenario_id: string; title: string; narrative: string }[];
  interventions: InterventionSpec[];
  metrics: string[];
  inverted_metrics: string[];
  safety_floor: number;
  morale_floor: number;
}

export interface SimTurnRecord {
  turn: number;
  hypothesis: string;
  hypothesis_assessment: {
    mentions_metrics: string[];
    targets_intervention_effects: boolean;
    directional: boolean;
    falsifiable: boolean;
    quality: string;
  };
  intervention_id: string;
  reverted_previous: boolean;
  applied_effects: Record<string, number>;
  delayed_landed: string[];
  events: string[];
  metrics_after: Record<string, number>;
  objective_after: number;
  objective_delta: number;
  verdict: string;
  timestamp: string;
}

export interface SimCompetencyMapping {
  competency_id: string;
  verdict: 'evidenced' | 'gap';
  reason: string;
  quotes: string[];
  evidence_id?: string;
}

export interface SimDebrief {
  objective_start: number;
  objective_end: number;
  objective_delta: number;
  incidents: number;
  turns: number;
  balanced_finish: boolean;
  competency_mappings: SimCompetencyMapping[];
}

export interface SimulationState {
  sim_id: string;
  user_id: string;
  scenario_id: string;
  seed: number;
  max_turns: number;
  turn: number;
  status: 'active' | 'complete';
  metrics: Record<string, number>;
  pending: { due_turn: number; effects: Record<string, number>; note: string }[];
  history: SimTurnRecord[];
  events: { turn: number; type: string; detail: string }[];
  objective_history: number[];
  debrief: SimDebrief | null;
}

// -------------------------------------------------------------------- interview

export interface InterviewTurn {
  index: number;
  question: string;
  question_type: 'opening' | 'probe' | 'depth_probe' | 'escalate' | 'advance';
  competency_id: string;
  exercise_id: string;
  difficulty: number;
  answer: string | null;
  evaluation: EvaluationResult | null;
  timestamp: string;
}

export interface InterviewSession {
  session_id: string;
  user_id: string;
  mode: 'technical' | 'management' | 'hybrid';
  seed: number;
  status: 'active' | 'complete';
  turns: InterviewTurn[];
  asked_competencies: string[];
  probe_count_on_current: number;
  evidence_id: string | null;
}

// -------------------------------------------------------- evidence / readiness

export interface EvidenceItem {
  evidence_id: string;
  user_id: string;
  competency_ids: string[];
  artifact_type: string;
  source: string;
  summary: string;
  quotes: string[];
  score: number;
  confidence: number;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface EvidenceListResponse {
  evidence: EvidenceItem[];
  count: number;
}

export interface ReadinessRow extends CompetencyReadiness {
  name: string;
  phase: number;
  dimension: Dimension;
  prerequisites: string[];
}

export interface DimensionSummary {
  dimension: string;
  avg_score: number;
  coverage: number;
  competent: number;
  total: number;
}

export interface Bottleneck {
  competency_id: string;
  name: string;
  phase: number;
  dimension: Dimension;
  readiness: CompetencyReadiness;
  blocked_competencies: string[];
  blocked_count: number;
  explanation: string;
}

export interface ReadinessResponse {
  matrix: ReadinessRow[];
  dimensions: DimensionSummary[];
  bottleneck: Bottleneck | null;
}

export interface NextBestAction {
  bottleneck: Bottleneck | null;
  explanation: string;
  concept: {
    competency_id: string;
    name: string;
    study: string[];
    description: string;
    phase_objective: string;
  };
  exercise: Exercise;
  assessment: {
    competency_id: string;
    kind: string;
    description: string;
  };
}

export interface Journey {
  user_id: string;
  state: JourneyStateId;
  current_phase: number;
  current_competency: string | null;
  remediation_target: string | null;
  history: Record<string, unknown>[];
  updated_at: string;
}
