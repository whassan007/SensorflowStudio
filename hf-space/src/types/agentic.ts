/** Types for the Agentic Launch Readiness subsystem (/api/agentic). */

export type EvidenceStatus = 'OBSERVED' | 'DERIVED' | 'HYPOTHESIS' | 'UNAVAILABLE';
export type EvidenceQuality = 'CONFIRMED' | 'LIKELY' | 'POSSIBLE' | 'INSUFFICIENT_EVIDENCE';
export type Severity = 'S0' | 'S1' | 'S2' | 'S3' | 'S4' | 'S5';
export type PolicyOutcome =
  | 'AUTOMATIC_STOP_SHIP'
  | 'LAUNCH_REVIEW_REQUIRED'
  | 'CONTINUE_INVESTIGATION'
  | 'NO_LAUNCH_IMPACT'
  | 'INDETERMINATE';
export type OptionCode =
  | 'STOP_SHIP'
  | 'OPTION_A_DELAY'
  | 'OPTION_B_MITIGATION'
  | 'OPTION_C_REDUCED_ODD'
  | 'EXPAND_EVALUATION'
  | 'HUMAN_SAFETY_REVIEW'
  | 'PROCEED';
export type StageStatus = 'pending' | 'running' | 'complete' | 'failed' | 'blocked';

export const STAGES = [
  'FAILURE_DETECTION',
  'EVIDENCE_AGGREGATION',
  'FAILURE_ANALYSIS',
  'LAUNCH_DECISION',
  'LEARNING_FLYWHEEL',
] as const;
export type Stage = (typeof STAGES)[number];

export interface FailureInstance {
  instance_id: string;
  sequence_id: string;
  frame_id: string;
  frame_index: number;
  object_instance_id: string;
  gt_class: string;
  predicted_class: string;
  confidence: number;
  distance_m: number;
  construction_zone: boolean;
  time_of_day: string;
  weather: string;
  geo_bucket: string;
  occluded: boolean;
  has_planner_trace: boolean;
}

export interface DetectionBasis {
  method: string;
  candidate_events: number;
  baseline_events: number;
  denominator: number;
  candidate_rate: number;
  baseline_rate: number;
  metric_deltas: Record<string, number>;
  notes: string[];
}

export interface FailureEvent {
  failure_id: string;
  kind: string;
  title: string;
  description: string;
  gt_class: string | null;
  predicted_class: string | null;
  detection_basis: DetectionBasis;
  instances: FailureInstance[];
  baseline_model: string;
  candidate_model: string;
  status: string;
  severity: Severity | null;
  policy_outcome: PolicyOutcome | null;
  validated: boolean;
  created_at: string;
}

export interface AgentEscalation {
  required: boolean;
  reasons: string[];
  human_review_triggers: string[];
}

export interface AgentResult {
  agent: string;
  agent_version: string;
  failure_id: string;
  status: 'ok' | 'failed' | 'escalated';
  authority: 'ADVISORY_ONLY';
  output: Record<string, unknown>;
  confidence: number;
  confidence_basis: string;
  epistemic_status: EvidenceStatus;
  escalation: AgentEscalation;
  llm_used: boolean;
  llm_rationale: string | null;
}

export interface EvidenceNode {
  node_id: string;
  node_type: string;
  status: EvidenceStatus;
  summary: string;
  fields: Record<string, unknown>;
  source: string;
  caveats: string[];
}

export interface EvidenceGraph {
  failure_id: string;
  nodes: EvidenceNode[];
  edges: { src: string; dst: string; relation: string }[];
  built_at: string;
}

export interface RateEstimate {
  events: number;
  denominator: number;
  rate: number;
  wilson_ci: number[];
  ci_method: string;
}

export interface StatisticalAssessment {
  baseline: RateEstimate;
  candidate: RateEstimate;
  absolute_delta: number;
  relative_delta: number | null;
  significant: boolean;
  significance_method: string;
  exact_binomial_p: number | null;
  seqeval: {
    delegated_to: string;
    test_method: string;
    delta_margin: number;
    alpha: number;
    decision: string;
    snapshot: Record<string, unknown>;
    clusters_fed: number;
    note: string;
  };
  power_mde: { mde_abs: number | null; observed_abs_delta: number; assessment: string; method: string };
  small_sample_flags: string[];
  rare_event_handling: string;
}

export interface StratumRisk {
  dimension: string;
  stratum: string;
  exposure: number;
  exposure_share: number;
  events: number;
  stratum_rate: number;
  baseline_rate: number;
  relative_risk: number | null;
  odds_ratio: number | null;
  risk_difference: number;
  rate_wilson_ci: number[];
  small_sample_flag: boolean;
}

export interface ConcentrationAnalysis {
  failure_id: string;
  determination: 'concentrated' | 'uniform' | 'insufficient_data';
  concentrated_dimensions: string[];
  strata: StratumRisk[];
  method: string;
}

export interface ExpectedLossRow {
  option: OptionCode;
  business_cost: number;
  residual_failure_rate: number;
  expected_incidents_per_release: number;
  expected_loss: number;
  feasible: boolean;
  infeasible_reason: string | null;
}

export interface PolicyEvaluation {
  outcome: PolicyOutcome;
  severity: Severity;
  severity_assignment?: { taxonomy_description?: string; criteria_fired?: string[] };
  policy_version: string;
  recommended_option: OptionCode;
  matrix_row_fired: { row: number; condition: string; option: OptionCode; description: string } | null;
  automatic_stop_ship_condition: { condition_id: string; description?: string } | null;
  option_c_evaluation: { feasible: boolean; checks: { check: string; passed: boolean; detail?: string }[] } | null;
  expected_loss_table: ExpectedLossRow[];
  mandatory_review_triggers?: { trigger: string; description: string; fired: boolean }[];
  indeterminate_reasons?: string[];
  input?: Record<string, unknown>;
  note?: string;
}

export interface StageRecord {
  stage: Stage;
  status: StageStatus;
  started_at: string | null;
  finished_at: string | null;
  detail: string;
}

export interface PipelineState {
  failure_id: string;
  stages: StageRecord[];
  agent_results: Record<string, AgentResult>;
  statistical: StatisticalAssessment | null;
  concentration: ConcentrationAnalysis | null;
  policy_evaluation: PolicyEvaluation | null;
  scorecard_id: string | null;
  suite_ids: string[];
  updated_at: string;
}

export interface ScorecardField {
  value: unknown;
  tag: 'OBSERVED' | 'PREDICTED' | 'HYPOTHETICAL' | 'REQUIRED_EVIDENCE';
  evidence_ref: string;
}

export interface Scorecard {
  scorecard_id: string;
  failure_id: string;
  title: string;
  failure_summary: ScorecardField;
  frequency: ScorecardField;
  exposure: ScorecardField;
  severity: ScorecardField;
  confidence: ScorecardField;
  novelty: ScorecardField;
  concentration: ScorecardField;
  downstream_impact: ScorecardField;
  mitigations: ScorecardField;
  residual_risk: ScorecardField;
  evidence_quality: EvidenceQuality;
  policy_outcome: PolicyOutcome | null;
  recommended_option: OptionCode | null;
  policy_version: string;
  generated_at: string;
  notes: string[];
}

export interface HumanReviewDecision {
  review_id: string;
  failure_id: string;
  reviewer: string;
  decision: string;
  approved_option: OptionCode | null;
  evidence_reviewed: string[];
  policy_version: string;
  rationale: string;
  override_reason: string | null;
  timestamp: string;
}

export interface SuiteMember {
  member_id: string;
  sequence_id: string;
  frame_id: string;
  object_instance_id: string;
  source_failure_id: string;
  training_eligible: boolean;
}

export interface EvaluationSuite {
  suite_id: string;
  name: string;
  version: number;
  creation_reason: string;
  source_failures: string[];
  taxonomy_tags: string[];
  sampling_policy: string;
  coverage: Record<string, unknown>;
  known_limitations: string[];
  approval_status: 'draft' | 'approved' | 'retired';
  members: SuiteMember[];
  contamination_guard: string;
  governance_overrides: { member_id: string; actor: string; reason: string; timestamp: string }[];
  created_at: string;
  updated_at: string;
}

export interface RegressionSuiteResult {
  suite: string;
  suite_id?: string | null;
  n: number;
  n_clusters?: number;
  baseline_rate?: number;
  candidate_rate?: number;
  delta?: number;
  delta_ci?: number[];
  decision: string;
  note?: string;
  stats_delegated_to?: string;
}

export interface AuditRecord {
  seq: number;
  event_type: string;
  failure_id: string | null;
  actor: string;
  detail: string;
  payload: Record<string, unknown>;
  timestamp: string;
  prev_hash: string;
  hash: string;
}

export interface LabeledValue<T = unknown> {
  label: 'OBSERVED' | 'HYPOTHETICAL' | 'REQUIRED-EVIDENCE' | 'DERIVED';
  value: T;
}

export interface Walkthrough {
  methodology_provenance: string;
  seed: number;
  deterministic: boolean;
  failure_id: string;
  title: string;
  layers: Record<string, Record<string, LabeledValue>>;
  scorecard_id: string | null;
  audit_records: LabeledValue<number>;
  audit_chain_valid: LabeledValue<boolean>;
  separation_of_powers: string;
}
