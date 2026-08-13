/** Types for the Regression RCA workbench (/api/rca). */

export type RootCause =
  | 'TRUE_MODEL_REGRESSION'
  | 'DISTRIBUTION_SHIFT'
  | 'FEATURE_SKEW'
  | 'SERVING_MISMATCH'
  | 'LABEL_LATENCY'
  | 'SAMPLING_BIAS'
  | 'STATISTICAL_NOISE'
  | 'OFFLINE_CONTAMINATION';

export type FindingStatus = 'PASS' | 'MISMATCH' | 'UNKNOWN';
export type Severity = 'INFO' | 'WARN' | 'CRITICAL';
export type StageStatus =
  | 'pending'
  | 'in_progress'
  | 'complete'
  | 'complete_with_unknowns';
export type Confidence = 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN';

export interface StageMeta {
  index: number;
  key: string;
  title: string;
  question: string;
}

export interface StageState {
  index: number;
  key: string;
  title: string;
  status: StageStatus;
  completed_at: number | null;
  acknowledged_unknowns: boolean;
  ack_note: string;
  skip_acknowledged: boolean;
}

export interface Finding {
  id: string;
  stage: string;
  code: string;
  title: string;
  status: FindingStatus;
  severity: Severity;
  detail: string;
  source: 'auto' | 'human';
  created_at: number;
  data: Record<string, unknown>;
}

export interface InvestigationClaims {
  metric: string;
  offline_delta_pp: number;
  shadow_delta_pp: number;
  offline_n: number;
  shadow_scored_n: number;
}

export interface Investigation {
  id: string;
  name: string;
  baseline_model: string;
  candidate_model: string;
  seed: number;
  training_mode: boolean;
  revealed: boolean;
  created_at: number;
  claims: InvestigationClaims;
  stages: StageState[];
  findings: Finding[];
  human_assessments: Record<string, { confidence: Confidence; note: string }>;
  events: { ts: number; kind: string; message: string; data: Record<string, unknown> }[];
  scenario_cause?: RootCause;
}

export interface InvestigationSummary {
  id: string;
  name: string;
  baseline_model: string;
  candidate_model: string;
  training_mode: boolean;
  revealed: boolean;
  created_at: number;
  claims: InvestigationClaims;
  stages_complete: number;
  stages_total: number;
  scenario_cause?: RootCause;
}

export interface DiagnosticsResponse {
  stage: string;
  data: Record<string, any>;
  findings: Finding[];
}

export interface EvidenceLink {
  code: string;
  stage: string;
  title: string;
  weight: number;
  severity: Severity;
  status: FindingStatus;
  finding_id: string;
}

export interface ScoreboardRow {
  hypothesis: RootCause;
  label: string;
  score: number;
  rank: number;
  evidence_for: EvidenceLink[];
  evidence_against: EvidenceLink[];
  auto_confidence: Confidence;
  human_confidence: Confidence | null;
  human_note: string;
  next_discriminating_test: string;
}

export interface Scoreboard {
  rows: ScoreboardRow[];
  top_hypothesis: RootCause;
  top_confidence: Confidence;
  score_gap: number;
  working_hypothesis_set: RootCause[];
  explainer: string;
}

export interface DecisionNode {
  id: string;
  question: string;
  answer: 'yes' | 'no' | 'unknown';
  basis: string[];
  conclusion_if_no: string | null;
  next: string | null;
}

export interface DecisionTree {
  nodes: DecisionNode[];
  path: string[];
  conclusion: string | null;
  conclusion_kind: 'root_cause' | 'insufficient_evidence' | 'no_regression';
  explainer: string;
}

export interface ExperimentDesign {
  id: string;
  design: string;
  description: string;
  discriminates: RootCause[];
  cost: 'low' | 'medium' | 'high';
  expected_days: number;
  information_gain: number;
  priority: number;
  rank: number;
}

export interface Experiments {
  experiments: ExperimentDesign[];
  minimum_additional_evidence: string;
  power: { effective_n: number; needed_effective_n: number; practical_margin_pp: number };
  explainer: string;
}

export interface RcaReport {
  investigation_id: string;
  name: string;
  generated_at: number;
  models: { baseline: string; candidate: string };
  claims: InvestigationClaims;
  executive_finding: {
    conclusion: string | null;
    conclusion_kind: string;
    top_hypothesis: RootCause;
    label: string;
    confidence: Confidence;
    score: number;
    score_gap_to_runner_up: number;
  };
  stage_summaries: {
    stage: string;
    headline: string;
    mismatches: string[];
    unknowns: string[];
    n_pass: number;
    verdict: 'mismatch' | 'unknown' | 'clean';
  }[];
  hypothesis_ranking: {
    rank: number;
    hypothesis: RootCause;
    score: number;
    confidence: Confidence;
    human_confidence: Confidence | null;
    evidence_for: number;
    evidence_against: number;
  }[];
  decision_path: string[];
  minimum_additional_evidence: string;
  recommended_experiments: ExperimentDesign[];
  remediation: { containment: string[]; short_term: string[]; long_term: string[] };
  acknowledged_unknowns: { ts: number; kind: string; message: string; data: Record<string, unknown> }[];
  human_findings: Finding[];
  markdown: string;
}

export interface RevealResponse {
  cause: RootCause;
  explanation: string;
}
