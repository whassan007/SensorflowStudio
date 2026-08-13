/** Types for the Agentic Retrospective Safety Analyzer (/api/retro). */

export type EvidenceTier =
  | 'TIER1_OBSERVED'
  | 'TIER2_DERIVED'
  | 'TIER3_RETRIEVED'
  | 'TIER4_AI_HYPOTHESIS';

export type RetroSeverity = 'BENIGN' | 'DISRUPTIVE' | 'CRITICAL' | 'FATAL';

export type LaunchRecommendation =
  | 'PASS'
  | 'CONDITIONAL_PASS'
  | 'FAIL'
  | 'INSUFFICIENT_EVIDENCE';

export interface EvidenceItem {
  tier: EvidenceTier;
  key: string;
  statement: string;
  value: string | null;
  provenance: string;
}

export interface RetrievedStandard {
  source: string;
  document: string;
  version: string;
  section: string;
  retrieved_text: string;
  relevance_score: number;
  doc_id: string;
  doc_type: string;
  jurisdiction: string;
  effective_date: string;
  synthetic: boolean;
  label: string;
  chunk_id: string;
}

export interface RootCauseHypothesis {
  hypothesis: string;
  confidence: number;
  supporting_evidence_keys: string[];
  missing_evidence: string[];
  tier: EvidenceTier;
}

export interface StatSignificance {
  method: string;
  significant: boolean | null;
  detail: string;
}

export interface UncertaintyReport {
  missing_fields: string[];
  unknown_metrics: string[];
  notes: string[];
}

export interface RetrospectiveScorecard {
  evaluation_id: string;
  created_at: string;
  policy_version: string;
  agent_version: string;
  backend_used: string;
  failure_type: string;
  severity: RetroSeverity;
  ai_proposed_severity: RetroSeverity | null;
  severity_divergence: boolean;
  safety_critical_recall_impact: number | null;
  scr_impact_detail: string;
  behavioral_consequence: string;
  launch_recommendation: LaunchRecommendation;
  launch_rationale: string[];
  baseline_model: string | null;
  candidate_model: string | null;
  scenario: Record<string, unknown>;
  object_class: string | null;
  ground_truth: Record<string, unknown> | null;
  prediction: Record<string, unknown> | null;
  confidence: number | null;
  ego_speed_mps: number | null;
  distance_to_object_m: number | null;
  relative_velocity_mps: number | null;
  stopping_distance_m: number | null;
  ttc_s: number | null;
  ttc_validity: string[];
  planner_response: Record<string, unknown> | null;
  disengagement_probability: number | null;
  metric_delta: Record<string, number> | null;
  statistical_significance: StatSignificance | null;
  distribution_shift: Record<string, unknown> | null;
  root_cause_hypotheses: RootCauseHypothesis[];
  evidence: EvidenceItem[];
  retrieved_standards: RetrievedStandard[];
  uncertainty: UncertaintyReport;
  human_review_required: boolean;
  human_review_reasons: string[];
}

export interface AnalyzeResponse {
  status: string;
  scorecard: RetrospectiveScorecard;
  markdown: string;
}

export interface FixtureInfo {
  fixture_id: string;
  evaluation_id: string;
  description: string;
  weather: string | null;
}

export interface BackendStatus {
  backend: string;
  available: boolean;
  endpoint: string | null;
  model: string | null;
  detail: string;
  health_latency_ms: number | null;
}

export interface CompatCheck {
  link: string;
  status: 'PASS' | 'FAIL' | 'SKIPPED';
  reason: string;
  remediation: string | null;
}

export interface CompatReport {
  vllm_supported: boolean;
  platform_summary: string;
  checks: CompatCheck[];
  failed_link: string | null;
  notes: string[];
}

export interface EnvironmentReport {
  os_name: string;
  os_version: string;
  machine_arch: string;
  is_macos: boolean;
  is_apple_silicon: boolean;
  python_version: string;
  gpus: { vendor: string; model: string; unified_memory: boolean; detail: string }[];
  vllm_installed: boolean;
  ollama_endpoint: string | null;
  ollama_models: string[];
  notes: string[];
}

export interface AuditRecord {
  call_id: string;
  analysis_id: string;
  tool: string;
  timestamp: string;
  args: Record<string, unknown>;
  status: string;
  result_hash: string | null;
  elapsed_ms: number;
  error: string | null;
  authorized_write: boolean;
}

export interface AnalysisSummary {
  evaluation_id: string;
  created_at: string | null;
  failure_type: string | null;
  severity: RetroSeverity | null;
  launch_recommendation: LaunchRecommendation | null;
  backend_used: string | null;
  human_review_required: boolean | null;
}

export interface ToolSpec {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  read_only: boolean;
  timeout_s: number;
  error_behavior: string;
}
