/** Types for the Next-Gen AV Perception Evaluation API (/api/nextgen). */

export type DataLabel = 'REAL' | 'REPLAYED' | 'SIMULATED' | 'GENERATED' | 'COUNTERFACTUAL';

export interface TransformationStep {
  kind: string;
  params: Record<string, unknown>;
}

export interface CatalogueEntry {
  kind: string;
  family: string;
  params: Record<string, unknown>;
}

export interface Provenance {
  source_scene_id: string;
  recipe: TransformationStep[];
  seed: number;
  generator: string;
  generator_version: string;
  data_label: DataLabel;
  created_at: string;
}

export interface ValidityCheck {
  check: string;
  passed: boolean;
  score: number;
  detail: string;
}

export interface ValidityReport {
  scenario_id: string;
  checks: ValidityCheck[];
  simulation_fidelity_score: number;
  counterfactual_validity: number;
  realism_confidence: number;
  accepted: boolean;
  evaluation_weight: number;
  weight_capped: boolean;
  reasons: string[];
}

export interface CounterfactualScenario {
  scenario_id: string;
  provenance: Provenance;
  n_frames: number;
  n_actors: number;
  environment: Record<string, string>;
  validity: ValidityReport | null;
}

export interface SuiteWeights {
  weights: Record<string, number>;
  low_fidelity_share: number;
  share_cap: number;
  scaled_down: boolean;
  policy: string;
}

export interface BehavioralMetrics {
  detection_latency_s: number | null;
  time_to_detection_s: number | null;
  min_ttc_s: number | null;
  stopping_distance_m: number | null;
  max_deceleration_mps2: number;
  max_steering_rate_radps: number;
  planner_interventions: number;
  collision: boolean;
  collision_probability: number;
  min_separation_m: number | null;
  safety_margin_m: number | null;
  final_speed_mps: number;
}

export interface TrajectoryPoint {
  t: number;
  x: number;
  y: number;
  v: number;
  a: number;
  n_detections: number;
}

export interface BehavioralAssessment {
  scenario_id: string;
  data_label: DataLabel;
  perception_mode: 'actual' | 'corrected';
  metrics: BehavioralMetrics;
  open_loop: Record<string, unknown> & { frame_recall?: number | null };
  trajectory: TrajectoryPoint[];
}

export interface CausalChainStep {
  question: string;
  answer: boolean;
  evidence: string;
}

export interface CausalReplayResult {
  scenario_id: string;
  data_label: DataLabel;
  actual: BehavioralAssessment;
  corrected: BehavioralAssessment;
  diffs: Record<string, number | boolean | null>;
  causal_chain: CausalChainStep[];
  verdict: 'METRIC_ONLY' | 'BEHAVIORALLY_CONSEQUENTIAL';
}

export interface SafetyReportSection {
  n_objects?: number;
  recall?: number | null;
  n_safety_critical?: number;
  safety_critical_recall?: number | null;
  risk_weighted_recall?: number | null;
}

export interface SafetyReport {
  data_label: string;
  region_params: Record<string, number>;
  region_definition: string;
  ego_speed_mps: number;
  open_loop: SafetyReportSection;
  safety_informed: SafetyReportSection;
  by_class: Record<string, SafetyReportSection & { n?: number }>;
  note: string;
}

export interface DivergenceDemo {
  baseline: SafetyReport;
  candidate: SafetyReport;
  deltas: Record<string, number | null>;
  headline: string;
  construction: string;
  data_label: string;
}

export interface GauntletStratum {
  stratum: string;
  priority: number;
  data_label: string;
  planted_effect: number;
  units_available: number;
  units_evaluated: number;
  units_saved_by_early_stop: number;
  n: number;
  n_clusters: number;
  delta_estimate: number | null;
  delta_ci: [number, number];
  e_regression: number;
  e_pass: number;
  decision: string;
}

export interface GauntletEvent {
  event: string;
  stratum?: string;
  detail: string;
  delta_estimate?: number;
  promoted?: string[];
}

export interface LaunchRecommendation {
  run_id: string;
  recommendation: string;
  blockers: string[];
  statistical_significance: Record<string, unknown>;
  safety_significance: Record<string, unknown>;
  lineage_valid: boolean;
  data_labels: string[];
}

export interface GauntletResult {
  run_id: string;
  candidate_version: string;
  baseline_version: string;
  scale: {
    total_units_defined: number;
    units_evaluated: number;
    units_saved: number;
    budget_units: number;
    budget_remaining: number;
  };
  timing: { wall_s: number; units_per_second: number | null };
  cache: { hits: number; misses: number; hit_rate: number | null };
  priority_order: string[];
  processed_order: string[];
  halted: boolean;
  events: GauntletEvent[];
  strata: GauntletStratum[];
  recommendation: LaunchRecommendation;
  statistical_validity: string;
  lineage_valid: boolean;
  status: string;
}

export interface ComputeReport {
  report_id: string;
  n_scenarios: number;
  n_models: number;
  naive_full_inferences: number;
  optimized_backbone_computes: number;
  optimized_head_computes: number;
  cache_hits: number;
  cache_misses: number;
  hit_rate: number;
  naive_cost_s: number;
  optimized_cost_s: number;
  savings_ratio: number;
  measured_backbone_s: number;
  measured_head_s: number;
  invalidation: string;
}

export interface ArchitectureDocs {
  docs: Record<string, { file: string; content: string | null }>;
  missing: string[];
}

export interface NextgenStatus {
  package: string;
  engines: string[];
  transformations: string[];
  counterfactuals: number;
  gauntlets: { run_id: string; status: string; recommendation: string }[];
  compute_report_available: boolean;
  agentic_scorecard_source: string;
  data_labels: DataLabel[];
  component_versions: Record<string, string>;
}
