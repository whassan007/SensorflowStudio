/** Contracts for the ROTR (right-of-the-road) API (/api/rotr/*). */

export interface RotrEnvironment {
  visibility: string;
  weather: string;
  lighting: string;
}

export interface RotrScenarioSummary {
  scenario_id: string;
  kind: string;
  committed: boolean;
  is_violation_opportunity: boolean;
  expected_rule_id: string | null;
  cause_layer: string | null;
  vulnerability: string;
  visibility: string;
  lighting: string;
  weather: string;
  n_violations_detected: number;
}

export interface RotrLayerEvidence {
  layer: string;
  status: 'SUPPORTED' | 'RULED_OUT' | 'UNKNOWN';
  evidence: string;
  confidence: number;
}

export interface RotrAttributionRow {
  violation_id: string;
  scenario_id: string;
  rule_id: string;
  primary_layer: string | null;
  layers: Record<string, RotrLayerEvidence>;
  note: string;
}

export interface RotrAttributionMatrix {
  run_id: string;
  rows: RotrAttributionRow[];
  invariant: string;
}

export interface RotrSafetyAssessment {
  min_ttc_s: number | null;
  pet_s: number | null;
  min_clearance_m: number | null;
  stopping_distance_m: number | null;
  max_braking_mps2: number;
  max_lateral_deviation_m: number;
  collision: boolean;
  surrogate_caveat: string;
}

export interface RotrTrajectoryPoint {
  t: number;
  x: number;
  y: number;
  v: number;
  a: number;
}

export interface RotrConsequenceSummary {
  consequence_class: string;
  corrected_layers: string[];
  engine: string;
  max_position_divergence_m: number;
  observed_safety: RotrSafetyAssessment;
  corrected_safety: RotrSafetyAssessment;
}

export interface RotrConsequenceDetail {
  counterfactual_id: string;
  violation_id: string;
  scenario_id: string;
  corrected_layers: string[];
  consequence_class: string;
  planner_evaluation: {
    engine: string;
    observed_trajectory: RotrTrajectoryPoint[];
    corrected_trajectory: RotrTrajectoryPoint[];
    max_position_divergence_m: number;
    max_speed_divergence_mps: number;
    corrected_max_braking_mps2: number;
    corrected_intervention: boolean;
  };
  observed_safety: RotrSafetyAssessment;
  corrected_safety: RotrSafetyAssessment;
  scenario_geometry?: {
    actual_context: {
      intersection_type: string;
      control: string;
      stop_line_x: number | null;
      intersection_x_min: number | null;
      intersection_x_max: number | null;
      lanes: { lane_id: string; center_y: number; restricted_to: string | null }[];
      crosswalks: { crosswalk_id: string; x_min: number; x_max: number }[];
    };
    actors: {
      actor_id: string;
      class_name: string;
      dims: number[];
      states: { t: number; x: number; y: number; yaw: number }[];
    }[];
  };
}

export interface RotrGateEvent {
  violation_id: string;
  legs: Record<string, boolean>;
  fired: boolean;
  note: string;
}

export interface RotrGate {
  gate_id: string;
  run_id: string;
  policy_version: string;
  outcome: 'GO' | 'NO_GO';
  events: RotrGateEvent[];
  agentic_advisory: Record<string, unknown> | null;
}

export interface RotrCluster {
  cluster_id: string;
  key: string;
  count: number;
  member_violation_ids: string[];
  exemplar_violation_id: string;
  environment_spread: string[];
  consequence_distribution: Record<string, number>;
}

export interface RotrWeightStratum {
  n: number;
  harm_fraction: number;
  weight: number;
  recall: number | null;
}

export interface RotrMetrics {
  n_scenarios: number;
  n_committed_violations: number;
  n_planted_non_violations: number;
  n_detected: number;
  n_violation_records: number;
  rotr_recall: number | null;
  false_accusation_rate: number | null;
  sc_rotr_recall: number | null;
  bcr: number | null;
  cfr: number | null;
  cfr_wilson_95: [number, number];
  weight_calibration: {
    version: string;
    method: string;
    strata: Record<string, RotrWeightStratum>;
  };
  cohorts: Record<string, { n: number; detected: number; recall: number | null }>;
  behavior_rates: Record<
    string,
    { opportunities: number; committed: number; violation_rate: number | null }
  >;
  surrogate_caveat: string;
}

export interface RotrRunSummary {
  run_id: string;
  bank_id: string;
  model_version: string;
  ruleset_version: string;
  n_scenarios: number;
  n_violations: number;
  scenario_summaries: RotrScenarioSummary[];
  metrics: RotrMetrics;
  clusters: RotrCluster[];
  gate: RotrGate;
}

export interface RotrRunListItem {
  run_id: string;
  bank_id: string;
  model_version: string;
  n_scenarios: number;
  n_violations: number;
  gate_outcome: string;
  rotr_recall: number | null;
}

export interface RotrQueryResultItem {
  violation_id: string;
  scenario_id: string;
  rule_id: string;
  rule_version: string;
  description: string;
  taxonomy: Record<string, string>;
  evidence: Record<string, unknown>;
  confidence: number;
  primary_layer: string | null;
  consequence_class: string | null;
  cluster_id: string | null;
  environment: RotrEnvironment;
  provenance: Record<string, unknown>;
}

export interface RotrQueryResponse {
  query: Record<string, string | null>;
  n_results: number;
  results: RotrQueryResultItem[];
}

export interface RotrViolationDetail {
  violation: {
    violation_id: string;
    scenario_id: string;
    rule_id: string;
    rule_version: string;
    description: string;
    actor_ids: string[];
    evidence: Record<string, unknown>;
    confidence: number;
    taxonomy: Record<string, string>;
  };
  attribution: {
    primary_layer: string | null;
    layers: Record<string, RotrLayerEvidence>;
    note: string;
  } | null;
  consequence: RotrConsequenceSummary | null;
}

export interface RotrRegressionResult {
  regression_id: string;
  baseline_run_id: string;
  candidate_run_id: string;
  baseline_model: string;
  candidate_model: string;
  metric_deltas: Record<string, number>;
  six_outcomes: Record<string, boolean>;
  primary_outcome: string;
  seqeval: Record<string, unknown> | null;
  distribution_note: string;
}

export interface RotrHITLReview {
  review_id: string;
  run_id: string;
  violation_id: string;
  cluster_id: string | null;
  status: 'PENDING' | 'VALIDATED' | 'REJECTED';
  action: string | null;
  actor: string | null;
  notes: string;
}

export interface RotrTrainingCandidate {
  candidate_id: string;
  run_id: string;
  violation_id: string;
  cluster_id: string | null;
  dataset_role: string;
  training_eligible: boolean;
  guard_state: string;
  override: { actor: string; reason: string; at: string } | null;
  studio2_entity_id?: string | null;
  registry_backend?: string;
}

export interface RotrSuite {
  suite_id: string;
  registry_backend: string;
  members: {
    candidate_id: string;
    violation_id: string;
    cluster_id: string | null;
    run_id: string;
    role: string;
    added_at: string;
    added_by: string | null;
  }[];
}

export interface RotrStopshipPolicy {
  policy: {
    policy_name: string;
    policy_semver: string;
    conjunction: Record<string, unknown>;
    notes: string;
  };
  policy_version: string;
  note: string;
}
