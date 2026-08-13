/** Types for the /api/hardening endpoints (Production Readiness page). */

export interface AuditFinding {
  id: string;
  area: string;
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  refs: string[];
  existing_approach: string;
  problem: string;
  correct_approach: string;
  disposition: 'fix_now' | 'fix_now_partial' | 'fix_now_layered' | 'follow_up';
  disposition_reason: string;
  effort: 'S' | 'M' | 'L';
}

export interface AuditSummary {
  critical: number;
  high: number;
  medium: number;
  low: number;
  fixed_now: number;
  fixed_now_partial: number;
  fixed_now_layered: number;
  deferred: number;
}

export interface AuditStrength {
  package: string;
  what: string;
}

export interface AuditDocument {
  audit_version: string;
  generated: string;
  scope: string;
  findings: AuditFinding[];
  strengths: AuditStrength[];
  summary: AuditSummary;
}

export interface ReadinessCategory {
  category: string;
  prototype: string;
  production_requirement: string;
  gap_count: number;
  open_finding_ids: string[];
  open_critical_ids: string[];
  partially_fixed_ids: string[];
  status:
    | 'closed'
    | 'blocked_critical'
    | 'gaps_open'
    | 'partially_hardened'
    | 'no_findings';
}

export interface ReadinessScorecard {
  categories: ReadinessCategory[];
  overall_status: 'NOT_PRODUCTION_READY' | 'HARDENING_IN_PROGRESS' | 'READY_CANDIDATE';
  rule: string;
  summary: AuditSummary;
}

export interface FunnelStage {
  stage: string;
  count: number;
}

export interface FunnelResponse {
  available: boolean;
  store: string;
  note?: string;
  stages?: FunnelStage[];
  triage_breakdown?: Record<string, number>;
  review_breakdown?: Record<string, number>;
  alerts?: number;
  audit_events?: number;
}

export interface HardeningSummary {
  summary: AuditSummary;
  strengths: AuditStrength[];
  readiness: ReadinessScorecard;
}
