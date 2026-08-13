/** Shared color/label mapping for the evidence hierarchy. */
import type { EvidenceTier, LaunchRecommendation, RetroSeverity } from '../../types/retro';

export const TIER_META: Record<EvidenceTier, { label: string; color: string }> = {
  TIER1_OBSERVED: { label: 'FACT', color: '#66bb6a' },
  TIER2_DERIVED: { label: 'DERIVED', color: '#4fc3f7' },
  TIER3_RETRIEVED: { label: 'RETRIEVED', color: '#ba68c8' },
  TIER4_AI_HYPOTHESIS: { label: 'AI HYPOTHESIS', color: '#ffb74d' },
};

export const DETERMINATION_COLOR = '#ef5350';

export const SEVERITY_COLORS: Record<RetroSeverity, string> = {
  BENIGN: '#66bb6a',
  DISRUPTIVE: '#ffb74d',
  CRITICAL: '#ef5350',
  FATAL: '#b71c1c',
};

export const LAUNCH_COLORS: Record<LaunchRecommendation, string> = {
  PASS: '#66bb6a',
  CONDITIONAL_PASS: '#ffb74d',
  FAIL: '#ef5350',
  INSUFFICIENT_EVIDENCE: '#90a4ae',
};
