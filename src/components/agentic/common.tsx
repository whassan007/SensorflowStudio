/** Shared chips + labels for the Launch Readiness UI. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import type { EvidenceStatus, PolicyOutcome, Severity } from '../../types/agentic';

export const SEVERITY_COLORS: Record<Severity, string> = {
  S0: '#455a64',
  S1: '#546e7a',
  S2: '#f9a825',
  S3: '#ef6c00',
  S4: '#d84315',
  S5: '#b71c1c',
};

export const OUTCOME_STYLE: Record<PolicyOutcome, { bg: string; fg: string; label: string }> = {
  AUTOMATIC_STOP_SHIP: { bg: '#b71c1c', fg: '#fff', label: 'AUTOMATIC STOP-SHIP' },
  LAUNCH_REVIEW_REQUIRED: { bg: '#e65100', fg: '#fff', label: 'LAUNCH REVIEW REQUIRED' },
  CONTINUE_INVESTIGATION: { bg: '#0d47a1', fg: '#90caf9', label: 'CONTINUE INVESTIGATION' },
  NO_LAUNCH_IMPACT: { bg: '#1b5e20', fg: '#a5d6a7', label: 'NO LAUNCH IMPACT' },
  INDETERMINATE: { bg: '#4a148c', fg: '#ce93d8', label: 'INDETERMINATE (FAIL-SAFE)' },
};

export const EVIDENCE_STATUS_COLORS: Record<EvidenceStatus, string> = {
  OBSERVED: '#2e7d32',
  DERIVED: '#0277bd',
  HYPOTHESIS: '#f9a825',
  UNAVAILABLE: '#757575',
};

export function SeverityChip({ severity }: { severity: Severity | null | undefined }) {
  if (!severity) return <Chip size="small" label="unrated" sx={{ height: 20, fontSize: 11 }} />;
  return (
    <Chip
      size="small"
      label={severity}
      sx={{
        height: 20,
        fontSize: 11,
        fontWeight: 800,
        bgcolor: SEVERITY_COLORS[severity],
        color: '#fff',
      }}
    />
  );
}

export function OutcomeChip({ outcome }: { outcome: PolicyOutcome | null | undefined }) {
  if (!outcome) return <Chip size="small" label="no decision" sx={{ height: 20, fontSize: 11 }} />;
  const s = OUTCOME_STYLE[outcome];
  return (
    <Chip
      size="small"
      label={s.label}
      sx={{ height: 20, fontSize: 10.5, fontWeight: 800, bgcolor: s.bg, color: s.fg }}
    />
  );
}

export function EvidenceStatusChip({ status }: { status: EvidenceStatus }) {
  return (
    <Chip
      size="small"
      label={status}
      sx={{
        height: 18,
        fontSize: 10,
        fontWeight: 700,
        bgcolor: EVIDENCE_STATUS_COLORS[status],
        color: '#fff',
      }}
    />
  );
}

/** Marks a value as AI analysis (advisory) vs a deterministic measurement. */
export function OriginBadge({ origin }: { origin: 'ai' | 'deterministic' }) {
  const isAi = origin === 'ai';
  return (
    <Tooltip
      title={
        isAi
          ? 'Produced by an advisory AI agent — a recommendation or hypothesis, never an authorization.'
          : 'Produced by deterministic code (metrics, statistics, policy) — reproducible and testable.'
      }
    >
      <Chip
        size="small"
        label={isAi ? 'AI ANALYSIS (ADVISORY)' : 'DETERMINISTIC'}
        sx={{
          height: 18,
          fontSize: 9.5,
          fontWeight: 800,
          letterSpacing: 0.4,
          bgcolor: isAi ? '#4a148c' : '#1b3a4b',
          color: isAi ? '#ce93d8' : '#4fc3f7',
        }}
      />
    </Tooltip>
  );
}

export function PanelTitle({
  title,
  origin,
  extra,
}: {
  title: string;
  origin?: 'ai' | 'deterministic';
  extra?: React.ReactNode;
}) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 800, flex: 1 }}>
        {title}
      </Typography>
      {origin ? <OriginBadge origin={origin} /> : null}
      {extra}
    </Box>
  );
}

export function KV({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <Box sx={{ display: 'flex', gap: 1, py: 0.25 }}>
      <Typography variant="caption" sx={{ color: '#8a949e', minWidth: 170, flexShrink: 0 }}>
        {k}
      </Typography>
      <Typography
        variant="caption"
        component="div"
        sx={{ fontFamily: mono ? 'monospace' : undefined, wordBreak: 'break-word' }}
      >
        {v}
      </Typography>
    </Box>
  );
}

export const fmtRate = (r: number | null | undefined): string =>
  r === null || r === undefined ? '—' : r === 0 ? '0' : r.toExponential(3);

export const fmtCi = (ci: number[] | null | undefined): string =>
  ci && ci.length === 2 ? `[${fmtRate(ci[0])}, ${fmtRate(ci[1])}]` : '—';
