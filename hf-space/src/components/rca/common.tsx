/** Shared visual language for the RCA workbench (dark theme, chip-heavy). */
import { type ReactNode } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import type { Confidence, Finding, FindingStatus, Severity } from '../../types/rca';

export const ACCENT = '#4fc3f7';
export const BG_PANEL = '#12171d';
export const BORDER = '#232a31';

export const STATUS_COLORS: Record<FindingStatus, string> = {
  PASS: '#66bb6a',
  MISMATCH: '#ef5350',
  UNKNOWN: '#ffb74d',
};

export const SEVERITY_COLORS: Record<Severity, string> = {
  INFO: '#78909c',
  WARN: '#ffb74d',
  CRITICAL: '#ef5350',
};

export const CONFIDENCE_COLORS: Record<Confidence, string> = {
  HIGH: '#66bb6a',
  MEDIUM: '#4fc3f7',
  LOW: '#90a4ae',
  UNKNOWN: '#ffb74d',
};

export function SectionCard({ title, subtitle, children, action }: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Paper variant="outlined" sx={{ p: 1.75, bgcolor: BG_PANEL, borderColor: BORDER, mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: subtitle ? 0.25 : 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, flex: 1, fontSize: 13.5 }}>
          {title}
        </Typography>
        {action}
      </Box>
      {subtitle ? (
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1.25 }}>
          {subtitle}
        </Typography>
      ) : null}
      {children}
    </Paper>
  );
}

export function StatusPill({ status }: { status: string }) {
  const norm = status.toUpperCase() as FindingStatus;
  const map: Record<string, { c: string; label: string }> = {
    PASS: { c: STATUS_COLORS.PASS, label: 'PASS' },
    MISMATCH: { c: STATUS_COLORS.MISMATCH, label: 'MISMATCH' },
    UNKNOWN: { c: STATUS_COLORS.UNKNOWN, label: 'UNKNOWN' },
    COMPARABLE: { c: STATUS_COLORS.PASS, label: 'comparable' },
    MATCH: { c: STATUS_COLORS.PASS, label: 'match' },
    EXPECTED_DIFFERENCE: { c: '#78909c', label: 'expected diff' },
  };
  const m = map[norm] ?? map[status.toUpperCase().replace(/ /g, '_')] ?? {
    c: SEVERITY_COLORS.WARN,
    label: status,
  };
  return (
    <Chip
      size="small"
      label={m.label}
      sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: `${m.c}22`, color: m.c, border: `1px solid ${m.c}55` }}
    />
  );
}

export function FindingChip({ finding, onClick }: { finding: Finding; onClick?: () => void }) {
  const c = STATUS_COLORS[finding.status];
  const sev = SEVERITY_COLORS[finding.severity];
  return (
    <Tooltip
      title={
        <Box sx={{ maxWidth: 380 }}>
          <Typography variant="caption" sx={{ fontWeight: 700, display: 'block' }}>
            [{finding.code}] {finding.title}
          </Typography>
          <Typography variant="caption">{finding.detail}</Typography>
        </Box>
      }
      arrow
    >
      <Chip
        size="small"
        onClick={onClick}
        label={
          <span>
            <b style={{ color: c }}>{finding.status}</b>
            {finding.severity !== 'INFO' ? (
              <b style={{ color: sev }}>{` · ${finding.severity}`}</b>
            ) : null}
            {` · ${finding.title}`}
            {finding.source === 'human' ? ' · 👤' : ''}
          </span>
        }
        sx={{
          height: 22,
          fontSize: 11,
          bgcolor: `${c}14`,
          border: `1px solid ${c}44`,
          mr: 0.5,
          mb: 0.5,
          '.MuiChip-label': { px: 1 },
        }}
      />
    </Tooltip>
  );
}

export function FindingChips({ findings, onClick }: {
  findings: Finding[];
  onClick?: (f: Finding) => void;
}) {
  if (!findings.length) return null;
  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 0.5 }}>
      {findings.map((f) => (
        <FindingChip key={f.id} finding={f} onClick={onClick ? () => onClick(f) : undefined} />
      ))}
    </Box>
  );
}

export function Explainer({ text }: { text: string }) {
  return (
    <Box sx={{ borderLeft: `3px solid ${ACCENT}55`, pl: 1.25, py: 0.25, mb: 1.5 }}>
      <Typography variant="caption" sx={{ color: '#aab4be' }}>
        {text}
      </Typography>
    </Box>
  );
}

/** Horizontal signed delta bar centered at zero (pp values). */
export function DeltaBar({ value, max, width = 140 }: { value: number; max: number; width?: number }) {
  const half = width / 2;
  const frac = Math.min(1, Math.abs(value) / Math.max(1e-9, max));
  const w = Math.max(1.5, frac * half);
  const color = value >= 0 ? '#66bb6a' : '#ef5350';
  return (
    <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75 }}>
      <svg width={width} height={14}>
        <line x1={half} y1={0} x2={half} y2={14} stroke={BORDER} strokeWidth={1} />
        <rect
          x={value >= 0 ? half : half - w}
          y={2.5}
          width={w}
          height={9}
          rx={2}
          fill={color}
          opacity={0.85}
        />
      </svg>
      <Typography
        variant="caption"
        sx={{ fontFamily: 'monospace', color, minWidth: 52, textAlign: 'right' }}
      >
        {value >= 0 ? '+' : ''}
        {value.toFixed(1)}pp
      </Typography>
    </Box>
  );
}

/** Simple 0..1 proportion bar for mix comparisons. */
export function ShareBar({ value, color = ACCENT, width = 90 }: {
  value: number;
  color?: string;
  width?: number;
}) {
  return (
    <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
      <Box sx={{ width, height: 8, bgcolor: '#0d1116', borderRadius: 1, overflow: 'hidden', border: `1px solid ${BORDER}` }}>
        <Box sx={{ width: `${Math.min(100, value * 100)}%`, height: '100%', bgcolor: color }} />
      </Box>
      <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#8a949e', minWidth: 40 }}>
        {(value * 100).toFixed(1)}%
      </Typography>
    </Box>
  );
}

export function PsiBadge({ psi }: { psi: number }) {
  const mag = psi < 0.02 ? 'negligible' : psi < 0.1 ? 'small' : psi < 0.25 ? 'moderate' : 'large';
  const color = mag === 'large' ? '#ef5350' : mag === 'moderate' ? '#ffb74d' : mag === 'small' ? '#4fc3f7' : '#78909c';
  return (
    <Tooltip title="Population Stability Index: practical shift magnitude, independent of p-values (<0.02 negligible, <0.1 small, <0.25 moderate, ≥0.25 large)" arrow>
      <Chip
        size="small"
        label={`PSI ${psi.toFixed(2)} · ${mag}`}
        sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: `${color}22`, color, border: `1px solid ${color}55` }}
      />
    </Tooltip>
  );
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <Box sx={{ display: 'flex', gap: 1, py: 0.2 }}>
      <Typography variant="caption" sx={{ color: '#8a949e', minWidth: 190 }}>
        {k}
      </Typography>
      <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace' }}>
        {v}
      </Typography>
    </Box>
  );
}

export const tableSx = {
  width: '100%',
  borderCollapse: 'collapse' as const,
  fontSize: 12,
  '& th': {
    textAlign: 'left' as const,
    color: '#8a949e',
    fontWeight: 700,
    borderBottom: `1px solid ${BORDER}`,
    px: 1,
    py: 0.6,
    fontSize: 11,
    whiteSpace: 'nowrap' as const,
  },
  '& td': {
    borderBottom: `1px solid ${BORDER}55`,
    px: 1,
    py: 0.55,
    verticalAlign: 'middle' as const,
  },
};
