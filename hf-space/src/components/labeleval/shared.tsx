import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { CheckCircle2, XCircle, MinusCircle } from 'lucide-react';
import type { GateLine } from '../../types/labeleval';
import { GATE_GLOSSARY, glossaryKeyForStatus } from '../../content/glossary';
import { ExplainTip, InfoDot } from '../help/InfoTip';

// ---------------------------------------------------------------- formatting

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Math.round(n).toLocaleString();
}

/** Continuous metrics: integers stay whole; otherwise up to `digits` (default 2). */
export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  if (Number.isInteger(n)) return n.toLocaleString();
  const d = Math.min(2, Math.max(0, digits));
  return n.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
}

/**
 * Formats a rate as a percentage (prefer 1 decimal like 10.0%; max 2).
 * Values <= 1 are treated as fractions (0..1); values > 1 are already percentages.
 */
export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const pct = v <= 1 ? v * 100 : v;
  const d = Math.min(2, Math.max(0, digits));
  return `${pct.toFixed(d)}%`;
}

export const formatNumber = fmtNum;
export const formatPercent = fmtPct;

export function pctFraction(v: number | null | undefined): number {
  if (v === null || v === undefined || Number.isNaN(v)) return 0;
  return v <= 1 ? v : v / 100;
}

// ---------------------------------------------------------------- MetricCard

export function MetricCard({
  label,
  value,
  sub,
  accent,
  term,
  info,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
  /** Glossary key: renders an info icon whose tooltip explains this metric. */
  term?: string;
  /** Ad-hoc explanation when no glossary key fits. */
  info?: string;
}) {
  return (
    <Card variant="outlined" sx={{ minWidth: 130, flex: '1 1 130px', bgcolor: '#161b21' }}>
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Typography variant="caption" sx={{ color: '#8a949e', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          {label}
          {term || info ? <InfoDot term={term} title={info ? label : undefined} detail={info} size={11} /> : null}
        </Typography>
        <Typography variant="h5" sx={{ fontWeight: 700, color: accent ?? '#e6e9ec', lineHeight: 1.3 }}>
          {value}
        </Typography>
        {sub ? (
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            {sub}
          </Typography>
        ) : null}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- StatusChip

const STATUS_COLORS: Record<string, { bg: string; fg: string }> = {
  // triage statuses
  AUTO_GRADED: { bg: '#1b5e20', fg: '#a5d6a7' },
  FLAGGED: { bg: '#e65100', fg: '#ffe0b2' },
  VERIFIED: { bg: '#2e7d32', fg: '#c8e6c9' },
  REJECTED: { bg: '#b71c1c', fg: '#ffcdd2' },
  PENDING: { bg: '#37474f', fg: '#cfd8dc' },
  // service states
  HEALTHY: { bg: '#1b5e20', fg: '#a5d6a7' },
  RUNNING: { bg: '#0d47a1', fg: '#90caf9' },
  DEGRADED: { bg: '#f9a825', fg: '#212121' },
  BLOCKED: { bg: '#e65100', fg: '#ffe0b2' },
  FAILED: { bg: '#b71c1c', fg: '#ffcdd2' },
  IDLE: { bg: '#37474f', fg: '#b0bec5' },
  // severities
  critical: { bg: '#b71c1c', fg: '#ffcdd2' },
  high: { bg: '#e65100', fg: '#ffe0b2' },
  medium: { bg: '#f9a825', fg: '#212121' },
  low: { bg: '#37474f', fg: '#cfd8dc' },
  info: { bg: '#0d47a1', fg: '#90caf9' },
  warning: { bg: '#e65100', fg: '#ffe0b2' },
  // model regression status
  improved: { bg: '#1b5e20', fg: '#a5d6a7' },
  regressed: { bg: '#b71c1c', fg: '#ffcdd2' },
  baseline: { bg: '#37474f', fg: '#cfd8dc' },
  unknown: { bg: '#37474f', fg: '#b0bec5' },
  // review task status
  open: { bg: '#e65100', fg: '#ffe0b2' },
  in_review: { bg: '#0d47a1', fg: '#90caf9' },
  resolved: { bg: '#1b5e20', fg: '#a5d6a7' },
  // job status
  queued: { bg: '#37474f', fg: '#cfd8dc' },
  running: { bg: '#0d47a1', fg: '#90caf9' },
  completed: { bg: '#1b5e20', fg: '#a5d6a7' },
  failed: { bg: '#b71c1c', fg: '#ffcdd2' },
  stopped: { bg: '#37474f', fg: '#b0bec5' },
};

export function StatusChip({ status, size = 'small' }: { status: string; size?: 'small' | 'medium' }) {
  const colors = STATUS_COLORS[status] ?? { bg: '#37474f', fg: '#cfd8dc' };
  const chip = (
    <Chip
      label={status.replace(/_/g, ' ')}
      size={size}
      sx={{ bgcolor: colors.bg, color: colors.fg, fontWeight: 600, fontSize: 11, height: 22, cursor: 'help' }}
    />
  );
  // Auto-explain known statuses / failure reasons from the glossary.
  const key = glossaryKeyForStatus(status);
  if (!key) return chip;
  return <ExplainTip term={key}>{chip}</ExplainTip>;
}

// ---------------------------------------------------------------- GateLineList

function fmtGateValue(v: number | string | boolean): string {
  if (typeof v === 'number') return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return String(v);
}

export function GateLineList({ checks }: { checks: GateLine[] }) {
  if (!checks.length) {
    return (
      <Typography variant="body2" sx={{ color: '#8a949e' }}>
        No gate results available.
      </Typography>
    );
  }
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>
            Gate
            <InfoDot term="quality_gate" />
          </TableCell>
          <TableCell align="right">
            Actual
            <InfoDot title="Actual" detail="The value measured for this label by the evaluation engines." />
          </TableCell>
          <TableCell align="right">
            Threshold
            <InfoDot term="quality_policy" />
          </TableCell>
          <TableCell align="center">Result</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {checks.map((c, i) => (
          <TableRow key={`${c.gate}-${i}`} sx={{ opacity: c.applicable ? 1 : 0.45 }}>
            <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
              {GATE_GLOSSARY[c.gate] ? (
                <ExplainTip term={GATE_GLOSSARY[c.gate]}>
                  <Box component="span" sx={{ borderBottom: '1px dotted #5c6873', cursor: 'help' }}>
                    {c.gate}
                  </Box>
                </ExplainTip>
              ) : (
                c.gate
              )}
            </TableCell>
            <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
              {fmtGateValue(c.actual)}
            </TableCell>
            <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
              {fmtGateValue(c.threshold)}
            </TableCell>
            <TableCell align="center">
              {!c.applicable ? (
                <MinusCircle size={16} color="#8a949e" />
              ) : c.passed ? (
                <CheckCircle2 size={16} color="#66bb6a" />
              ) : (
                <XCircle size={16} color="#ef5350" />
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ---------------------------------------------------------------- layout helpers

export function SectionCard({
  title,
  action,
  children,
  sx,
  help,
  helpTerm,
}: {
  title: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  sx?: object;
  /** Ad-hoc "what is this panel" explanation shown behind an info icon. */
  help?: string;
  /** Glossary key alternative to `help`. */
  helpTerm?: string;
}) {
  return (
    <Card variant="outlined" sx={{ bgcolor: '#161b21', ...sx }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            {title}
            {help || helpTerm ? (
              <InfoDot term={helpTerm} title={helpTerm ? undefined : 'About this panel'} detail={help} size={13} />
            ) : null}
          </Typography>
          {action}
        </Box>
        {children}
      </CardContent>
    </Card>
  );
}

export function LoadingBox({ label = 'Loading…' }: { label?: string }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, p: 3, justifyContent: 'center' }}>
      <CircularProgress size={20} />
      <Typography variant="body2" sx={{ color: '#8a949e' }}>
        {label}
      </Typography>
    </Box>
  );
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <Alert severity="warning" variant="outlined" sx={{ my: 1 }}>
      {error.startsWith('API') ? error : `Backend unavailable: ${error}`}
    </Alert>
  );
}

export function EmptyState({
  title,
  message,
  action,
}: {
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <Box sx={{ textAlign: 'center', py: 6, px: 2 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>
        {title}
      </Typography>
      <Typography variant="body2" sx={{ color: '#8a949e', mb: 2, maxWidth: 520, mx: 'auto' }}>
        {message}
      </Typography>
      {action}
    </Box>
  );
}

// ---------------------------------------------------------------- horizontal bar

export function HBar({
  label,
  value,
  max,
  color = '#4fc3f7',
  valueLabel,
  term,
  info,
}: {
  label: string;
  value: number;
  max: number;
  color?: string;
  valueLabel?: string;
  /** Glossary key explaining this bar's meaning on hover. */
  term?: string;
  /** Ad-hoc explanation when no glossary key fits. */
  info?: string;
}) {
  const width = max > 0 ? Math.max(1.5, (value / max) * 100) : 0;
  const labelNode =
    term || info ? (
      <ExplainTip term={term} title={info ? label : undefined} detail={info}>
        <Box component="span" sx={{ borderBottom: '1px dotted #3d4650', cursor: 'help' }}>
          {label}
        </Box>
      </ExplainTip>
    ) : (
      label
    );
  return (
    <Box sx={{ mb: 0.75 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.25 }}>
        <Typography variant="caption" sx={{ color: '#aab4be' }}>
          {labelNode}
        </Typography>
        <Typography variant="caption" sx={{ color: '#e6e9ec', fontFamily: 'monospace' }}>
          {valueLabel ?? value.toLocaleString()}
        </Typography>
      </Box>
      <Box sx={{ height: 8, bgcolor: '#232a31', borderRadius: 1, overflow: 'hidden' }}>
        <Box sx={{ height: '100%', width: `${width}%`, bgcolor: color, borderRadius: 1 }} />
      </Box>
    </Box>
  );
}
