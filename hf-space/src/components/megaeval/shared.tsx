/**
 * Shared UI helpers for the Evaluation Command Center (megaeval).
 * Complements components/labeleval/shared.tsx with aggregate-first idioms:
 * query provenance badges, exact/approx honesty tags, delta coloring,
 * outcome chips and sketch histogram sparklines.
 */
import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { CheckCircle2, AlertTriangle, OctagonAlert } from 'lucide-react';
import type { ErrorType, HistogramSketch, QueryMeta, QueryRow } from '../../types/megaeval';
import { fmtCompact } from '../../services/megaeval';
import { fmtPct } from '../labeleval/shared';

// ---------------------------------------------------------------- query provenance

const SOURCE_COLORS: Record<QueryMeta['source'], { bg: string; fg: string }> = {
  cache: { bg: '#1b5e20', fg: '#a5d6a7' },
  cube: { bg: '#0d47a1', fg: '#90caf9' },
  scan: { bg: '#e65100', fg: '#ffe0b2' },
};

function fmtLatency(ms: number): string {
  if (ms < 1) return `${ms.toFixed(1)}ms`;
  if (ms < 100) return `${ms.toFixed(1)}ms`;
  return `${Math.round(ms)}ms`;
}

/** "cube · 74ms" / "cache · 0.1ms" provenance badge for POST /api/evaluations/query panels. */
export function QueryBadge({ meta }: { meta: QueryMeta | null | undefined }) {
  if (!meta) return null;
  const colors = SOURCE_COLORS[meta.source] ?? SOURCE_COLORS.cube;
  const label = `${meta.source} · ${fmtLatency(meta.latency_ms)}`;
  return (
    <Tooltip
      title={`source=${meta.source} · cache_hit=${meta.cache_hit ? 'yes' : 'no'} · cells=${meta.cells_touched}${
        meta.exact ? ' · exact' : ` · approx (${meta.approximate_fields.join(', ') || 'sketch'})`
      }`}
    >
      <Chip
        size="small"
        label={meta.cache_hit ? `${label} ✓` : label}
        sx={{
          height: 18,
          fontSize: 10,
          fontFamily: 'monospace',
          fontWeight: 700,
          bgcolor: colors.bg,
          color: colors.fg,
        }}
      />
    </Tooltip>
  );
}

/** Exact vs sketch-derived honesty tag. */
export function ExactnessTag({ approx, sx }: { approx: boolean; sx?: object }) {
  return (
    <Chip
      size="small"
      label={approx ? 'approx' : 'exact'}
      sx={{
        height: 16,
        fontSize: 9.5,
        fontFamily: 'monospace',
        bgcolor: approx ? '#4a3b12' : '#12303f',
        color: approx ? '#ffd54f' : '#81d4fa',
        ...sx,
      }}
    />
  );
}

// ---------------------------------------------------------------- query row helpers

/** Reads a numeric metric out of a QueryRow (values arrive as string | number | null). */
export function rowNum(row: QueryRow, key: string): number | null {
  const v = row[key];
  if (v === null || v === undefined) return null;
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isNaN(n) ? null : n;
}

export function rowStr(row: QueryRow, key: string): string {
  const v = row[key];
  return v === null || v === undefined ? '—' : String(v);
}

// ---------------------------------------------------------------- inline percent bar

/** Compact percentage cell with an inline bar, for dense aggregate tables. */
export function PctBarCell({
  value,
  color = '#4fc3f7',
  width = 84,
}: {
  value: number | null | undefined;
  color?: string;
  width?: number;
}) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return (
      <Typography variant="caption" sx={{ color: '#8a949e', fontFamily: 'monospace' }}>
        —
      </Typography>
    );
  }
  const frac = Math.max(0, Math.min(1, value <= 1 ? value : value / 100));
  return (
    <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75 }}>
      <Box sx={{ width, height: 6, bgcolor: '#232a31', borderRadius: 1, overflow: 'hidden' }}>
        <Box sx={{ height: '100%', width: `${frac * 100}%`, bgcolor: color, borderRadius: 1 }} />
      </Box>
      <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#e6e9ec', minWidth: 44 }}>
        {fmtPct(value)}
      </Typography>
    </Box>
  );
}

// ---------------------------------------------------------------- outcome / error type chips

export const OUTCOME_COLORS: Record<string, { bg: string; fg: string }> = {
  TP: { bg: '#1b5e20', fg: '#a5d6a7' },
  FN: { bg: '#b71c1c', fg: '#ffcdd2' },
  FP: { bg: '#e65100', fg: '#ffe0b2' },
  LOCALIZATION: { bg: '#4a148c', fg: '#ce93d8' },
  ANOMALY: { bg: '#880e4f', fg: '#f48fb1' },
  LOW_CONF: { bg: '#f9a825', fg: '#212121' },
};

export function OutcomeChip({ outcome }: { outcome: string }) {
  const colors = OUTCOME_COLORS[outcome] ?? { bg: '#37474f', fg: '#cfd8dc' };
  return (
    <Chip
      size="small"
      label={outcome}
      sx={{ height: 18, fontSize: 10, fontWeight: 700, fontFamily: 'monospace', bgcolor: colors.bg, color: colors.fg }}
    />
  );
}

export const ERROR_TYPE_COLORS: Record<ErrorType, string> = {
  FN: '#ef5350',
  FP: '#ffa726',
  LOCALIZATION: '#ab47bc',
  ANOMALY: '#ec407a',
  LOW_CONF: '#ffee58',
};

// ---------------------------------------------------------------- run status

const RUN_STATUS_COLORS: Record<string, { bg: string; fg: string }> = {
  created: { bg: '#37474f', fg: '#cfd8dc' },
  queued: { bg: '#37474f', fg: '#cfd8dc' },
  running: { bg: '#0d47a1', fg: '#90caf9' },
  reducing: { bg: '#4a148c', fg: '#ce93d8' },
  materializing: { bg: '#004d40', fg: '#80cbc4' },
  published: { bg: '#1b5e20', fg: '#a5d6a7' },
  failed: { bg: '#b71c1c', fg: '#ffcdd2' },
};

export function RunStatusChip({ status }: { status: string }) {
  const colors = RUN_STATUS_COLORS[status] ?? { bg: '#37474f', fg: '#cfd8dc' };
  return (
    <Chip
      size="small"
      label={status.toUpperCase()}
      sx={{ height: 20, fontSize: 10, fontWeight: 700, letterSpacing: 0.5, bgcolor: colors.bg, color: colors.fg }}
    />
  );
}

// ---------------------------------------------------------------- container status

export function ContainerStatusIcon({ status }: { status: 'ok' | 'warn' | 'critical' }) {
  if (status === 'critical') return <OctagonAlert size={15} color="#ef5350" />;
  if (status === 'warn') return <AlertTriangle size={15} color="#ffa726" />;
  return <CheckCircle2 size={15} color="#66bb6a" />;
}

// ---------------------------------------------------------------- deltas

/** Colored delta text. Set higherIsBetter=false for metrics like anomaly_rate / error_rate / fp. */
export function DeltaText({
  delta,
  higherIsBetter = true,
  asPct = true,
}: {
  delta: number;
  higherIsBetter?: boolean;
  asPct?: boolean;
}) {
  const good = higherIsBetter ? delta >= 0 : delta <= 0;
  const neutral = Math.abs(delta) < 1e-9;
  const magnitude = asPct ? (Math.abs(delta) * 100).toFixed(1) : Math.abs(delta).toFixed(3);
  const sign = delta > 0 ? '+' : delta < 0 ? '−' : '±';
  const text = `${sign}${magnitude}`;
  return (
    <Typography
      component="span"
      variant="caption"
      sx={{
        fontFamily: 'monospace',
        fontWeight: 700,
        color: neutral ? '#8a949e' : good ? '#66bb6a' : '#ef5350',
      }}
    >
      {text}
    </Typography>
  );
}

// ---------------------------------------------------------------- dimension chips

export function DimChips({ values, sx }: { values: (string | undefined)[]; sx?: object }) {
  return (
    <Box sx={{ display: 'inline-flex', gap: 0.5, flexWrap: 'wrap', ...sx }}>
      {values
        .filter((v): v is string => Boolean(v))
        .map((v, i) => (
          <Chip
            key={`${v}-${i}`}
            size="small"
            label={v}
            sx={{ height: 18, fontSize: 10, bgcolor: '#232a31', color: '#aab4be' }}
          />
        ))}
    </Box>
  );
}

// ---------------------------------------------------------------- histogram sparkline (sketch)

/** SVG histogram sparkline with p10 / p50 / p90 markers. Sketch-derived — always labeled approx. */
export function HistogramSparkline({
  sketch,
  color = '#4fc3f7',
  width = 260,
  height = 56,
}: {
  sketch: HistogramSketch;
  color?: string;
  width?: number;
  height?: number;
}) {
  const maxCount = Math.max(1, ...sketch.counts);
  const barW = width / Math.max(1, sketch.counts.length);
  const span = sketch.hi - sketch.lo || 1;
  const xFor = (v: number) => ((v - sketch.lo) / span) * width;
  const markers: Array<{ key: string; v: number }> = ['p10', 'p50', 'p90']
    .filter((k) => sketch.percentiles[k] !== undefined)
    .map((k) => ({ key: k, v: sketch.percentiles[k] }));
  return (
    <Box>
      <svg width={width} height={height + 14}>
        {sketch.counts.map((c, i) => {
          const h = (c / maxCount) * height;
          return (
            <rect
              key={i}
              x={i * barW + 0.5}
              y={height - h}
              width={Math.max(1, barW - 1)}
              height={Math.max(0.5, h)}
              fill={color}
              opacity={0.75}
            />
          );
        })}
        {markers.map((m) => (
          <g key={m.key}>
            <line x1={xFor(m.v)} x2={xFor(m.v)} y1={0} y2={height} stroke="#ffd54f" strokeDasharray="2,2" />
            <text x={xFor(m.v)} y={height + 11} fontSize={9} fill="#ffd54f" textAnchor="middle" fontFamily="monospace">
              {m.key} {m.v.toFixed(2)}
            </text>
          </g>
        ))}
      </svg>
    </Box>
  );
}

// ---------------------------------------------------------------- misc

export function CompactStat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minWidth: 90 }}>
      <Typography variant="caption" sx={{ color: '#8a949e', textTransform: 'uppercase', fontSize: 9.5 }}>
        {label}
      </Typography>
      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
        {value}
      </Typography>
    </Box>
  );
}

export function compactWithExact(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—';
  return `${fmtCompact(n)} (${Math.round(n).toLocaleString()})`;
}
