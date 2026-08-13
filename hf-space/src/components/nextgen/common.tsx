/** Shared building blocks for the Closed-Loop Lab page. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { DataLabel } from '../../types/nextgen';

export const PANEL_SX = {
  p: 1.5,
  bgcolor: '#161c22',
  border: '1px solid #232a31',
  borderRadius: 1.5,
} as const;

const LABEL_COLORS: Record<DataLabel, string> = {
  REAL: '#2e7d32',
  REPLAYED: '#0277bd',
  SIMULATED: '#6a1b9a',
  GENERATED: '#ef6c00',
  COUNTERFACTUAL: '#c62828',
};

export function DataLabelChip({ label }: { label: DataLabel | string }) {
  const color = LABEL_COLORS[label as DataLabel] ?? '#455a64';
  return (
    <Chip
      size="small"
      label={label}
      sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: color, color: '#fff' }}
    />
  );
}

export function ScoreChip({ name, value }: { name: string; value: number }) {
  const color = value >= 0.8 ? '#2e7d32' : value >= 0.5 ? '#e65100' : '#b71c1c';
  return (
    <Chip
      size="small"
      label={`${name} ${(value * 100).toFixed(0)}%`}
      sx={{ height: 20, fontSize: 10.5, fontWeight: 700, bgcolor: '#12171d', color, border: `1px solid ${color}` }}
    />
  );
}

export function MetricCard({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  tone?: 'good' | 'bad' | 'neutral';
}) {
  const color = tone === 'good' ? '#81c784' : tone === 'bad' ? '#ef9a9a' : '#e0e3e7';
  return (
    <Paper sx={{ ...PANEL_SX, minWidth: 130, flex: '1 1 130px' }}>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', fontSize: 10.5 }}>
        {label}
      </Typography>
      <Typography variant="h6" sx={{ fontWeight: 800, fontSize: 18, color }}>
        {value === null || value === undefined ? '—' : value}
        {unit && value !== null && value !== undefined ? (
          <Typography component="span" variant="caption" sx={{ color: '#8a949e', ml: 0.5 }}>
            {unit}
          </Typography>
        ) : null}
      </Typography>
    </Paper>
  );
}

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
}

/** Dependency-free SVG line chart (time series comparison). */
export function LineChart({
  series,
  height = 160,
  xLabel,
  yLabel,
}: {
  series: Series[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
}) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;
  const w = 560;
  const pad = { l: 42, r: 10, t: 10, b: 24 };
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs) || 1;
  const yMin = Math.min(0, ...ys);
  const yMax = Math.max(...ys) * 1.08 || 1;
  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * (w - pad.l - pad.r);
  const sy = (y: number) => height - pad.b - ((y - yMin) / (yMax - yMin || 1)) * (height - pad.t - pad.b);
  const yTicks = [yMin, (yMin + yMax) / 2, yMax];
  const xTicks = [xMin, (xMin + xMax) / 2, xMax];
  return (
    <Box>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', maxWidth: w }}>
        {yTicks.map((t, i) => (
          <g key={`y${i}`}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="#232a31" strokeDasharray="3 3" />
            <text x={pad.l - 6} y={sy(t) + 3} fill="#8a949e" fontSize={9} textAnchor="end">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {xTicks.map((t, i) => (
          <text key={`x${i}`} x={sx(t)} y={height - 8} fill="#8a949e" fontSize={9} textAnchor="middle">
            {t.toFixed(1)}
          </text>
        ))}
        {series.map((s) => (
          <polyline
            key={s.name}
            fill="none"
            stroke={s.color}
            strokeWidth={1.8}
            points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')}
          />
        ))}
        {yLabel ? (
          <text x={10} y={pad.t + 4} fill="#8a949e" fontSize={9}>
            {yLabel}
          </text>
        ) : null}
        {xLabel ? (
          <text x={w - pad.r} y={height - 8} fill="#8a949e" fontSize={9} textAnchor="end">
            {xLabel}
          </text>
        ) : null}
      </svg>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {series.map((s) => (
          <Box key={s.name} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 14, height: 3, bgcolor: s.color, borderRadius: 1 }} />
            <Typography variant="caption" sx={{ color: '#c7ccd1', fontSize: 10.5 }}>
              {s.name}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
