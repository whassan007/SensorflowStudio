/**
 * Shared lightweight SVG chart components used across the new visual pages:
 * HeatmapGrid (ODD coverage, regression map), Donut (composition), DeltaBars
 * (paired baseline/candidate deltas), and SeriesChart (multi-line with
 * reference boundaries — the base of the sequential-evidence chart).
 * No chart library: everything is plain SVG in theme colors.
 */
import { useMemo, useState, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { tokens } from '../../theme';

// ---------------------------------------------------------------- HeatmapGrid

export interface HeatCell {
  row: string;
  col: string;
  value: number | null;
  /** Fully resolved fill color; the page owns the color semantics. */
  color: string;
  label?: string;
  tooltip?: ReactNode;
}

interface HeatmapGridProps {
  rows: string[];
  cols: string[];
  cells: HeatCell[];
  onCellClick?: (cell: HeatCell) => void;
  selectedKey?: string | null; // `${row}|${col}`
  cellH?: number;
  minCellW?: number;
  rowLabelW?: number;
}

export function HeatmapGrid({
  rows,
  cols,
  cells,
  onCellClick,
  selectedKey,
  cellH = 34,
  minCellW = 56,
  rowLabelW = 120,
}: HeatmapGridProps) {
  const byKey = useMemo(() => {
    const m = new Map<string, HeatCell>();
    cells.forEach((c) => m.set(`${c.row}|${c.col}`, c));
    return m;
  }, [cells]);

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: `${rowLabelW}px repeat(${cols.length}, minmax(${minCellW}px, 1fr))`, gap: '3px', minWidth: rowLabelW + cols.length * minCellW }}>
        <Box />
        {cols.map((c) => (
          <Typography key={c} variant="caption" sx={{ color: tokens.color.textDim, textAlign: 'center', fontSize: 10.5, alignSelf: 'end', pb: 0.25, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {c}
          </Typography>
        ))}
        {rows.map((r) => (
          <Box key={r} sx={{ display: 'contents' }}>
            <Typography variant="caption" sx={{ color: tokens.color.textDim, fontSize: 10.5, alignSelf: 'center', pr: 0.5, textAlign: 'right', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {r}
            </Typography>
            {cols.map((c) => {
              const cell = byKey.get(`${r}|${c}`);
              const selected = selectedKey === `${r}|${c}`;
              const body = (
                <Box
                  key={c}
                  role={onCellClick ? 'button' : undefined}
                  tabIndex={onCellClick && cell ? 0 : undefined}
                  onClick={cell && onCellClick ? () => onCellClick(cell) : undefined}
                  onKeyDown={cell && onCellClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') onCellClick(cell); } : undefined}
                  sx={{
                    height: cellH,
                    borderRadius: '3px',
                    bgcolor: cell ? cell.color : tokens.color.surfaceSunken,
                    border: selected ? `2px solid #fff` : `1px solid rgba(0,0,0,0.35)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: cell && onCellClick ? 'pointer' : 'default',
                    transition: `transform ${tokens.motion.fast}, border-color ${tokens.motion.fast}`,
                    '&:hover': cell && onCellClick ? { transform: 'scale(1.06)', zIndex: 1 } : undefined,
                    '&:focus-visible': { outline: `2px solid ${tokens.color.info}`, outlineOffset: 1 },
                  }}
                >
                  <Typography variant="caption" sx={{ fontSize: 9.5, fontFamily: 'monospace', color: 'rgba(255,255,255,0.92)', fontWeight: 600 }}>
                    {cell?.label ?? ''}
                  </Typography>
                </Box>
              );
              return cell?.tooltip ? (
                <Tooltip key={c} title={cell.tooltip} enterDelay={150} slotProps={{ tooltip: { sx: { bgcolor: '#1d242c', border: `1px solid ${tokens.color.borderStrong}`, p: 1.25, maxWidth: 360 } } }}>
                  {body}
                </Tooltip>
              ) : (
                body
              );
            })}
          </Box>
        ))}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------- Donut

export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

export function Donut({ segments, size = 150, thickness = 22, centerLabel, centerSub }: { segments: DonutSegment[]; size?: number; thickness?: number; centerLabel?: string; centerSub?: string }) {
  const total = segments.reduce((s, x) => s + x.value, 0);
  const r = (size - thickness) / 2;
  const cx = size / 2;
  const circumference = 2 * Math.PI * r;
  let acc = 0;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cx} r={r} fill="none" stroke={tokens.color.border} strokeWidth={thickness} />
        {total > 0
          ? segments.map((s) => {
              const frac = s.value / total;
              const dash = frac * circumference;
              const el = (
                <circle
                  key={s.label}
                  cx={cx}
                  cy={cx}
                  r={r}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={thickness}
                  strokeDasharray={`${dash} ${circumference - dash}`}
                  strokeDashoffset={-acc * circumference + circumference / 4}
                  style={{ transition: `stroke-dasharray ${tokens.motion.slow}` }}
                />
              );
              acc += frac;
              return el;
            })
          : null}
        {centerLabel ? (
          <text x={cx} y={cx + (centerSub ? -2 : 5)} textAnchor="middle" fill={tokens.color.text} fontSize={18} fontWeight={800}>
            {centerLabel}
          </text>
        ) : null}
        {centerSub ? (
          <text x={cx} y={cx + 15} textAnchor="middle" fill={tokens.color.neutral} fontSize={9.5}>
            {centerSub}
          </text>
        ) : null}
      </svg>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        {segments.map((s) => (
          <Box key={s.label} sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <Box sx={{ width: 10, height: 10, borderRadius: '2px', bgcolor: s.color }} />
            <Typography variant="caption" sx={{ color: tokens.color.textDim }}>
              {s.label}
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: 'monospace', color: tokens.color.text }}>
              {s.value.toLocaleString()}
              {total > 0 ? ` (${((s.value / total) * 100).toFixed(1)}%)` : ''}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------- DeltaBars

export interface DeltaBarRow {
  label: string;
  baseline: number;
  candidate: number;
  delta: number;
  n?: number;
  improved: boolean;
  tooltip?: ReactNode;
}

/** Paired horizontal bars (baseline vs candidate) with a signed delta chip. */
export function DeltaBars({ rows, max = 1, fmt = (v: number) => v.toFixed(3) }: { rows: DeltaBarRow[]; max?: number; fmt?: (v: number) => string }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {rows.map((r) => {
        const body = (
          <Box key={r.label} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="caption" sx={{ width: 130, flexShrink: 0, color: tokens.color.textDim, textAlign: 'right', pr: 0.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {r.label}
              {r.n !== undefined ? <span style={{ color: tokens.color.textFaint }}> · n={r.n.toLocaleString()}</span> : null}
            </Typography>
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '3px' }}>
              <Box sx={{ height: 7, bgcolor: tokens.color.border, borderRadius: 1, overflow: 'hidden' }}>
                <Box sx={{ height: '100%', width: `${Math.min(100, (r.baseline / max) * 100)}%`, bgcolor: tokens.color.textFaint, transition: `width ${tokens.motion.slow}` }} />
              </Box>
              <Box sx={{ height: 7, bgcolor: tokens.color.border, borderRadius: 1, overflow: 'hidden' }}>
                <Box sx={{ height: '100%', width: `${Math.min(100, (r.candidate / max) * 100)}%`, bgcolor: r.improved ? tokens.color.success : tokens.color.danger, transition: `width ${tokens.motion.slow}` }} />
              </Box>
            </Box>
            <Typography
              variant="caption"
              sx={{
                width: 88,
                flexShrink: 0,
                fontFamily: 'monospace',
                fontWeight: 700,
                color: r.improved ? tokens.color.success : tokens.color.danger,
              }}
            >
              {r.delta >= 0 ? '+' : ''}
              {fmt(r.delta)}
            </Typography>
          </Box>
        );
        return r.tooltip ? (
          <Tooltip key={r.label} title={r.tooltip} enterDelay={150} placement="top" slotProps={{ tooltip: { sx: { bgcolor: '#1d242c', border: `1px solid ${tokens.color.borderStrong}`, p: 1.25, maxWidth: 380 } } }}>
            {body}
          </Tooltip>
        ) : (
          body
        );
      })}
    </Box>
  );
}

// ---------------------------------------------------------------- SeriesChart

export interface Series {
  id: string;
  color: string;
  points: Array<{ x: number; y: number; meta?: unknown }>;
  width?: number;
  dashed?: boolean;
}

interface SeriesChartProps {
  series: Series[];
  height?: number;
  xLabel: string;
  yLabel: string;
  refLinesY?: Array<{ y: number; label: string; color?: string; dashed?: boolean }>;
  /** Vertical marker, e.g. "stopped here". */
  markerX?: { x: number; label: string; color?: string } | null;
  onHoverPoint?: (seriesId: string | null, point: { x: number; y: number; meta?: unknown } | null) => void;
  yDomain?: [number, number];
  xLog?: boolean;
}

const SPAD = { l: 52, r: 14, t: 12, b: 34 };
const SVIEW_W = 720;

export function SeriesChart({ series, height = 300, xLabel, yLabel, refLinesY = [], markerX = null, onHoverPoint, yDomain }: SeriesChartProps) {
  const [hover, setHover] = useState<{ sid: string; x: number; y: number } | null>(null);

  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  const allY = [...series.flatMap((s) => s.points.map((p) => p.y)), ...refLinesY.map((r) => r.y)];
  const xLo = allX.length ? Math.min(...allX) : 0;
  const xHi = allX.length ? Math.max(...allX, xLo + 1) : 1;
  let yLo: number;
  let yHi: number;
  if (yDomain) {
    [yLo, yHi] = yDomain;
  } else {
    yLo = allY.length ? Math.min(...allY) : 0;
    yHi = allY.length ? Math.max(...allY, yLo + 1) : 1;
    const pad = (yHi - yLo) * 0.1 || 1;
    yLo -= pad;
    yHi += pad;
  }

  const sx = (x: number) => SPAD.l + ((x - xLo) / (xHi - xLo || 1)) * (SVIEW_W - SPAD.l - SPAD.r);
  const sy = (y: number) => height - SPAD.b - ((y - yLo) / (yHi - yLo || 1)) * (height - SPAD.t - SPAD.b);

  const xTicks = Array.from({ length: 6 }, (_, i) => xLo + ((xHi - xLo) * i) / 5);
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + ((yHi - yLo) * i) / 4);

  return (
    <svg width="100%" viewBox={`0 0 ${SVIEW_W} ${height}`} style={{ display: 'block' }}>
      <line x1={SPAD.l} y1={height - SPAD.b} x2={SVIEW_W - SPAD.r} y2={height - SPAD.b} stroke={tokens.color.borderStrong} />
      <line x1={SPAD.l} y1={SPAD.t} x2={SPAD.l} y2={height - SPAD.b} stroke={tokens.color.borderStrong} />
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} y1={height - SPAD.b} x2={sx(t)} y2={height - SPAD.b + 4} stroke={tokens.color.borderStrong} />
          <text x={sx(t)} y={height - SPAD.b + 15} textAnchor="middle" fontSize={9.5} fill={tokens.color.neutral}>
            {Math.round(t).toLocaleString()}
          </text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={SPAD.l} y1={sy(t)} x2={SVIEW_W - SPAD.r} y2={sy(t)} stroke={tokens.color.border} strokeDasharray="2 4" />
          <text x={SPAD.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9.5} fill={tokens.color.neutral}>
            {Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(1)}
          </text>
        </g>
      ))}
      <text x={(SPAD.l + SVIEW_W - SPAD.r) / 2} y={height - 5} textAnchor="middle" fontSize={11} fill={tokens.color.textDim}>
        {xLabel}
      </text>
      <text x={13} y={(SPAD.t + height - SPAD.b) / 2} textAnchor="middle" fontSize={11} fill={tokens.color.textDim} transform={`rotate(-90 13 ${(SPAD.t + height - SPAD.b) / 2})`}>
        {yLabel}
      </text>

      {refLinesY.map((r) => (
        <g key={`ref${r.label}`}>
          <line x1={SPAD.l} y1={sy(r.y)} x2={SVIEW_W - SPAD.r} y2={sy(r.y)} stroke={r.color ?? tokens.color.warn} strokeDasharray={r.dashed === false ? undefined : '7 4'} strokeWidth={1.4} />
          <text x={SVIEW_W - SPAD.r - 4} y={sy(r.y) - 5} textAnchor="end" fontSize={10} fontWeight={700} fill={r.color ?? tokens.color.warn}>
            {r.label}
          </text>
        </g>
      ))}

      {markerX ? (
        <g>
          <line x1={sx(markerX.x)} y1={SPAD.t} x2={sx(markerX.x)} y2={height - SPAD.b} stroke={markerX.color ?? tokens.color.danger} strokeWidth={1.6} strokeDasharray="3 3" />
          <text x={sx(markerX.x) + 5} y={SPAD.t + 11} fontSize={10} fontWeight={700} fill={markerX.color ?? tokens.color.danger}>
            {markerX.label}
          </text>
        </g>
      ) : null}

      {series.map((s) => (
        <g key={s.id}>
          <polyline
            points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width ?? 1.8}
            strokeDasharray={s.dashed ? '5 4' : undefined}
            opacity={hover && hover.sid !== s.id ? 0.25 : 1}
            style={{ transition: `opacity ${tokens.motion.fast}` }}
          />
          {s.points.map((p, i) => (
            <circle
              key={i}
              cx={sx(p.x)}
              cy={sy(p.y)}
              r={hover?.sid === s.id && hover.x === p.x ? 5 : 3}
              fill={s.color}
              opacity={hover && hover.sid !== s.id ? 0.25 : 1}
              onPointerEnter={() => {
                setHover({ sid: s.id, x: p.x, y: p.y });
                onHoverPoint?.(s.id, p);
              }}
              onPointerLeave={() => {
                setHover(null);
                onHoverPoint?.(null, null);
              }}
              style={{ cursor: 'pointer', transition: `r ${tokens.motion.fast}` }}
            />
          ))}
        </g>
      ))}
    </svg>
  );
}
