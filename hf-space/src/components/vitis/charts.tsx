/** Dependency-free SVG charts for the Hardware Acceleration page. */
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const AXIS = '#8a949e';
const GRID = '#232a31';

export function HBarChart({
  items,
  format,
  height = 30,
}: {
  items: { label: string; value: number; color?: string }[];
  format?: (v: number) => string;
  height?: number;
}) {
  const max = Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
      {items.map((it) => (
        <Box key={it.label} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" sx={{ width: 150, color: AXIS, flexShrink: 0 }}>
            {it.label}
          </Typography>
          <Box sx={{ flex: 1, bgcolor: GRID, borderRadius: 0.5, height: height * 0.5, position: 'relative' }}>
            <Box
              sx={{
                width: `${Math.max(1.5, (it.value / max) * 100)}%`,
                height: '100%',
                bgcolor: it.color ?? '#4fc3f7',
                borderRadius: 0.5,
              }}
            />
          </Box>
          <Typography variant="caption" sx={{ width: 62, textAlign: 'right', fontFamily: 'monospace' }}>
            {format ? format(it.value) : it.value.toFixed(3)}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

export function LineChart({
  points,
  xLabel,
  yLabel,
  color = '#4fc3f7',
  width = 460,
  height = 190,
  markers,
  yFormat,
}: {
  points: { x: number; y: number }[];
  xLabel: string;
  yLabel: string;
  color?: string;
  width?: number;
  height?: number;
  /** Optional per-point marker colors (e.g. verdict-coded). */
  markers?: (string | undefined)[];
  yFormat?: (v: number) => string;
}) {
  if (points.length === 0) return null;
  const pad = { l: 52, r: 12, t: 10, b: 32 };
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = 0;
  const yMax = Math.max(...ys) * 1.1 || 1;
  const sx = (x: number) =>
    pad.l + ((x - xMin) / Math.max(xMax - xMin, 1e-9)) * (width - pad.l - pad.r);
  const sy = (y: number) =>
    height - pad.b - ((y - yMin) / Math.max(yMax - yMin, 1e-9)) * (height - pad.t - pad.b);
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x)},${sy(p.y)}`).join(' ');
  const yTicks = [0, yMax / 2, yMax];
  const fmt = yFormat ?? ((v: number) => v.toPrecision(2));
  return (
    <svg width={width} height={height} role="img" aria-label={`${yLabel} vs ${xLabel}`}>
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={width - pad.r} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
          <text x={pad.l - 6} y={sy(t) + 3} fill={AXIS} fontSize={10} textAnchor="end">
            {fmt(t)}
          </text>
        </g>
      ))}
      {xs.map((x) => (
        <text key={x} x={sx(x)} y={height - pad.b + 16} fill={AXIS} fontSize={10} textAnchor="middle">
          {x}
        </text>
      ))}
      <text x={(pad.l + width - pad.r) / 2} y={height - 4} fill={AXIS} fontSize={10} textAnchor="middle">
        {xLabel}
      </text>
      <path d={path} fill="none" stroke={color} strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={`${p.x}-${i}`} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={markers?.[i] ?? color} stroke="#101418" strokeWidth={1} />
      ))}
    </svg>
  );
}
