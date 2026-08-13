/**
 * Brush-selectable scatter chart (direct-manipulation primitive).
 *
 * Drag a rectangle over the plot to select points — the selection is
 * reported via `onBrush` and highlighted in place. Click empty space to
 * clear. Used by the calibration residual scatter and the visual query
 * builder preview.
 */
import { useMemo, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { tokens } from '../../theme';

export interface BrushPoint {
  id: string;
  x: number;
  y: number;
  color?: string;
  label?: string;
  r?: number;
}

interface BrushChartProps {
  points: BrushPoint[];
  width?: number | string;
  height?: number;
  xLabel: string;
  yLabel: string;
  onBrush?: (selectedIds: string[] | null) => void;
  /** Horizontal reference lines in y units, e.g. thresholds. */
  refLinesY?: Array<{ y: number; label: string; color?: string }>;
  /** Vertical reference lines in x units. */
  refLinesX?: Array<{ x: number; label: string; color?: string }>;
  xDomain?: [number, number];
  yDomain?: [number, number];
}

const PAD = { l: 46, r: 12, t: 10, b: 32 };
const VIEW_W = 640;

function niceDomain(vals: number[], fixed?: [number, number]): [number, number] {
  if (fixed) return fixed;
  if (!vals.length) return [0, 1];
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

export default function BrushChart({
  points,
  width = '100%',
  height = 260,
  xLabel,
  yLabel,
  onBrush,
  refLinesY = [],
  refLinesX = [],
  xDomain,
  yDomain,
}: BrushChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [brush, setBrush] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [selected, setSelected] = useState<Set<string> | null>(null);
  const [hover, setHover] = useState<BrushPoint | null>(null);

  const viewH = height;
  const [xLo, xHi] = useMemo(
    () => niceDomain([...points.map((p) => p.x), ...refLinesX.map((r) => r.x)], xDomain),
    [points, refLinesX, xDomain]
  );
  const [yLo, yHi] = useMemo(
    () => niceDomain([...points.map((p) => p.y), ...refLinesY.map((r) => r.y)], yDomain),
    [points, refLinesY, yDomain]
  );

  const sx = (x: number) => PAD.l + ((x - xLo) / (xHi - xLo)) * (VIEW_W - PAD.l - PAD.r);
  const sy = (y: number) => viewH - PAD.b - ((y - yLo) / (yHi - yLo)) * (viewH - PAD.t - PAD.b);

  const toLocal = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const rect = svg.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * VIEW_W,
      y: ((e.clientY - rect.top) / rect.height) * viewH,
    };
  };

  const finishBrush = (b: { x0: number; y0: number; x1: number; y1: number }) => {
    const [bx0, bx1] = [Math.min(b.x0, b.x1), Math.max(b.x0, b.x1)];
    const [by0, by1] = [Math.min(b.y0, b.y1), Math.max(b.y0, b.y1)];
    if (bx1 - bx0 < 4 && by1 - by0 < 4) {
      setSelected(null);
      onBrush?.(null);
      return;
    }
    const ids = points.filter((p) => {
      const px = sx(p.x);
      const py = sy(p.y);
      return px >= bx0 && px <= bx1 && py >= by0 && py <= by1;
    }).map((p) => p.id);
    setSelected(new Set(ids));
    onBrush?.(ids);
  };

  const xTicks = useMemo(() => {
    const n = 5;
    return Array.from({ length: n + 1 }, (_, i) => xLo + ((xHi - xLo) * i) / n);
  }, [xLo, xHi]);
  const yTicks = useMemo(() => {
    const n = 4;
    return Array.from({ length: n + 1 }, (_, i) => yLo + ((yHi - yLo) * i) / n);
  }, [yLo, yHi]);

  return (
    <Box sx={{ width, position: 'relative' }}>
      <svg
        ref={svgRef}
        width="100%"
        viewBox={`0 0 ${VIEW_W} ${viewH}`}
        style={{ display: 'block', cursor: 'crosshair', touchAction: 'none' }}
        onPointerDown={(e) => {
          const p = toLocal(e);
          (e.target as Element).setPointerCapture(e.pointerId);
          setBrush({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
        }}
        onPointerMove={(e) => {
          if (!brush || !(e.target as Element).hasPointerCapture?.(e.pointerId)) return;
          const p = toLocal(e);
          setBrush({ ...brush, x1: p.x, y1: p.y });
        }}
        onPointerUp={(e) => {
          (e.target as Element).releasePointerCapture(e.pointerId);
          if (brush) finishBrush(brush);
          setBrush(null);
        }}
      >
        {/* axes */}
        <line x1={PAD.l} y1={viewH - PAD.b} x2={VIEW_W - PAD.r} y2={viewH - PAD.b} stroke={tokens.color.borderStrong} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={viewH - PAD.b} stroke={tokens.color.borderStrong} />
        {xTicks.map((t) => (
          <g key={`xt${t}`}>
            <line x1={sx(t)} y1={viewH - PAD.b} x2={sx(t)} y2={viewH - PAD.b + 4} stroke={tokens.color.borderStrong} />
            <text x={sx(t)} y={viewH - PAD.b + 15} textAnchor="middle" fontSize={9.5} fill={tokens.color.neutral}>
              {Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(2)}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`yt${t}`}>
            <line x1={PAD.l} y1={sy(t)} x2={VIEW_W - PAD.r} y2={sy(t)} stroke={tokens.color.border} strokeDasharray="2 4" />
            <text x={PAD.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9.5} fill={tokens.color.neutral}>
              {Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={(PAD.l + VIEW_W - PAD.r) / 2} y={viewH - 4} textAnchor="middle" fontSize={10.5} fill={tokens.color.textDim}>
          {xLabel}
        </text>
        <text x={12} y={(PAD.t + viewH - PAD.b) / 2} textAnchor="middle" fontSize={10.5} fill={tokens.color.textDim} transform={`rotate(-90 12 ${(PAD.t + viewH - PAD.b) / 2})`}>
          {yLabel}
        </text>

        {/* reference lines */}
        {refLinesY.map((r) => (
          <g key={`ry${r.label}`}>
            <line x1={PAD.l} y1={sy(r.y)} x2={VIEW_W - PAD.r} y2={sy(r.y)} stroke={r.color ?? tokens.color.warn} strokeDasharray="6 4" strokeWidth={1.2} />
            <text x={VIEW_W - PAD.r - 4} y={sy(r.y) - 4} textAnchor="end" fontSize={9.5} fill={r.color ?? tokens.color.warn}>
              {r.label}
            </text>
          </g>
        ))}
        {refLinesX.map((r) => (
          <g key={`rx${r.label}`}>
            <line x1={sx(r.x)} y1={PAD.t} x2={sx(r.x)} y2={viewH - PAD.b} stroke={r.color ?? tokens.color.warn} strokeDasharray="6 4" strokeWidth={1.2} />
            <text x={sx(r.x) + 4} y={PAD.t + 10} fontSize={9.5} fill={r.color ?? tokens.color.warn}>
              {r.label}
            </text>
          </g>
        ))}

        {/* points */}
        {points.map((p) => {
          const dim = selected !== null && !selected.has(p.id);
          return (
            <circle
              key={p.id}
              cx={sx(p.x)}
              cy={sy(p.y)}
              r={p.r ?? 4}
              fill={p.color ?? tokens.color.info}
              opacity={dim ? 0.18 : 0.85}
              stroke={hover?.id === p.id ? '#fff' : 'none'}
              strokeWidth={1.2}
              onPointerEnter={() => setHover(p)}
              onPointerLeave={() => setHover((h) => (h?.id === p.id ? null : h))}
              style={{ transition: `opacity ${tokens.motion.fast}` }}
            />
          );
        })}

        {/* live brush rect */}
        {brush ? (
          <rect
            x={Math.min(brush.x0, brush.x1)}
            y={Math.min(brush.y0, brush.y1)}
            width={Math.abs(brush.x1 - brush.x0)}
            height={Math.abs(brush.y1 - brush.y0)}
            fill="rgba(79,195,247,0.12)"
            stroke={tokens.color.info}
            strokeDasharray="4 3"
            pointerEvents="none"
          />
        ) : null}
      </svg>

      {hover ? (
        <Box
          sx={{
            position: 'absolute',
            top: 4,
            right: 8,
            px: 1,
            py: 0.5,
            bgcolor: '#1d242c',
            border: `1px solid ${tokens.color.borderStrong}`,
            borderRadius: 1,
            pointerEvents: 'none',
          }}
        >
          <Typography variant="caption" sx={{ fontFamily: 'monospace', fontSize: 11 }}>
            {hover.label ?? hover.id}: ({hover.x.toFixed(2)}, {hover.y.toFixed(2)})
          </Typography>
        </Box>
      ) : null}
    </Box>
  );
}
