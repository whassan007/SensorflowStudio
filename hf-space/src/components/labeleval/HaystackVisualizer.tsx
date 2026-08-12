import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import type { HaystackCategory, HaystackPoint } from '../../types/labeleval';
import { SectionCard, fmtInt } from './shared';

const WIDTH = 700;
const HEIGHT = 420;
const PAD = 24;

const CATEGORY_STYLE: Record<HaystackCategory, { color: string; label: string }> = {
  normal: { color: '#5c6773', label: 'Normal' },
  anomaly: { color: '#ffa726', label: 'Anomaly' },
  rare_event: { color: '#ef5350', label: 'Rare event' },
  false_positive: { color: '#ab47bc', label: 'False positive' },
  false_negative: { color: '#42a5f5', label: 'False negative' },
  verified: { color: '#66bb6a', label: 'Verified' },
};

const CATEGORY_ORDER: HaystackCategory[] = [
  'normal',
  'verified',
  'false_positive',
  'false_negative',
  'anomaly',
  'rare_event',
];

export default function HaystackVisualizer({
  points,
  onPointClick,
}: {
  points: HaystackPoint[];
  onPointClick: (point: HaystackPoint) => void;
}) {
  const [hovered, setHovered] = useState<HaystackPoint | null>(null);

  const { scaled, counts } = useMemo(() => {
    const counts = new Map<HaystackCategory, number>();
    for (const p of points) counts.set(p.category, (counts.get(p.category) ?? 0) + 1);
    if (points.length === 0) return { scaled: [] as Array<HaystackPoint & { sx: number; sy: number }>, counts };
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const scaled = points.map((p) => ({
      ...p,
      sx: PAD + ((p.x - minX) / spanX) * (WIDTH - 2 * PAD),
      sy: HEIGHT - PAD - ((p.y - minY) / spanY) * (HEIGHT - 2 * PAD),
    }));
    // Draw interesting categories on top of the normal haystack.
    scaled.sort((a, b) => CATEGORY_ORDER.indexOf(a.category) - CATEGORY_ORDER.indexOf(b.category));
    return { scaled, counts };
  }, [points]);

  return (
    <SectionCard title="Haystack — finding the needles">
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Box sx={{ position: 'relative' }}>
          <svg
            width={WIDTH}
            height={HEIGHT}
            style={{ background: '#0c1014', border: '1px solid #232a31', borderRadius: 6, maxWidth: '100%' }}
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          >
            {/* subtle grid */}
            {Array.from({ length: 7 }, (_, i) => (
              <line
                key={`v${i}`}
                x1={PAD + (i * (WIDTH - 2 * PAD)) / 6}
                y1={PAD}
                x2={PAD + (i * (WIDTH - 2 * PAD)) / 6}
                y2={HEIGHT - PAD}
                stroke="#1a2027"
              />
            ))}
            {Array.from({ length: 5 }, (_, i) => (
              <line
                key={`h${i}`}
                x1={PAD}
                y1={PAD + (i * (HEIGHT - 2 * PAD)) / 4}
                x2={WIDTH - PAD}
                y2={PAD + (i * (HEIGHT - 2 * PAD)) / 4}
                stroke="#1a2027"
              />
            ))}
            {scaled.map((p) => {
              const style = CATEGORY_STYLE[p.category];
              const interesting = p.category !== 'normal';
              const common = {
                cursor: 'pointer',
                onClick: () => onPointClick(p),
                onMouseEnter: () => setHovered(p),
                onMouseLeave: () => setHovered(null),
              };
              if (p.category === 'rare_event') {
                const r = 6;
                return (
                  <polygon
                    key={p.id}
                    points={`${p.sx},${p.sy - r} ${p.sx + r},${p.sy} ${p.sx},${p.sy + r} ${p.sx - r},${p.sy}`}
                    fill={style.color}
                    stroke="#fff"
                    strokeWidth={0.8}
                    {...common}
                  />
                );
              }
              return (
                <circle
                  key={p.id}
                  cx={p.sx}
                  cy={p.sy}
                  r={interesting ? 4.5 : 2}
                  fill={style.color}
                  fillOpacity={interesting ? 0.95 : 0.55}
                  stroke={interesting ? '#101418' : 'none'}
                  strokeWidth={interesting ? 0.6 : 0}
                  {...common}
                />
              );
            })}
            {points.length === 0 ? (
              <text x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" fill="#8a949e" fontSize={13}>
                No haystack points yet — run the evaluation pipeline.
              </text>
            ) : null}
          </svg>
          {hovered ? (
            <Box
              sx={{
                position: 'absolute',
                left: 8,
                top: 8,
                bgcolor: 'rgba(16,20,24,0.92)',
                border: '1px solid #2c353e',
                borderRadius: 1,
                px: 1,
                py: 0.5,
                pointerEvents: 'none',
              }}
            >
              <Typography variant="caption" sx={{ display: 'block', fontFamily: 'monospace' }}>
                {hovered.id}
              </Typography>
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                {hovered.class_name} · {CATEGORY_STYLE[hovered.category].label} · score{' '}
                {hovered.anomaly_score.toFixed(3)} · {hovered.frame_id}
              </Typography>
            </Box>
          ) : null}
        </Box>

        <Box sx={{ minWidth: 170 }}>
          <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
            LEGEND
          </Typography>
          {CATEGORY_ORDER.map((cat) => {
            const style = CATEGORY_STYLE[cat];
            return (
              <Box key={cat} sx={{ display: 'flex', alignItems: 'center', gap: 1, my: 0.75 }}>
                {cat === 'rare_event' ? (
                  <Box sx={{ width: 10, height: 10, bgcolor: style.color, transform: 'rotate(45deg)' }} />
                ) : (
                  <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: style.color }} />
                )}
                <Typography variant="body2" sx={{ flex: 1 }}>
                  {style.label}
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {fmtInt(counts.get(cat) ?? 0)}
                </Typography>
              </Box>
            );
          })}
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
            Click any point to open its full evidence panel.
          </Typography>
        </Box>
      </Box>
    </SectionCard>
  );
}
