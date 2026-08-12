import Typography from '@mui/material/Typography';
import { SectionCard, fmtInt } from './shared';

interface FlywheelCounts {
  verified: number;
  datasets: number;
  activeJobs: number;
  models: number;
  evaluated: number;
}

const NODES = [
  { key: 'verified', label: 'Verified Data' },
  { key: 'datasets', label: 'Training Dataset' },
  { key: 'activeJobs', label: 'Train' },
  { key: 'models', label: 'New Model' },
  { key: 'evaluated', label: 'New Evaluations' },
] as const;

export default function TrainingFlywheel({ counts }: { counts: FlywheelCounts }) {
  const W = 520;
  const H = 380;
  const cx = W / 2;
  const cy = H / 2;
  const R = 135;
  const nodeR = 44;

  const positions = NODES.map((_, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / NODES.length;
    return { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) };
  });

  return (
    <SectionCard title="Training Flywheel — the data engine loop">
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: 560 }}>
        <defs>
          <marker id="fly-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4fc3f7" />
          </marker>
        </defs>
        {/* edges around the circle */}
        {positions.map((p, i) => {
          const next = positions[(i + 1) % positions.length]!;
          const dx = next.x - p.x;
          const dy = next.y - p.y;
          const dist = Math.hypot(dx, dy) || 1;
          const startX = p.x + (dx / dist) * (nodeR + 4);
          const startY = p.y + (dy / dist) * (nodeR + 4);
          const endX = next.x - (dx / dist) * (nodeR + 8);
          const endY = next.y - (dy / dist) * (nodeR + 8);
          // bow the edge outward for a circular feel
          const mx = (startX + endX) / 2 + ((startX + endX) / 2 - cx) * 0.22;
          const my = (startY + endY) / 2 + ((startY + endY) / 2 - cy) * 0.22;
          return (
            <path
              key={i}
              d={`M ${startX} ${startY} Q ${mx} ${my} ${endX} ${endY}`}
              fill="none"
              stroke="#4fc3f7"
              strokeWidth={1.6}
              strokeOpacity={0.75}
              markerEnd="url(#fly-arrow)"
            />
          );
        })}
        {/* nodes */}
        {NODES.map((node, i) => {
          const p = positions[i]!;
          const value = counts[node.key];
          return (
            <g key={node.key}>
              <circle cx={p.x} cy={p.y} r={nodeR} fill="#12171d" stroke="#2c353e" strokeWidth={1.5} />
              <text x={p.x} y={p.y - 6} textAnchor="middle" fill="#e6e9ec" fontSize={11} fontWeight={700}>
                {node.label.split(' ').map((word, j) => (
                  <tspan key={j} x={p.x} dy={j === 0 ? 0 : 12}>
                    {word}
                  </tspan>
                ))}
              </text>
              <text x={p.x} y={p.y + nodeR - 14} textAnchor="middle" fill="#4fc3f7" fontSize={13} fontWeight={800}>
                {fmtInt(value)}
              </text>
            </g>
          );
        })}
        <text x={cx} y={cy - 4} textAnchor="middle" fill="#8a949e" fontSize={11}>
          verified labels feed
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill="#8a949e" fontSize={11}>
          better models feed better labels
        </text>
      </svg>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
        Only VERIFIED labels enter training datasets; every new model is evaluated (and regression-checked) before its
        outputs feed back into the loop.
      </Typography>
    </SectionCard>
  );
}
