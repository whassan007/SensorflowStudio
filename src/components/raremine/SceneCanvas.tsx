/**
 * Schematic BEV rendering of a scene from metadata: ego vehicle at the bottom,
 * forward distance up the canvas, lateral offset across. The mined candidate
 * object is highlighted with a bounding ring; other objects are gray. This is
 * a schematic — no imagery exists, only the generator's object metadata.
 */
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import type { SceneView } from '../../types/raremine';

const W = 300;
const H = 260;
const MAX_FORWARD = 60; // metres shown
const MAX_LATERAL = 14;

function projX(lateral: number): number {
  return W / 2 + (lateral / MAX_LATERAL) * (W / 2 - 16);
}

function projY(forward: number): number {
  return H - 26 - (Math.min(forward, MAX_FORWARD) / MAX_FORWARD) * (H - 50);
}

export default function SceneCanvas({ scene }: { scene: SceneView }) {
  return (
    <Box>
      <svg width={W} height={H} style={{ background: '#0d1117', borderRadius: 6, border: '1px solid #232a31' }}>
        {/* road corridor */}
        <rect x={W / 2 - 42} y={0} width={84} height={H} fill="#161d24" />
        <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="#2c3640" strokeDasharray="6 6" />
        {/* range rings */}
        {[15, 30, 45].map((d) => (
          <g key={d}>
            <line x1={10} y1={projY(d)} x2={W - 10} y2={projY(d)} stroke="#1c242d" />
            <text x={12} y={projY(d) - 3} fill="#4a5560" fontSize={9}>
              {d} m
            </text>
          </g>
        ))}
        {/* ego vehicle */}
        <rect x={W / 2 - 9} y={H - 24} width={18} height={20} rx={3} fill="#4fc3f7" opacity={0.9} />
        <text x={W / 2} y={H - 8} fill="#4fc3f7" fontSize={9} textAnchor="middle">
          EGO
        </text>
        {/* objects */}
        {scene.objects.map((o) => {
          const forward = o.position[0] ?? o.distance_m;
          const lateral = o.position[1] ?? 0;
          const x = projX(lateral);
          const y = projY(forward);
          const r = Math.max(5, 11 - forward / 8);
          return (
            <g key={o.object_id}>
              {o.is_candidate ? (
                <>
                  <rect
                    x={x - r - 5}
                    y={y - r - 5}
                    width={(r + 5) * 2}
                    height={(r + 5) * 2}
                    fill="none"
                    stroke="#e53935"
                    strokeWidth={1.6}
                    strokeDasharray="4 3"
                    rx={3}
                  />
                  <circle cx={x} cy={y} r={r} fill="#e53935" opacity={0.85} />
                  <text x={x} y={y - r - 9} fill="#e53935" fontSize={9.5} fontWeight={700} textAnchor="middle">
                    CANDIDATE {forward.toFixed(0)} m
                  </text>
                </>
              ) : (
                <circle cx={x} cy={y} r={r} fill="#5c6a76" opacity={0.8} />
              )}
              {o.gt ? (
                <text x={x} y={y + r + 10} fill="#66bb6a" fontSize={8} textAnchor="middle">
                  GT: {o.gt.class_name}
                  {o.gt.is_costumed ? ' (costumed)' : ''}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <Typography variant="caption" sx={{ color: '#5c6a76', display: 'block', mt: 0.5 }}>
        {scene.lighting} · {scene.weather} · frame {scene.frame_index} · modalities:{' '}
        {Object.entries(scene.modalities)
          .filter(([, on]) => on)
          .map(([m]) => m)
          .join(', ')}
      </Typography>
    </Box>
  );
}
