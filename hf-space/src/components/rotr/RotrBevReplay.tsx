/**
 * BEV schematic replay of one violation: top-down road geometry (lanes,
 * stop line, crosswalk, intersection box), actor positions at the scrubbed
 * time, and the OBSERVED vs CORRECTED ego trajectories from the
 * counterfactual replay.
 */

import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import type { RotrConsequenceDetail, RotrTrajectoryPoint } from '../../types/rotr';

const W = 860;
const H = 340;
const PAD = 24;

const ACTOR_COLOR: Record<string, string> = {
  pedestrian: '#ffb300',
  cyclist: '#ff7043',
  vehicle: '#ab47bc',
  bus: '#7e57c2',
};

interface Props {
  detail: RotrConsequenceDetail;
}

function nearest(traj: RotrTrajectoryPoint[], t: number): RotrTrajectoryPoint | null {
  if (!traj.length) return null;
  let best = traj[0];
  for (const p of traj) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
  return best;
}

export default function RotrBevReplay({ detail }: Props) {
  const obs = detail.planner_evaluation.observed_trajectory;
  const corr = detail.planner_evaluation.corrected_trajectory;
  const geom = detail.scenario_geometry;
  const tMax = obs.length ? obs[obs.length - 1].t : 0;
  const [t, setT] = useState(tMax / 2);

  const bounds = useMemo(() => {
    let xMin = -12;
    let xMax = 62;
    let yMin = -9;
    let yMax = 9;
    for (const p of [...obs, ...corr]) {
      xMin = Math.min(xMin, p.x - 3);
      xMax = Math.max(xMax, p.x + 3);
      yMin = Math.min(yMin, p.y - 3);
      yMax = Math.max(yMax, p.y + 3);
    }
    for (const a of geom?.actors ?? []) {
      for (const s of a.states) {
        if (s.y > -12 && s.y < 24) {
          yMin = Math.min(yMin, s.y - 2);
          yMax = Math.max(yMax, s.y + 2);
        }
      }
    }
    return { xMin, xMax, yMin, yMax };
  }, [obs, corr, geom]);

  const sx = (x: number) =>
    PAD + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * (W - 2 * PAD);
  const sy = (y: number) =>
    PAD + ((bounds.yMax - y) / (bounds.yMax - bounds.yMin)) * (H - 2 * PAD);

  const path = (traj: RotrTrajectoryPoint[]) =>
    traj.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');

  const obsNow = nearest(obs, t);
  const corrNow = nearest(corr, t);
  const ctx = geom?.actual_context;

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap', alignItems: 'center' }}>
        <Chip size="small" label="observed ego" sx={{ bgcolor: '#c62828', color: '#fff', height: 20 }} />
        <Chip size="small" label="corrected ego (counterfactual)" sx={{ bgcolor: '#2e7d32', color: '#fff', height: 20 }} />
        <Chip
          size="small"
          label={`engine: ${detail.planner_evaluation.engine}`}
          sx={{ bgcolor: '#232a31', height: 20, fontFamily: 'monospace', fontSize: 10 }}
        />
        <Chip
          size="small"
          label={detail.consequence_class}
          sx={{
            bgcolor: detail.consequence_class === 'SAFETY_CRITICAL' ? '#b71c1c' : '#37474f',
            color: '#fff',
            height: 20,
            fontWeight: 700,
          }}
        />
      </Box>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: '#0d1117', borderRadius: 8 }}>
        {/* lanes */}
        {ctx?.lanes.map((lane) => (
          <g key={lane.lane_id}>
            <rect
              x={sx(bounds.xMin)}
              y={sy(lane.center_y + 1.75)}
              width={sx(bounds.xMax) - sx(bounds.xMin)}
              height={sy(lane.center_y - 1.75) - sy(lane.center_y + 1.75)}
              fill={lane.restricted_to ? 'rgba(255,112,67,0.10)' : 'rgba(255,255,255,0.045)'}
              stroke="#2c333b"
              strokeDasharray="6 5"
            />
            <text x={sx(bounds.xMin) + 4} y={sy(lane.center_y) + 4} fontSize={10} fill="#5c6670">
              {lane.lane_id}
              {lane.restricted_to ? ` (${lane.restricted_to}-only)` : ''}
            </text>
          </g>
        ))}
        {/* intersection box */}
        {ctx?.intersection_x_min != null && ctx?.intersection_x_max != null ? (
          <rect
            x={sx(ctx.intersection_x_min)}
            y={PAD}
            width={sx(ctx.intersection_x_max) - sx(ctx.intersection_x_min)}
            height={H - 2 * PAD}
            fill="rgba(79,195,247,0.05)"
            stroke="#31424e"
          />
        ) : null}
        {/* crosswalks */}
        {ctx?.crosswalks.map((cw) => (
          <g key={cw.crosswalk_id}>
            {Array.from({ length: 7 }).map((_, i) => (
              <rect
                key={i}
                x={sx(cw.x_min) + (i * (sx(cw.x_max) - sx(cw.x_min))) / 7 + 2}
                y={sy(5)}
                width={(sx(cw.x_max) - sx(cw.x_min)) / 7 - 4}
                height={sy(-5) - sy(5)}
                fill="rgba(255,255,255,0.10)"
              />
            ))}
          </g>
        ))}
        {/* stop line */}
        {ctx?.stop_line_x != null ? (
          <line
            x1={sx(ctx.stop_line_x)}
            y1={PAD}
            x2={sx(ctx.stop_line_x)}
            y2={H - PAD}
            stroke="#e53935"
            strokeWidth={2}
            strokeDasharray="8 4"
          />
        ) : null}
        {/* trajectories */}
        <path d={path(obs)} fill="none" stroke="#c62828" strokeWidth={2.2} />
        <path d={path(corr)} fill="none" stroke="#2e7d32" strokeWidth={2.2} strokeDasharray="7 4" />
        {/* actors at t */}
        {geom?.actors.map((a) => {
          const s = a.states.length
            ? a.states.reduce((b, c) => (Math.abs(c.t - t) < Math.abs(b.t - t) ? c : b))
            : null;
          if (!s) return null;
          const color = ACTOR_COLOR[a.class_name] ?? '#90a4ae';
          const isPed = a.class_name === 'pedestrian' || a.class_name === 'cyclist';
          return (
            <g key={a.actor_id}>
              {isPed ? (
                <circle cx={sx(s.x)} cy={sy(s.y)} r={6} fill={color} />
              ) : (
                <rect
                  x={sx(s.x) - 11}
                  y={sy(s.y) - 6}
                  width={22}
                  height={12}
                  fill={color}
                  rx={2}
                  transform={`rotate(${(-s.yaw * 180) / Math.PI} ${sx(s.x)} ${sy(s.y)})`}
                />
              )}
              <text x={sx(s.x) + 9} y={sy(s.y) - 8} fontSize={10} fill={color}>
                {a.actor_id}
              </text>
            </g>
          );
        })}
        {/* ego markers at t */}
        {obsNow ? (
          <rect x={sx(obsNow.x) - 13} y={sy(obsNow.y) - 7} width={26} height={14} fill="#c62828" rx={3} stroke="#fff" strokeWidth={0.8} />
        ) : null}
        {corrNow ? (
          <rect x={sx(corrNow.x) - 13} y={sy(corrNow.y) - 7} width={26} height={14} fill="#2e7d32" rx={3} stroke="#fff" strokeWidth={0.8} opacity={0.9} />
        ) : null}
      </svg>
      <Box sx={{ px: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="caption" sx={{ color: '#8a949e', minWidth: 70 }}>
          t = {t.toFixed(1)} s
        </Typography>
        <Slider size="small" min={0} max={tMax} step={0.1} value={t} onChange={(_, v) => setT(v as number)} />
      </Box>
      <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', mt: 0.5 }}>
        {(
          [
            ['observed', detail.observed_safety, obsNow],
            ['corrected', detail.corrected_safety, corrNow],
          ] as const
        ).map(([label, sa, now]) => (
          <Typography key={label} variant="caption" sx={{ fontFamily: 'monospace', color: label === 'observed' ? '#ef9a9a' : '#a5d6a7' }}>
            {label}: v={now ? now.v.toFixed(1) : '—'} m/s · minTTC=
            {sa.min_ttc_s != null ? `${sa.min_ttc_s.toFixed(2)}s` : '—'} · clearance=
            {sa.min_clearance_m != null ? `${sa.min_clearance_m.toFixed(2)}m` : '—'} · PET=
            {sa.pet_s != null ? `${sa.pet_s.toFixed(2)}s` : '—'} · brake=
            {sa.max_braking_mps2.toFixed(1)} m/s² {sa.collision ? '· COLLISION' : ''}
          </Typography>
        ))}
      </Box>
      <Typography variant="caption" sx={{ color: '#6b7681', display: 'block', mt: 0.5 }}>
        divergence: {detail.planner_evaluation.max_position_divergence_m.toFixed(2)} m position ·{' '}
        {detail.planner_evaluation.max_speed_divergence_mps.toFixed(2)} m/s speed · corrected layers:{' '}
        {detail.corrected_layers.join(', ')} — {detail.observed_safety.surrogate_caveat}
      </Typography>
    </Box>
  );
}
