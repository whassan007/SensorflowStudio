/**
 * Interactive top-down BEV canvas for one perception frame: GT boxes, camera
 * detections with along-ray uncertainty ellipses, LiDAR detections, fused
 * boxes (masklet-propagated ones dashed), with hover details and
 * click-to-follow-track across frames. Built on the shared CanvasSurface
 * (pan/zoom) primitive.
 */
import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import CanvasSurface from '../visual/canvas';
import type { BevDetection, BevFusedBox, BevGtBox, BevReplayFrame } from '../../services/bevfusion';
import { tokens } from '../../theme';

export const LAYER_META = {
  gt: { label: 'Ground truth', color: '#e6e9ec' },
  camera: { label: 'Camera dets + uncertainty', color: '#ffa726' },
  lidar: { label: 'LiDAR dets', color: '#26c6da' },
  fused: { label: 'Fused (tracked)', color: '#66bb6a' },
  masklet: { label: 'Masklet-propagated', color: '#ab47bc' },
  baseline: { label: 'Camera baseline', color: '#ef5350' },
} as const;

export type LayerId = keyof typeof LAYER_META;

const CLASS_COLORS: Record<string, string> = {
  vehicle: '#66bb6a',
  truck: '#9ccc65',
  pedestrian: '#ff7043',
  cyclist: '#ffca28',
  motorcycle: '#ec407a',
};

export interface HoverInfo {
  kind: 'gt' | 'camera' | 'lidar' | 'fused' | 'baseline';
  title: string;
  lines: string[];
}

// World: x forward 0..80m (drawn rightward), y left/right ±25m (drawn vertically).
const WORLD = { x: -4, y: -25, w: 88, h: 50 };

/** 1-sigma covariance ellipse parameters from a 2x2 covariance matrix. */
function covEllipse(cov: [[number, number], [number, number]]): { rx: number; ry: number; angleDeg: number } {
  const [[a, b], [, d]] = cov;
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  const angle = Math.atan2(l1 - a, b || 1e-9);
  return { rx: Math.sqrt(Math.max(l1, 1e-6)), ry: Math.sqrt(Math.max(l2, 1e-6)), angleDeg: (angle * 180) / Math.PI };
}

function OrientedBox({
  x,
  y,
  l,
  w,
  yaw,
  stroke,
  fill = 'none',
  dashed = false,
  strokeWidth = 0.22,
  onEnter,
  onLeave,
  onClick,
  highlight = false,
}: {
  x: number;
  y: number;
  l: number;
  w: number;
  yaw: number;
  stroke: string;
  fill?: string;
  dashed?: boolean;
  strokeWidth?: number;
  onEnter?: () => void;
  onLeave?: () => void;
  onClick?: () => void;
  highlight?: boolean;
}) {
  // ego frame: +x forward, +y left. Screen: x → right, y → down = -world y.
  const deg = (-yaw * 180) / Math.PI;
  return (
    <g
      transform={`translate(${x}, ${-y}) rotate(${deg})`}
      onPointerEnter={onEnter}
      onPointerLeave={onLeave}
      onClick={(e) => {
        if (onClick) {
          e.stopPropagation();
          onClick();
        }
      }}
      style={{ cursor: onClick ? 'pointer' : undefined }}
    >
      <rect
        x={-l / 2}
        y={-w / 2}
        width={l}
        height={w}
        fill={fill}
        stroke={stroke}
        strokeWidth={highlight ? strokeWidth * 2 : strokeWidth}
        strokeDasharray={dashed ? '0.7 0.45' : undefined}
        rx={0.15}
      />
      {/* heading tick */}
      <line x1={l / 2} y1={0} x2={l / 2 + 0.9} y2={0} stroke={stroke} strokeWidth={strokeWidth} />
    </g>
  );
}

interface BevFrameCanvasProps {
  frame: BevReplayFrame;
  layers: Record<LayerId, boolean>;
  selectedTrackId: number | string | null;
  onSelectTrack: (id: number | string | null) => void;
  /** Track trail of the selected track: positions up to the current frame. */
  trail: Array<{ x: number; y: number }>;
  onHover: (info: HoverInfo | null) => void;
  height?: number;
}

export default function BevFrameCanvas({ frame, layers, selectedTrackId, onSelectTrack, trail, onHover, height = 460 }: BevFrameCanvasProps) {
  const [hoverKey, setHoverKey] = useState<string | null>(null);

  const hover = (key: string | null, info: HoverInfo | null) => {
    setHoverKey(key);
    onHover(info);
  };

  const gtBoxes = useMemo(() => frame.gt, [frame]);

  return (
    <CanvasSurface world={WORLD} height={height} gridStep={10} ariaLabel="Top-down bird's-eye-view of the perception frame" onBackgroundClick={() => onSelectTrack(null)}>
      {() => (
        <>
          {/* range rings + ego */}
          {[20, 40, 60, 80].map((r) => (
            <g key={r}>
              <circle cx={0} cy={0} r={r} fill="none" stroke={tokens.color.border} strokeWidth={0.12} strokeDasharray="1 1" />
              <text x={r - 0.5} y={-0.8} fontSize={1.6} fill={tokens.color.textFaint} textAnchor="end">{r}m</text>
            </g>
          ))}
          {/* ego vehicle */}
          <g>
            <polygon points="2.4,0 -1.4,1.1 -1.4,-1.1" fill={tokens.color.info} stroke="#fff" strokeWidth={0.12} />
            <text x={0} y={2.6} fontSize={1.6} fill={tokens.color.info} textAnchor="middle" fontWeight={700}>EGO</text>
          </g>

          {/* selected-track trail */}
          {trail.length > 1 ? (
            <polyline
              points={trail.map((p) => `${p.x},${-p.y}`).join(' ')}
              fill="none"
              stroke="#fff"
              strokeWidth={0.18}
              strokeDasharray="0.5 0.5"
              opacity={0.8}
            />
          ) : null}

          {/* GT */}
          {layers.gt
            ? gtBoxes.map((g: BevGtBox) => {
                const [x, y, , l, w, , yaw] = g.bbox_3d;
                const key = `gt-${g.instance_id}`;
                return (
                  <OrientedBox
                    key={key}
                    x={x}
                    y={y}
                    l={l}
                    w={w}
                    yaw={yaw}
                    stroke={g.occluded ? tokens.color.warn : 'rgba(230,233,236,0.75)'}
                    dashed={g.occluded}
                    highlight={hoverKey === key}
                    onEnter={() =>
                      hover(key, {
                        kind: 'gt',
                        title: `GT · ${g.class_name}`,
                        lines: [
                          `instance ${g.instance_id.split('-').slice(-2).join('-')}`,
                          `position (${x.toFixed(1)}, ${y.toFixed(1)}) m · ${g.distance.toFixed(1)} m away`,
                          g.occluded ? 'camera line-of-sight OCCLUDED this frame' : 'visible to camera',
                        ],
                      })
                    }
                    onLeave={() => hover(null, null)}
                  />
                );
              })
            : null}

          {/* camera detections + uncertainty ellipses */}
          {layers.camera
            ? frame.camera.map((d: BevDetection, i: number) => {
                const e = covEllipse(d.cov);
                const key = `cam-${i}`;
                return (
                  <g key={key}
                    onPointerEnter={() =>
                      hover(key, {
                        kind: 'camera',
                        title: `Camera det · ${d.class_name}`,
                        lines: [
                          `position (${d.x.toFixed(1)}, ${d.y.toFixed(1)}) m · conf ${d.confidence.toFixed(2)}`,
                          `1σ uncertainty ${e.rx.toFixed(1)} m along ray × ${e.ry.toFixed(1)} m across`,
                          'monocular depth: uncertainty stretches along the camera ray',
                        ],
                      })
                    }
                    onPointerLeave={() => hover(null, null)}
                  >
                    <ellipse
                      cx={d.x}
                      cy={-d.y}
                      rx={e.rx}
                      ry={e.ry}
                      transform={`rotate(${-e.angleDeg} ${d.x} ${-d.y})`}
                      fill="rgba(255,167,38,0.13)"
                      stroke={LAYER_META.camera.color}
                      strokeWidth={hoverKey === key ? 0.22 : 0.12}
                    />
                    <circle cx={d.x} cy={-d.y} r={0.45} fill={LAYER_META.camera.color} />
                  </g>
                );
              })
            : null}

          {/* lidar detections */}
          {layers.lidar
            ? frame.lidar.map((d: BevDetection, i: number) => {
                const key = `lid-${i}`;
                const s = 0.55;
                return (
                  <g key={key}
                    onPointerEnter={() =>
                      hover(key, {
                        kind: 'lidar',
                        title: `LiDAR det · ${d.class_name}`,
                        lines: [
                          `position (${d.x.toFixed(1)}, ${d.y.toFixed(1)}) m · conf ${d.confidence.toFixed(2)}`,
                          'tight isotropic position uncertainty; weak semantics (geometry templates)',
                        ],
                      })
                    }
                    onPointerLeave={() => hover(null, null)}
                  >
                    <line x1={d.x - s} y1={-d.y - s} x2={d.x + s} y2={-d.y + s} stroke={LAYER_META.lidar.color} strokeWidth={hoverKey === key ? 0.28 : 0.18} />
                    <line x1={d.x - s} y1={-d.y + s} x2={d.x + s} y2={-d.y - s} stroke={LAYER_META.lidar.color} strokeWidth={hoverKey === key ? 0.28 : 0.18} />
                  </g>
                );
              })
            : null}

          {/* camera baseline boxes */}
          {layers.baseline
            ? frame.baseline.map((b: BevFusedBox, i: number) => {
                const [x, y, , l, w, , yaw] = b.bbox_3d;
                const key = `base-${i}`;
                return (
                  <OrientedBox
                    key={key}
                    x={x}
                    y={y}
                    l={l}
                    w={w}
                    yaw={yaw}
                    stroke={LAYER_META.baseline.color}
                    strokeWidth={0.16}
                    highlight={hoverKey === key}
                    onEnter={() =>
                      hover(key, {
                        kind: 'baseline',
                        title: `Baseline (camera-only) · ${b.class_name}`,
                        lines: [
                          `track #${b.track_id} · conf ${b.confidence.toFixed(2)}`,
                          'greedy frame-to-frame association — IDs churn on dropouts',
                        ],
                      })
                    }
                    onLeave={() => hover(null, null)}
                  />
                );
              })
            : null}

          {/* fused boxes (masklet propagated = dashed) */}
          {frame.fused.map((b: BevFusedBox, i: number) => {
            const show = b.propagated ? layers.masklet : layers.fused;
            if (!show) return null;
            const [x, y, , l, w, , yaw] = b.bbox_3d;
            const key = `fused-${i}`;
            const selected = selectedTrackId !== null && b.track_id === selectedTrackId;
            const color = b.propagated ? LAYER_META.masklet.color : CLASS_COLORS[b.class_name] ?? LAYER_META.fused.color;
            return (
              <g key={key}>
                <OrientedBox
                  x={x}
                  y={y}
                  l={l}
                  w={w}
                  yaw={yaw}
                  stroke={selected ? '#ffffff' : color}
                  fill={selected ? 'rgba(255,255,255,0.12)' : `${color}22`}
                  dashed={b.propagated}
                  strokeWidth={0.26}
                  highlight={hoverKey === key || selected}
                  onEnter={() =>
                    hover(key, {
                      kind: 'fused',
                      title: `Fused · ${b.class_name}${b.propagated ? ' (masklet-propagated)' : ''}`,
                      lines: [
                        `track #${b.track_id} · conf ${b.confidence.toFixed(2)} · (${x.toFixed(1)}, ${y.toFixed(1)}) m`,
                        b.propagated
                          ? 'no detection this frame — box carried by the masklet motion model'
                          : 'decoded from the fused camera+LiDAR BEV grid',
                        'click to follow this track across frames',
                      ],
                    })
                  }
                  onLeave={() => hover(null, null)}
                  onClick={() => onSelectTrack(selected ? null : b.track_id)}
                />
                <text x={x} y={-y - w / 2 - 0.5} fontSize={1.5} fill={selected ? '#fff' : color} textAnchor="middle" fontWeight={700} pointerEvents="none">
                  #{b.track_id}
                </text>
              </g>
            );
          })}
        </>
      )}
    </CanvasSurface>
  );
}

export function CanvasLegend({ layers, onToggle }: { layers: Record<LayerId, boolean>; onToggle: (l: LayerId) => void }) {
  return (
    <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center' }}>
      {(Object.keys(LAYER_META) as LayerId[]).map((l) => (
        <Chip
          key={l}
          size="small"
          label={LAYER_META[l].label}
          onClick={() => onToggle(l)}
          sx={{
            height: 22,
            fontSize: 11,
            cursor: 'pointer',
            bgcolor: layers[l] ? `${LAYER_META[l].color}26` : 'transparent',
            color: layers[l] ? LAYER_META[l].color : tokens.color.textFaint,
            border: `1px solid ${layers[l] ? LAYER_META[l].color : tokens.color.border}`,
            textDecoration: layers[l] ? 'none' : 'line-through',
            transition: `all ${tokens.motion.fast}`,
          }}
        />
      ))}
      <Typography variant="caption" sx={{ color: tokens.color.neutral, ml: 0.5 }}>
        scroll to zoom · drag background to pan · hover any object · click a fused box to follow its track
      </Typography>
    </Box>
  );
}
