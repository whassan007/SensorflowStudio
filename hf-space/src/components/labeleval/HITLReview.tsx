import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import Typography from '@mui/material/Typography';
import type { Annotation, FrameSummary, ReviewTask } from '../../types/labeleval';
import { SectionCard, StatusChip, GateLineList, LoadingBox, MetricCard, fmtNum, fmtPct } from './shared';

// ---------------------------------------------------------------- geometry helpers

type Pt = [number, number];

/** Rotated BEV rectangle corners for bbox_3d = [x,y,z,l,w,h,yaw]. */
function boxCornersBEV(bbox: readonly number[]): Pt[] {
  const [x, y, , l, w, , yaw] = bbox as [number, number, number, number, number, number, number];
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const dl = l / 2;
  const dw = w / 2;
  const local: Pt[] = [
    [dl, dw],
    [dl, -dw],
    [-dl, -dw],
    [-dl, dw],
  ];
  return local.map(([lx, ly]) => [x + lx * cos - ly * sin, y + lx * sin + ly * cos]);
}

interface BevTransform {
  toSvg: (p: Pt) => Pt;
}

function makeBevTransform(frame: FrameSummary, width: number, height: number, pad = 14): BevTransform {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const [px, py] of frame.lidar_points_bev) {
    xs.push(px);
    ys.push(py);
  }
  for (const a of frame.annotations) {
    if (a.bbox_3d) {
      xs.push(a.bbox_3d[0]);
      ys.push(a.bbox_3d[1]);
    }
  }
  for (const g of frame.gt_boxes) {
    xs.push(g.bbox_3d[0]);
    ys.push(g.bbox_3d[1]);
  }
  xs.push(frame.ego_pose.x);
  ys.push(frame.ego_pose.y);
  const minX = xs.length ? Math.min(...xs) : -50;
  const maxX = xs.length ? Math.max(...xs) : 50;
  const minY = ys.length ? Math.min(...ys) : -50;
  const maxY = ys.length ? Math.max(...ys) : 50;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return {
    toSvg: ([px, py]) => [width / 2 + (px - cx) * scale, height / 2 - (py - cy) * scale],
  };
}

function polygonPoints(corners: Pt[], t: BevTransform): string {
  return corners.map((c) => t.toSvg(c).map((v) => v.toFixed(1)).join(',')).join(' ');
}

// ---------------------------------------------------------------- individual views

const VIEW_W = 440;
const VIEW_H = 250;

function ViewFrame({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Box sx={{ flex: '1 1 440px', minWidth: 320 }}>
      <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
        {title}
      </Typography>
      <Box sx={{ mt: 0.5 }}>{children}</Box>
    </Box>
  );
}

function CameraView({ frame, highlightId }: { frame: FrameSummary; highlightId: string }) {
  const sx = VIEW_W / (frame.camera.width || 1);
  const sy = VIEW_H / (frame.camera.height || 1);
  return (
    <svg width="100%" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} style={{ borderRadius: 6, border: '1px solid #232a31' }}>
      <defs>
        <linearGradient id="cam-bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1c2733" />
          <stop offset="55%" stopColor="#131a22" />
          <stop offset="56%" stopColor="#0e1318" />
          <stop offset="100%" stopColor="#161c14" />
        </linearGradient>
      </defs>
      <rect width={VIEW_W} height={VIEW_H} fill="url(#cam-bg)" />
      {/* horizon + lane hints for the synthetic camera image */}
      <line x1={0} y1={VIEW_H * 0.55} x2={VIEW_W} y2={VIEW_H * 0.55} stroke="#26313c" />
      <line x1={VIEW_W * 0.5} y1={VIEW_H} x2={VIEW_W * 0.42} y2={VIEW_H * 0.55} stroke="#2c353e" strokeDasharray="6 6" />
      <line x1={VIEW_W * 0.62} y1={VIEW_H} x2={VIEW_W * 0.47} y2={VIEW_H * 0.55} stroke="#2c353e" strokeDasharray="6 6" />
      {frame.annotations.map((a) => {
        if (!a.bbox_2d) return null;
        const [bx, by, bw, bh] = a.bbox_2d;
        const highlighted = a.annotation_id === highlightId;
        return (
          <g key={a.annotation_id}>
            <rect
              x={bx * sx}
              y={by * sy}
              width={bw * sx}
              height={bh * sy}
              fill={highlighted ? 'rgba(239,83,80,0.15)' : 'none'}
              stroke={highlighted ? '#ef5350' : '#4fc3f7'}
              strokeWidth={highlighted ? 2 : 1}
            />
            <text x={bx * sx} y={by * sy - 3} fill={highlighted ? '#ef5350' : '#4fc3f7'} fontSize={9}>
              {a.class_name} {(a.confidence * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LidarView({ frame, highlightId }: { frame: FrameSummary; highlightId: string }) {
  const t = makeBevTransform(frame, VIEW_W, VIEW_H);
  return (
    <svg width="100%" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} style={{ borderRadius: 6, border: '1px solid #232a31', background: '#0c1014' }}>
      {frame.lidar_points_bev.map((p, i) => {
        const [x, y] = t.toSvg([p[0], p[1]]);
        return <circle key={i} cx={x} cy={y} r={0.9} fill="#3f6d8a" fillOpacity={0.8} />;
      })}
      {frame.gt_boxes.map((g) => (
        <polygon
          key={g.gt_id}
          points={polygonPoints(boxCornersBEV(g.bbox_3d), t)}
          fill="none"
          stroke="#66bb6a"
          strokeWidth={1.2}
          strokeDasharray="4 3"
        />
      ))}
      {frame.annotations.map((a) => {
        if (!a.bbox_3d) return null;
        const highlighted = a.annotation_id === highlightId;
        return (
          <polygon
            key={a.annotation_id}
            points={polygonPoints(boxCornersBEV(a.bbox_3d), t)}
            fill={highlighted ? 'rgba(239,83,80,0.2)' : 'none'}
            stroke={highlighted ? '#ef5350' : '#90a4ae'}
            strokeWidth={highlighted ? 2 : 1}
          />
        );
      })}
    </svg>
  );
}

function BirdsEyeView({
  frame,
  prevFrame,
  nextFrame,
  highlightId,
}: {
  frame: FrameSummary;
  prevFrame: FrameSummary | null;
  nextFrame: FrameSummary | null;
  highlightId: string;
}) {
  const t = makeBevTransform(frame, VIEW_W, VIEW_H);
  const ego = t.toSvg([frame.ego_pose.x, frame.ego_pose.y]);

  // Track trails: connect centers of same track_id across prev/current/next.
  const trails: Array<{ trackId: string; pts: Pt[]; highlighted: boolean }> = [];
  for (const a of frame.annotations) {
    if (!a.track_id || !a.bbox_3d) continue;
    const pts: Pt[] = [];
    const prevMatch = prevFrame?.annotations.find((p) => p.track_id === a.track_id && p.bbox_3d);
    if (prevMatch?.bbox_3d) pts.push([prevMatch.bbox_3d[0], prevMatch.bbox_3d[1]]);
    pts.push([a.bbox_3d[0], a.bbox_3d[1]]);
    const nextMatch = nextFrame?.annotations.find((n) => n.track_id === a.track_id && n.bbox_3d);
    if (nextMatch?.bbox_3d) pts.push([nextMatch.bbox_3d[0], nextMatch.bbox_3d[1]]);
    if (pts.length > 1) {
      trails.push({ trackId: a.track_id, pts, highlighted: a.annotation_id === highlightId });
    }
  }

  return (
    <svg width="100%" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} style={{ borderRadius: 6, border: '1px solid #232a31', background: '#0c1014' }}>
      {trails.map((tr) => (
        <polyline
          key={tr.trackId}
          points={tr.pts.map((p) => t.toSvg(p).map((v) => v.toFixed(1)).join(',')).join(' ')}
          fill="none"
          stroke={tr.highlighted ? '#ef5350' : '#7e57c2'}
          strokeWidth={tr.highlighted ? 2 : 1.2}
          strokeOpacity={0.85}
        />
      ))}
      {frame.annotations.map((a) => {
        if (!a.bbox_3d) return null;
        const [x, y] = t.toSvg([a.bbox_3d[0], a.bbox_3d[1]]);
        const highlighted = a.annotation_id === highlightId;
        return (
          <g key={a.annotation_id}>
            <circle cx={x} cy={y} r={highlighted ? 5 : 3.5} fill={highlighted ? '#ef5350' : '#4fc3f7'} />
            <text x={x + 6} y={y + 3} fill="#8a949e" fontSize={8}>
              {a.class_name}
            </text>
          </g>
        );
      })}
      {/* ego marker */}
      <g transform={`translate(${ego[0]},${ego[1]}) rotate(${(-frame.ego_pose.yaw * 180) / Math.PI})`}>
        <polygon points="0,-8 6,8 -6,8" fill="#ffca28" stroke="#101418" strokeWidth={1} />
      </g>
      <text x={ego[0] + 9} y={ego[1] + 4} fill="#ffca28" fontSize={9}>
        EGO
      </text>
    </svg>
  );
}

function MiniBev({ frame, highlightTrack, label }: { frame: FrameSummary | null; highlightTrack: string | null; label: string }) {
  const W = 140;
  const H = 220;
  if (!frame) {
    return (
      <Box sx={{ flex: 1, textAlign: 'center' }}>
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ borderRadius: 6, border: '1px solid #232a31', background: '#0c1014' }}>
          <text x={W / 2} y={H / 2} textAnchor="middle" fill="#5c6773" fontSize={10}>
            n/a
          </text>
        </svg>
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          {label}
        </Typography>
      </Box>
    );
  }
  const t = makeBevTransform(frame, W, H, 8);
  return (
    <Box sx={{ flex: 1, textAlign: 'center' }}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ borderRadius: 6, border: '1px solid #232a31', background: '#0c1014' }}>
        {frame.lidar_points_bev.map((p, i) => {
          const [x, y] = t.toSvg([p[0], p[1]]);
          return <circle key={i} cx={x} cy={y} r={0.6} fill="#3f6d8a" fillOpacity={0.7} />;
        })}
        {frame.annotations.map((a) => {
          if (!a.bbox_3d) return null;
          const highlighted = highlightTrack !== null && a.track_id === highlightTrack;
          return (
            <polygon
              key={a.annotation_id}
              points={polygonPoints(boxCornersBEV(a.bbox_3d), t)}
              fill="none"
              stroke={highlighted ? '#ef5350' : '#90a4ae'}
              strokeWidth={highlighted ? 1.6 : 0.8}
            />
          );
        })}
      </svg>
      <Typography variant="caption" sx={{ color: '#8a949e' }}>
        {label}
      </Typography>
    </Box>
  );
}

// ---------------------------------------------------------------- main component

export default function HITLReview({
  tasks,
  selectedTask,
  onSelectTask,
  frame,
  prevFrame,
  nextFrame,
  frameLoading,
}: {
  tasks: ReviewTask[];
  selectedTask: ReviewTask | null;
  onSelectTask: (taskId: string) => void;
  frame: FrameSummary | null;
  prevFrame: FrameSummary | null;
  nextFrame: FrameSummary | null;
  frameLoading: boolean;
}) {
  const evidence = selectedTask?.evidence ?? null;
  const annotation: Annotation | null =
    (frame && selectedTask ? frame.annotations.find((a) => a.annotation_id === selectedTask.annotation_id) : null) ??
    null;
  const highlightId = selectedTask?.annotation_id ?? '';

  return (
    <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', flexWrap: 'wrap' }}>
      {/* Left: task list */}
      <SectionCard title={`Review Tasks (${tasks.length})`} sx={{ flex: '0 1 300px', minWidth: 270 }}>
        {tasks.length === 0 ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No review tasks — flagged labels will appear here.
          </Typography>
        ) : (
          <List dense sx={{ maxHeight: 620, overflowY: 'auto', p: 0 }}>
            {tasks.map((t) => (
              <ListItemButton
                key={t.task_id}
                selected={t.task_id === selectedTask?.task_id}
                onClick={() => onSelectTask(t.task_id)}
                sx={{ display: 'block', borderBottom: '1px solid #1c232b', px: 1 }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {t.annotation_id}
                  </Typography>
                  <StatusChip status={t.status} />
                </Box>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                  {t.failure_reasons.map((r) => (
                    <Chip
                      key={r}
                      size="small"
                      label={r.replace(/_/g, ' ')}
                      sx={{
                        bgcolor: r === t.primary_failure_reason ? '#4a1f1f' : '#232a31',
                        color: r === t.primary_failure_reason ? '#ef9a9a' : '#aab4be',
                        fontSize: 10,
                        height: 18,
                      }}
                    />
                  ))}
                </Box>
              </ListItemButton>
            ))}
          </List>
        )}
      </SectionCard>

      {/* Right: sensor views + evidence */}
      <Box sx={{ flex: '1 1 560px', minWidth: 480, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {!selectedTask ? (
          <SectionCard title="Sensor Views">
            <Typography variant="body2" sx={{ color: '#8a949e' }}>
              Select a task to inspect its frame across camera, LiDAR, BEV and temporal views.
            </Typography>
          </SectionCard>
        ) : frameLoading || !frame ? (
          <SectionCard title={`Sensor Views — ${selectedTask.frame_id}`}>
            {frameLoading ? (
              <LoadingBox label="Loading frame…" />
            ) : (
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                Frame data unavailable (backend offline?).
              </Typography>
            )}
          </SectionCard>
        ) : (
          <SectionCard
            title={`Sensor Views — ${frame.frame_id} · seq ${frame.sequence_id} · ${frame.num_lidar_points.toLocaleString()} lidar pts`}
          >
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
              <ViewFrame title="CAMERA VIEW (synthetic image, 2D boxes)">
                <CameraView frame={frame} highlightId={highlightId} />
              </ViewFrame>
              <ViewFrame title="LIDAR VIEW (BEV points + 3D boxes · annotation red, GT green dashed)">
                <LidarView frame={frame} highlightId={highlightId} />
              </ViewFrame>
              <ViewFrame title="BIRD'S-EYE VIEW (positions + track trails, ego marker)">
                <BirdsEyeView frame={frame} prevFrame={prevFrame} nextFrame={nextFrame} highlightId={highlightId} />
              </ViewFrame>
              <ViewFrame title="TEMPORAL VIEW (prev / current / next)">
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <MiniBev frame={prevFrame} highlightTrack={annotation?.track_id ?? null} label="prev" />
                  <MiniBev frame={frame} highlightTrack={annotation?.track_id ?? null} label="current" />
                  <MiniBev frame={nextFrame} highlightTrack={annotation?.track_id ?? null} label="next" />
                </Box>
              </ViewFrame>
            </Box>
          </SectionCard>
        )}

        {selectedTask ? (
          <SectionCard title="Evidence">
            {evidence ? (
              <>
                <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
                  <MetricCard label="3D IoU" value={fmtNum(evidence.geometry.iou_3d)} />
                  <MetricCard label="Position err (m)" value={fmtNum(evidence.geometry.position_error)} />
                  <MetricCard label="Orientation err (°)" value={fmtNum(evidence.geometry.orientation_error_deg, 1)} />
                  <MetricCard
                    label="Anomaly score"
                    value={fmtNum(evidence.anomaly.score)}
                    accent={evidence.anomaly.is_anomaly ? '#ffa726' : undefined}
                  />
                  <MetricCard label="Consensus" value={fmtPct(evidence.grading.consensus)} />
                  <MetricCard label="Track quality" value={fmtNum(evidence.tracking.track_quality)} />
                </Box>
                <GateLineList
                  checks={evidence.decision ? evidence.decision.gate_lines : evidence.validation.checks}
                />
              </>
            ) : (
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                No evaluation evidence attached to this task.
              </Typography>
            )}
          </SectionCard>
        ) : null}
      </Box>
    </Box>
  );
}
