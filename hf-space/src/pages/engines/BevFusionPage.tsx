/**
 * Perception Engines (BEV fusion) — engine comparison card (headline delta
 * chips, PROMOTE badge, blockers), per-cohort paired delta bars with
 * explanation tooltips, run form, and the interactive top-down BEV canvas of
 * one frame with a frame scrubber and track following.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import MenuItem from '@mui/material/MenuItem';
import Slider from '@mui/material/Slider';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { Pause, Play, PlayCircle } from 'lucide-react';
import {
  getBevReplay,
  getBevReport,
  runBevComparison,
  type BevReplayResponse,
  type BevReport,
} from '../../services/bevfusion';
import BevFrameCanvas, { CanvasLegend, type HoverInfo, type LayerId } from '../../components/engines/BevFrameCanvas';
import { DeltaBars, type DeltaBarRow } from '../../components/visual/charts';
import { ChartSkeleton, IllustratedEmpty, PanelSkeleton, TileSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, SectionCard, fmtNum, fmtPct } from '../../components/labeleval/shared';
import { InfoDot } from '../../components/help/InfoTip';
import { deltaColor, tokens, verdictColor } from '../../theme';

const LOWER_IS_BETTER = new Set(['position_error_m', 'id_switch_rate', 'fragmentation_rate']);

const METRIC_HELP: Record<string, string> = {
  precision: 'Fraction of produced labels that match a GT object.',
  recall: 'Fraction of GT objects that got a label — the headline miss rate.',
  f1: 'Harmonic mean of precision and recall.',
  mean_iou: 'Mean BEV overlap of matched boxes — geometric quality.',
  position_error_m: 'Mean center distance of matched boxes (meters, lower is better).',
  class_accuracy: 'Fraction of matched boxes with the correct class.',
  safety_recall: 'Recall over pedestrians/cyclists/motorcycles only.',
  id_switch_rate: 'Track identity switches per GT track (lower is better).',
  fragmentation_rate: 'How often GT tracks break into pieces (lower is better).',
  idf1: 'Identity-F1: how consistently the same object keeps the same track id.',
};

function DeltaChip({ metric, baseline, candidate, delta, improved }: { metric: string; baseline: number; candidate: number; delta: number; improved: boolean }) {
  const color = improved ? tokens.color.success : tokens.color.danger;
  const pct = !LOWER_IS_BETTER.has(metric) && Math.abs(baseline) <= 1;
  const fmt = (v: number) => (pct ? fmtPct(v) : fmtNum(v));
  return (
    <Box
      title={METRIC_HELP[metric] ?? metric}
      sx={{
        px: 1.25,
        py: 0.75,
        borderRadius: 1,
        border: `1px solid ${color}55`,
        bgcolor: `${color}12`,
        minWidth: 118,
        cursor: 'help',
        transition: `transform ${tokens.motion.fast}`,
        '&:hover': { transform: 'translateY(-2px)' },
      }}
    >
      <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', fontSize: 10.5 }}>
        {metric.replace(/_/g, ' ')}
      </Typography>
      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 800, color }}>
        {delta >= 0 ? '+' : ''}
        {fmt(delta)}
      </Typography>
      <Typography variant="caption" sx={{ color: tokens.color.textFaint, fontFamily: 'monospace', fontSize: 10 }}>
        {fmt(baseline)} → {fmt(candidate)}
      </Typography>
    </Box>
  );
}

export default function BevFusionPage() {
  const [report, setReport] = useState<BevReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // run form
  const [nSeq, setNSeq] = useState(6);
  const [framesPerSeq, setFramesPerSeq] = useState(24);
  const [seed, setSeed] = useState(7);

  // canvas state
  const [replay, setReplay] = useState<BevReplayResponse | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [sequence, setSequence] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [layers, setLayers] = useState<Record<LayerId, boolean>>({ gt: true, camera: true, lidar: true, fused: true, masklet: true, baseline: false });
  const [selectedTrack, setSelectedTrack] = useState<number | string | null>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getBevReport()
      .then(setReport)
      .catch(() => setReport(null))
      .finally(() => setLoading(false));
  }, []);

  // replay follows the report params + chosen sequence
  useEffect(() => {
    if (!report) return;
    let cancelled = false;
    setReplay(null);
    setReplayError(null);
    setSelectedTrack(null);
    setFrameIdx(0);
    getBevReplay(report.params, sequence)
      .then((r) => {
        if (!cancelled) setReplay(r);
      })
      .catch((e) => {
        if (!cancelled) setReplayError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [report, sequence]);

  // playback
  useEffect(() => {
    if (playRef.current) clearInterval(playRef.current);
    if (playing && replay) {
      playRef.current = setInterval(() => {
        setFrameIdx((i) => (i + 1) % replay.frames.length);
      }, 220);
    }
    return () => {
      if (playRef.current) clearInterval(playRef.current);
    };
  }, [playing, replay]);

  const run = useCallback(() => {
    setRunning(true);
    setError(null);
    runBevComparison({ n_sequences: nSeq, frames_per_sequence: framesPerSeq, seed })
      .then((r) => {
        setReport(r);
        setSequence(0);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setRunning(false));
  }, [nSeq, framesPerSeq, seed]);

  const cohortRows: DeltaBarRow[] = useMemo(
    () =>
      (report?.per_cohort ?? []).map((c) => ({
        label: c.cohort,
        baseline: c.recall_baseline,
        candidate: c.recall_candidate,
        delta: c.recall_delta,
        n: c.n_gt,
        improved: c.recall_delta >= 0,
        tooltip: (
          <Box>
            <Typography variant="caption" sx={{ fontWeight: 700, color: tokens.color.info, display: 'block' }}>
              Why the {c.cohort} cohort moved
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', color: '#e6e9ec' }}>{c.explanation}</Typography>
            <Typography variant="caption" sx={{ display: 'block', color: tokens.color.textDim, mt: 0.5 }}>
              recall {fmtPct(c.recall_baseline)} → {fmtPct(c.recall_candidate)} · position error{' '}
              {c.position_error_baseline_m.toFixed(2)} → {c.position_error_candidate_m.toFixed(2)} m · IoU{' '}
              {c.mean_iou_baseline.toFixed(2)} → {c.mean_iou_candidate.toFixed(2)}
            </Typography>
          </Box>
        ),
      })),
    [report]
  );

  const classRows: DeltaBarRow[] = useMemo(
    () =>
      (report?.per_class ?? []).map((c) => ({
        label: c.class,
        baseline: c.recall_baseline,
        candidate: c.recall_candidate,
        delta: c.recall_delta,
        n: c.n_gt,
        improved: c.recall_delta >= 0,
      })),
    [report]
  );

  const frame = replay?.frames[Math.min(frameIdx, (replay?.frames.length ?? 1) - 1)] ?? null;

  const trail = useMemo(() => {
    if (!replay || selectedTrack === null) return [];
    const pts: Array<{ x: number; y: number }> = [];
    for (let i = 0; i <= frameIdx && i < replay.frames.length; i += 1) {
      const b = replay.frames[i].fused.find((f) => f.track_id === selectedTrack);
      if (b) pts.push({ x: b.bbox_3d[0], y: b.bbox_3d[1] });
    }
    return pts;
  }, [replay, selectedTrack, frameIdx]);

  const recoColor = verdictColor(report?.recommendation);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}
      {loading ? <TileSkeleton n={5} /> : null}

      {!loading && !report ? (
        <SectionCard title="Engine comparison">
          <IllustratedEmpty
            art="canvas"
            title="No BEV-fusion comparison yet"
            message="Run one: deterministic multi-sensor scenes are generated, both engines label every frame (camera-only baseline vs camera+LiDAR BEV fusion with masklet tracking), and the evaluation compares them per cohort."
            action={
              <Button variant="contained" startIcon={<PlayCircle size={16} />} disabled={running} onClick={run}>
                {running ? 'Running…' : 'Run comparison'}
              </Button>
            }
          />
        </SectionCard>
      ) : null}

      {report ? (
        <>
          <SectionCard
            title={
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
                {report.engines.baseline} <span style={{ color: tokens.color.neutral }}>vs</span> {report.engines.candidate}
                <Chip size="small" label={report.recommendation.replace(/_/g, ' ')} sx={{ fontWeight: 900, bgcolor: `${recoColor}22`, color: recoColor, border: `2px solid ${recoColor}` }} />
              </Box>
            }
            help="Headline metric deltas, candidate minus baseline (green = improved, red = regressed — direction-aware for error metrics). The promotion recommendation applies the policy: bounded drops on recall/precision/safety-recall and per-cohort recall. Hover a chip for the metric definition."
            action={
              <Typography variant="caption" sx={{ color: tokens.color.neutral, fontFamily: 'monospace' }}>
                {report.run_id} · {report.scale.frames} frames · {report.scale.gt_boxes.toLocaleString()} GT boxes
              </Typography>
            }
          >
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {report.headline_deltas.map((d) => (
                <DeltaChip key={d.metric} {...d} />
              ))}
            </Box>
            {report.blockers.length ? (
              <Alert severity="error" variant="outlined" sx={{ mt: 1.5 }}>
                <strong>Blockers:</strong> {report.blockers.join(' · ')}
              </Alert>
            ) : (
              <Typography variant="caption" sx={{ color: tokens.color.success, display: 'block', mt: 1.5 }}>
                No policy blockers — every gated metric is within tolerance.
              </Typography>
            )}
            <Typography variant="caption" sx={{ color: tokens.color.textFaint, display: 'block', mt: 0.5 }}>
              {report.notes}
            </Typography>
          </SectionCard>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <SectionCard
              title="Per-cohort recall — where the fusion wins and why"
              sx={{ flex: '3 1 460px' }}
              help="Paired bars: gray = camera baseline, colored = fused candidate, sorted by delta. Hover a row for the causal explanation — each cohort's gain traces to a specific sensor-physics mechanism (occlusion, darkness, range, rain), not a tuning constant."
            >
              <DeltaBars rows={cohortRows} max={1} fmt={(v) => `${(v * 100).toFixed(1)}pp`} />
            </SectionCard>
            <SectionCard title="Per-class recall" sx={{ flex: '2 1 340px' }} help="Same paired-delta view per object class. Small vulnerable classes (pedestrian, cyclist, motorcycle) matter most — they drive safety recall.">
              <DeltaBars rows={classRows} max={1} fmt={(v) => `${(v * 100).toFixed(1)}pp`} />
            </SectionCard>
          </Box>

          <SectionCard
            title={
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
                Top-down BEV canvas — one frame, every hypothesis
                <InfoDot title="BEV canvas" detail="A deterministic replay of the selected scene sequence through both engines (same seeds as the scored run). White outlines = ground truth (dashed amber = camera-occluded this frame). Orange dots + ellipses = camera detections with their along-ray depth uncertainty. Teal crosses = LiDAR detections. Filled boxes = fused+tracked output (dashed purple = masklet-propagated through a detection gap). Red outlines = the camera-only baseline for contrast." />
              </Box>
            }
            action={
              replay ? (
                <Chip size="small" label={`${replay.time_of_day} · ${replay.weather}`} sx={{ bgcolor: replay.time_of_day === 'night' ? '#1a237e' : tokens.color.surfaceRaised, fontWeight: 700 }} />
              ) : undefined
            }
          >
            {replayError ? (
              <Alert severity="warning" variant="outlined">
                Frame replay unavailable: {replayError} — the /api/studio-ux router may not be loaded on the backend.
              </Alert>
            ) : null}
            {!replay && !replayError ? <ChartSkeleton height={380} /> : null}
            {replay && frame ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
                  <TextField select size="small" label="Sequence" value={sequence} onChange={(e) => setSequence(Number(e.target.value))} sx={{ width: 200 }}>
                    {Array.from({ length: report.params.n_sequences }, (_, i) => {
                      const tod = i % 3 === 1 ? 'night' : 'day';
                      const wthr = i % 3 === 2 ? 'rain' : 'clear';
                      return (
                        <MenuItem key={i} value={i}>
                          seq {i} · {tod}/{wthr}
                        </MenuItem>
                      );
                    })}
                  </TextField>
                  <IconButton size="small" onClick={() => setPlaying((p) => !p)} sx={{ border: `1px solid ${tokens.color.border}` }} aria-label={playing ? 'Pause playback' : 'Play frames'}>
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                  </IconButton>
                  <Box sx={{ flex: 1, minWidth: 220, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <Slider
                      size="small"
                      value={frameIdx}
                      min={0}
                      max={replay.frames.length - 1}
                      onChange={(_, v) => {
                        setPlaying(false);
                        setFrameIdx(v as number);
                      }}
                      aria-label="Frame scrubber"
                    />
                    <Typography variant="caption" sx={{ fontFamily: 'monospace', width: 72, flexShrink: 0 }}>
                      frame {frame.index + 1}/{replay.frames.length}
                    </Typography>
                  </Box>
                  {selectedTrack !== null ? (
                    <Chip size="small" label={`following track #${selectedTrack}`} onDelete={() => setSelectedTrack(null)} sx={{ bgcolor: tokens.color.infoBg, color: tokens.color.info }} />
                  ) : null}
                </Box>
                <CanvasLegend layers={layers} onToggle={(l) => setLayers((s) => ({ ...s, [l]: !s[l] }))} />
                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'stretch', flexWrap: 'wrap' }}>
                  <Box sx={{ flex: '3 1 520px', minWidth: 0 }}>
                    <BevFrameCanvas
                      frame={frame}
                      layers={layers}
                      selectedTrackId={selectedTrack}
                      onSelectTrack={setSelectedTrack}
                      trail={trail}
                      onHover={setHover}
                    />
                  </Box>
                  <Box sx={{ flex: '1 1 220px', border: `1px solid ${tokens.color.border}`, borderRadius: 1, p: 1.5, bgcolor: tokens.color.surfaceSunken, minHeight: 120 }}>
                    {hover ? (
                      <>
                        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>{hover.title}</Typography>
                        {hover.lines.map((l, i) => (
                          <Typography key={i} variant="caption" sx={{ display: 'block', color: tokens.color.textDim, mb: 0.25 }}>
                            {l}
                          </Typography>
                        ))}
                      </>
                    ) : (
                      <>
                        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5, color: tokens.color.neutral }}>
                          Object details
                        </Typography>
                        <Typography variant="caption" sx={{ color: tokens.color.textFaint }}>
                          Hover any box, ellipse or cross on the canvas. Click a fused box to follow its track — scrub the
                          frames to watch masklet propagation carry identity through occlusion gaps (dashed purple boxes).
                        </Typography>
                      </>
                    )}
                    {frame ? (
                      <Box sx={{ mt: 1.5, pt: 1, borderTop: `1px solid ${tokens.color.border}` }}>
                        <Typography variant="caption" sx={{ display: 'block', color: tokens.color.neutral, fontFamily: 'monospace', fontSize: 10.5 }}>
                          GT {frame.gt.length} · cam {frame.camera.length} · lidar {frame.lidar.length} · fused {frame.fused.length}
                          {frame.fused.some((b) => b.propagated) ? ` (${frame.fused.filter((b) => b.propagated).length} propagated)` : ''}
                        </Typography>
                        {frame.gt.some((g) => g.occluded) ? (
                          <Typography variant="caption" sx={{ color: tokens.color.warn, fontSize: 10.5 }}>
                            ⚠ {frame.gt.filter((g) => g.occluded).length} object(s) camera-occluded this frame
                          </Typography>
                        ) : null}
                      </Box>
                    ) : null}
                  </Box>
                </Box>
              </Box>
            ) : null}
          </SectionCard>
        </>
      ) : null}

      {report || !loading ? (
        <SectionCard
          title="Run a new comparison"
          help="Generates fresh deterministic scenes (seed-controlled), runs both engines on every frame, evaluates and persists the report. Sequences cycle day/clear → night/clear → day/rain so all cohorts are represented."
        >
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <TextField size="small" label="Sequences" type="number" value={nSeq} inputProps={{ min: 1, max: 24 }} onChange={(e) => setNSeq(Math.max(1, Math.min(24, Number(e.target.value) || 6)))} sx={{ width: 110 }} />
            <TextField size="small" label="Frames / sequence" type="number" value={framesPerSeq} inputProps={{ min: 8, max: 60 }} onChange={(e) => setFramesPerSeq(Math.max(8, Math.min(60, Number(e.target.value) || 24)))} sx={{ width: 140 }} />
            <TextField size="small" label="Seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 7)} sx={{ width: 90 }} />
            <Button variant="contained" startIcon={<PlayCircle size={16} />} disabled={running} onClick={run}>
              {running ? 'Running engines…' : 'Run comparison'}
            </Button>
            {running ? <PanelSkeleton rows={1} header={false} /> : null}
          </Box>
        </SectionCard>
      ) : null}

      {report ? (
        <Typography variant="caption" sx={{ color: tokens.color.textFaint }}>
          Deltas colored by {`direction-aware improvement (`}
          <span style={{ color: deltaColor(1) }}>green</span>
          {` = better, `}
          <span style={{ color: deltaColor(-1) }}>red</span>
          {` = worse). Engine comparison recorded at ${new Date(report.created_at).toLocaleString()}.`}
        </Typography>
      ) : null}
    </Box>
  );
}
