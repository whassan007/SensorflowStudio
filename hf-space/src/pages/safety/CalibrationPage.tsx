/**
 * Calibration Panel — multi-sensor calibration validation (Deepen-style):
 * three validation modes, a brush-selectable residual scatter, per-check
 * table and a status badge.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Slider from '@mui/material/Slider';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { CheckCircle2, PlayCircle, XCircle } from 'lucide-react';
import {
  getCalibrationStatus,
  validateCalibration,
  type CalibrationMode,
  type CalibrationResult,
} from '../../services/safety';
import BrushChart, { type BrushPoint } from '../../components/visual/BrushChart';
import { PanelSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, MetricCard, SectionCard } from '../../components/labeleval/shared';
import { InfoDot } from '../../components/help/InfoTip';
import { tokens, verdictColor } from '../../theme';

const MODES: Array<{ id: CalibrationMode; label: string; desc: string }> = [
  {
    id: 'clean',
    label: 'Clean sensors',
    desc: 'Well-calibrated rig: residuals should be small, unbiased and isotropic — the healthy baseline.',
  },
  {
    id: 'miscalibrated',
    label: 'Miscalibration',
    desc: 'Injects a rotation/translation extrinsic offset: residuals become systematically biased in one direction — the classic drifted-mount signature.',
  },
  {
    id: 'perception_failure',
    label: 'Perception failure',
    desc: 'Tampers a fraction of objects: most residuals stay clean but a subset are wild outliers — distinguishable from miscalibration by the outlier fraction, not the bias.',
  },
];

export default function CalibrationPage() {
  const [mode, setMode] = useState<CalibrationMode>('clean');
  const [rotation, setRotation] = useState(2.0);
  const [tamper, setTamper] = useState(0.25);
  const [result, setResult] = useState<CalibrationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[] | null>(null);

  useEffect(() => {
    getCalibrationStatus()
      .then((s) => {
        if (s.status !== 'NEVER_RUN') setResult(s);
      })
      .catch(() => undefined);
  }, []);

  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    setSelectedIds(null);
    validateCalibration({ mode, rotation_offset_deg: rotation, tamper_fraction: tamper })
      .then(setResult)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [mode, rotation, tamper]);

  const points: BrushPoint[] = useMemo(
    () =>
      (result?.per_object ?? []).map((o) => ({
        id: o.object_id,
        x: o.residual_px[0],
        y: o.residual_px[1],
        color: o.flagged ? tokens.color.danger : tokens.color.success,
        label: `${o.class_name} ${o.object_id}`,
        r: o.flagged ? 5 : 4,
      })),
    [result]
  );

  const visibleObjects = useMemo(() => {
    const all = result?.per_object ?? [];
    if (!selectedIds) return all;
    const set = new Set(selectedIds);
    return all.filter((o) => set.has(o.object_id));
  }, [result, selectedIds]);

  const statusColor = verdictColor(result?.status);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}

      <SectionCard
        title="Validation mode"
        help="The validator projects LiDAR box centers into the camera and measures pixel residuals against camera detections. The three modes simulate the three states a real rig can be in — the point is that the checks can TELL THEM APART from residual statistics alone (bias vs scatter vs outlier fraction)."
      >
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          {MODES.map((m) => (
            <Box
              key={m.id}
              onClick={() => setMode(m.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter') setMode(m.id); }}
              sx={{
                flex: '1 1 240px',
                p: 1.5,
                borderRadius: 1,
                border: `2px solid ${mode === m.id ? tokens.color.info : tokens.color.border}`,
                bgcolor: mode === m.id ? tokens.color.infoBg : tokens.color.surfaceSunken,
                cursor: 'pointer',
                transition: `border-color ${tokens.motion.fast}, background-color ${tokens.motion.fast}`,
                '&:focus-visible': { outline: `2px solid ${tokens.color.info}` },
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{m.label}</Typography>
              <Typography variant="caption" sx={{ color: tokens.color.neutral }}>{m.desc}</Typography>
            </Box>
          ))}
        </Box>
        <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', mt: 1.5, flexWrap: 'wrap' }}>
          {mode === 'miscalibrated' ? (
            <Box sx={{ width: 240 }}>
              <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                Injected rotation offset: {rotation.toFixed(1)}°
              </Typography>
              <Slider size="small" value={rotation} min={0.5} max={10} step={0.5} onChange={(_, v) => setRotation(v as number)} sx={{ py: 0.5 }} />
            </Box>
          ) : null}
          {mode === 'perception_failure' ? (
            <Box sx={{ width: 240 }}>
              <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                Tampered fraction: {(tamper * 100).toFixed(0)}%
              </Typography>
              <Slider size="small" value={tamper} min={0.05} max={0.8} step={0.05} onChange={(_, v) => setTamper(v as number)} sx={{ py: 0.5 }} />
            </Box>
          ) : null}
          <Button variant="contained" startIcon={<PlayCircle size={16} />} disabled={loading} onClick={run}>
            {loading ? 'Validating…' : 'Run validation'}
          </Button>
        </Box>
      </SectionCard>

      {loading && !result ? <PanelSkeleton rows={5} /> : null}

      {result && result.status !== 'NEVER_RUN' ? (
        <>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'stretch' }}>
            <Box
              sx={{
                flex: '0 0 auto',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                px: 3,
                borderRadius: 1,
                border: `2px solid ${statusColor}`,
                bgcolor: `${statusColor}18`,
              }}
            >
              <Typography variant="h5" sx={{ fontWeight: 900, color: statusColor, letterSpacing: 0.5 }}>
                {result.status}
              </Typography>
              <Typography variant="caption" sx={{ color: tokens.color.textDim, maxWidth: 220, textAlign: 'center' }}>
                {result.diagnosis}
              </Typography>
              {result.simulated ? (
                <Chip size="small" label="simulated scene" sx={{ mt: 0.5, height: 18, fontSize: 10, bgcolor: tokens.color.warnBg, color: tokens.color.warn }} />
              ) : null}
            </Box>
            {result.metrics ? (
              <>
                <MetricCard label="Bias" value={`${result.metrics.bias_px.toFixed(2)} px`} sub={`vector (${result.metrics.bias_vector_px[0]}, ${result.metrics.bias_vector_px[1]})`} info="Mean residual vector magnitude — systematic bias is the miscalibration signature." accent={result.metrics.bias_px > (result.thresholds?.bias_px ?? 6) ? tokens.color.danger : tokens.color.success} />
                <MetricCard label="Scatter" value={`${result.metrics.scatter_px.toFixed(2)} px`} info="Residual standard deviation around the bias — noise level of the projection." />
                <MetricCard label="Outlier fraction" value={`${(result.metrics.outlier_fraction * 100).toFixed(0)}%`} info="Share of objects with residuals past the outlier threshold — high fraction with low bias indicates perception failure, not miscalibration." accent={result.metrics.outlier_fraction > 0.1 ? tokens.color.danger : tokens.color.success} />
                <MetricCard label="Est. rotation offset" value={`${result.metrics.estimated_rotation_offset_deg.toFixed(2)}°`} sub={result.injected ? `injected: ${result.injected.rotation_offset_deg}°` : undefined} info="Rotation offset back-estimated from the residual field — compare against the injected value to see the estimator working." />
              </>
            ) : null}
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <SectionCard
              title="Residual scatter"
              sx={{ flex: '2 1 460px' }}
              help="Each point is one object's camera-projection residual (pixels). A healthy rig clusters at the origin; miscalibration shifts the whole cloud off-center; perception failure keeps the cluster centered but sprays outliers. Drag a box to select points — the table filters to your selection."
            >
              <BrushChart
                points={points}
                height={280}
                xLabel="residual x (px)"
                yLabel="residual y (px)"
                onBrush={setSelectedIds}
                refLinesX={[{ x: 0, label: '', color: tokens.color.borderStrong }]}
                refLinesY={[{ y: 0, label: '', color: tokens.color.borderStrong }]}
              />
              <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                {selectedIds ? `${selectedIds.length} objects selected — click empty space to clear.` : 'Drag to brush-select objects; green = inlier, red = flagged.'}
              </Typography>
            </SectionCard>

            <SectionCard title="Checks" sx={{ flex: '1 1 340px' }} help="Deterministic checks with their thresholds. The combination of which checks fail is what separates CALIBRATED / MISCALIBRATED / PERCEPTION_FAILURE.">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Check</TableCell>
                    <TableCell align="right">Actual</TableCell>
                    <TableCell align="right">Threshold</TableCell>
                    <TableCell align="center">OK</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(result.checks ?? []).map((c) => (
                    <TableRow key={c.name}>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>{c.name}</TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{c.actual.toFixed(3)}</TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12, color: tokens.color.neutral }}>{c.threshold}</TableCell>
                      <TableCell align="center">
                        {c.passed ? <CheckCircle2 size={15} color={tokens.color.success} /> : <XCircle size={15} color={tokens.color.danger} />}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </SectionCard>
          </Box>

          <SectionCard
            title={
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
                Per-object residuals {selectedIds ? `(${visibleObjects.length} selected)` : `(${visibleObjects.length})`}
                <InfoDot title="Per-object residuals" detail="One row per validation object. Flagged objects exceed the outlier threshold; their failure reason distinguishes projection outliers from inlier-ratio failures." />
              </Box>
            }
          >
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Object</TableCell>
                  <TableCell>Class</TableCell>
                  <TableCell align="right">Residual (px)</TableCell>
                  <TableCell align="right">|r|</TableCell>
                  <TableCell align="right">Inlier ratio</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {visibleObjects.map((o) => (
                  <TableRow key={o.object_id} sx={{ bgcolor: o.flagged ? tokens.color.dangerBg : undefined }}>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>{o.object_id}</TableCell>
                    <TableCell>{o.class_name}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                      ({o.residual_px[0].toFixed(1)}, {o.residual_px[1].toFixed(1)})
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{o.residual_magnitude_px.toFixed(2)}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{o.inlier_ratio.toFixed(2)}</TableCell>
                    <TableCell>
                      {o.flagged ? (
                        <Chip size="small" label={o.failure_reason ?? 'flagged'} sx={{ height: 18, fontSize: 10, bgcolor: tokens.color.dangerBg, color: tokens.color.danger }} />
                      ) : (
                        <Chip size="small" label="inlier" sx={{ height: 18, fontSize: 10, bgcolor: tokens.color.successBg, color: tokens.color.success }} />
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </SectionCard>
        </>
      ) : null}
    </Box>
  );
}
