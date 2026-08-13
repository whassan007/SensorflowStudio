import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { MetricCard } from '../../components/labeleval/shared';
import { VitisSection } from '../../components/vitis/Section';
import { HBarChart, LineChart } from '../../components/vitis/charts';
import { runHil, runHilSweep } from '../../services/vitis';
import type { HilRun, HilSweep } from '../../types/vitis';

const VERDICT_STYLE: Record<string, { color: 'error' | 'success' | 'warning'; label: string }> = {
  REGRESSION: { color: 'error', label: 'QUANTIZATION-GAP REGRESSION DETECTED' },
  PASS: { color: 'success', label: 'NO QUANTIZATION-GAP REGRESSION' },
  INSUFFICIENT_EVIDENCE: { color: 'warning', label: 'INSUFFICIENT EVIDENCE — collect more frames' },
};

const ABLATION_LABELS: Record<string, string> = {
  precision_only: 'Precision (bit-width)',
  streaming_only: 'Streaming (XFCVDEPTH)',
  hls_approx_only: 'HLS LUT approx.',
};

export default function HilTab({ device }: { device: string }) {
  const [widthBits, setWidthBits] = useState(10);
  const [depth, setDepth] = useState(2048);
  const [seed, setSeed] = useState(7);
  const [nSequences, setNSequences] = useState(4);
  const [frames, setFrames] = useState(14);
  const [useLut, setUseLut] = useState(true);
  const [busy, setBusy] = useState<'run' | 'sweep' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<HilRun | null>(null);
  const [sweep, setSweep] = useState<HilSweep | null>(null);

  const doRun = async () => {
    setBusy('run');
    setError(null);
    try {
      setRun(
        await runHil({
          n_sequences: nSequences,
          frames_per_sequence: frames,
          seed,
          width_bits: widthBits,
          int_bits: 4,
          max_line_buffer_depth: depth,
          use_lut_approx: useLut,
          device,
          run_ablation: true,
        })
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const doSweep = async () => {
    setBusy('sweep');
    setError(null);
    try {
      setSweep(
        await runHilSweep({
          n_sequences: nSequences,
          frames_per_sequence: frames,
          seed,
          widths: [6, 8, 10, 12, 14, 16],
          int_bits: 4,
          max_line_buffer_depth: depth,
          use_lut_approx: useLut,
          device,
        })
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const verdict = run ? VERDICT_STYLE[run.verdict.decision] : null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <VitisSection title="Run configuration" subtitle="Same frames, two backends: float32 reference vs constraint-faithful Vitis emulation.">
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField select size="small" label="Bit width W (ap_fixed<W,4>)" value={widthBits}
            onChange={(e) => setWidthBits(Number(e.target.value))} sx={{ width: 190 }}>
            {[6, 8, 10, 12, 14, 16].map((w) => (
              <MenuItem key={w} value={w}>{w} bits</MenuItem>
            ))}
          </TextField>
          <TextField select size="small" label="Line buffer depth (XFCVDEPTH)" value={depth}
            onChange={(e) => setDepth(Number(e.target.value))} sx={{ width: 210 }}>
            {[64, 128, 256, 2048].map((d) => (
              <MenuItem key={d} value={d}>{d} px{d < 256 ? ' (exceeded → strips)' : ''}</MenuItem>
            ))}
          </TextField>
          <TextField size="small" type="number" label="Sequences" value={nSequences}
            onChange={(e) => setNSequences(Math.max(1, Math.min(12, Number(e.target.value))))} sx={{ width: 100 }} />
          <TextField size="small" type="number" label="Frames/seq" value={frames}
            onChange={(e) => setFrames(Math.max(4, Math.min(40, Number(e.target.value))))} sx={{ width: 100 }} />
          <TextField size="small" type="number" label="Seed" value={seed}
            onChange={(e) => setSeed(Number(e.target.value))} sx={{ width: 90 }} />
          <FormControlLabel
            control={<Switch size="small" checked={useLut} onChange={(e) => setUseLut(e.target.checked)} />}
            label={<Typography variant="caption">LUT div/sqrt</Typography>}
          />
          <Button variant="contained" size="small" onClick={doRun} disabled={busy !== null}>
            {busy === 'run' ? <CircularProgress size={16} /> : 'Run paired comparison'}
          </Button>
          <Button variant="outlined" size="small" onClick={doSweep} disabled={busy !== null}>
            {busy === 'sweep' ? <CircularProgress size={16} /> : 'Bit-width sweep'}
          </Button>
        </Box>
      </VitisSection>

      {error ? <Alert severity="error">{error}</Alert> : null}

      {run && verdict ? (
        <>
          <Alert severity={verdict.color} sx={{ fontWeight: 700 }}>
            {verdict.label}
            <Typography variant="caption" sx={{ display: 'block', fontWeight: 400 }}>
              method: {run.verdict.method} · mean paired Δ correctness ={' '}
              {run.verdict.mean_delta.toFixed(4)} · CI [{run.verdict.ci[0].toFixed(3)},{' '}
              {run.verdict.ci[1].toFixed(3)}] · {run.verdict.n} frame clusters
            </Typography>
          </Alert>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <MetricCard label="Gap score" value={run.comparison.gap_score.toFixed(4)}
              info="Scalar severity of the reference-vs-emulated gap: 0.45·drop rate + 0.2·class-flip rate + 0.2·|confidence drift| + 0.15·IoU loss." />
            <MetricCard label="Dropped by Vitis path" value={run.comparison.totals.dropped_by_vitis}
              sub={`of ${run.comparison.totals.gt_objects} GT objects`} accent="#ef9a9a"
              info="Objects the float32 path detected but the fixed-point path lost." />
            <MetricCard label="Class flips" value={run.comparison.totals.class_flips}
              info="Paired detections whose predicted class differs between backends." />
            <MetricCard label="Mean |confidence drift|" value={run.comparison.drift.mean_abs_confidence_drift.toFixed(4)}
              info="Mean absolute per-object confidence difference between backends." />
            <MetricCard label="Position drift" value={`${run.comparison.drift.mean_position_drift_px.toFixed(3)} px`}
              info="Mean centroid distance between the paired detections of the two backends." />
            <MetricCard label="Pair IoU" value={run.comparison.drift.mean_pair_iou.toFixed(3)}
              info="Mean IoU between the reference and emulated boxes for the same object." />
          </Box>
          {run.ablation ? (
            <VitisSection
              title="Ablation attribution — why does the gap exist?"
              subtitle={run.ablation.note}
            >
              <HBarChart
                items={Object.entries(run.ablation.attribution).map(([k, v]) => ({
                  label: ABLATION_LABELS[k] ?? k,
                  value: v,
                  color: k === 'precision_only' ? '#4fc3f7' : k === 'streaming_only' ? '#ffb74d' : '#ba68c8',
                }))}
                format={(v) => `${(v * 100).toFixed(1)}%`}
              />
            </VitisSection>
          ) : null}
          <VitisSection title="Per-cohort paired delta" subtitle="Mean per-frame correctness delta (emulated − reference) by scene condition.">
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {Object.entries(run.comparison.cohort_delta).map(([cohort, d]) => (
                <Chip key={cohort} size="small"
                  label={`${cohort}: ${d >= 0 ? '+' : ''}${d.toFixed(4)}`}
                  sx={{ bgcolor: d < -0.02 ? '#4a1f1f' : '#1f2a1f', fontFamily: 'monospace' }} />
              ))}
            </Box>
          </VitisSection>
        </>
      ) : null}

      {sweep ? (
        <VitisSection
          title="Bit-width sweep"
          subtitle="Gap score vs total bits W (I=4). Marker color = sequential verdict at that width."
        >
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <LineChart
              points={sweep.points.map((p) => ({ x: p.width_bits, y: p.gap_score }))}
              markers={sweep.points.map((p) =>
                p.decision === 'REGRESSION' ? '#ef5350' : p.decision === 'PASS' ? '#66bb6a' : '#ffb74d'
              )}
              xLabel="total bits W" yLabel="gap score"
            />
            <Box sx={{ flex: 1, minWidth: 260 }}>
              {sweep.minimal_passing_config ? (
                <Alert severity={sweep.minimal_passing_config.decision === 'PASS' ? 'success' : 'info'}>
                  <strong>
                    Minimal viable config: ap_fixed&lt;{sweep.minimal_passing_config.width_bits},
                    {sweep.minimal_passing_config.int_bits}&gt;
                  </strong>
                  <Typography variant="caption" sx={{ display: 'block' }}>
                    verdict at this width: {sweep.minimal_passing_config.decision}
                    {sweep.minimal_passing_config.note ? ` — ${sweep.minimal_passing_config.note}` : ''}
                  </Typography>
                </Alert>
              ) : (
                <Alert severity="error">Every swept width regressed.</Alert>
              )}
              <Box component="table" sx={{ mt: 1.5, borderCollapse: 'collapse', width: '100%', '& td, & th': { border: '1px solid #232a31', px: 1, py: 0.4, fontSize: 12 } }}>
                <thead>
                  <tr><th>W</th><th>verdict</th><th>gap</th><th>dropped</th><th>flips</th><th>pair IoU</th></tr>
                </thead>
                <tbody>
                  {sweep.points.map((p) => (
                    <tr key={p.width_bits}>
                      <td style={{ fontFamily: 'monospace' }}>{p.width_bits}</td>
                      <td style={{ color: p.decision === 'REGRESSION' ? '#ef5350' : p.decision === 'PASS' ? '#66bb6a' : '#ffb74d' }}>{p.decision}</td>
                      <td style={{ fontFamily: 'monospace' }}>{p.gap_score.toFixed(4)}</td>
                      <td style={{ fontFamily: 'monospace' }}>{p.dropped}</td>
                      <td style={{ fontFamily: 'monospace' }}>{p.class_flips}</td>
                      <td style={{ fontFamily: 'monospace' }}>{p.mean_pair_iou.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </Box>
            </Box>
          </Box>
        </VitisSection>
      ) : null}

      {!run && !sweep ? (
        <Alert severity="info">
          Run a paired comparison to see per-object drift, ablation attribution and the sequential verdict, or a
          bit-width sweep to find the minimal ap_fixed width that still passes.
        </Alert>
      ) : null}
    </Box>
  );
}
