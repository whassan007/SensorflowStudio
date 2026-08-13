import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { MetricCard } from '../../components/labeleval/shared';
import { VitisSection } from '../../components/vitis/Section';
import { runTemporal } from '../../services/vitis';
import type { EngineStability, TemporalRun } from '../../types/vitis';

function scoreColor(score: number): string {
  if (score >= 70) return '#66bb6a';
  if (score >= 45) return '#ffb74d';
  return '#ef5350';
}

function EngineRow({ name, m }: { name: string; m: EngineStability }) {
  return (
    <tr>
      <td style={{ fontFamily: 'monospace' }}>{name}</td>
      <td style={{ fontFamily: 'monospace', color: scoreColor(m.stability_score), fontWeight: 700 }}>
        {m.stability_score.toFixed(1)}
      </td>
      <td style={{ fontFamily: 'monospace' }}>{(m.flicker_rate * 100).toFixed(2)}%</td>
      <td style={{ fontFamily: 'monospace' }}>{m.mean_jitter.toFixed(3)}</td>
      <td style={{ fontFamily: 'monospace' }}>{m.fragmentation_per_track.toFixed(3)}</td>
      <td style={{ fontFamily: 'monospace' }}>
        {m.id_switches} ({m.id_switches_at_flow_break} at flow break)
      </td>
    </tr>
  );
}

export default function TemporalTab({ device }: { device: string }) {
  const [nSequences, setNSequences] = useState(3);
  const [frames, setFrames] = useState(18);
  const [seed, setSeed] = useState(7);
  const [widthBits, setWidthBits] = useState(12);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<TemporalRun | null>(null);

  const doRun = async () => {
    setBusy(true);
    setError(null);
    try {
      setRun(
        await runTemporal({
          n_sequences: nSequences,
          frames_per_sequence: frames,
          seed,
          width_bits: widthBits,
          device,
        })
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const ref = run?.results.reference;
  const emu = run?.results.vitis_emulated;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <VitisSection
        title="Temporal stability profile"
        subtitle="Dense optical flow gives a model-independent motion baseline; engines are penalized for flicker, jitter, fragmentation and ID switches the flow cannot excuse. Runs on both backends."
      >
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField size="small" type="number" label="Sequences" value={nSequences}
            onChange={(e) => setNSequences(Math.max(1, Math.min(8, Number(e.target.value))))} sx={{ width: 100 }} />
          <TextField size="small" type="number" label="Frames/seq" value={frames}
            onChange={(e) => setFrames(Math.max(8, Math.min(40, Number(e.target.value))))} sx={{ width: 100 }} />
          <TextField size="small" type="number" label="Seed" value={seed}
            onChange={(e) => setSeed(Number(e.target.value))} sx={{ width: 90 }} />
          <TextField size="small" type="number" label="Flow bit width" value={widthBits}
            onChange={(e) => setWidthBits(Math.max(6, Math.min(24, Number(e.target.value))))} sx={{ width: 120 }} />
          <Button variant="contained" size="small" onClick={doRun} disabled={busy}>
            {busy ? <CircularProgress size={16} /> : 'Profile engines (both backends)'}
          </Button>
        </Box>
      </VitisSection>

      {error ? <Alert severity="error">{error}</Alert> : null}

      {run && ref && emu ? (
        <>
          <Alert severity={run.backend_agreement.ranking_agrees ? 'success' : 'error'}>
            <strong>
              Backend agreement meta-check:{' '}
              {run.backend_agreement.ranking_agrees
                ? 'fixed-point flow preserves the engine ranking'
                : 'RANKING CHANGED under fixed-point flow — do not trust the accelerated metric at this precision'}
            </strong>
            <Typography variant="caption" sx={{ display: 'block' }}>
              max |stability-score Δ| between backends:{' '}
              {run.backend_agreement.max_abs_score_delta.toFixed(2)} ·{' '}
              {Object.entries(run.backend_agreement.stability_score_delta_by_engine)
                .map(([e, d]) => `${e}: ${d >= 0 ? '+' : ''}${d}`)
                .join(' · ')}
            </Typography>
          </Alert>

          <VitisSection title="Engine comparison (reference flow)" subtitle="Higher stability score is better. Hover the score formula in the page help.">
            <Box component="table" sx={{ borderCollapse: 'collapse', width: '100%', '& td, & th': { border: '1px solid #232a31', px: 1, py: 0.5, fontSize: 12.5, textAlign: 'left' } }}>
              <thead>
                <tr>
                  <th>engine</th><th>stability score</th><th>flicker rate</th>
                  <th>jitter (vs flow)</th><th>fragmentation/track</th><th>ID switches</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(ref.engines).map(([name, m]) => (
                  <EngineRow key={name} name={name} m={m} />
                ))}
              </tbody>
            </Box>
            <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
              Same metrics under vitis_emulated (fixed-point) flow:{' '}
              {Object.entries(emu.engines)
                .map(([name, m]) => `${name}: ${m.stability_score.toFixed(1)}`)
                .join(' · ')}
            </Typography>
          </VitisSection>

          <VitisSection title="Per-cohort breakdown" subtitle="Stability score by scene condition (reference flow). 'occluded' restricts to instances with planted occlusion windows.">
            <Box component="table" sx={{ borderCollapse: 'collapse', width: '100%', '& td, & th': { border: '1px solid #232a31', px: 1, py: 0.5, fontSize: 12.5, textAlign: 'left' } }}>
              <thead>
                <tr>
                  <th>engine</th>
                  {Object.keys(Object.values(ref.engines)[0]?.cohorts ?? {}).map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(ref.engines).map(([name, m]) => (
                  <tr key={name}>
                    <td style={{ fontFamily: 'monospace' }}>{name}</td>
                    {Object.entries(m.cohorts ?? {}).map(([c, cm]) => (
                      <td key={c} style={{ fontFamily: 'monospace', color: scoreColor(cm.stability_score) }}>
                        {cm.stability_score.toFixed(1)}
                        <span style={{ color: '#8a949e' }}> ({(cm.flicker_rate * 100).toFixed(1)}% flicker)</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Box>
          </VitisSection>

          {run.timeline_sample ? (
            <VitisSection
              title={`Flow vs detection timeline — ${run.timeline_sample.instance_id}`}
              subtitle={`Engine ${run.timeline_sample.engine}, sequence ${run.timeline_sample.sequence_id}. Red frames: detection dropped while the flow says the object stayed observable.`}
            >
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {run.timeline_sample.frames.map((f) => {
                  const dropWhileContinuous = !f.detected && f.flow_continuous;
                  const bg = dropWhileContinuous ? '#b71c1c' : f.detected ? '#1b5e20' : '#232a31';
                  return (
                    <Tooltip
                      key={f.frame_index}
                      title={`frame ${f.frame_index} · ${f.detected ? `detected (track ${f.track_id})` : 'NOT detected'} · flow ${f.flow_continuous ? 'continuous' : 'discontinuous'}${f.occluded ? ' · occluded (planted)' : ''}${f.flow_residual_m !== null ? ` · residual ${f.flow_residual_m} m` : ''}`}
                    >
                      <Box sx={{ width: 26, height: 34, bgcolor: bg, borderRadius: 0.5, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: f.occluded ? '1px dashed #ffb74d' : '1px solid #232a31', cursor: 'default' }}>
                        <Typography variant="caption" sx={{ fontSize: 9, lineHeight: 1 }}>{f.frame_index}</Typography>
                        <Typography variant="caption" sx={{ fontSize: 8, lineHeight: 1.4, color: '#8a949e' }}>
                          {f.flow_continuous ? '~' : '×'}
                        </Typography>
                      </Box>
                    </Tooltip>
                  );
                })}
              </Box>
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <Chip size="small" label="detected" sx={{ bgcolor: '#1b5e20', height: 18, fontSize: 10 }} />
                <Chip size="small" label="dropped while flow-continuous" sx={{ bgcolor: '#b71c1c', height: 18, fontSize: 10 }} />
                <Chip size="small" label="not detected (excused)" sx={{ bgcolor: '#232a31', height: 18, fontSize: 10 }} />
                <Chip size="small" label="dashed = planted occlusion" variant="outlined" sx={{ height: 18, fontSize: 10, borderColor: '#ffb74d', color: '#ffb74d' }} />
              </Box>
            </VitisSection>
          ) : null}

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <MetricCard label="Stereo objects checked" value={ref.stereo.objects_checked}
              info="Objects with a valid block-matching disparity measurement in the synthetic stereo pairs." />
            <MetricCard label="Disparity error" value={`${ref.stereo.median_abs_disparity_error_px?.toFixed(2) ?? '—'} px`}
              info="Median |measured − geometric| disparity across checked objects (reference backend)." />
            <MetricCard label="Depth error (near <30 m)" value={`${ref.stereo.median_abs_depth_error_near_m?.toFixed(2) ?? '—'} m`}
              sub={`relative (all ranges): ${ref.stereo.median_rel_depth_error !== null ? (ref.stereo.median_rel_depth_error * 100).toFixed(1) : '—'}%`}
              info={ref.stereo.note ?? 'Disparity-derived depth vs exact scene geometry.'} />
            <MetricCard label="Emulated-backend disparity error" value={`${emu.stereo.median_abs_disparity_error_px?.toFixed(2) ?? '—'} px`}
              info="Same stereo check with fixed-point SAD costs on the emulated backend." />
          </Box>
        </>
      ) : (
        <Alert severity="info">
          Profile the engines to compare flicker, jitter, fragmentation and stereo-depth consistency across backends.
        </Alert>
      )}
    </Box>
  );
}
