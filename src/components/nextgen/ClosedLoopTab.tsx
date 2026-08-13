/** Tab 2: closed-loop behavioral run + causal counterfactual replay diff view. */
import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { GitCompareArrows, Play } from 'lucide-react';
import * as api from '../../services/nextgen';
import type { BehavioralAssessment, CausalReplayResult } from '../../types/nextgen';
import { DataLabelChip, LineChart, MetricCard, PANEL_SX } from './common';

function fmt(v: number | null | undefined, digits = 2): string | null {
  return v === null || v === undefined ? null : v.toFixed(digits);
}

function MetricsPanel({ a }: { a: BehavioralAssessment }) {
  const m = a.metrics;
  return (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
      <MetricCard label="Detection latency" value={fmt(m.detection_latency_s)} unit="s" />
      <MetricCard label="Min TTC" value={fmt(m.min_ttc_s)} unit="s" tone={m.min_ttc_s !== null && m.min_ttc_s < 1.5 ? 'bad' : 'neutral'} />
      <MetricCard label="Min separation" value={fmt(m.min_separation_m)} unit="m" tone={m.min_separation_m !== null && m.min_separation_m < 1 ? 'bad' : 'neutral'} />
      <MetricCard label="Safety margin" value={fmt(m.safety_margin_m)} unit="m" />
      <MetricCard label="Stopping distance" value={fmt(m.stopping_distance_m)} unit="m" />
      <MetricCard label="Max decel" value={fmt(m.max_deceleration_mps2)} unit="m/s²" />
      <MetricCard label="Planner interventions" value={m.planner_interventions} />
      <MetricCard label="Collision prob" value={fmt(m.collision_probability)} tone={m.collision ? 'bad' : 'good'} />
      <MetricCard label="Frame recall (open-loop)" value={fmt(a.open_loop.frame_recall as number | null)} />
    </Box>
  );
}

const FAULT_PRESETS: Record<string, { label: string; faults?: Record<string, unknown>[] }> = {
  default: { label: 'Demo faults (miss emergent pedestrian)' },
  none: { label: 'No faults (nominal perception)', faults: [] },
  cosmetic: {
    label: 'Cosmetic class flip (pedestrian → cyclist)',
    faults: [{ type: 'misclassify', instance_id: 'demo-crossing-ped', as_class: 'cyclist' }],
  },
};

export default function ClosedLoopTab() {
  const [scenarioIds, setScenarioIds] = useState<string[]>(['demo']);
  const [scenarioId, setScenarioId] = useState('demo');
  const [preset, setPreset] = useState('default');
  const [assessment, setAssessment] = useState<BehavioralAssessment | null>(null);
  const [causal, setCausal] = useState<CausalReplayResult | null>(null);
  const [busy, setBusy] = useState<'replay' | 'causal' | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api
      .listCounterfactuals()
      .then((r) => setScenarioIds(['demo', ...r.scenarios.map((s) => s.scenario_id)]))
      .catch(() => {});
  }, []);

  const run = async (kind: 'replay' | 'causal') => {
    setBusy(kind);
    setError('');
    try {
      const faults = FAULT_PRESETS[preset].faults;
      const body = { scenario_id: scenarioId, ...(faults !== undefined ? { faults } : {}) };
      if (kind === 'replay') setAssessment(await api.runReplay(body));
      else setCausal(await api.runCausalReplay(body));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  };

  const behavioral = causal?.verdict === 'BEHAVIORALLY_CONSEQUENTIAL';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Paper sx={PANEL_SX}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField select size="small" label="scenario" value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} sx={{ minWidth: 220 }}>
            {scenarioIds.map((s) => (
              <MenuItem key={s} value={s} sx={{ fontSize: 12 }}>
                {s === 'demo' ? 'demo (occluded emergence testbed)' : s}
              </MenuItem>
            ))}
          </TextField>
          <TextField select size="small" label="perception faults" value={preset} onChange={(e) => setPreset(e.target.value)} sx={{ minWidth: 280 }}>
            {Object.entries(FAULT_PRESETS).map(([k, v]) => (
              <MenuItem key={k} value={k} sx={{ fontSize: 12 }}>
                {v.label}
              </MenuItem>
            ))}
          </TextField>
          <Button
            variant="outlined"
            size="small"
            disabled={busy !== null}
            startIcon={busy === 'replay' ? <CircularProgress size={13} /> : <Play size={13} />}
            onClick={() => run('replay')}
            sx={{ textTransform: 'none' }}
          >
            Run closed-loop
          </Button>
          <Button
            variant="contained"
            size="small"
            disabled={busy !== null}
            startIcon={busy === 'causal' ? <CircularProgress size={13} /> : <GitCompareArrows size={13} />}
            onClick={() => run('causal')}
            sx={{ textTransform: 'none' }}
          >
            Causal replay (actual vs corrected)
          </Button>
        </Box>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      {assessment && (
        <Paper sx={PANEL_SX}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
              Behavioral metrics — {assessment.perception_mode} perception
            </Typography>
            <DataLabelChip label={assessment.data_label} />
          </Box>
          <MetricsPanel a={assessment} />
          <Box sx={{ mt: 1.5 }}>
            <LineChart
              height={150}
              xLabel="t (s)"
              yLabel="ego speed (m/s)"
              series={[
                {
                  name: 'ego speed',
                  color: '#4fc3f7',
                  points: assessment.trajectory.map((p) => ({ x: p.t, y: p.v })),
                },
              ]}
            />
          </Box>
        </Paper>
      )}

      {causal && (
        <>
          <Paper
            sx={{
              ...PANEL_SX,
              borderColor: behavioral ? '#b71c1c' : '#2e7d32',
              bgcolor: behavioral ? '#2a1416' : '#14241a',
            }}
          >
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
              <Chip
                label={causal.verdict}
                sx={{
                  fontWeight: 900,
                  fontSize: 13,
                  bgcolor: behavioral ? '#b71c1c' : '#2e7d32',
                  color: '#fff',
                }}
              />
              <DataLabelChip label={causal.data_label} />
              <Typography variant="body2" sx={{ color: '#c7ccd1' }}>
                {behavioral
                  ? 'Correcting perception changes the safety outcome — this regression matters behaviorally.'
                  : 'Perception error is visible in open-loop metrics but does not change behavior or safety outcome.'}
              </Typography>
            </Box>
          </Paper>

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <Paper sx={{ ...PANEL_SX, flex: '1 1 420px' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
                Trajectory diff — ego speed
              </Typography>
              <LineChart
                height={160}
                xLabel="t (s)"
                yLabel="v (m/s)"
                series={[
                  { name: 'actual perception', color: '#ef5350', points: causal.actual.trajectory.map((p) => ({ x: p.t, y: p.v })) },
                  { name: 'corrected perception', color: '#66bb6a', points: causal.corrected.trajectory.map((p) => ({ x: p.t, y: p.v })) },
                ]}
              />
            </Paper>
            <Paper sx={{ ...PANEL_SX, flex: '1 1 420px' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
                Trajectory diff — ego position
              </Typography>
              <LineChart
                height={160}
                xLabel="t (s)"
                yLabel="x (m)"
                series={[
                  { name: 'actual perception', color: '#ef5350', points: causal.actual.trajectory.map((p) => ({ x: p.t, y: p.x })) },
                  { name: 'corrected perception', color: '#66bb6a', points: causal.corrected.trajectory.map((p) => ({ x: p.t, y: p.x })) },
                ]}
              />
            </Paper>
          </Box>

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <Paper sx={{ ...PANEL_SX, flex: '1 1 380px' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                Causal chain
              </Typography>
              {causal.causal_chain.map((c, i) => (
                <Box key={i} sx={{ display: 'flex', gap: 1, mb: 0.75, alignItems: 'baseline' }}>
                  <Chip
                    size="small"
                    label={c.answer ? 'YES' : 'NO'}
                    sx={{
                      height: 18,
                      fontSize: 9.5,
                      fontWeight: 800,
                      width: 40,
                      bgcolor: c.answer ? '#3a1b1b' : '#1b3a22',
                      color: c.answer ? '#ef9a9a' : '#81c784',
                    }}
                  />
                  <Box>
                    <Typography variant="caption" sx={{ color: '#e0e3e7', fontWeight: 700, display: 'block', fontSize: 11 }}>
                      {c.question}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 10.5 }}>
                      {c.evidence}
                    </Typography>
                  </Box>
                </Box>
              ))}
            </Paper>
            <Paper sx={{ ...PANEL_SX, flex: '1 1 380px' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                Outcome diffs (actual − corrected)
              </Typography>
              {Object.entries(causal.diffs).map(([k, v]) => (
                <Box key={k} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.25 }}>
                  <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 11 }}>
                    {k}
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#e0e3e7', fontFamily: 'monospace', fontSize: 11 }}>
                    {typeof v === 'boolean' ? String(v) : v === null ? '—' : v.toFixed(3)}
                  </Typography>
                </Box>
              ))}
            </Paper>
          </Box>
        </>
      )}
    </Box>
  );
}
