/** Tab 4: launch-evaluation gauntlet — priority strata, early stopping, budget, compute savings. */
import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import LinearProgress from '@mui/material/LinearProgress';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { Cpu, Rocket } from 'lucide-react';
import * as api from '../../services/nextgen';
import type { ComputeReport, GauntletResult } from '../../types/nextgen';
import { DataLabelChip, MetricCard, PANEL_SX } from './common';

const EFFECT_PRESETS: Record<string, { label: string; effects: Record<string, number> }> = {
  clean: { label: 'Clean candidate (no planted regressions)', effects: {} },
  catastrophic: { label: 'Catastrophic safety regression (-8pp safety-critical)', effects: { safety_critical: -0.08 } },
  odd: { label: 'New-ODD regression (-3pp)', effects: { new_odd: -0.03 } },
  improvement: { label: 'Improvement (+2pp safety-critical)', effects: { safety_critical: 0.02 } },
};

const DECISION_COLORS: Record<string, string> = {
  REGRESSION: '#ef9a9a',
  PASS: '#81c784',
  INSUFFICIENT_EVIDENCE: '#ffb74d',
};

function RecommendationChip({ rec }: { rec: string }) {
  const color = rec === 'LAUNCH' ? '#2e7d32' : rec === 'DO_NOT_LAUNCH' ? '#b71c1c' : '#e65100';
  return <Chip label={rec} sx={{ fontWeight: 900, bgcolor: color, color: '#fff' }} />;
}

function ComputeSavingsCard({ report }: { report: ComputeReport }) {
  return (
    <Paper sx={{ ...PANEL_SX, flex: '1 1 340px' }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
        Compute savings (measured benchmark)
      </Typography>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
        <MetricCard label="Naive cost" value={report.naive_cost_s.toFixed(2)} unit="s" tone="bad" />
        <MetricCard label="Optimized cost" value={report.optimized_cost_s.toFixed(2)} unit="s" tone="good" />
        <MetricCard label="Savings" value={`${(report.savings_ratio * 100).toFixed(1)}%`} tone="good" />
        <MetricCard label="Cache hit rate" value={`${((report.hit_rate ?? 0) * 100).toFixed(0)}%`} />
      </Box>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
        naive = {report.n_scenarios} scenarios × {report.n_models} models × full inference (
        {report.naive_full_inferences} runs); optimized = {report.optimized_backbone_computes} backbone computes
        + {report.optimized_head_computes} head-only runs. Backbone {report.measured_backbone_s.toFixed(3)}s vs
        head {report.measured_head_s.toFixed(3)}s per scenario (measured).
      </Typography>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 0.5 }}>
        Invalidation: {report.invalidation}
      </Typography>
    </Paper>
  );
}

export default function GauntletTab() {
  const [preset, setPreset] = useState('catastrophic');
  const [budget, setBudget] = useState(120000);
  const [result, setResult] = useState<GauntletResult | null>(null);
  const [compute, setCompute] = useState<ComputeReport | null>(null);
  const [busy, setBusy] = useState<'run' | 'bench' | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.getComputeReport().then(setCompute).catch(() => {});
  }, []);

  const run = async () => {
    setBusy('run');
    setError('');
    try {
      setResult(await api.runGauntlet({ effects: EFFECT_PRESETS[preset].effects, budget_units: budget }));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  };

  const bench = async () => {
    setBusy('bench');
    setError('');
    try {
      setCompute(await api.runComputeBenchmark({ n_scenarios: 3, frames_per_sequence: 10 }));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  };

  const scale = result?.scale;
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Paper sx={PANEL_SX}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField select size="small" label="candidate profile" value={preset} onChange={(e) => setPreset(e.target.value)} sx={{ minWidth: 340 }}>
            {Object.entries(EFFECT_PRESETS).map(([k, v]) => (
              <MenuItem key={k} value={k} sx={{ fontSize: 12 }}>
                {v.label}
              </MenuItem>
            ))}
          </TextField>
          <TextField size="small" type="number" label="budget (units)" value={budget} onChange={(e) => setBudget(Number(e.target.value))} sx={{ width: 140 }} />
          <Button
            variant="contained"
            size="small"
            disabled={busy !== null}
            startIcon={busy === 'run' ? <CircularProgress size={13} /> : <Rocket size={13} />}
            onClick={run}
            sx={{ textTransform: 'none' }}
          >
            Run gauntlet
          </Button>
          <Button
            variant="outlined"
            size="small"
            disabled={busy !== null}
            startIcon={busy === 'bench' ? <CircularProgress size={13} /> : <Cpu size={13} />}
            onClick={bench}
            sx={{ textTransform: 'none' }}
          >
            Run compute benchmark
          </Button>
        </Box>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      {result && (
        <>
          <Paper sx={PANEL_SX}>
            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
              <RecommendationChip rec={result.recommendation.recommendation} />
              {result.halted && <Chip label="HALTED EARLY" sx={{ fontWeight: 800, bgcolor: '#3a1b1b', color: '#ef9a9a' }} />}
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                {result.run_id} · {result.candidate_version} vs {result.baseline_version} · wall{' '}
                {result.timing.wall_s.toFixed(2)}s ·{' '}
                {result.timing.units_per_second ? `${Math.round(result.timing.units_per_second).toLocaleString()} units/s` : ''}
              </Typography>
            </Box>
            {result.recommendation.blockers.length > 0 && (
              <Typography variant="caption" sx={{ color: '#ef9a9a', display: 'block', mt: 0.5 }}>
                {result.recommendation.blockers.join(' · ')}
              </Typography>
            )}
            <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 0.5 }}>
              {result.statistical_validity}
            </Typography>
          </Paper>

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <Paper sx={{ ...PANEL_SX, flex: '2 1 460px' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                Priority strata progress
              </Typography>
              {result.strata.map((s) => {
                const frac = s.units_available ? s.units_evaluated / s.units_available : 0;
                return (
                  <Box key={s.stratum} sx={{ mb: 1.25 }}>
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.25, flexWrap: 'wrap' }}>
                      <Typography variant="caption" sx={{ fontWeight: 800, color: '#e0e3e7', minWidth: 150 }}>
                        P{s.priority} {s.stratum}
                      </Typography>
                      <DataLabelChip label={s.data_label} />
                      <Chip
                        size="small"
                        label={s.decision}
                        sx={{ height: 18, fontSize: 9.5, fontWeight: 800, bgcolor: '#12171d', color: DECISION_COLORS[s.decision] ?? '#c7ccd1' }}
                      />
                      <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 10 }}>
                        {s.units_evaluated.toLocaleString()} / {s.units_available.toLocaleString()} units
                        {s.units_saved_by_early_stop > 0 ? ` · ${s.units_saved_by_early_stop.toLocaleString()} saved by early stop` : ''}
                        {s.delta_estimate !== null ? ` · Δ ${(s.delta_estimate * 100).toFixed(2)}pp` : ''}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={frac * 100}
                      sx={{
                        height: 8,
                        borderRadius: 1,
                        bgcolor: '#0d1117',
                        '& .MuiLinearProgress-bar': { bgcolor: DECISION_COLORS[s.decision] ?? '#4fc3f7' },
                      }}
                    />
                  </Box>
                );
              })}
              {result.processed_order.length < result.priority_order.length && (
                <Typography variant="caption" sx={{ color: '#ffb74d' }}>
                  {result.priority_order.filter((p) => !result.processed_order.includes(p)).join(', ')} never
                  reached (halted).
                </Typography>
              )}
            </Paper>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, flex: '1 1 320px' }}>
              <Paper sx={PANEL_SX}>
                <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                  Budget funnel
                </Typography>
                {scale && (
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <MetricCard label="Units defined" value={scale.total_units_defined.toLocaleString()} />
                    <MetricCard label="Evaluated" value={scale.units_evaluated.toLocaleString()} />
                    <MetricCard label="Saved (early stop)" value={scale.units_saved.toLocaleString()} tone="good" />
                    <MetricCard label="Budget left" value={scale.budget_remaining.toLocaleString()} />
                    <MetricCard
                      label="Outcome-cache hit rate"
                      value={result.cache.hit_rate === null ? '—' : `${(result.cache.hit_rate * 100).toFixed(0)}%`}
                    />
                  </Box>
                )}
              </Paper>
              <Paper sx={PANEL_SX}>
                <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                  Early-stop events
                </Typography>
                {result.events.length === 0 && (
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    No early-stop events — all strata ran to their evidence thresholds.
                  </Typography>
                )}
                {result.events.map((e, i) => (
                  <Box key={i} sx={{ mb: 0.75 }}>
                    <Chip
                      size="small"
                      label={e.event}
                      sx={{
                        height: 18,
                        fontSize: 9.5,
                        fontWeight: 800,
                        bgcolor: e.event === 'CATASTROPHIC_HALT' ? '#b71c1c' : '#12171d',
                        color: e.event === 'CATASTROPHIC_HALT' ? '#fff' : '#ffb74d',
                        mr: 0.5,
                      }}
                    />
                    <Typography variant="caption" sx={{ color: '#c7ccd1', fontSize: 10.5 }}>
                      {e.stratum ? `${e.stratum}: ` : ''}
                      {e.detail}
                    </Typography>
                  </Box>
                ))}
              </Paper>
            </Box>
          </Box>
        </>
      )}

      {compute ? (
        <ComputeSavingsCard report={compute} />
      ) : (
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          No compute benchmark yet — run one to see the naive-vs-optimized savings card.
        </Typography>
      )}
    </Box>
  );
}
