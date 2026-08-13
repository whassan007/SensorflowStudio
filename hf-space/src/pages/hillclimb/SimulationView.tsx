/** Hill-climbing simulation: state gauges, hypothesis input, intervention picker, event log, debrief. */

import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Checkbox from '@mui/material/Checkbox';
import Chip from '@mui/material/Chip';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ErrorNote, HBar, LoadingBox, SectionCard } from '../../components/labeleval/shared';
import { useAsync } from '../../components/hillclimb/shared';
import { getSimulationCatalog, startSimulation, stepSimulation } from '../../services/hillclimb';
import type { SimulationState } from '../../types/hillclimb';

function metricColor(name: string, value: number, inverted: string[]): string {
  const effective = inverted.includes(name) ? 100 - value : value;
  if (effective >= 60) return '#66bb6a';
  if (effective >= 40) return '#f9a825';
  return '#ef5350';
}

export default function SimulationView() {
  const catalog = useAsync(getSimulationCatalog);
  const [sim, setSim] = useState<SimulationState | null>(null);
  const [hypothesis, setHypothesis] = useState('');
  const [interventionId, setInterventionId] = useState('add_monitoring');
  const [revert, setRevert] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = () => {
    setBusy(true);
    setError(null);
    startSimulation(Math.floor(Math.random() * 100000), 8)
      .then(setSim)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  const step = () => {
    if (!sim || !hypothesis.trim()) return;
    setBusy(true);
    setError(null);
    stepSimulation(sim.sim_id, hypothesis, interventionId, revert)
      .then((s) => {
        setSim(s);
        setHypothesis('');
        setRevert(false);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  if (catalog.loading && !catalog.data) return <LoadingBox label="Loading scenario…" />;
  if (catalog.error && !catalog.data) return <ErrorNote error={catalog.error} />;

  const inverted = catalog.data?.inverted_metrics ?? [];
  const interventions = catalog.data?.interventions ?? [];
  const scenario = catalog.data?.scenarios[0];
  const lastTurn = sim?.history[sim.history.length - 1] ?? null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}

      {!sim ? (
        <SectionCard title={scenario?.title ?? 'Scenario'} help="A stateful multi-objective exercise: each turn you write a falsifiable hypothesis, pick one intervention, measure the seeded (deterministic per seed) effect, and keep or reject. Safety has a hard floor — breach it and an incident fires.">
          <Typography variant="body2" sx={{ color: '#e6e9ec', lineHeight: 1.7, mb: 2 }}>
            {scenario?.narrative}
          </Typography>
          <Button variant="contained" onClick={start} disabled={busy}>
            Take over the team (8 turns)
          </Button>
        </SectionCard>
      ) : null}

      {sim ? (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Chip size="small" label={`turn ${sim.turn}/${sim.max_turns}`} sx={{ bgcolor: '#0d47a1', color: '#90caf9', fontWeight: 700 }} />
            <Chip
              size="small"
              label={`balanced objective ${sim.objective_history[sim.objective_history.length - 1].toFixed(1)}`}
              sx={{ bgcolor: '#232a31', color: '#e6e9ec', fontFamily: 'monospace' }}
            />
            {lastTurn ? (
              <Chip
                size="small"
                label={`Δ ${lastTurn.objective_delta >= 0 ? '+' : ''}${lastTurn.objective_delta.toFixed(1)} → ${
                  lastTurn.verdict === 'keep_recommended' ? 'KEEP' : 'REJECT'
                }`}
                sx={{
                  bgcolor: lastTurn.objective_delta >= 0 ? '#1b5e20' : '#5d1414',
                  color: lastTurn.objective_delta >= 0 ? '#a5d6a7' : '#ffcdd2',
                  fontWeight: 700,
                }}
              />
            ) : null}
            {sim.status === 'complete' ? (
              <Chip size="small" label="COMPLETE — see debrief below" sx={{ bgcolor: '#00695c', color: '#80cbc4', fontWeight: 700 }} />
            ) : null}
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <SectionCard
              title="Competing objectives"
              help="All ten metrics matter; cost and risk are inverted (lower is better). Safety below the hard floor triggers an incident with second-order damage."
              sx={{ flex: '1 1 360px' }}
            >
              {(catalog.data?.metrics ?? []).map((m) => (
                <HBar
                  key={m}
                  label={`${m.replace(/_/g, ' ')}${inverted.includes(m) ? ' (lower=better)' : ''}`}
                  value={sim.metrics[m] ?? 0}
                  max={100}
                  color={metricColor(m, sim.metrics[m] ?? 0, inverted)}
                  valueLabel={(sim.metrics[m] ?? 0).toFixed(0)}
                />
              ))}
            </SectionCard>

            <Box sx={{ flex: '1 1 380px', display: 'flex', flexDirection: 'column', gap: 2 }}>
              {sim.status === 'active' ? (
                <SectionCard title={`Turn ${sim.turn + 1}: hypothesis → intervention`}>
                  <TextField
                    label="Hypothesis (falsifiable: metric + direction + magnitude)"
                    placeholder="e.g. 'Adding monitoring will raise safety by ~5 within 2 turns'"
                    multiline
                    minRows={2}
                    fullWidth
                    value={hypothesis}
                    onChange={(e) => setHypothesis(e.target.value)}
                    sx={{ mb: 1.5 }}
                  />
                  <TextField
                    select
                    fullWidth
                    size="small"
                    label="Intervention"
                    value={interventionId}
                    onChange={(e) => setInterventionId(e.target.value)}
                    sx={{ mb: 1 }}
                  >
                    {interventions.map((i) => (
                      <MenuItem key={i.id} value={i.id}>
                        {i.label}
                      </MenuItem>
                    ))}
                  </TextField>
                  <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
                    {interventions.find((i) => i.id === interventionId)?.description}{' '}
                    <i>{interventions.find((i) => i.id === interventionId)?.second_order}</i>
                  </Typography>
                  {sim.history.length > 0 ? (
                    <FormControlLabel
                      control={<Checkbox size="small" checked={revert} onChange={(e) => setRevert(e.target.checked)} />}
                      label={
                        <Typography variant="caption" sx={{ color: '#ffb74d' }}>
                          Reject previous intervention first (roll back its immediate effects)
                        </Typography>
                      }
                    />
                  ) : null}
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Button variant="contained" onClick={step} disabled={busy || !hypothesis.trim()}>
                      {busy ? 'Measuring…' : 'Intervene & measure'}
                    </Button>
                  </Box>
                </SectionCard>
              ) : null}

              <SectionCard title="Event log" sx={{ maxHeight: 420, overflowY: 'auto' }}>
                {sim.history.length === 0 ? (
                  <Typography variant="body2" sx={{ color: '#8a949e' }}>
                    No turns yet.
                  </Typography>
                ) : null}
                {[...sim.history].reverse().map((t) => (
                  <Box key={t.turn} sx={{ mb: 1.5, pb: 1, borderBottom: '1px solid #232a31' }}>
                    <Typography variant="caption" sx={{ color: '#4fc3f7', fontWeight: 700 }}>
                      TURN {t.turn} · {t.intervention_id} · Δobj {t.objective_delta >= 0 ? '+' : ''}
                      {t.objective_delta.toFixed(1)}
                      {'  '}
                      <Chip
                        size="small"
                        label={`hypothesis: ${t.hypothesis_assessment.quality}`}
                        sx={{
                          height: 16, fontSize: 9.5, ml: 0.5,
                          bgcolor: t.hypothesis_assessment.quality === 'strong' ? '#1b5e20' : t.hypothesis_assessment.quality === 'directional' ? '#f9a825' : '#5d1414',
                          color: t.hypothesis_assessment.quality === 'directional' ? '#212121' : '#e6e9ec',
                        }}
                      />
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#aab4be', display: 'block', fontStyle: 'italic' }}>
                      “{t.hypothesis}”
                    </Typography>
                    {t.delayed_landed.map((d, i) => (
                      <Typography key={i} variant="caption" sx={{ color: '#9ccc65', display: 'block' }}>
                        ⏲ {d}
                      </Typography>
                    ))}
                    {t.events.map((e, i) => (
                      <Typography key={i} variant="caption" sx={{ color: '#ef9a9a', display: 'block' }}>
                        ⚠ {e}
                      </Typography>
                    ))}
                  </Box>
                ))}
              </SectionCard>
            </Box>
          </Box>

          {sim.debrief ? (
            <SectionCard title="Debrief — decisions mapped to competency evidence">
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1.5 }}>
                <Chip size="small" label={`objective ${sim.debrief.objective_start} → ${sim.debrief.objective_end} (${sim.debrief.objective_delta >= 0 ? '+' : ''}${sim.debrief.objective_delta})`} sx={{ bgcolor: '#232a31', fontFamily: 'monospace' }} />
                <Chip size="small" label={`${sim.debrief.incidents} incident(s)`} sx={{ bgcolor: sim.debrief.incidents ? '#5d1414' : '#1b5e20', color: sim.debrief.incidents ? '#ffcdd2' : '#a5d6a7' }} />
                <Chip size="small" label={sim.debrief.balanced_finish ? 'balanced finish' : 'unbalanced finish'} sx={{ bgcolor: sim.debrief.balanced_finish ? '#1b5e20' : '#e65100', color: '#e6e9ec' }} />
              </Box>
              {sim.debrief.competency_mappings.map((m) => (
                <Alert
                  key={m.competency_id}
                  severity={m.verdict === 'evidenced' ? 'success' : 'warning'}
                  variant="outlined"
                  sx={{ mb: 1, py: 0.25 }}
                >
                  <b>{m.competency_id}</b> — {m.reason}
                  {m.quotes.map((q, i) => (
                    <Typography key={i} variant="caption" sx={{ display: 'block', fontStyle: 'italic', color: '#aab4be' }}>
                      “{q}”
                    </Typography>
                  ))}
                </Alert>
              ))}
              <Button size="small" onClick={start}>
                Run again (new seed)
              </Button>
            </SectionCard>
          ) : null}
        </>
      ) : null}
    </Box>
  );
}
