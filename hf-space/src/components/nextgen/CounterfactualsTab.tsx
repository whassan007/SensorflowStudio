/** Tab 1: transformation recipe builder + generated scenarios with validity gate results. */
import { useCallback, useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ChevronDown, ChevronUp, FlaskConical, Plus, ShieldCheck, Trash2 } from 'lucide-react';
import * as api from '../../services/nextgen';
import type { CatalogueEntry, CounterfactualScenario, SuiteWeights } from '../../types/nextgen';
import { DataLabelChip, PANEL_SX, ScoreChip } from './common';

interface RecipeRow {
  kind: string;
  paramsJson: string;
}

function ValidityDetails({ scenario }: { scenario: CounterfactualScenario }) {
  const v = scenario.validity;
  if (!v) return null;
  return (
    <Box sx={{ mt: 1, pl: 1, borderLeft: '2px solid #232a31' }}>
      {v.checks.map((c) => (
        <Box key={c.check} sx={{ display: 'flex', gap: 1, alignItems: 'baseline', mb: 0.25 }}>
          <Chip
            size="small"
            label={c.passed ? 'PASS' : 'FAIL'}
            sx={{
              height: 16,
              fontSize: 9,
              fontWeight: 800,
              bgcolor: c.passed ? '#1b3a22' : '#3a1b1b',
              color: c.passed ? '#81c784' : '#ef9a9a',
              width: 44,
            }}
          />
          <Typography variant="caption" sx={{ color: '#c7ccd1', fontSize: 10.5, minWidth: 190 }}>
            {c.check}
          </Typography>
          <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 10 }}>
            {c.detail}
          </Typography>
        </Box>
      ))}
      {v.reasons.length > 0 && (
        <Typography variant="caption" sx={{ color: '#ef9a9a', fontSize: 10.5 }}>
          {v.reasons.join(' · ')}
        </Typography>
      )}
    </Box>
  );
}

function ScenarioCard({
  scenario,
  onValidate,
  validating,
}: {
  scenario: CounterfactualScenario;
  onValidate: (id: string) => void;
  validating: boolean;
}) {
  const [open, setOpen] = useState(false);
  const v = scenario.validity;
  return (
    <Paper sx={{ ...PANEL_SX, mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        <Typography variant="body2" sx={{ fontWeight: 700, fontFamily: 'monospace', fontSize: 12 }}>
          {scenario.scenario_id}
        </Typography>
        <DataLabelChip label={scenario.provenance.data_label} />
        {v ? (
          <>
            <Chip
              size="small"
              label={v.accepted ? 'GATE: ACCEPTED' : 'GATE: REJECTED'}
              sx={{
                height: 20,
                fontSize: 10,
                fontWeight: 800,
                bgcolor: v.accepted ? '#1b3a22' : '#3a1b1b',
                color: v.accepted ? '#81c784' : '#ef9a9a',
              }}
            />
            <ScoreChip name="fidelity" value={v.simulation_fidelity_score} />
            <ScoreChip name="validity" value={v.counterfactual_validity} />
            <ScoreChip name="realism" value={v.realism_confidence} />
            <Chip
              size="small"
              label={`weight ${v.evaluation_weight.toFixed(2)}${v.weight_capped ? ' (capped)' : ''}`}
              sx={{ height: 20, fontSize: 10, bgcolor: '#12171d', color: v.weight_capped ? '#ffb74d' : '#c7ccd1' }}
            />
          </>
        ) : (
          <Button
            size="small"
            variant="outlined"
            startIcon={validating ? <CircularProgress size={12} /> : <ShieldCheck size={13} />}
            disabled={validating}
            onClick={() => onValidate(scenario.scenario_id)}
            sx={{ height: 24, fontSize: 11, textTransform: 'none' }}
          >
            Run validity gate
          </Button>
        )}
        <Box sx={{ flex: 1 }} />
        <IconButton size="small" onClick={() => setOpen((o) => !o)}>
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </IconButton>
      </Box>
      <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 10.5 }}>
        source {scenario.provenance.source_scene_id} · seed {scenario.provenance.seed} ·{' '}
        {scenario.provenance.recipe.map((r) => r.kind).join(' → ')} · {scenario.n_frames} frames ·{' '}
        {scenario.n_actors} actors · {scenario.environment.time_of_day}/{scenario.environment.weather}
      </Typography>
      <Collapse in={open}>
        <ValidityDetails scenario={scenario} />
      </Collapse>
    </Paper>
  );
}

export default function CounterfactualsTab() {
  const [catalogue, setCatalogue] = useState<CatalogueEntry[]>([]);
  const [recipe, setRecipe] = useState<RecipeRow[]>([
    { kind: 'actors.occluded_emergence', paramsJson: '{}' },
  ]);
  const [seed, setSeed] = useState(7);
  const [nScenarios, setNScenarios] = useState(1);
  const [scenarios, setScenarios] = useState<CounterfactualScenario[]>([]);
  const [weights, setWeights] = useState<SuiteWeights | null>(null);
  const [busy, setBusy] = useState(false);
  const [validatingId, setValidatingId] = useState<string | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    const [list, w] = await Promise.all([api.listCounterfactuals(), api.getSuiteWeights()]);
    setScenarios(list.scenarios);
    setWeights(w);
  }, []);

  useEffect(() => {
    api.getCatalogue().then((c) => setCatalogue(c.transformations)).catch(() => {});
    refresh().catch((e) => setError(String(e.message ?? e)));
  }, [refresh]);

  const generate = async () => {
    setBusy(true);
    setError('');
    try {
      const steps = recipe.map((r) => ({ kind: r.kind, params: JSON.parse(r.paramsJson || '{}') }));
      await api.generateCounterfactuals({ recipe: steps, seed, n_scenarios: nScenarios });
      await refresh();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const validate = async (id: string) => {
    setValidatingId(id);
    setError('');
    try {
      await api.validateCounterfactual(id);
      await refresh();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setValidatingId(null);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Paper sx={PANEL_SX}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
          Transformation recipe
        </Typography>
        {recipe.map((row, idx) => (
          <Box key={idx} sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center' }}>
            <TextField
              select
              size="small"
              value={row.kind}
              onChange={(e) =>
                setRecipe((rs) => rs.map((r, j) => (j === idx ? { ...r, kind: e.target.value } : r)))
              }
              sx={{ minWidth: 260 }}
            >
              {catalogue.map((c) => (
                <MenuItem key={c.kind} value={c.kind} sx={{ fontSize: 12 }}>
                  {c.kind}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              size="small"
              label="params (JSON)"
              value={row.paramsJson}
              onChange={(e) =>
                setRecipe((rs) => rs.map((r, j) => (j === idx ? { ...r, paramsJson: e.target.value } : r)))
              }
              sx={{ flex: 1 }}
              inputProps={{ style: { fontFamily: 'monospace', fontSize: 12 } }}
            />
            <IconButton size="small" onClick={() => setRecipe((rs) => rs.filter((_, j) => j !== idx))}>
              <Trash2 size={14} />
            </IconButton>
          </Box>
        ))}
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          <Button
            size="small"
            startIcon={<Plus size={13} />}
            onClick={() => setRecipe((rs) => [...rs, { kind: catalogue[0]?.kind ?? 'environment.clear_to_fog', paramsJson: '{}' }])}
            sx={{ textTransform: 'none' }}
          >
            Add step
          </Button>
          <TextField
            size="small"
            type="number"
            label="seed"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            sx={{ width: 90 }}
          />
          <TextField
            size="small"
            type="number"
            label="scenarios"
            value={nScenarios}
            onChange={(e) => setNScenarios(Math.max(1, Math.min(8, Number(e.target.value))))}
            sx={{ width: 90 }}
          />
          <Button
            variant="contained"
            size="small"
            disabled={busy || recipe.length === 0}
            startIcon={busy ? <CircularProgress size={13} /> : <FlaskConical size={14} />}
            onClick={generate}
            sx={{ textTransform: 'none' }}
          >
            Generate counterfactuals
          </Button>
        </Box>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      {weights && Object.keys(weights.weights).length > 0 && (
        <Paper sx={PANEL_SX}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
            Suite weight policy
          </Typography>
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 0.5 }}>
            {weights.policy}
          </Typography>
          <Typography variant="caption" sx={{ color: '#c7ccd1' }}>
            low-fidelity share {(weights.low_fidelity_share * 100).toFixed(1)}% (cap{' '}
            {(weights.share_cap * 100).toFixed(0)}%){' '}
            {weights.scaled_down ? '— scaled down to respect cap' : '— within cap'}
          </Typography>
        </Paper>
      )}

      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
          Generated scenarios ({scenarios.length})
        </Typography>
        {scenarios.length === 0 && (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No counterfactuals yet — build a recipe and generate.
          </Typography>
        )}
        {scenarios.map((s) => (
          <ScenarioCard
            key={s.scenario_id}
            scenario={s}
            onValidate={validate}
            validating={validatingId === s.scenario_id}
          />
        ))}
      </Box>
    </Box>
  );
}
