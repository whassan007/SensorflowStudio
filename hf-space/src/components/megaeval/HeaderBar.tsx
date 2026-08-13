/**
 * Command Center header: population + run + baseline selectors, run status,
 * cache stats chip, and the two entry-point actions (generate population,
 * launch evaluation run). The action buttons are exported so the page-level
 * empty state can reuse them front and center.
 */
import { useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { Database, FlaskConical } from 'lucide-react';
import type { CacheStats, EvaluationRunInfo, PopulationMeta } from '../../types/megaeval';
import { createRun, fmtCompact, generatePopulation } from '../../services/megaeval';
import { fmtPct } from '../labeleval/shared';
import { GLOSSARY } from '../../content/glossary';
import { GlossaryContent } from '../help/InfoTip';
import { RunStatusChip } from './shared';

// ---------------------------------------------------------------- action buttons

export function GeneratePopulationButton({
  onGenerated,
  variant = 'outlined',
  size = 'small',
}: {
  onGenerated: (pop: PopulationMeta) => void;
  variant?: 'outlined' | 'contained';
  size?: 'small' | 'medium' | 'large';
}) {
  const [open, setOpen] = useState(false);
  const [numObjects, setNumObjects] = useState('320000');
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const parsed = Number(numObjects);
      const pop = await generatePopulation({
        ...(name.trim() ? { name: name.trim() } : {}),
        num_objects: Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : 320000,
      });
      setOpen(false);
      onGenerated(pop);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Tooltip title="Create a synthetic annotated population for aggregate-first Command Center evaluation">
        <Button
          variant={variant}
          size={size}
          startIcon={<Database size={16} />}
          onClick={() => setOpen(true)}
          aria-label="Generate evaluation population"
        >
          Generate population
        </Button>
      </Tooltip>
      <Dialog open={open} onClose={() => (busy ? undefined : setOpen(false))} maxWidth="xs" fullWidth>
        <DialogTitle>Generate evaluation population</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '12px !important' }}>
          {error ? (
            <Alert severity="error" variant="outlined">
              {error}
            </Alert>
          ) : null}
          <TextField
            label="Name (optional)"
            size="small"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. q3-highway-mix"
          />
          <TextField
            label="Objects"
            size="small"
            type="number"
            value={numObjects}
            onChange={(e) => setNumObjects(e.target.value)}
            helperText="1K – 1.2M synthetic annotated objects (default 320K)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)} disabled={busy}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={() => void submit()}
            disabled={busy}
            startIcon={busy ? <CircularProgress size={16} color="inherit" /> : <Database size={16} />}
          >
            {busy ? 'Generating…' : 'Generate'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export function NewRunButton({
  populationId,
  onCreated,
  variant = 'contained',
  size = 'small',
}: {
  populationId: string | null;
  onCreated: (run: EvaluationRunInfo) => void;
  variant?: 'outlined' | 'contained';
  size?: 'small' | 'medium' | 'large';
}) {
  const [open, setOpen] = useState(false);
  const [modelVersion, setModelVersion] = useState('perception-v2.1');
  const [injectRegression, setInjectRegression] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!populationId) return;
    setBusy(true);
    setError(null);
    try {
      const run = await createRun({
        population_id: populationId,
        model_version: modelVersion.trim() || 'perception-v2.1',
        ...(injectRegression ? { overrides: { night_penalty: 0.3, vru_penalty: 0.1 } } : {}),
      });
      setOpen(false);
      onCreated(run);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Tooltip title={populationId ? 'Launch an evaluation run that scores the population against a model version' : 'Generate a population first'}>
        <span>
          <Button
            variant={variant}
            size={size}
            startIcon={<FlaskConical size={16} />}
            disabled={!populationId}
            onClick={() => setOpen(true)}
            aria-label="New evaluation run"
          >
            New evaluation run
          </Button>
        </span>
      </Tooltip>
      <Dialog open={open} onClose={() => (busy ? undefined : setOpen(false))} maxWidth="xs" fullWidth>
        <DialogTitle>Launch evaluation run</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '12px !important' }}>
          {error ? (
            <Alert severity="error" variant="outlined">
              {error}
            </Alert>
          ) : null}
          <TextField
            label="Model version"
            size="small"
            value={modelVersion}
            onChange={(e) => setModelVersion(e.target.value)}
          />
          <FormControlLabel
            control={
              <Switch checked={injectRegression} onChange={(e) => setInjectRegression(e.target.checked)} />
            }
            label={
              <Box>
                <Typography variant="body2">Inject regression (demo)</Typography>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  Applies overrides night_penalty=0.30, vru_penalty=0.10
                </Typography>
              </Box>
            }
          />
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Runs are asynchronous: the run is queued, evaluated across distributed workers, then published.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)} disabled={busy}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={() => void submit()}
            disabled={busy || !populationId}
            startIcon={busy ? <CircularProgress size={16} color="inherit" /> : <FlaskConical size={16} />}
          >
            {busy ? 'Queueing…' : 'Launch run'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ---------------------------------------------------------------- header bar

export default function HeaderBar({
  populations,
  populationId,
  onPopulationChange,
  runs,
  runId,
  onRunChange,
  baselineRunId,
  onBaselineChange,
  selectedRun,
  cache,
  onGenerated,
  onRunCreated,
}: {
  populations: PopulationMeta[];
  populationId: string | null;
  onPopulationChange: (id: string) => void;
  runs: EvaluationRunInfo[];
  runId: string | null;
  onRunChange: (id: string) => void;
  baselineRunId: string | null;
  onBaselineChange: (id: string | null) => void;
  selectedRun: EvaluationRunInfo | null;
  cache: CacheStats | null;
  onGenerated: (pop: PopulationMeta) => void;
  onRunCreated: (run: EvaluationRunInfo) => void;
}) {
  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? '')),
    [runs]
  );
  const publishedRuns = useMemo(() => sortedRuns.filter((r) => r.status === 'published'), [sortedRuns]);

  const runLabel = (r: EvaluationRunInfo) =>
    `${r.model_version} · ${r.run_id.slice(0, 8)}${r.status === 'published' ? '' : ` (${r.status})`}`;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.25,
        flexWrap: 'wrap',
        p: 1.5,
        bgcolor: '#161b21',
        border: '1px solid #232a31',
        borderRadius: 1,
      }}
    >
      <Tooltip title={<GlossaryContent entry={GLOSSARY.population} />} enterDelay={500}>
        <TextField
          select
          size="small"
          label="Population"
          value={populationId ?? ''}
          onChange={(e) => onPopulationChange(e.target.value)}
          sx={{ minWidth: 220 }}
          disabled={populations.length === 0}
        >
          {populations.map((p) => (
            <MenuItem key={p.population_id} value={p.population_id}>
              {p.name} · {fmtCompact(p.num_objects)} objects
            </MenuItem>
          ))}
        </TextField>
      </Tooltip>

      <Tooltip title={<GlossaryContent entry={GLOSSARY.evaluation_run} />} enterDelay={500}>
        <TextField
          select
          size="small"
          label="Evaluation run"
          value={runId && sortedRuns.some((r) => r.run_id === runId) ? runId : ''}
          onChange={(e) => onRunChange(e.target.value)}
          sx={{ minWidth: 240 }}
          disabled={sortedRuns.length === 0}
        >
          {sortedRuns.map((r) => (
            <MenuItem key={r.run_id} value={r.run_id}>
              {runLabel(r)}
            </MenuItem>
          ))}
        </TextField>
      </Tooltip>

      <Tooltip
        title="The published run the Compare tab measures the current candidate against. Promotion verdicts and regression scans use this as the reference."
        enterDelay={500}
      >
        <TextField
          select
          size="small"
          label="Baseline (compare)"
          value={
            baselineRunId && publishedRuns.some((r) => r.run_id === baselineRunId) ? baselineRunId : ''
          }
          onChange={(e) => onBaselineChange(e.target.value === '' ? null : e.target.value)}
          sx={{ minWidth: 220 }}
          disabled={publishedRuns.length === 0}
        >
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
          {publishedRuns
            .filter((r) => r.run_id !== runId)
            .map((r) => (
              <MenuItem key={r.run_id} value={r.run_id}>
                {runLabel(r)}
              </MenuItem>
            ))}
        </TextField>
      </Tooltip>

      {selectedRun ? <RunStatusChip status={selectedRun.status} /> : null}

      <Box sx={{ flex: 1 }} />

      {cache ? (
        <Tooltip
          title={
            <Box>
              <GlossaryContent entry={GLOSSARY.query_cache} />
              <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 0.5, fontFamily: 'monospace' }}>
                {`now: ${cache.entries} entries · ${cache.hits} hits / ${cache.misses} misses`}
              </Typography>
            </Box>
          }
        >
          <Chip
            size="small"
            label={`cache ${cache.entries} · hit ${cache.hit_rate === null ? '—' : fmtPct(cache.hit_rate)}`}
            sx={{ height: 20, fontSize: 10.5, fontFamily: 'monospace', bgcolor: '#232a31', color: '#aab4be', cursor: 'help' }}
          />
        </Tooltip>
      ) : null}

      <GeneratePopulationButton onGenerated={onGenerated} />
      <NewRunButton populationId={populationId} onCreated={onRunCreated} />
    </Box>
  );
}
