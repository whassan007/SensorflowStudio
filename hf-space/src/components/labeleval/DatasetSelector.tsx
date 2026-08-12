import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import { DatabaseZap, Play, ShieldQuestion } from 'lucide-react';
import type { DatasetSummary } from '../../types/labeleval';
import {
  generateDataset,
  precheckDataset,
  runPipeline,
  type PrecheckResponse,
} from '../../services/labeleval';
import { SectionCard, StatusChip, GateLineList, fmtInt, fmtPct, EmptyState } from './shared';

function lineageChips(d: DatasetSummary) {
  const chips: string[] = [];
  if (d.lineage.generated_from_model) chips.push(`from ${d.lineage.generated_from_model}`);
  if (d.lineage.corrected_by_review_batch) chips.push(`corrected by ${d.lineage.corrected_by_review_batch}`);
  if (d.lineage.validated_by_policy) chips.push(`policy ${d.lineage.validated_by_policy}`);
  if (d.lineage.parent_dataset) chips.push(`parent ${d.lineage.parent_dataset}`);
  return chips;
}

export default function DatasetSelector({
  datasets,
  selectedId,
  onSelect,
  onChanged,
  verificationRates,
}: {
  datasets: DatasetSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** Called after generate / run so the parent can refresh. */
  onChanged: () => void;
  /** Optional map dataset_id -> verification rate (0..1). */
  verificationRates?: Record<string, number>;
}) {
  const [genOpen, setGenOpen] = useState(false);
  const [genForm, setGenForm] = useState({ name: '', num_sequences: 8, frames_per_sequence: 25, seed: 42 });
  const [busy, setBusy] = useState<string | null>(null); // 'generate' | `precheck:${id}` | `run:${id}`
  const [precheck, setPrecheck] = useState<{ datasetId: string; result: PrecheckResponse } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const doGenerate = async () => {
    setBusy('generate');
    setErrorMsg(null);
    try {
      const ds = await generateDataset({
        ...(genForm.name ? { name: genForm.name } : {}),
        num_sequences: genForm.num_sequences,
        frames_per_sequence: genForm.frames_per_sequence,
        seed: genForm.seed,
      });
      setGenOpen(false);
      setNotice(`Generated dataset ${ds.name} (${ds.dataset_id})`);
      onSelect(ds.dataset_id);
      onChanged();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const doPrecheck = async (id: string) => {
    setBusy(`precheck:${id}`);
    setErrorMsg(null);
    try {
      const result = await precheckDataset(id);
      setPrecheck({ datasetId: id, result });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const doRun = async (id: string) => {
    setBusy(`run:${id}`);
    setErrorMsg(null);
    try {
      const res = await runPipeline(id);
      setNotice(`Pipeline run ${res.run_id} started (${res.status}). Watch progress on the Overview / Pipeline pages.`);
      onChanged();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <SectionCard
      title="Datasets"
      action={
        <Button
          size="small"
          variant="contained"
          startIcon={<DatabaseZap size={16} />}
          onClick={() => setGenOpen(true)}
        >
          Generate Synthetic Dataset
        </Button>
      }
    >
      {notice ? (
        <Alert severity="success" variant="outlined" onClose={() => setNotice(null)} sx={{ mb: 1 }}>
          {notice}
        </Alert>
      ) : null}
      {errorMsg ? (
        <Alert severity="error" variant="outlined" onClose={() => setErrorMsg(null)} sx={{ mb: 1 }}>
          {errorMsg}
        </Alert>
      ) : null}

      {datasets.length === 0 ? (
        <EmptyState
          title="No datasets yet"
          message="Generate a synthetic dataset to seed the evaluation pipeline."
          action={
            <Button variant="contained" startIcon={<DatabaseZap size={16} />} onClick={() => setGenOpen(true)}>
              Generate Synthetic Dataset
            </Button>
          }
        />
      ) : (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Version</TableCell>
              <TableCell align="right">Frames</TableCell>
              <TableCell align="right">Annotations</TableCell>
              <TableCell>Ground truth</TableCell>
              <TableCell align="right">Verification</TableCell>
              <TableCell>Lineage</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {datasets.map((d) => {
              const rate = verificationRates?.[d.dataset_id];
              return (
                <TableRow
                  key={d.dataset_id}
                  hover
                  selected={d.dataset_id === selectedId}
                  onClick={() => onSelect(d.dataset_id)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell sx={{ fontWeight: 600 }}>{d.name}</TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{d.version}</TableCell>
                  <TableCell align="right">{fmtInt(d.num_frames)}</TableCell>
                  <TableCell align="right">{fmtInt(d.num_annotations)}</TableCell>
                  <TableCell>
                    {d.gt_availability.has_reference && d.gt_availability.gt_type ? (
                      <Chip
                        size="small"
                        label={`${d.gt_availability.gt_type.replace(/_/g, ' ')} · ${fmtPct(d.gt_availability.coverage)}`}
                        sx={{
                          bgcolor: d.gt_availability.gt_type === 'PSEUDO_GROUND_TRUTH' ? '#4a2c00' : '#1b3a24',
                          color: d.gt_availability.gt_type === 'PSEUDO_GROUND_TRUTH' ? '#ffcc80' : '#a5d6a7',
                          fontSize: 10,
                        }}
                      />
                    ) : (
                      <Chip size="small" label="no reference" sx={{ bgcolor: '#232a31', fontSize: 10 }} />
                    )}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {rate !== undefined ? fmtPct(rate) : '—'}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {lineageChips(d).map((c) => (
                        <Chip key={c} size="small" label={c} sx={{ bgcolor: '#232a31', fontSize: 10 }} />
                      ))}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <StatusChip status={d.status} />
                  </TableCell>
                  <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                    <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'flex-end' }}>
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={
                          busy === `precheck:${d.dataset_id}` ? (
                            <CircularProgress size={12} />
                          ) : (
                            <ShieldQuestion size={14} />
                          )
                        }
                        disabled={busy !== null}
                        onClick={() => void doPrecheck(d.dataset_id)}
                      >
                        Precheck
                      </Button>
                      <Button
                        size="small"
                        variant="contained"
                        startIcon={
                          busy === `run:${d.dataset_id}` ? <CircularProgress size={12} /> : <Play size={14} />
                        }
                        disabled={busy !== null}
                        onClick={() => void doRun(d.dataset_id)}
                      >
                        Run Evaluation
                      </Button>
                    </Box>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      {/* Generate dialog */}
      <Dialog open={genOpen} onClose={() => setGenOpen(false)}>
        <DialogTitle>Generate Synthetic Dataset</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '12px !important', minWidth: 340 }}>
          <TextField
            label="Name (optional)"
            size="small"
            value={genForm.name}
            onChange={(e) => setGenForm({ ...genForm, name: e.target.value })}
          />
          <TextField
            label="Sequences"
            size="small"
            type="number"
            value={genForm.num_sequences}
            onChange={(e) => setGenForm({ ...genForm, num_sequences: Number(e.target.value) })}
          />
          <TextField
            label="Frames per sequence"
            size="small"
            type="number"
            value={genForm.frames_per_sequence}
            onChange={(e) => setGenForm({ ...genForm, frames_per_sequence: Number(e.target.value) })}
          />
          <TextField
            label="Seed"
            size="small"
            type="number"
            value={genForm.seed}
            onChange={(e) => setGenForm({ ...genForm, seed: Number(e.target.value) })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGenOpen(false)}>Cancel</Button>
          <Button variant="contained" disabled={busy === 'generate'} onClick={() => void doGenerate()}>
            {busy === 'generate' ? 'Generating…' : 'Generate'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Precheck results dialog */}
      <Dialog open={precheck !== null} onClose={() => setPrecheck(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Precheck — {precheck?.datasetId}</DialogTitle>
        <DialogContent>
          {precheck ? (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>{precheck.result.status}</strong> — {precheck.result.message}
              </Typography>
              <GateLineList checks={precheck.result.checks} />
            </>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPrecheck(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </SectionCard>
  );
}
