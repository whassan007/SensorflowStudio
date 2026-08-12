import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import { Check, X, Wrench, Merge, Split, EyeOff } from 'lucide-react';
import type { Annotation, ReviewAction, ReviewActionResponse, ReviewTask } from '../../types/labeleval';
import { postReviewAction } from '../../services/labeleval';
import { SectionCard, GateLineList, StatusChip } from './shared';

const CLASSES = ['pedestrian', 'cyclist', 'vehicle', 'motorcycle', 'truck'];
const BBOX_FIELDS = ['x', 'y', 'z', 'l', 'w', 'h', 'yaw'] as const;

export default function RelabelingPanel({
  task,
  annotation,
  onResolved,
}: {
  task: ReviewTask | null;
  annotation: Annotation | null;
  onResolved: () => void;
}) {
  const [busy, setBusy] = useState<ReviewAction | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<ReviewActionResponse | null>(null);
  const [correctOpen, setCorrectOpen] = useState(false);
  const [bbox, setBbox] = useState<number[]>([0, 0, 0, 4, 2, 1.6, 0]);
  const [klass, setKlass] = useState('vehicle');
  const [mergeOpen, setMergeOpen] = useState(false);
  const [mergeTrackId, setMergeTrackId] = useState('');

  // Clear the previous re-validation result when switching tasks.
  useEffect(() => {
    setResult(null);
    setErrorMsg(null);
  }, [task?.task_id]);

  const submit = async (action: ReviewAction, extra?: { corrected_bbox_3d?: number[]; corrected_class?: string; merge_with_track_id?: string }) => {
    if (!task) return;
    setBusy(action);
    setErrorMsg(null);
    try {
      const res = await postReviewAction(task.task_id, { action, ...extra });
      setResult(res);
      setCorrectOpen(false);
      setMergeOpen(false);
      onResolved();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const openCorrect = () => {
    setBbox(annotation?.bbox_3d ? [...annotation.bbox_3d] : [0, 0, 0, 4, 2, 1.6, 0]);
    setKlass(annotation?.class_name && CLASSES.includes(annotation.class_name) ? annotation.class_name : 'vehicle');
    setCorrectOpen(true);
  };

  const revalidationPassed = result?.revalidation.validation.passed ?? null;

  return (
    <SectionCard title="Relabeling Actions">
      <Typography variant="caption" sx={{ color: '#ffcc80', display: 'block', mb: 1.5 }}>
        Corrected labels are never verified without re-running the quality gates — every action below triggers
        re-validation.
      </Typography>

      {errorMsg ? (
        <Alert severity="error" variant="outlined" onClose={() => setErrorMsg(null)} sx={{ mb: 1 }}>
          {errorMsg}
        </Alert>
      ) : null}

      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
        <Button
          variant="contained"
          color="success"
          size="small"
          startIcon={busy === 'approve' ? <CircularProgress size={14} /> : <Check size={14} />}
          disabled={!task || busy !== null}
          onClick={() => void submit('approve')}
        >
          Approve
        </Button>
        <Button
          variant="contained"
          color="error"
          size="small"
          startIcon={busy === 'reject' ? <CircularProgress size={14} /> : <X size={14} />}
          disabled={!task || busy !== null}
          onClick={() => void submit('reject')}
        >
          Reject
        </Button>
        <Button
          variant="outlined"
          size="small"
          startIcon={<Wrench size={14} />}
          disabled={!task || busy !== null}
          onClick={openCorrect}
        >
          Correct
        </Button>
        <Button
          variant="outlined"
          size="small"
          startIcon={busy === 'merge_tracks' ? <CircularProgress size={14} /> : <Merge size={14} />}
          disabled={!task || busy !== null}
          onClick={() => setMergeOpen(true)}
        >
          Merge Tracks
        </Button>
        <Button
          variant="outlined"
          size="small"
          startIcon={busy === 'split_track' ? <CircularProgress size={14} /> : <Split size={14} />}
          disabled={!task || busy !== null}
          onClick={() => void submit('split_track')}
        >
          Split Track
        </Button>
        <Button
          variant="outlined"
          size="small"
          startIcon={busy === 'mark_ignore' ? <CircularProgress size={14} /> : <EyeOff size={14} />}
          disabled={!task || busy !== null}
          onClick={() => void submit('mark_ignore')}
        >
          Mark Ignore
        </Button>
      </Box>

      {result ? (
        <Box
          sx={{
            border: `2px solid ${revalidationPassed ? '#66bb6a' : '#ef5350'}`,
            borderRadius: 1,
            p: 2,
            bgcolor: revalidationPassed ? 'rgba(102,187,106,0.08)' : 'rgba(239,83,80,0.08)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
              {result.message}
            </Typography>
            {result.task.resolution?.final_status ? <StatusChip status={result.task.resolution.final_status} /> : null}
          </Box>
          <Typography variant="body2" sx={{ color: '#aab4be', mb: 1 }}>
            Re-validation re-ran the quality gates:{' '}
            <strong style={{ color: revalidationPassed ? '#66bb6a' : '#ef5350' }}>
              {revalidationPassed ? 'PASSED → VERIFIED' : 'FAILED'}
            </strong>
            {!revalidationPassed && result.revalidation.decision?.primary_failure_reason
              ? ` — ${result.revalidation.decision.primary_failure_reason.replace(/_/g, ' ')}`
              : ''}
          </Typography>
          <GateLineList checks={result.revalidation.validation.checks} />
        </Box>
      ) : null}

      {/* Correct dialog */}
      <Dialog open={correctOpen} onClose={() => setCorrectOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Correct annotation {task?.annotation_id}</DialogTitle>
        <DialogContent sx={{ pt: '12px !important' }}>
          <FormControl size="small" fullWidth sx={{ mb: 2 }}>
            <InputLabel>Class</InputLabel>
            <Select value={klass} label="Class" onChange={(e) => setKlass(e.target.value)}>
              {CLASSES.map((c) => (
                <MenuItem key={c} value={c}>
                  {c}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1 }}>
            {BBOX_FIELDS.map((field, i) => (
              <TextField
                key={field}
                size="small"
                type="number"
                label={field}
                inputProps={{ step: 0.1 }}
                value={bbox[i]}
                onChange={(e) => {
                  const next = [...bbox];
                  next[i] = Number(e.target.value);
                  setBbox(next);
                }}
              />
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCorrectOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            disabled={busy !== null}
            onClick={() => void submit('correct', { corrected_bbox_3d: bbox, corrected_class: klass })}
          >
            {busy === 'correct' ? 'Submitting…' : 'Submit correction'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Merge dialog */}
      <Dialog open={mergeOpen} onClose={() => setMergeOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Merge tracks</DialogTitle>
        <DialogContent sx={{ pt: '12px !important' }}>
          <TextField
            size="small"
            fullWidth
            label="Merge with track id"
            value={mergeTrackId}
            onChange={(e) => setMergeTrackId(e.target.value)}
            helperText={annotation?.track_id ? `Current track: ${annotation.track_id}` : undefined}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMergeOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            disabled={busy !== null || !mergeTrackId}
            onClick={() => void submit('merge_tracks', { merge_with_track_id: mergeTrackId })}
          >
            {busy === 'merge_tracks' ? 'Submitting…' : 'Merge'}
          </Button>
        </DialogActions>
      </Dialog>
    </SectionCard>
  );
}
