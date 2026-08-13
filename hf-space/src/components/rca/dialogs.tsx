/** Dialogs: record a human finding; acknowledge unknowns to complete a stage. */
import { useState } from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import type { FindingStatus, Severity } from '../../types/rca';

export function RecordFindingDialog({ open, stageTitle, onClose, onSubmit }: {
  open: boolean;
  stageTitle: string;
  onClose: () => void;
  onSubmit: (f: { title: string; status: FindingStatus; severity: Severity; detail: string }) => void;
}) {
  const [title, setTitle] = useState('');
  const [status, setStatus] = useState<FindingStatus>('MISMATCH');
  const [severity, setSeverity] = useState<Severity>('WARN');
  const [detail, setDetail] = useState('');
  const reset = () => { setTitle(''); setStatus('MISMATCH'); setSeverity('WARN'); setDetail(''); };
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ fontSize: 15, fontWeight: 800 }}>Record finding — {stageTitle}</DialogTitle>
      <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, pt: '8px !important' }}>
        <TextField
          label="Finding" size="small" fullWidth autoFocus value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Confirmed with infra team: feature pipeline fp-2.5 deployed online only"
        />
        <TextField select label="Status" size="small" value={status} onChange={(e) => setStatus(e.target.value as FindingStatus)}>
          {(['PASS', 'MISMATCH', 'UNKNOWN'] as FindingStatus[]).map((s) => <MenuItem key={s} value={s}>{s}</MenuItem>)}
        </TextField>
        <TextField select label="Severity" size="small" value={severity} onChange={(e) => setSeverity(e.target.value as Severity)}>
          {(['INFO', 'WARN', 'CRITICAL'] as Severity[]).map((s) => <MenuItem key={s} value={s}>{s}</MenuItem>)}
        </TextField>
        <TextField
          label="Detail / evidence" size="small" fullWidth multiline minRows={2} value={detail}
          onChange={(e) => setDetail(e.target.value)}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained" disabled={!title.trim()}
          onClick={() => { onSubmit({ title: title.trim(), status, severity, detail }); reset(); }}
        >
          Record
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export function AcknowledgeUnknownsDialog({ open, stageTitle, unknownTitles, onClose, onAcknowledge }: {
  open: boolean;
  stageTitle: string;
  unknownTitles: string[];
  onClose: () => void;
  onAcknowledge: (note: string) => void;
}) {
  const [note, setNote] = useState('');
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ fontSize: 15, fontWeight: 800, color: '#ffb74d' }}>
        Proceed with unknowns? — {stageTitle}
      </DialogTitle>
      <DialogContent sx={{ pt: '4px !important' }}>
        <Typography variant="body2" sx={{ mb: 1 }}>
          This stage has critical UNKNOWN findings. Completing it without resolving them
          will be permanently recorded on the investigation ("proceeding with unknowns")
          and will appear in the final report.
        </Typography>
        {unknownTitles.map((t) => (
          <Typography key={t} variant="caption" sx={{ display: 'block', color: '#ffb74d' }}>
            ? {t}
          </Typography>
        ))}
        <TextField
          label="Why is it acceptable to proceed? (recorded)" size="small" fullWidth multiline minRows={2}
          sx={{ mt: 1.5 }} value={note} onChange={(e) => setNote(e.target.value)}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Go back and resolve</Button>
        <Button color="warning" variant="contained" disabled={!note.trim()} onClick={() => { onAcknowledge(note.trim()); setNote(''); }}>
          Acknowledge and proceed
        </Button>
      </DialogActions>
    </Dialog>
  );
}
