import { useState } from 'react';
import Box from '@mui/material/Box';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { ModelSummary } from '../types/labeleval';
import { getModels, usePoll } from '../services/labeleval';
import { SectionCard, StatusChip, LoadingBox, ErrorNote, MetricCard, fmtPct } from '../components/labeleval/shared';

export default function ModelsPage() {
  const models = usePoll(getModels, 10000);
  const [selected, setSelected] = useState<ModelSummary | null>(null);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {models.loading && !models.data ? <LoadingBox label="Loading models…" /> : null}
      {models.error && !models.data ? <ErrorNote error={models.error} /> : null}

      <SectionCard title={`Models (${models.data?.models.length ?? 0})`}>
        {!models.data || models.data.models.length === 0 ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No models registered yet — train one from the Training page.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Version</TableCell>
                <TableCell>Name</TableCell>
                <TableCell>Trained on</TableCell>
                <TableCell align="right">Precision</TableCell>
                <TableCell align="right">Recall</TableCell>
                <TableCell align="right">mAP 3D</TableCell>
                <TableCell align="right">Safety recall</TableCell>
                <TableCell align="right">Rare recall</TableCell>
                <TableCell>Regression</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.data.models.map((m) => (
                <TableRow key={m.model_id} hover sx={{ cursor: 'pointer' }} onClick={() => setSelected(m)}>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700 }}>
                    {m.model_version}
                  </TableCell>
                  <TableCell>{m.name}</TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{m.trained_on_dataset ?? '—'}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(m.metrics.precision)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(m.metrics.recall)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(m.metrics.map_3d)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(m.metrics.safety_critical_recall)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(m.metrics.rare_recall)}
                  </TableCell>
                  <TableCell>
                    <StatusChip status={m.regression_status} />
                  </TableCell>
                  <TableCell>
                    <StatusChip status={m.status} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </SectionCard>

      <Dialog open={selected !== null} onClose={() => setSelected(null)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selected?.name} — {selected?.model_version}
        </DialogTitle>
        <DialogContent>
          {selected ? (
            <>
              <Typography variant="body2" sx={{ color: '#8a949e', mb: 2 }}>
                Model id {selected.model_id} · trained on {selected.trained_on_dataset ?? '—'} · created{' '}
                {new Date(selected.created_at).toLocaleString()} · status {selected.status}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
                <MetricCard label="Precision" value={fmtPct(selected.metrics.precision)} />
                <MetricCard label="Recall" value={fmtPct(selected.metrics.recall)} />
                <MetricCard label="F1" value={fmtPct(selected.metrics.f1)} />
                <MetricCard label="mAP 3D" value={fmtPct(selected.metrics.map_3d)} />
                <MetricCard label="Safety recall" value={fmtPct(selected.metrics.safety_critical_recall)} accent="#ffa726" />
                <MetricCard label="Rare recall" value={fmtPct(selected.metrics.rare_recall)} accent="#ef5350" />
              </Box>
              <StatusChip status={selected.regression_status} />
            </>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelected(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
