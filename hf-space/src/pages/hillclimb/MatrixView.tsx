/** Competency matrix: explainable scores — click a row to see the evidence behind it. */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { ErrorNote, LoadingBox, SectionCard } from '../../components/labeleval/shared';
import { ReadinessChip, useAsync } from '../../components/hillclimb/shared';
import { getEvidence, getReadiness } from '../../services/hillclimb';
import type { EvidenceItem, ReadinessRow } from '../../types/hillclimb';
import type { GoFn } from './HillClimbSection';

function fmtScore(v: number): string {
  return v > 0 ? v.toFixed(1) : '—';
}

export default function MatrixView({ go }: { go: GoFn }) {
  const readiness = useAsync(getReadiness);
  const [selected, setSelected] = useState<ReadinessRow | null>(null);
  const [evidence, setEvidence] = useState<EvidenceItem[] | null>(null);

  const openRow = (row: ReadinessRow) => {
    setSelected(row);
    setEvidence(null);
    getEvidence(row.competency_id)
      .then((r) => setEvidence(r.evidence))
      .catch(() => setEvidence([]));
  };

  if (readiness.loading && !readiness.data) return <LoadingBox label="Loading matrix…" />;
  if (readiness.error && !readiness.data) return <ErrorNote error={readiness.error} />;

  const rows = readiness.data?.matrix ?? [];
  const bottleneckId = readiness.data?.bottleneck?.competency_id;

  return (
    <SectionCard
      title={`Competency matrix (${rows.length})`}
      help="Knowledge / application / evidence are tracked separately per competency and never collapsed into one number. Click a row to see the evidence behind its scores. The flagged row is your current bottleneck."
    >
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Competency</TableCell>
            <TableCell>Phase</TableCell>
            <TableCell>Dimension</TableCell>
            <TableCell align="right">Knowledge</TableCell>
            <TableCell align="right">Application</TableCell>
            <TableCell align="right">Evidence</TableCell>
            <TableCell>State</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((r) => (
            <TableRow
              key={r.competency_id}
              hover
              sx={{ cursor: 'pointer', bgcolor: r.competency_id === bottleneckId ? 'rgba(230,81,0,0.12)' : undefined }}
              onClick={() => openRow(r)}
            >
              <TableCell>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {r.name}
                  {r.competency_id === bottleneckId ? (
                    <Chip size="small" label="BOTTLENECK" sx={{ ml: 1, height: 16, fontSize: 9, bgcolor: '#e65100', color: '#ffe0b2', fontWeight: 800 }} />
                  ) : null}
                </Typography>
                <Typography variant="caption" sx={{ color: '#5c6873', fontFamily: 'monospace' }}>
                  {r.competency_id}
                </Typography>
              </TableCell>
              <TableCell>{r.phase}</TableCell>
              <TableCell>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  {r.dimension}
                </Typography>
              </TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{fmtScore(r.knowledge_score)}</TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{fmtScore(r.application_score)}</TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{fmtScore(r.evidence_score)}</TableCell>
              <TableCell>
                <ReadinessChip state={r.readiness_state} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Dialog open={selected !== null} onClose={() => setSelected(null)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selected?.name}
          <Typography variant="caption" sx={{ display: 'block', color: '#8a949e', fontFamily: 'monospace' }}>
            {selected?.competency_id} · prerequisites: {selected?.prerequisites.join(', ') || 'none'}
          </Typography>
        </DialogTitle>
        <DialogContent>
          {selected ? (
            <>
              <Typography variant="body2" sx={{ mb: 1.5 }}>
                knowledge <b>{fmtScore(selected.knowledge_score)}</b> · application{' '}
                <b>{fmtScore(selected.application_score)}</b> · evidence{' '}
                <b>{fmtScore(selected.evidence_score)}</b> — <ReadinessChip state={selected.readiness_state} />
              </Typography>
              <Typography variant="caption" sx={{ color: '#4fc3f7', fontWeight: 700 }}>
                EVIDENCE BEHIND THIS SCORE
              </Typography>
              {evidence === null ? (
                <LoadingBox label="Loading evidence…" />
              ) : evidence.length === 0 ? (
                <Typography variant="body2" sx={{ color: '#8a949e', mt: 1 }}>
                  No evidence yet — every score requires evidence, which is why this competency is
                  still {selected.readiness_state.replace(/_/g, ' ')}.
                </Typography>
              ) : (
                evidence.map((e) => (
                  <Box key={e.evidence_id} sx={{ my: 1, p: 1, bgcolor: '#12171d', borderRadius: 1, border: '1px solid #232a31' }}>
                    <Typography variant="caption" sx={{ color: '#aab4be', fontWeight: 700 }}>
                      {e.artifact_type.replace(/_/g, ' ')} · score {e.score}/5 · {new Date(e.timestamp).toLocaleString()}
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#cfd8dc' }}>
                      {e.summary}
                    </Typography>
                    {e.quotes.slice(0, 2).map((q, i) => (
                      <Typography key={i} variant="caption" sx={{ display: 'block', fontStyle: 'italic', color: '#8a949e' }}>
                        “{q}”
                      </Typography>
                    ))}
                  </Box>
                ))
              )}
            </>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => go('practice', null, selected?.competency_id ?? null)}>Practice this</Button>
          <Button onClick={() => setSelected(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </SectionCard>
  );
}
