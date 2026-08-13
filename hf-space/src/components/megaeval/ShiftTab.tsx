/**
 * Distribution shift panel: cohorts whose share of the evaluation population
 * diverges from the training mix, annotated with the recall gap so a shift
 * that also hurts quality stands out.
 */
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { fmtCompact, getRunShift } from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, LoadingBox, SectionCard, fmtPct } from '../labeleval/shared';
import { HeadCell } from '../help/InfoTip';

function gapColor(gap: number | null): string | undefined {
  if (gap === null) return undefined;
  const g = Math.abs(gap);
  if (g >= 0.1) return '#ef5350';
  if (g >= 0.05) return '#ffa726';
  return undefined;
}

export default function ShiftTab({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const shift = usePoll(() => getRunShift(runId), null, [runId, refreshKey]);
  const data = shift.data;

  return (
    <SectionCard title="Distribution shift (train mix vs evaluation population)" helpTerm="distribution_shift">
      {shift.error ? <ErrorNote error={shift.error} /> : null}
      {shift.loading && !data ? <LoadingBox /> : null}
      {data ? (
        <>
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
            {data.method} · thresholds: min eval count {fmtCompact(data.thresholds.min_eval_count)}, relative change ≥{' '}
            {fmtPct(data.thresholds.rel_threshold)}
          </Typography>
          {data.shifts.length === 0 ? (
            <Typography variant="body2" sx={{ color: '#8a949e' }}>
              No significant distribution shifts detected.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>
                    <HeadCell label="Cohort" term="cohort" />
                  </TableCell>
                  <TableCell align="right">
                    <HeadCell
                      label="Train → eval share"
                      title="Train → eval share"
                      detail="This cohort's share of the training corpus vs its share of the evaluation population. A jump means the model is being asked about conditions it rarely saw during training."
                    />
                  </TableCell>
                  <TableCell align="right">
                    <HeadCell
                      label="Relative change"
                      title="Relative change"
                      detail="(eval share − train share) / train share. Only cohorts above the relative-change threshold with enough eval objects are listed."
                    />
                  </TableCell>
                  <TableCell align="right">Eval count</TableCell>
                  <TableCell align="right">
                    <HeadCell
                      label="Cohort recall vs overall"
                      title="Cohort recall vs overall"
                      detail="Recall inside this cohort compared to the run's overall recall — shows whether the shift actually hurts quality."
                    />
                  </TableCell>
                  <TableCell align="right">
                    <HeadCell
                      label="Recall gap"
                      title="Recall gap"
                      detail="Cohort recall minus overall recall, in points. Strongly negative + over-represented = highest-priority data collection target (highlighted rows)."
                    />
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.shifts.map((s) => {
                  const color = gapColor(s.recall_gap);
                  return (
                    <TableRow key={s.cohort} hover sx={color === '#ef5350' ? { bgcolor: '#2a1214' } : undefined}>
                      <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>{s.cohort}</TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {fmtPct(s.train_share)} → {fmtPct(s.eval_share)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontFamily: 'monospace', color: s.relative_change > 0 ? '#ffa726' : '#4fc3f7' }}
                      >
                        {s.relative_change > 0 ? '+' : ''}
                        {Math.round(s.relative_change * 100)}%
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {fmtCompact(s.eval_count)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {fmtPct(s.cohort_recall)} vs {fmtPct(s.overall_recall)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace', fontWeight: 700, color }}>
                        {s.recall_gap === null ? '—' : `${(s.recall_gap * 100).toFixed(1)} pts`}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              Highlighted rows are cohorts that are over-represented vs training <em>and</em> underperforming on
              recall — prime candidates for targeted data collection.
            </Typography>
          </Box>
        </>
      ) : null}
    </SectionCard>
  );
}
