/**
 * Multi-model compare panel (A vs B vs C) — wraps /api/models/compare.
 */
import { useMemo } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { compareModels } from '../../services/platform';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, LoadingBox, SectionCard, fmtPct } from '../labeleval/shared';

export default function ModelComparePanel({
  runId,
  baselineRunId,
  extraRunIds = [],
  refreshKey = 0,
}: {
  runId: string | null;
  baselineRunId: string | null;
  extraRunIds?: string[];
  refreshKey?: number;
}) {
  const runIds = useMemo(() => {
    const ids = [baselineRunId, runId, ...extraRunIds].filter(Boolean) as string[];
    return [...new Set(ids)];
  }, [runId, baselineRunId, extraRunIds]);

  const cmp = usePoll(
    () =>
      runIds.length >= 2
        ? compareModels(runIds, baselineRunId)
        : Promise.resolve(null),
    null,
    [runIds.join(','), baselineRunId, refreshKey]
  );

  if (runIds.length < 2) {
    return (
      <Alert severity="info" variant="outlined">
        Select a baseline run (and optionally more candidates) to compare models A / B / C.
      </Alert>
    );
  }
  if (cmp.error && !cmp.data) return <ErrorNote error={cmp.error} />;
  if (cmp.loading && !cmp.data) return <LoadingBox label="Comparing models…" />;
  const data = cmp.data;
  if (!data) return null;

  const modelNames = data.models.map((m) => m.model_version);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title="Model compare (A / B / C)"
        action={
          <Typography variant="caption" color="text.secondary">
            Baseline {data.baseline_run_id.slice(0, 8)}…
          </Typography>
        }
      >
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
          {data.recommendations.map((r) => (
            <Alert
              key={r.candidate}
              severity={r.recommendation === 'PROMOTE' ? 'success' : 'warning'}
              variant="outlined"
              sx={{ py: 0 }}
            >
              {r.candidate}: {r.recommendation}
              {r.blocker_count ? ` (${r.blocker_count} blockers)` : ''}
            </Alert>
          ))}
        </Box>

        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Metric</TableCell>
              {modelNames.map((m) => (
                <TableCell key={m} align="right">
                  {m}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.metric_matrix.map((row) => (
              <TableRow key={row.metric}>
                <TableCell>{row.metric}</TableCell>
                {modelNames.map((m) => {
                  const v = row[m] as number | null | undefined;
                  const d = row.deltas_vs_baseline?.[m];
                  return (
                    <TableCell key={m} align="right">
                      {fmtPct(v ?? null)}
                      {d !== undefined && d !== null ? (
                        <Typography component="span" variant="caption" sx={{ ml: 0.5, color: d < 0 ? '#c62828' : '#2e7d32' }}>
                          {d >= 0 ? '+' : ''}
                          {(d * 100).toFixed(1)}pp
                        </Typography>
                      ) : null}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionCard>
    </Box>
  );
}
