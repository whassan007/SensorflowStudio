/**
 * Model-vs-model comparison: candidate (selected run) against the baseline
 * chosen in the header. Headline deltas, per-class deltas, top regressions,
 * and the PROMOTE / DO_NOT_PROMOTE recommendation with blockers.
 */
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { compareRuns, fmtCompact } from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, LoadingBox, SectionCard, fmtPct } from '../labeleval/shared';
import { ExplainTip, InfoDot } from '../help/InfoTip';
import { DeltaText } from './shared';

/** Metrics where a positive delta is a regression rather than an improvement. */
const LOWER_IS_BETTER = new Set(['anomaly_rate', 'error_rate', 'fp', 'fn', 'loc_err', 'anomalies']);

function fmtMetricValue(v: number): string {
  // Rates arrive as fractions; counts as integers > 1.
  return Math.abs(v) <= 1 ? fmtPct(v) : fmtCompact(v);
}

export default function CompareTab({
  runId,
  baselineRunId,
  refreshKey,
}: {
  runId: string;
  baselineRunId: string | null;
  refreshKey: number;
}) {
  const cmp = usePoll(
    () =>
      baselineRunId
        ? compareRuns({ candidate_run_id: runId, baseline_run_id: baselineRunId })
        : Promise.resolve(null),
    null,
    [runId, baselineRunId, refreshKey]
  );

  if (!baselineRunId) {
    return (
      <Alert severity="info" variant="outlined">
        Select a <strong>baseline run</strong> in the header to compare the current candidate against it.
      </Alert>
    );
  }
  if (cmp.error) return <ErrorNote error={cmp.error} />;
  if (cmp.loading && !cmp.data) return <LoadingBox label="Comparing runs…" />;
  const data = cmp.data;
  if (!data) return null;

  const promote = data.recommendation === 'PROMOTE';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Alert severity={promote ? 'success' : 'error'} variant="filled" sx={{ fontWeight: 700 }}>
        <AlertTitle sx={{ fontWeight: 800, display: 'flex', alignItems: 'center' }}>
          <ExplainTip term="do_not_promote">
            <Box component="span" sx={{ borderBottom: '1px dotted rgba(255,255,255,0.6)', cursor: 'help' }}>
              {promote ? 'PROMOTE' : 'DO NOT PROMOTE'}
            </Box>
          </ExplainTip>
          <Box component="span" sx={{ ml: 0.75 }}>
            — {data.candidate.model_version} vs baseline {data.baseline.model_version}
          </Box>
        </AlertTitle>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.5, opacity: 0.85 }}>
          {promote
            ? 'Every policy rule passed: no headline or safety metric dropped beyond tolerance and no cohort regressed materially.'
            : 'Each line below is a violated promotion-policy rule, with the measured drop and the maximum the policy allows.'}
        </Typography>
        {data.blockers.length ? (
          <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
            {data.blockers.map((b, i) => (
              <li key={i}>
                <Typography variant="body2">{b}</Typography>
              </li>
            ))}
          </Box>
        ) : (
          <Typography variant="body2">No blockers under the current promotion policy.</Typography>
        )}
      </Alert>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <SectionCard
          title="Headline deltas"
          help="Population-level metric comparison: baseline value, candidate value, and the delta (green = better, red = worse; direction-aware, e.g. lower anomaly rate is good). Hover metric names for definitions."
          sx={{ flex: '1 1 380px' }}
        >
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Metric</TableCell>
                <TableCell align="right">Baseline</TableCell>
                <TableCell align="right">Candidate</TableCell>
                <TableCell align="right">Δ</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.headline_deltas.map((d) => (
                <TableRow key={d.metric} hover>
                  <TableCell sx={{ fontFamily: 'monospace' }}>{d.metric}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtMetricValue(d.baseline)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtMetricValue(d.candidate)}
                  </TableCell>
                  <TableCell align="right">
                    <DeltaText
                      delta={d.delta}
                      higherIsBetter={!LOWER_IS_BETTER.has(d.metric)}
                      asPct={Math.abs(d.baseline) <= 1 && Math.abs(d.candidate) <= 1}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </SectionCard>

        <SectionCard
          title="Per-class deltas"
          help="The same comparison broken out per object class — a stable global average can hide a collapsed class. Vulnerable-road-user classes (pedestrian, cyclist, motorcycle) deserve the closest look."
          sx={{ flex: '1 1 480px' }}
        >
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Class</TableCell>
                <TableCell align="right">n</TableCell>
                <TableCell align="right">Recall</TableCell>
                <TableCell align="right">Δ recall</TableCell>
                <TableCell align="right">Precision</TableCell>
                <TableCell align="right">Δ precision</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.per_class.map((c) => (
                <TableRow key={c.class} hover>
                  <TableCell sx={{ fontWeight: 600 }}>{c.class}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtCompact(c.n)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(c.recall_baseline)} → {fmtPct(c.recall_candidate)}
                  </TableCell>
                  <TableCell align="right">
                    <DeltaText delta={c.recall_delta} />
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(c.precision_baseline)} → {fmtPct(c.precision_candidate)}
                  </TableCell>
                  <TableCell align="right">
                    <DeltaText delta={c.precision_delta} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </SectionCard>
      </Box>

      <SectionCard
        title={
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
            {`Top regressions (${data.regressions.length})`}
            <InfoDot term="regression_detected" size={13} />
          </Box>
        }
      >
        {data.regressions.length === 0 ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No cohort regressions detected against the baseline.
          </Typography>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
            {data.regressions.map((r) => (
              <Box
                key={r.cohort}
                sx={{
                  p: 1,
                  border: '1px solid #5c1a1a',
                  bgcolor: '#2a1214',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  flexWrap: 'wrap',
                }}
              >
                <ExplainTip term="regression_detected">
                  <Typography variant="body2" sx={{ fontWeight: 700, color: '#ef9a9a', cursor: 'help' }}>
                    🚨 REGRESSION:
                  </Typography>
                </ExplainTip>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {r.cohort} recall {fmtPct(r.recall_baseline)} → {fmtPct(r.recall_candidate)} (Δ{' '}
                  <DeltaText delta={r.recall_delta} />)
                </Typography>
                <Typography variant="caption" sx={{ color: '#8a949e', ml: 'auto', fontFamily: 'monospace' }}>
                  n={fmtCompact(r.n)}
                </Typography>
              </Box>
            ))}
          </Box>
        )}
      </SectionCard>

      {data.worst_cohorts.length ? (
        <SectionCard
          title="Worst cohorts (candidate)"
          help="The candidate's lowest-recall cohorts with enough objects to matter, regardless of whether they regressed — chronic weak spots as opposed to new ones."
        >
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Cohort</TableCell>
                <TableCell align="right">n</TableCell>
                <TableCell align="right">Recall (baseline → candidate)</TableCell>
                <TableCell align="right">Δ</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.worst_cohorts.map((c) => (
                <TableRow key={c.cohort} hover>
                  <TableCell sx={{ fontFamily: 'monospace' }}>{c.cohort}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtCompact(c.n)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(c.recall_baseline)} → {fmtPct(c.recall_candidate)}
                  </TableCell>
                  <TableCell align="right">
                    <DeltaText delta={c.recall_delta} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </SectionCard>
      ) : null}
    </Box>
  );
}
