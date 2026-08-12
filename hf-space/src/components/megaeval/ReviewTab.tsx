/**
 * Review sampling panel: statistically grounded human review. Build a
 * stratified sampling plan, execute (simulated) reviews, then read precision
 * and recall estimates with Wilson confidence intervals per stratum.
 */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { ClipboardList, UserCheck } from 'lucide-react';
import type { MetricEstimate, ReviewState, SamplingFunnel, StratumPlan } from '../../types/megaeval';
import { executeReview, fmtCompact, planReview } from '../../services/megaeval';
import { ErrorNote, HBar, SectionCard, fmtPct } from '../labeleval/shared';

const FUNNEL_STAGES: Array<{ key: keyof SamplingFunnel; label: string }> = [
  { key: 'population_objects', label: 'Population objects' },
  { key: 'containers', label: 'Containers' },
  { key: 'suspicious_containers', label: 'Suspicious containers' },
  { key: 'candidate_pool', label: 'Candidate pool' },
  { key: 'statistically_selected', label: 'Statistically selected' },
  { key: 'reviewed', label: 'Reviewed' },
];

function EstimateCard({ label, est }: { label: string; est: MetricEstimate }) {
  return (
    <SectionCard title={`${label} estimate`} sx={{ flex: '1 1 420px' }}>
      <Typography variant="h5" sx={{ fontWeight: 800, color: '#a5d6a7' }}>
        {fmtPct(est.estimate)}
        <Typography component="span" variant="body2" sx={{ color: '#8a949e', ml: 1 }}>
          95% CI {fmtPct(est.ci_low)}–{fmtPct(est.ci_high)} · n={est.n_reviewed.toLocaleString()} reviewed · frame{' '}
          {fmtCompact(est.frame_size)}
        </Typography>
      </Typography>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
        {est.method}
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Stratum</TableCell>
            <TableCell align="right">N</TableCell>
            <TableCell align="right">n</TableCell>
            <TableCell align="right">p</TableCell>
            <TableCell align="right">Wilson 95% CI</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {est.strata.map((s) => (
            <TableRow key={s.stratum} hover>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{s.stratum}</TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                {fmtCompact(s.N)}
              </TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                {s.n.toLocaleString()}
              </TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                {fmtPct(s.p)}
              </TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                {fmtPct(s.wilson_ci[0])}–{fmtPct(s.wilson_ci[1])}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </SectionCard>
  );
}

function PlanTable({ title, strata }: { title: string; strata: Record<string, StratumPlan> }) {
  const entries = Object.entries(strata);
  return (
    <SectionCard title={title} sx={{ flex: '1 1 340px' }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Stratum</TableCell>
            <TableCell align="right">N</TableCell>
            <TableCell align="right">Allocated</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {entries.map(([key, plan]) => (
            <TableRow key={key} hover>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{plan.label || key}</TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                {fmtCompact(plan.N)}
              </TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                {plan.allocated.toLocaleString()}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </SectionCard>
  );
}

export default function ReviewTab({
  runId,
  review,
  onReview,
}: {
  runId: string;
  review: ReviewState | null;
  onReview: (r: ReviewState) => void;
}) {
  const [busy, setBusy] = useState<'plan' | 'execute' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const act = async (kind: 'plan' | 'execute') => {
    setBusy(kind);
    setError(null);
    try {
      onReview(kind === 'plan' ? await planReview(runId) : await executeReview(runId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const funnel = review?.funnel;
  const max = funnel ? Math.max(1, funnel.population_objects) : 1;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title="Statistical review sampling"
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              variant="outlined"
              startIcon={busy === 'plan' ? <CircularProgress size={14} color="inherit" /> : <ClipboardList size={14} />}
              disabled={busy !== null}
              onClick={() => void act('plan')}
            >
              Build sampling plan
            </Button>
            <Button
              size="small"
              variant="contained"
              startIcon={busy === 'execute' ? <CircularProgress size={14} color="inherit" /> : <UserCheck size={14} />}
              disabled={busy !== null || !review?.planned}
              onClick={() => void act('execute')}
            >
              Execute reviews (simulated)
            </Button>
          </Box>
        }
      >
        {error ? <ErrorNote error={error} /> : null}
        {!review || !review.planned ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No sampling plan yet. Reviewing everything at this scale is impossible — build a stratified plan to
            estimate precision and recall from a small, statistically selected sample.
          </Typography>
        ) : (
          <>
            <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
              {review.method ?? 'stratified sampling'}
              {review.target_n !== undefined ? ` · target n=${review.target_n.toLocaleString()}` : ''}
              {review.executed ? ' · executed' : ' · planned, not executed'}
            </Typography>
            {funnel
              ? FUNNEL_STAGES.map((s) => (
                  <HBar
                    key={s.key}
                    label={s.label}
                    value={funnel[s.key]}
                    max={max}
                    color={s.key === 'reviewed' ? '#66bb6a' : '#4fc3f7'}
                    valueLabel={`${fmtCompact(funnel[s.key])} (${fmtPct(funnel[s.key] / max)})`}
                  />
                ))
              : null}
          </>
        )}
      </SectionCard>

      {review?.planned && !review.executed && review.strata ? (
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <PlanTable title="Precision strata plan" strata={review.strata.precision} />
          <PlanTable title="Recall strata plan" strata={review.strata.recall} />
        </Box>
      ) : null}

      {review?.results ? (
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <EstimateCard label="Precision" est={review.results.precision} />
          <EstimateCard label="Recall" est={review.results.recall} />
        </Box>
      ) : null}
    </Box>
  );
}
