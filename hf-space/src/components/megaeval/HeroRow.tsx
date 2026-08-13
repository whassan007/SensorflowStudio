/**
 * Hero aggregates for a published run: population-scale counts followed by the
 * headline quality metrics. Precision/recall are annotated with sampling-verified
 * confidence intervals when review results exist.
 */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import type { EvaluationRunInfo, MetricEstimate, ReviewState } from '../../types/megaeval';
import { fmtCompact, queryEvaluations } from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { MetricCard, fmtNum, fmtPct } from '../labeleval/shared';
import { QueryBadge, rowNum } from './shared';
import { ExplainTip } from '../help/InfoTip';

function ciSub(est: MetricEstimate | undefined): string | null {
  if (!est) return null;
  return `95% CI ${fmtPct(est.ci_low)}–${fmtPct(est.ci_high)}, n=${est.n_reviewed.toLocaleString()} reviewed`;
}

const VERIFIED_CHIP = (
  <ExplainTip term="sampling_verified">
    <Chip
      size="small"
      label="sampling-verified"
      sx={{ height: 16, fontSize: 9, fontWeight: 700, bgcolor: '#12303f', color: '#81d4fa', ml: 0.5, cursor: 'help' }}
    />
  </ExplainTip>
);

export default function HeroRow({
  run,
  review,
  refreshKey,
}: {
  run: EvaluationRunInfo;
  review: ReviewState | null;
  refreshKey: number;
}) {
  const h = run.headline;
  // Reviewed / verified totals come from the aggregate query API (also shows provenance).
  const totals = usePoll(
    () =>
      queryEvaluations({
        evaluation_id: run.run_id,
        metrics: ['n', 'reviewed', 'verified', 'review_coverage'],
      }),
    null,
    [run.run_id, refreshKey]
  );
  const totalsRow = totals.data?.rows[0];
  const reviewed = totalsRow ? rowNum(totalsRow, 'reviewed') : null;
  const verified = totalsRow ? rowNum(totalsRow, 'verified') : null;

  const results = review?.results ?? null;
  const precisionCi = ciSub(results?.precision);
  const recallCi = ciSub(results?.recall);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        <MetricCard label="Objects evaluated" value={fmtCompact(h.n)} sub="exact (cube)" term="metric_cube" />
        <MetricCard
          label="Containers"
          term="container"
          value={fmtCompact(h.containers)}
          sub={
            h.containers_hll_estimate !== undefined ? (
              <ExplainTip term="hll">
                <Box component="span" sx={{ borderBottom: '1px dotted #5c6873', cursor: 'help' }}>
                  {`HLL est ${fmtCompact(h.containers_hll_estimate)} (approx)`}
                </Box>
              </ExplainTip>
            ) : (
              'exact'
            )
          }
        />
        <MetricCard
          label="Reviewed"
          term="review_sample"
          value={fmtCompact(reviewed)}
          sub={
            <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
              <span>human reviews</span>
              <QueryBadge meta={totals.data?.meta} />
            </Box>
          }
          accent="#42a5f5"
        />
        <MetricCard label="Verified" value={fmtCompact(verified)} accent="#66bb6a" sub="human-verified" term="status_verified" />
      </Box>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        <MetricCard
          label="Precision"
          term="precision"
          value={fmtPct(h.precision)}
          sub={
            precisionCi ? (
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', flexWrap: 'wrap' }}>
                <span>
                  {fmtPct(results?.precision.estimate)} ({precisionCi})
                </span>
                {VERIFIED_CHIP}
              </Box>
            ) : (
              'point estimate'
            )
          }
        />
        <MetricCard
          label="Recall"
          term="recall"
          value={fmtPct(h.recall)}
          sub={
            recallCi ? (
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', flexWrap: 'wrap' }}>
                <span>
                  {fmtPct(results?.recall.estimate)} ({recallCi})
                </span>
                {VERIFIED_CHIP}
              </Box>
            ) : (
              'point estimate'
            )
          }
        />
        <MetricCard label="F1" value={fmtPct(h.f1)} term="f1" />
        <MetricCard label="Mean IoU" value={fmtNum(h.mean_iou, 3)} term="iou_3d" />
        <MetricCard label="Anomaly rate" value={fmtPct(h.anomaly_rate)} accent="#ffa726" term="anomaly_rate" />
        <MetricCard label="Safety recall" value={fmtPct(h.safety_recall)} accent="#ef5350" term="safety_recall" />
      </Box>
    </Box>
  );
}
