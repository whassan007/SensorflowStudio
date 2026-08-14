/**
 * Quality tab: per-class quality bars, error-type distribution, the quality
 * funnel, cross-run quality trend, and sketch-based distributions.
 * Every aggregate-query panel surfaces its provenance badge.
 */
import { useMemo } from 'react';
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { ErrorType, EvaluationRunInfo } from '../../types/megaeval';
import {
  fmtCompact,
  getRunDistributions,
  getRunFunnel,
  queryEvaluations,
  searchErrors,
} from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, HBar, LoadingBox, SectionCard, fmtNum, fmtPct } from '../labeleval/shared';
import { HeadCell } from '../help/InfoTip';
import {
  ERROR_TYPE_COLORS,
  ExactnessTag,
  HistogramSparkline,
  PctBarCell,
  QueryBadge,
  rowNum,
  rowStr,
} from './shared';

// ---------------------------------------------------------------- per-class quality

function QualityByClass({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const q = usePoll(
    () =>
      queryEvaluations({
        evaluation_id: runId,
        group_by: ['class'],
        metrics: ['n', 'precision', 'recall', 'f1', 'mean_iou'],
      }),
    null,
    [runId, refreshKey]
  );
  return (
    <SectionCard
      title="Quality by class"
      help="Headline metrics broken down per object class, aggregated from the metric cube. Classes with markedly lower recall than the population are where the model under-detects — start drilling there."
      action={<QueryBadge meta={q.data?.meta} />}
      sx={{ flex: '1 1 460px' }}
    >
      {q.error ? <ErrorNote error={q.error} /> : null}
      {q.loading && !q.data ? <LoadingBox /> : null}
      {q.data ? (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Class</TableCell>
              <TableCell align="right">
                <HeadCell label="n" title="n" detail="Number of evaluated objects of this class in the population (exact cube count)." />
              </TableCell>
              <TableCell>
                <HeadCell label="Recall" term="recall" />
              </TableCell>
              <TableCell>
                <HeadCell label="Precision" term="precision" />
              </TableCell>
              <TableCell align="right">
                <HeadCell label="F1" term="f1" />
              </TableCell>
              <TableCell align="right">
                <HeadCell label="Mean IoU" term="iou_3d" />
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {[...q.data.rows]
              .sort((a, b) => b.n - a.n)
              .map((row) => (
                <TableRow key={rowStr(row, 'class')} hover>
                  <TableCell sx={{ fontWeight: 600 }}>{rowStr(row, 'class')}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtCompact(row.n)}
                  </TableCell>
                  <TableCell>
                    <PctBarCell value={rowNum(row, 'recall')} color="#66bb6a" />
                  </TableCell>
                  <TableCell>
                    <PctBarCell value={rowNum(row, 'precision')} color="#4fc3f7" />
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(rowNum(row, 'f1'))}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtNum(rowNum(row, 'mean_iou'))}
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      ) : null}
    </SectionCard>
  );
}

// ---------------------------------------------------------------- error distribution

const ERROR_TYPE_ORDER: ErrorType[] = ['FN', 'FP', 'LOCALIZATION', 'ANOMALY', 'LOW_CONF'];

function ErrorDistribution({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const res = usePoll(() => searchErrors({ run_id: runId }), null, [runId, refreshKey]);
  const byType = res.data?.by_type ?? {};
  const total = ERROR_TYPE_ORDER.reduce((acc, t) => acc + (byType[t] ?? 0), 0);
  const max = Math.max(1, ...ERROR_TYPE_ORDER.map((t) => byType[t] ?? 0));
  const ERROR_TERM_KEYS: Record<ErrorType, string> = {
    FN: 'error_fn',
    FP: 'error_fp',
    LOCALIZATION: 'error_localization',
    ANOMALY: 'reason_anomaly',
    LOW_CONF: 'error_low_conf',
  };
  return (
    <SectionCard
      title="Error distribution"
      helpTerm="error_index"
      sx={{ flex: '1 1 320px' }}
    >
      {res.error ? <ErrorNote error={res.error} /> : null}
      {res.loading && !res.data ? <LoadingBox /> : null}
      {res.data ? (
        <>
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
            {fmtCompact(res.data.matched_errors)} potential errors indexed (exact)
          </Typography>
          {ERROR_TYPE_ORDER.map((t) => {
            const count = byType[t] ?? 0;
            return (
              <HBar
                key={t}
                label={t}
                term={ERROR_TERM_KEYS[t]}
                value={count}
                max={max}
                color={ERROR_TYPE_COLORS[t]}
                valueLabel={`${fmtCompact(count)} (${total > 0 ? fmtPct(count / total) : '—'})`}
              />
            );
          })}
        </>
      ) : null}
    </SectionCard>
  );
}

// ---------------------------------------------------------------- quality funnel

function QualityFunnel({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const funnel = usePoll(() => getRunFunnel(runId), null, [runId, refreshKey]);
  const stages = funnel.data?.stages ?? [];
  const max = stages.length ? Math.max(...stages.map((s) => s.count)) : 1;
  const est = funnel.data?.estimated_precision ?? null;
  return (
    <SectionCard
      title="Quality funnel"
      help="How the population narrows toward human verification: Objects Evaluated → Containers → Potential Errors (error index) → High-Confidence Errors → Review Sample → Human Verified. Each stage shows its count and share of the population; the precision estimate at the bottom carries a Wilson 95% CI from the review sample."
      sx={{ flex: '1 1 420px' }}
    >
      {funnel.error ? <ErrorNote error={funnel.error} /> : null}
      {funnel.loading && !funnel.data ? <LoadingBox /> : null}
      {funnel.data ? (
        <>
          {stages.map((s) => (
            <HBar
              key={s.stage}
              label={s.stage}
              value={s.count}
              max={max}
              color="#4fc3f7"
              valueLabel={`${fmtCompact(s.count)} (${fmtPct(s.pct_of_population)})`}
            />
          ))}
          {est && est.estimate !== null ? (
            <Typography variant="body2" sx={{ mt: 1.5, color: '#a5d6a7' }}>
              Estimated precision {fmtPct(est.estimate)}
              {est.ci_low !== null && est.ci_high !== null
                ? ` (95% CI ${fmtPct(est.ci_low)}–${fmtPct(est.ci_high)}${
                    est.n_reviewed !== null ? `, n=${est.n_reviewed.toLocaleString()} reviewed` : ''
                  })`
                : ''}
            </Typography>
          ) : null}
        </>
      ) : null}
    </SectionCard>
  );
}

// ---------------------------------------------------------------- quality trend

const TREND_SERIES: Array<{ key: 'precision' | 'recall' | 'safety_recall'; label: string; color: string }> = [
  { key: 'precision', label: 'precision', color: '#4fc3f7' },
  { key: 'recall', label: 'recall', color: '#66bb6a' },
  { key: 'safety_recall', label: 'safety recall', color: '#ffa726' },
];

function QualityTrend({ runs }: { runs: EvaluationRunInfo[] }) {
  const published = useMemo(
    () =>
      runs
        .filter((r) => r.status === 'published')
        .sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? '')),
    [runs]
  );

  const width = 520;
  const height = 150;
  const padL = 40;
  const padB = 30;
  const padT = 10;

  const values = published.flatMap((r) =>
    TREND_SERIES.map((s) => r.headline[s.key]).filter((v): v is number => v !== null && v !== undefined)
  );
  const lo = values.length ? Math.max(0, Math.min(...values) - 0.05) : 0;
  const hi = values.length ? Math.min(1, Math.max(...values) + 0.02) : 1;
  const xFor = (i: number) =>
    padL + (published.length <= 1 ? (width - padL) / 2 : (i / (published.length - 1)) * (width - padL - 10));
  const yFor = (v: number) => padT + (1 - (v - lo) / Math.max(1e-9, hi - lo)) * (height - padT - padB);

  return (
    <SectionCard
      title="Quality trend across published runs"
      help="Headline precision / recall / safety recall for every published evaluation run on this population, ordered by creation time. A dip in the newest point is exactly what the Compare tab's promotion policy guards against."
      sx={{ flex: '1 1 460px' }}
    >
      {published.length < 2 ? (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          {published.length === 0
            ? 'No published runs yet.'
            : 'Only one published run — launch another run to see the trend.'}
        </Typography>
      ) : null}
      {published.length >= 1 ? (
        <>
          <svg width={width} height={height}>
            {[lo, (lo + hi) / 2, hi].map((v, i) => (
              <g key={i}>
                <line x1={padL} x2={width - 5} y1={yFor(v)} y2={yFor(v)} stroke="#232a31" />
                <text x={2} y={yFor(v) + 3} fontSize={9} fill="#8a949e" fontFamily="monospace">
                  {fmtPct(v)}
                </text>
              </g>
            ))}
            {TREND_SERIES.map((s) => {
              const pts = published
                .map((r, i) => ({ v: r.headline[s.key], i }))
                .filter((p): p is { v: number; i: number } => p.v !== null && p.v !== undefined);
              if (!pts.length) return null;
              return (
                <g key={s.key}>
                  <polyline
                    fill="none"
                    stroke={s.color}
                    strokeWidth={2}
                    points={pts.map((p) => `${xFor(p.i)},${yFor(p.v)}`).join(' ')}
                  />
                  {pts.map((p) => (
                    <circle key={p.i} cx={xFor(p.i)} cy={yFor(p.v)} r={3} fill={s.color} />
                  ))}
                </g>
              );
            })}
            {published.map((r, i) => (
              <text
                key={r.run_id}
                x={xFor(i)}
                y={height - 6}
                fontSize={9}
                fill="#aab4be"
                textAnchor="middle"
                fontFamily="monospace"
              >
                {r.model_version.length > 14 ? `${r.model_version.slice(0, 13)}…` : r.model_version}
              </text>
            ))}
          </svg>
          <Box sx={{ display: 'flex', gap: 2 }}>
            {TREND_SERIES.map((s) => (
              <Box key={s.key} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 10, height: 3, bgcolor: s.color, borderRadius: 1 }} />
                <Typography variant="caption" sx={{ color: '#aab4be' }}>
                  {s.label}
                </Typography>
              </Box>
            ))}
          </Box>
        </>
      ) : null}
    </SectionCard>
  );
}

// ---------------------------------------------------------------- distributions (sketches)

function DistributionsCard({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const dist = usePoll(() => getRunDistributions(runId), null, [runId, refreshKey]);
  const d = dist.data;
  return (
    <SectionCard
      title={
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>Distributions</span>
          <ExactnessTag approx method="quantile_sketch" />
        </Box>
      }
      help="Population-scale distributions from mergeable sketches: confidence and IoU histograms come from fixed-bin quantile sketches (p10/p50/p90 markers), container cardinality from a HyperLogLog. Approximate by design — exact counts shown alongside where available."
      sx={{ flex: '1 1 320px' }}
    >
      {dist.error ? <ErrorNote error={dist.error} /> : null}
      {dist.loading && !d ? <LoadingBox /> : null}
      {d ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {d.confidence ? (
            <Box>
              <Typography variant="caption" sx={{ color: '#aab4be' }}>
                Confidence — approximate (sketch)
              </Typography>
              <HistogramSparkline sketch={d.confidence} color="#4fc3f7" />
            </Box>
          ) : null}
          {d.iou ? (
            <Box>
              <Typography variant="caption" sx={{ color: '#aab4be' }}>
                IoU — approximate (sketch)
              </Typography>
              <HistogramSparkline sketch={d.iou} color="#66bb6a" />
            </Box>
          ) : null}
          <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  Containers (HLL)
                </Typography>
                <ExactnessTag approx method="hll" />
              </Box>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                {d.containers_hll_estimate === null ? '—' : fmtCompact(d.containers_hll_estimate)}
              </Typography>
            </Box>
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  Containers (exact)
                </Typography>
                <ExactnessTag approx={false} />
              </Box>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                {d.containers_exact === null ? '—' : fmtCompact(d.containers_exact)}
              </Typography>
            </Box>
          </Box>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            {d.note}
          </Typography>
        </Box>
      ) : null}
    </SectionCard>
  );
}

// ---------------------------------------------------------------- tab

export default function QualityTab({
  runId,
  runs,
  refreshKey,
}: {
  runId: string;
  runs: EvaluationRunInfo[];
  refreshKey: number;
}) {
  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <QualityByClass runId={runId} refreshKey={refreshKey} />
      <ErrorDistribution runId={runId} refreshKey={refreshKey} />
      <QualityFunnel runId={runId} refreshKey={refreshKey} />
      <QualityTrend runs={runs} />
      <DistributionsCard runId={runId} refreshKey={refreshKey} />
    </Box>
  );
}
