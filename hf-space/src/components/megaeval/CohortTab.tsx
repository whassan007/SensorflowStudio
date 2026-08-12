/**
 * Cohort explorer: pick a dimension tab, click a cohort value to drill —
 * the value becomes a filter breadcrumb (e.g. Pedestrian → Night → Occluded)
 * and the next level groups by the newly selected dimension. Every node shows
 * cube metrics via POST /api/evaluations/query. "Why?" runs the factor
 * decomposition for the current filter set.
 */
import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Tab from '@mui/material/Tab';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import { HelpCircle, ChevronRight } from 'lucide-react';
import type { DimName, WhyResponse } from '../../types/megaeval';
import { DIM_NAMES } from '../../types/megaeval';
import { fmtCompact, queryEvaluations, whyAnalysis } from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, HBar, LoadingBox, SectionCard, fmtNum, fmtPct } from '../labeleval/shared';
import { PctBarCell, QueryBadge, rowNum, rowStr } from './shared';

interface Crumb {
  dim: DimName;
  value: string;
}

function crumbsToFilters(crumbs: Crumb[]): Partial<Record<DimName, string[]>> {
  const filters: Partial<Record<DimName, string[]>> = {};
  for (const c of crumbs) {
    filters[c.dim] = [...(filters[c.dim] ?? []), c.value];
  }
  return filters;
}

function WhyPanel({ why }: { why: WhyResponse }) {
  const maxCohortN = Math.max(1, ...why.top_cohorts.map((c) => c.n));
  return (
    <SectionCard
      title={`Why is ${why.metric} degraded here?`}
      sx={{ flex: '1 1 340px', borderColor: '#4a3b12' }}
    >
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
        {fmtCompact(why.failure_count)} failures decomposed · {why.method}
      </Typography>
      {why.factors.map((f) => (
        <HBar
          key={f.factor}
          label={f.factor.replace(/_/g, ' ')}
          value={f.share <= 1 ? f.share * 100 : f.share}
          max={100}
          color="#ffd54f"
          valueLabel={`${fmtPct(f.share)} (${fmtCompact(f.count)})`}
        />
      ))}
      {why.top_cohorts.length ? (
        <>
          <Typography variant="caption" sx={{ color: '#aab4be', display: 'block', mt: 1.5, mb: 0.5 }}>
            Top affected cohorts
          </Typography>
          {why.top_cohorts.map((c) => (
            <HBar
              key={c.cohort}
              label={c.cohort}
              value={c.n}
              max={maxCohortN}
              color="#ef5350"
              valueLabel={`n=${fmtCompact(c.n)}${c.fn !== undefined && c.fn !== null ? ` · fn=${fmtCompact(c.fn)}` : ''}${
                c.fp !== undefined && c.fp !== null ? ` · fp=${fmtCompact(c.fp)}` : ''
              }`}
            />
          ))}
        </>
      ) : null}
    </SectionCard>
  );
}

export default function CohortTab({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const [crumbs, setCrumbs] = useState<Crumb[]>([]);
  const [dim, setDim] = useState<DimName>('class');
  const [why, setWhy] = useState<WhyResponse | null>(null);
  const [whyBusy, setWhyBusy] = useState(false);
  const [whyError, setWhyError] = useState<string | null>(null);

  const filters = useMemo(() => crumbsToFilters(crumbs), [crumbs]);
  const filtersKey = JSON.stringify(filters);

  const query = usePoll(
    () =>
      queryEvaluations({
        evaluation_id: runId,
        filters,
        group_by: [dim],
        metrics: ['n', 'recall', 'precision', 'mean_iou', 'anomaly_rate'],
      }),
    null,
    [runId, refreshKey, dim, filtersKey]
  );

  const usedDims = new Set(crumbs.map((c) => c.dim));

  const drillInto = (value: string) => {
    const next = [...crumbs, { dim, value }];
    setCrumbs(next);
    setWhy(null);
    // Auto-advance the group-by dimension to the next unused one.
    const nextUsed = new Set(next.map((c) => c.dim));
    const nextDim = DIM_NAMES.find((d) => !nextUsed.has(d));
    if (nextDim) setDim(nextDim);
  };

  const removeCrumb = (idx: number) => {
    setCrumbs((prev) => prev.filter((_, i) => i !== idx));
    setWhy(null);
  };

  const runWhy = async () => {
    setWhyBusy(true);
    setWhyError(null);
    try {
      setWhy(await whyAnalysis({ run_id: runId, filters, metric: 'recall' }));
    } catch (err) {
      setWhyError(err instanceof Error ? err.message : String(err));
    } finally {
      setWhyBusy(false);
    }
  };

  const rows = useMemo(
    () =>
      query.data
        ? // usePoll keeps the previous response while a new query is in flight;
          // drop rows that don't carry the currently-selected dimension so stale
          // groupings never render (and never collide on duplicate row keys).
          query.data.rows
            .filter((r) => typeof r[dim] === 'string')
            .sort((a, b) => b.n - a.n)
        : [],
    [query.data, dim]
  );

  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <SectionCard
        title="Cohort explorer"
        action={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <QueryBadge meta={query.data?.meta} />
            <Button
              size="small"
              variant="outlined"
              color="warning"
              startIcon={whyBusy ? <CircularProgress size={14} color="inherit" /> : <HelpCircle size={14} />}
              disabled={crumbs.length === 0 || whyBusy}
              onClick={() => void runWhy()}
            >
              Why?
            </Button>
          </Box>
        }
        sx={{ flex: '2 1 560px' }}
      >
        {/* Filter breadcrumb */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
          <Chip
            size="small"
            label="All objects"
            onClick={() => {
              setCrumbs([]);
              setWhy(null);
            }}
            sx={{ height: 22, bgcolor: crumbs.length ? '#232a31' : '#12303f', color: '#81d4fa', fontWeight: 700 }}
          />
          {crumbs.map((c, i) => (
            <Box key={`${c.dim}-${c.value}-${i}`} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <ChevronRight size={13} color="#8a949e" />
              <Chip
                size="small"
                label={`${c.dim}: ${c.value}`}
                onDelete={() => removeCrumb(i)}
                sx={{ height: 22, bgcolor: '#12303f', color: '#81d4fa', fontFamily: 'monospace', fontSize: 11 }}
              />
            </Box>
          ))}
        </Box>

        {/* Dimension tabs */}
        <Tabs
          value={dim}
          onChange={(_, v: DimName) => setDim(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ minHeight: 32, mb: 1, '& .MuiTab-root': { minHeight: 32, py: 0.5, fontSize: 12 } }}
        >
          {DIM_NAMES.map((d) => (
            <Tab key={d} label={d.replace(/_/g, ' ')} value={d} disabled={usedDims.has(d)} />
          ))}
        </Tabs>

        {query.error ? <ErrorNote error={query.error} /> : null}
        {query.loading && !query.data ? <LoadingBox /> : null}
        {query.data ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{dim.replace(/_/g, ' ')}</TableCell>
                <TableCell align="right">n</TableCell>
                <TableCell>Recall</TableCell>
                <TableCell>Precision</TableCell>
                <TableCell align="right">Mean IoU</TableCell>
                <TableCell align="right">Anomaly rate</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row, i) => {
                const value = rowStr(row, dim);
                return (
                  <TableRow
                    key={`${value}-${i}`}
                    hover
                    onClick={() => drillInto(value)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell sx={{ fontWeight: 600, color: '#4fc3f7' }}>{value}</TableCell>
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
                      {fmtNum(rowNum(row, 'mean_iou'), 3)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {fmtPct(rowNum(row, 'anomaly_rate'))}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        ) : null}
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
          Click a row to drill into that cohort; the next level groups by the selected dimension tab.
        </Typography>
      </SectionCard>

      {whyError ? (
        <Box sx={{ flex: '1 1 340px' }}>
          <ErrorNote error={whyError} />
        </Box>
      ) : null}
      {why ? <WhyPanel why={why} /> : null}
    </Box>
  );
}
