/**
 * Visual Query Builder — an alternate, direct-manipulation front end for the
 * metric-cube query API (/api/evaluations/query).
 *
 * Drag dimension chips from the tray into the "Group by" and "Filter" wells,
 * pick filter values with visual chips, toggle metrics, and watch the live
 * result preview (table + bars) update as you build. The generated request
 * JSON is always visible, so the visual query stays auditable against the
 * manual API.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { GripVertical, X } from 'lucide-react';
import { DIM_NAMES, type DimName, type EvaluationQueryResponse, type MetricName } from '../../types/megaeval';
import { getDimensions, queryEvaluations } from '../../services/megaeval';
import { SectionCard, HBar, fmtPct, fmtInt } from '../../components/labeleval/shared';
import { InfoDot } from '../help/InfoTip';
import { PanelSkeleton, IllustratedEmpty } from '../visual/Feedback';
import DragList from '../visual/DragList';
import { tokens } from '../../theme';

const DIM_COLORS: Record<DimName, string> = {
  class: '#66bb6a',
  weather: '#4fc3f7',
  lighting: '#f9a825',
  road_type: '#ab47bc',
  scenario: '#ff7043',
  sensor: '#26c6da',
  distance_band: '#9ccc65',
  speed_band: '#ec407a',
  occlusion: '#8a949e',
};

const METRIC_CHOICES: Array<{ name: MetricName; label: string; help: string; pct: boolean }> = [
  { name: 'recall', label: 'recall', help: 'TP / (TP+FN) — fraction of real objects labeled.', pct: true },
  { name: 'precision', label: 'precision', help: 'TP / (TP+FP) — fraction of produced labels that are correct.', pct: true },
  { name: 'f1', label: 'f1', help: 'Harmonic mean of precision and recall.', pct: true },
  { name: 'mean_iou', label: 'mean IoU', help: 'Mean 3D overlap of matched boxes.', pct: true },
  { name: 'safety_recall', label: 'safety recall', help: 'Recall restricted to safety-critical objects.', pct: true },
  { name: 'error_rate', label: 'error rate', help: '(FP+FN) / n.', pct: true },
  { name: 'anomaly_rate', label: 'anomaly rate', help: 'Fraction of objects flagged by the anomaly ensemble.', pct: true },
  { name: 'conf_mean', label: 'conf mean', help: 'Mean detection confidence.', pct: false },
];

type Well = 'group' | 'filter';

interface FilterClause {
  dim: DimName;
  values: string[];
}

interface VisualQueryTabProps {
  runId: string;
}

export default function VisualQueryTab({ runId }: VisualQueryTabProps) {
  const [dimValues, setDimValues] = useState<Record<string, string[]> | null>(null);
  const [groupBy, setGroupBy] = useState<DimName[]>(['class']);
  const [filters, setFilters] = useState<FilterClause[]>([]);
  const [metrics, setMetrics] = useState<MetricName[]>(['recall', 'precision']);
  const [result, setResult] = useState<EvaluationQueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOverWell, setDragOverWell] = useState<Well | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getDimensions()
      .then((d) => setDimValues(d.dimensions as Record<string, string[]>))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const request = useMemo(
    () => ({
      evaluation_id: runId,
      group_by: groupBy,
      filters: Object.fromEntries(filters.filter((f) => f.values.length).map((f) => [f.dim, f.values])),
      metrics,
      limit: 40,
    }),
    [runId, groupBy, filters, metrics]
  );

  // live preview: debounce 350 ms after any change
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setLoading(true);
    debounceRef.current = setTimeout(() => {
      queryEvaluations(request)
        .then((r) => {
          setResult(r);
          setError(null);
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setLoading(false));
    }, 350);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [request]);

  const dropDim = useCallback(
    (well: Well, dim: DimName) => {
      if (well === 'group') {
        setGroupBy((g) => (g.includes(dim) ? g : [...g, dim]));
      } else {
        setFilters((fs) => (fs.some((f) => f.dim === dim) ? fs : [...fs, { dim, values: [] }]));
      }
      setDragOverWell(null);
    },
    []
  );

  const wellProps = (well: Well) => ({
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      setDragOverWell(well);
    },
    onDragLeave: () => setDragOverWell(null),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      const dim = e.dataTransfer.getData('application/x-dim') as DimName;
      if (dim && DIM_NAMES.includes(dim)) dropDim(well, dim);
    },
  });

  const firstMetric = metrics[0];
  const firstMetricMeta = METRIC_CHOICES.find((m) => m.name === firstMetric);
  const rows = result?.rows ?? [];
  const barMax = useMemo(() => {
    if (!firstMetric) return 1;
    const vals = rows.map((r) => (typeof r[firstMetric] === 'number' ? (r[firstMetric] as number) : 0));
    return firstMetricMeta?.pct ? 1 : Math.max(1e-9, ...vals);
  }, [rows, firstMetric, firstMetricMeta]);

  const rowLabel = (r: Record<string, unknown>): string => groupBy.map((d) => String(r[d] ?? '—')).join(' · ') || 'all';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title={
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
            Build the query visually
            <InfoDot
              title="Visual query builder"
              detail="Drag a dimension chip into the Group-by well to break results down by it, or into the Filter well to restrict the population (then pick the allowed values). Toggle metrics below. The preview runs the real /api/evaluations/query request — shown as JSON on the right — after every change."
            />
          </Box>
        }
      >
        {/* dimension tray */}
        <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mb: 1.5, alignItems: 'center' }}>
          <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
            Dimensions — drag into a well:
          </Typography>
          {DIM_NAMES.map((d) => (
            <Chip
              key={d}
              size="small"
              icon={<GripVertical size={12} />}
              label={d.replace(/_/g, ' ')}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('application/x-dim', d);
                e.dataTransfer.effectAllowed = 'copy';
              }}
              onClick={() => dropDim('group', d)}
              sx={{
                cursor: 'grab',
                bgcolor: `${DIM_COLORS[d]}1a`,
                color: DIM_COLORS[d],
                border: `1px solid ${DIM_COLORS[d]}55`,
                fontWeight: 700,
                '&:active': { cursor: 'grabbing' },
              }}
            />
          ))}
        </Box>

        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          {/* group-by well */}
          <Box
            {...wellProps('group')}
            sx={{
              flex: '1 1 260px',
              minHeight: 74,
              border: `2px dashed ${dragOverWell === 'group' ? tokens.color.info : tokens.color.borderStrong}`,
              borderRadius: 1.5,
              p: 1.25,
              bgcolor: dragOverWell === 'group' ? tokens.color.infoBg : tokens.color.surfaceSunken,
              transition: `border-color ${tokens.motion.fast}, background-color ${tokens.motion.fast}`,
            }}
          >
            <Typography variant="caption" sx={{ color: tokens.color.neutral, fontWeight: 700, letterSpacing: 0.5, display: 'block', mb: 0.75 }}>
              GROUP BY {groupBy.length ? `(${groupBy.length} — drag the handle to reorder)` : '— drop a dimension here'}
            </Typography>
            <DragList
              dense
              items={groupBy.map((d) => ({
                id: d,
                content: (
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Chip size="small" label={d.replace(/_/g, ' ')} sx={{ bgcolor: `${DIM_COLORS[d]}26`, color: DIM_COLORS[d], fontWeight: 700, height: 22 }} />
                    <Box
                      component="span"
                      onClick={() => setGroupBy((g) => g.filter((x) => x !== d))}
                      sx={{ display: 'inline-flex', cursor: 'pointer', color: tokens.color.textFaint, '&:hover': { color: tokens.color.danger } }}
                      title="Remove from group-by"
                    >
                      <X size={13} />
                    </Box>
                  </Box>
                ),
              }))}
              onReorder={(ids) => setGroupBy(ids as DimName[])}
            />
          </Box>

          {/* filter well */}
          <Box
            {...wellProps('filter')}
            sx={{
              flex: '2 1 380px',
              minHeight: 74,
              border: `2px dashed ${dragOverWell === 'filter' ? tokens.color.warn : tokens.color.borderStrong}`,
              borderRadius: 1.5,
              p: 1.25,
              bgcolor: dragOverWell === 'filter' ? tokens.color.warnBg : tokens.color.surfaceSunken,
              transition: `border-color ${tokens.motion.fast}, background-color ${tokens.motion.fast}`,
            }}
          >
            <Typography variant="caption" sx={{ color: tokens.color.neutral, fontWeight: 700, letterSpacing: 0.5, display: 'block', mb: 0.75 }}>
              FILTERS {filters.length ? `(${filters.length})` : '— drop a dimension here, then pick values'}
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
              {filters.map((f) => (
                <Box key={f.dim} sx={{ display: 'flex', gap: 0.75, alignItems: 'center', flexWrap: 'wrap' }}>
                  <Chip size="small" label={f.dim.replace(/_/g, ' ')} sx={{ bgcolor: `${DIM_COLORS[f.dim]}26`, color: DIM_COLORS[f.dim], fontWeight: 700 }} />
                  <Typography variant="caption" sx={{ color: tokens.color.textFaint, fontStyle: 'italic' }}>
                    is any of
                  </Typography>
                  {(dimValues?.[f.dim] ?? []).map((v) => {
                    const on = f.values.includes(v);
                    return (
                      <Chip
                        key={v}
                        size="small"
                        label={v}
                        onClick={() =>
                          setFilters((fs) =>
                            fs.map((x) => (x.dim === f.dim ? { ...x, values: on ? x.values.filter((y) => y !== v) : [...x.values, v] } : x))
                          )
                        }
                        sx={{
                          height: 21,
                          fontSize: 10.5,
                          cursor: 'pointer',
                          bgcolor: on ? tokens.color.infoBg : 'transparent',
                          color: on ? tokens.color.info : tokens.color.textFaint,
                          border: `1px solid ${on ? tokens.color.info : tokens.color.border}`,
                          transition: `all ${tokens.motion.fast}`,
                        }}
                      />
                    );
                  })}
                  <Box
                    component="span"
                    onClick={() => setFilters((fs) => fs.filter((x) => x.dim !== f.dim))}
                    sx={{ display: 'inline-flex', alignItems: 'center', cursor: 'pointer', color: tokens.color.textFaint, '&:hover': { color: tokens.color.danger } }}
                    title="Remove this filter"
                  >
                    <X size={13} />
                  </Box>
                </Box>
              ))}
            </Box>
          </Box>
        </Box>

        {/* metric toggles */}
        <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mt: 1.5, alignItems: 'center' }}>
          <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
            Metrics:
          </Typography>
          {METRIC_CHOICES.map((m) => {
            const on = metrics.includes(m.name);
            return (
              <Chip
                key={m.name}
                size="small"
                label={m.label}
                title={m.help}
                onClick={() => setMetrics((ms) => (on ? ms.filter((x) => x !== m.name) : [...ms, m.name]))}
                sx={{
                  cursor: 'pointer',
                  bgcolor: on ? tokens.color.successBg : 'transparent',
                  color: on ? tokens.color.success : tokens.color.textFaint,
                  border: `1px solid ${on ? tokens.color.success : tokens.color.border}`,
                  fontWeight: on ? 700 : 400,
                  transition: `all ${tokens.motion.fast}`,
                }}
              />
            );
          })}
        </Box>
      </SectionCard>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <SectionCard
          title="Live result preview"
          sx={{ flex: '3 1 460px' }}
          action={
            result ? (
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                <Chip size="small" label={result.meta.source} sx={{ height: 20, fontSize: 10, bgcolor: tokens.color.surfaceRaised, fontFamily: 'monospace' }} title="Query provenance: answered from cache, the metric cube, or a raw scan" />
                <Chip size="small" label={`${result.meta.latency_ms.toFixed(0)} ms`} sx={{ height: 20, fontSize: 10, bgcolor: tokens.color.surfaceRaised, fontFamily: 'monospace' }} />
                <Chip size="small" label={result.meta.exact ? 'exact' : 'approx'} sx={{ height: 20, fontSize: 10, fontFamily: 'monospace', bgcolor: result.meta.exact ? tokens.color.successBg : tokens.color.warnBg, color: result.meta.exact ? tokens.color.success : tokens.color.warn }} />
              </Box>
            ) : undefined
          }
        >
          {error ? (
            <Typography variant="caption" sx={{ color: tokens.color.warn }}>
              {error}
            </Typography>
          ) : null}
          {loading && !result ? <PanelSkeleton rows={5} header={false} /> : null}
          {result && rows.length === 0 ? (
            <IllustratedEmpty art="search" title="No matching population" message="The current filters exclude every object — loosen a filter value to see results." />
          ) : null}
          {rows.length > 0 ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, opacity: loading ? 0.6 : 1, transition: `opacity ${tokens.motion.fast}` }}>
              {firstMetric ? (
                <Box>
                  <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mb: 0.5 }}>
                    {firstMetricMeta?.label ?? firstMetric} by {groupBy.map((d) => d.replace(/_/g, ' ')).join(' × ') || 'population'}
                  </Typography>
                  {rows.slice(0, 14).map((r, i) => {
                    const v = typeof r[firstMetric] === 'number' ? (r[firstMetric] as number) : 0;
                    return (
                      <HBar
                        key={i}
                        label={rowLabel(r)}
                        value={v}
                        max={barMax}
                        color={tokens.color.info}
                        valueLabel={firstMetricMeta?.pct ? fmtPct(v) : v.toFixed(3)}
                      />
                    );
                  })}
                </Box>
              ) : null}
              <Box sx={{ overflowX: 'auto' }}>
                <Box component="table" sx={{ borderCollapse: 'collapse', width: '100%', '& td, & th': { border: `1px solid ${tokens.color.border}`, px: 1, py: 0.4, fontSize: 12, textAlign: 'right' }, '& th': { bgcolor: tokens.color.surfaceRaised, color: tokens.color.textDim }, '& td:first-of-type, & th:first-of-type': { textAlign: 'left' } }}>
                  <thead>
                    <tr>
                      <th>group</th>
                      <th>n</th>
                      {metrics.map((m) => (
                        <th key={m}>{m.replace(/_/g, ' ')}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 20).map((r, i) => (
                      <tr key={i}>
                        <td style={{ fontFamily: 'monospace' }}>{rowLabel(r)}</td>
                        <td style={{ fontFamily: 'monospace' }}>{fmtInt(r.n)}</td>
                        {metrics.map((m) => {
                          const v = r[m];
                          const meta = METRIC_CHOICES.find((x) => x.name === m);
                          return (
                            <td key={m} style={{ fontFamily: 'monospace' }}>
                              {typeof v === 'number' ? (meta?.pct ? fmtPct(v) : v.toFixed(3)) : '—'}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </Box>
              </Box>
            </Box>
          ) : null}
        </SectionCard>

        <SectionCard
          title="Equivalent API request"
          sx={{ flex: '1 1 280px' }}
          help="The exact JSON body sent to POST /api/evaluations/query for the preview — build the same query manually and you will get the same rows."
        >
          <Box component="pre" sx={{ m: 0, p: 1, bgcolor: tokens.color.surfaceSunken, border: `1px solid ${tokens.color.border}`, borderRadius: 1, fontSize: 11, fontFamily: 'monospace', overflow: 'auto', maxHeight: 340, color: tokens.color.textDim }}>
            {JSON.stringify(request, null, 2)}
          </Box>
        </SectionCard>
      </Box>
    </Box>
  );
}
