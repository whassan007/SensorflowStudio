/**
 * ODD Coverage — interactive combinatorial coverage heatmap over the ODD
 * taxonomy (ISO 34503-inspired) with adequacy coloring, a risk-sorted gap
 * table, and a wired "Fill gap" action showing before/after cell metrics.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import MenuItem from '@mui/material/MenuItem';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ArrowRight, Wand2 } from 'lucide-react';
import {
  fillOddGap,
  getOddCoverage,
  getOddTaxonomy,
  type OddCell,
  type OddCoverageResponse,
  type OddFillRequest,
  type OddTaxonomy,
} from '../../services/safety';
import { RunSelect, usePublishedRuns } from '../../components/safety/shared';
import { HeatmapGrid, type HeatCell } from '../../components/visual/charts';
import { PanelSkeleton, TileSkeleton, IllustratedEmpty } from '../../components/visual/Feedback';
import { ErrorNote, MetricCard, SectionCard, fmtPct } from '../../components/labeleval/shared';
import { HeadCell, InfoDot } from '../../components/help/InfoTip';
import { tokens } from '../../theme';

// Coverage-adequacy palette: green adequate, amber = statistical gap, red = performance gap.
function cellColor(cell: OddCell): string {
  if (cell.n === 0) return '#20262d';
  if (cell.adequate) {
    const r = cell.recall ?? 0;
    return r >= 0.9 ? '#1e5c28' : r >= 0.8 ? '#2e6b33' : '#3f7a3f';
  }
  if (cell.gap_reasons.includes('recall_below_target') || cell.performance_deficit > 0.1) return '#8e2f2b';
  return '#8a6215';
}

function GapReasonChips({ reasons }: { reasons: string[] }) {
  const meta: Record<string, { label: string; color: string; help: string }> = {
    insufficient_samples: { label: 'too few samples', color: tokens.color.warn, help: 'The cell has fewer evaluated objects than the min-samples threshold — its metrics are statistically unreliable.' },
    wide_ci: { label: 'wide CI', color: tokens.color.warn, help: 'The Wilson confidence interval on recall is wider than the max-CI-width threshold.' },
    recall_below_target: { label: 'recall below target', color: tokens.color.danger, help: 'Even with enough data, recall in this cell is below the target — a performance gap, not a data gap.' },
  };
  return (
    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
      {reasons.map((r) => {
        const m = meta[r] ?? { label: r, color: tokens.color.neutral, help: '' };
        return (
          <Chip
            key={r}
            size="small"
            label={m.label}
            title={m.help}
            sx={{ height: 18, fontSize: 10, bgcolor: 'transparent', border: `1px solid ${m.color}`, color: m.color }}
          />
        );
      })}
    </Box>
  );
}

function CellStats({ cell, title }: { cell: OddCell; title: string }) {
  return (
    <Box sx={{ flex: 1, p: 1.5, border: `1px solid ${tokens.color.border}`, borderRadius: 1, bgcolor: tokens.color.surfaceSunken }}>
      <Typography variant="caption" sx={{ color: tokens.color.neutral, fontWeight: 700, textTransform: 'uppercase' }}>
        {title}
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 1.5, rowGap: 0.25, mt: 0.5 }}>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>n</Typography>
        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>{cell.n.toLocaleString()}</Typography>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>recall</Typography>
        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
          {cell.recall !== null ? fmtPct(cell.recall) : '—'}
          {cell.wilson_ci ? ` (${fmtPct(cell.wilson_ci[0])} – ${fmtPct(cell.wilson_ci[1])})` : ''}
        </Typography>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>CI width</Typography>
        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>{cell.ci_width?.toFixed(4) ?? '—'}</Typography>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>adequate</Typography>
        <Typography variant="caption" sx={{ fontFamily: 'monospace', color: cell.adequate ? tokens.color.success : tokens.color.danger, fontWeight: 700 }}>
          {cell.adequate ? 'YES' : 'NO'}
        </Typography>
      </Box>
      {!cell.adequate ? <Box sx={{ mt: 0.75 }}><GapReasonChips reasons={cell.gap_reasons} /></Box> : null}
    </Box>
  );
}

export default function OddCoveragePage() {
  const { runs, error: runsError } = usePublishedRuns();
  const [runId, setRunId] = useState<string | null>(null);
  const [taxonomy, setTaxonomy] = useState<OddTaxonomy | null>(null);
  const [rowDim, setRowDim] = useState('class');
  const [colDim, setColDim] = useState('weather');
  const [coverage, setCoverage] = useState<OddCoverageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<OddCell | null>(null);
  const [filling, setFilling] = useState<string | null>(null); // cell_id in flight
  const [fillResult, setFillResult] = useState<{ before: OddCell; after: OddCell; added: number; dataset: string } | null>(null);

  useEffect(() => {
    getOddTaxonomy().then(setTaxonomy).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!runId && runs?.length) setRunId(runs[0].run_id);
  }, [runs, runId]);

  const refresh = useCallback(() => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    setSelectedCell(null);
    getOddCoverage(runId, [rowDim, colDim], true)
      .then((c) => setCoverage(c))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [runId, rowDim, colDim]);

  useEffect(refresh, [refresh]);

  const dimNames = useMemo(() => Object.keys(taxonomy?.dimensions ?? { class: 0, weather: 0, lighting: 0, road_type: 0 }), [taxonomy]);

  const heat = useMemo(() => {
    if (!coverage?.cells) return null;
    const rows = taxonomy?.dimensions[rowDim]?.values ?? [...new Set(coverage.cells.map((c) => c.cell[rowDim]))];
    const cols = taxonomy?.dimensions[colDim]?.values ?? [...new Set(coverage.cells.map((c) => c.cell[colDim]))];
    const cells: HeatCell[] = coverage.cells.map((c) => ({
      row: c.cell[rowDim],
      col: c.cell[colDim],
      value: c.recall,
      color: cellColor(c),
      label: c.n === 0 ? '·' : c.recall !== null ? `${(c.recall * 100).toFixed(0)}%` : '—',
      tooltip: (
        <Box>
          <Typography variant="caption" sx={{ fontWeight: 700, color: tokens.color.info, display: 'block' }}>
            {c.cell_id}
          </Typography>
          <Typography variant="caption" sx={{ display: 'block' }}>
            n={c.n.toLocaleString()} · recall {c.recall !== null ? fmtPct(c.recall) : '—'}
            {c.wilson_ci ? ` [${fmtPct(c.wilson_ci[0])}–${fmtPct(c.wilson_ci[1])}]` : ''}
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', color: tokens.color.textDim }}>
            production share {fmtPct(c.production_share)} · {c.adequate ? 'adequate' : `gap: ${c.gap_reasons.join(', ')}`}
          </Typography>
          <Typography variant="caption" sx={{ color: tokens.color.neutral }}>click for details & fill action</Typography>
        </Box>
      ),
    }));
    return { rows, cols, cells };
  }, [coverage, rowDim, colDim, taxonomy]);

  const cellByKey = useMemo(() => {
    const m = new Map<string, OddCell>();
    coverage?.cells?.forEach((c) => m.set(`${c.cell[rowDim]}|${c.cell[colDim]}`, c));
    return m;
  }, [coverage, rowDim, colDim]);

  const riskByCellId = useMemo(() => {
    const m = new Map<string, OddFillRequest>();
    coverage?.fill_requests.forEach((f) => m.set(f.cell_id, f));
    return m;
  }, [coverage]);

  const sortedGaps = useMemo(() => {
    if (!coverage) return [];
    return [...coverage.gaps].sort((a, b) => {
      const ra = riskByCellId.get(a.cell_id)?.risk ?? a.production_share * Math.max(a.performance_deficit, 0.01);
      const rb = riskByCellId.get(b.cell_id)?.risk ?? b.production_share * Math.max(b.performance_deficit, 0.01);
      return rb - ra;
    });
  }, [coverage, riskByCellId]);

  const doFill = useCallback(
    (cell: OddCell) => {
      if (!runId) return;
      setFilling(cell.cell_id);
      setFillResult(null);
      fillOddGap(runId, cell.cell)
        .then((res) => {
          const r = res as unknown as { cell_before: OddCell; cell_after: OddCell; objects_added: number; generated_dataset_id: string };
          setFillResult({ before: r.cell_before, after: r.cell_after, added: r.objects_added, dataset: r.generated_dataset_id });
          refresh();
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setFilling(null));
    },
    [runId, refresh]
  );

  const s = coverage?.summary;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {runsError ? <ErrorNote error={runsError} /> : null}
      {error ? <ErrorNote error={error} /> : null}

      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <RunSelect label="Evaluation run" value={runId} onChange={setRunId} runs={runs} />
        <TextField select size="small" label="Rows" value={rowDim} onChange={(e) => setRowDim(e.target.value)} sx={{ minWidth: 140 }}>
          {dimNames.filter((d) => d !== colDim).map((d) => (
            <MenuItem key={d} value={d}>{d}</MenuItem>
          ))}
        </TextField>
        <TextField select size="small" label="Columns" value={colDim} onChange={(e) => setColDim(e.target.value)} sx={{ minWidth: 140 }}>
          {dimNames.filter((d) => d !== rowDim).map((d) => (
            <MenuItem key={d} value={d}>{d}</MenuItem>
          ))}
        </TextField>
        {taxonomy ? (
          <Typography variant="caption" sx={{ color: tokens.color.neutral, maxWidth: 420 }}>
            {taxonomy.standard_basis}
          </Typography>
        ) : null}
      </Box>

      {runs !== null && runs.length === 0 ? (
        <SectionCard title="ODD coverage needs a published evaluation run">
          <IllustratedEmpty
            art="gauge"
            title="No published evaluation runs"
            message="Coverage is computed from the metric cube of a published mega-scale evaluation run. Generate a population and launch a run from the Command Center, then come back here."
          />
        </SectionCard>
      ) : null}

      {loading && !coverage ? <TileSkeleton n={5} /> : null}

      {s ? (
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          <MetricCard label="Coverage rate" value={fmtPct(s.coverage_rate)} sub={`${s.adequate_cells}/${s.total_cells} cells adequate`} accent={s.coverage_rate >= 0.9 ? tokens.color.success : s.coverage_rate >= 0.7 ? tokens.color.warn : tokens.color.danger} term="odd_coverage" />
          <MetricCard label="Production-weighted" value={fmtPct(s.production_weighted_coverage)} sub="weighted by exposure share" accent={tokens.color.info} info="Coverage where each cell counts proportionally to its production exposure share — a gap in a common condition hurts more than one in a rare condition." />
          <MetricCard label="Gap cells" value={s.gap_cells} sub={`${s.empty_cells} empty`} accent={s.gap_cells > 0 ? tokens.color.warn : tokens.color.success} info="Cells that fail the adequacy criteria (min samples, max CI width, recall target)." />
          <MetricCard label="Overall recall" value={fmtPct(s.overall_recall)} term="recall" />
        </Box>
      ) : null}

      {heat && coverage ? (
        <SectionCard
          title={`Coverage heatmap — ${rowDim} × ${colDim}`}
          help={`Each cell is one ODD combination. Green = adequate (enough samples, tight CI, recall at target — darker green is higher recall). Amber = statistical gap (too little data / wide CI). Red = performance gap (recall below the ${fmtPct(coverage.thresholds.target_recall)} target). Click a cell for details and the fill-gap action.`}
          action={
            <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
              thresholds: n ≥ {coverage.thresholds.min_samples}, CI width ≤ {coverage.thresholds.max_ci_width}, recall target {fmtPct(coverage.thresholds.target_recall)}
            </Typography>
          }
        >
          <HeatmapGrid
            rows={heat.rows}
            cols={heat.cols}
            cells={heat.cells}
            selectedKey={selectedCell ? `${selectedCell.cell[rowDim]}|${selectedCell.cell[colDim]}` : null}
            onCellClick={(hc) => setSelectedCell(cellByKey.get(`${hc.row}|${hc.col}`) ?? null)}
          />
          {selectedCell ? (
            <Box sx={{ mt: 1.5, display: 'flex', gap: 1.5, alignItems: 'stretch', flexWrap: 'wrap' }}>
              <CellStats cell={selectedCell} title={selectedCell.cell_id} />
              <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 1 }}>
                {selectedCell.is_gap ? (
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<Wand2 size={15} />}
                    disabled={filling !== null}
                    onClick={() => doFill(selectedCell)}
                  >
                    {filling === selectedCell.cell_id ? 'Generating…' : 'Fill gap'}
                  </Button>
                ) : (
                  <Chip size="small" label="adequate — no action needed" sx={{ bgcolor: tokens.color.successBg, color: tokens.color.success }} />
                )}
                <Typography variant="caption" sx={{ color: tokens.color.neutral, maxWidth: 220 }}>
                  Fill gap generates targeted synthetic sequences for this cell via the platform generator and re-computes coverage.
                </Typography>
              </Box>
            </Box>
          ) : null}
        </SectionCard>
      ) : null}

      {coverage && loading ? <PanelSkeleton rows={3} header={false} /> : null}

      {sortedGaps.length ? (
        <SectionCard
          title="Coverage gaps, risk-sorted"
          help="Risk = production exposure share × performance deficit: how much real-world harm the gap could cause. The Fill gap action generates targeted synthetic data for the cell and shows the before/after effect."
        >
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell><HeadCell label="Cell" title="ODD cell" detail="The combination of ODD dimension values defining this coverage cell." /></TableCell>
                <TableCell align="right"><HeadCell label="n" title="Samples" detail="Evaluated objects in this cell." /></TableCell>
                <TableCell align="right"><HeadCell label="Recall (95% CI)" term="wilson_ci" /></TableCell>
                <TableCell align="right"><HeadCell label="Prod. share" title="Production share" detail="Estimated fraction of real-world exposure falling in this cell." /></TableCell>
                <TableCell align="right"><HeadCell label="Risk" title="Gap risk" detail="Production share × performance deficit — the sort key of this table." /></TableCell>
                <TableCell><HeadCell label="Why it's a gap" title="Gap reasons" detail="Which adequacy criteria the cell fails." /></TableCell>
                <TableCell align="center">Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedGaps.map((g) => (
                <TableRow key={g.cell_id} hover selected={selectedCell?.cell_id === g.cell_id} onClick={() => setSelectedCell(g)} sx={{ cursor: 'pointer' }}>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>
                    {Object.entries(g.cell).map(([k, v]) => (
                      <Chip key={k} size="small" label={`${k}=${v}`} sx={{ height: 18, fontSize: 10, mr: 0.5, bgcolor: tokens.color.surfaceRaised }} />
                    ))}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{g.n.toLocaleString()}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {g.recall !== null ? fmtPct(g.recall) : '—'}
                    {g.wilson_ci ? (
                      <Typography component="span" variant="caption" sx={{ color: tokens.color.neutral }}>
                        {' '}[{fmtPct(g.wilson_ci[0])}–{fmtPct(g.wilson_ci[1])}]
                      </Typography>
                    ) : null}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{fmtPct(g.production_share)}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {(riskByCellId.get(g.cell_id)?.risk ?? g.production_share * Math.max(g.performance_deficit, 0.01)).toExponential(2)}
                  </TableCell>
                  <TableCell><GapReasonChips reasons={g.gap_reasons} /></TableCell>
                  <TableCell align="center">
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<Wand2 size={13} />}
                      disabled={filling !== null}
                      onClick={(e) => {
                        e.stopPropagation();
                        doFill(g);
                      }}
                    >
                      {filling === g.cell_id ? 'Generating…' : 'Fill gap'}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </SectionCard>
      ) : null}

      {coverage && !sortedGaps.length && !loading ? (
        <Alert severity="success" variant="outlined">
          No coverage gaps at the current thresholds — every populated ODD cell is adequate.
        </Alert>
      ) : null}

      <Dialog open={fillResult !== null} onClose={() => setFillResult(null)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontWeight: 800 }}>
          Gap filled — before / after
          <InfoDot title="What happened" detail="Targeted synthetic sequences were generated for the gap cell (conditions retargeted to the cell), labeled by the current model, persisted as a supplement, and coverage was recomputed." />
        </DialogTitle>
        <DialogContent>
          {fillResult ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Typography variant="body2" sx={{ color: tokens.color.textDim }}>
                Added <strong>{fillResult.added}</strong> objects via synthetic dataset{' '}
                <Box component="code" sx={{ fontFamily: 'monospace' }}>{fillResult.dataset}</Box>. Supplements are
                tracked separately from the immutable evaluation population.
              </Typography>
              <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                <CellStats cell={fillResult.before} title="Before" />
                <ArrowRight size={28} color={tokens.color.info} style={{ flexShrink: 0 }} />
                <CellStats cell={fillResult.after} title="After" />
              </Box>
            </Box>
          ) : null}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
