/**
 * Discrepancy Dashboard — auto-label discrepancy mining: diff a simulated
 * online perception pass against the stored offline auto-labels. Mine
 * action, by-type donut, cohort table, sample discrepancy list.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { Pickaxe } from 'lucide-react';
import { getDiscrepancySummary, mineDiscrepancies, type DiscrepancyReport } from '../../services/safety';
import { Donut } from '../../components/visual/charts';
import { IllustratedEmpty, TileSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, MetricCard, SectionCard, StatusChip, fmtPct } from '../../components/labeleval/shared';
import { HeadCell } from '../../components/help/InfoTip';
import { useLabelEval } from '../../context/LabelEvalContext';
import { tokens } from '../../theme';

const TYPE_COLORS: Record<string, string> = {
  online_miss: '#ef5350',
  class_flip: '#ffa726',
  position_drift: '#4fc3f7',
  online_phantom: '#ab47bc',
};

interface Totals {
  objects: number;
  online_detections: number;
  offline_annotations: number;
  discrepancies: number;
  discrepancy_rate: number;
}

interface CohortRow {
  cohort: string;
  class: string;
  weather: string;
  time_of_day: string;
  objects: number;
  discrepancies: number;
  discrepancy_rate: number;
}

interface DiscRow {
  discrepancy_id: string;
  type: string;
  class_name: string;
  weather: string;
  time_of_day: string;
  safety_critical: boolean;
  severity: string;
  details: Record<string, unknown>;
}

export default function DiscrepancyPage() {
  const { activeDatasetId } = useLabelEval();
  const [report, setReport] = useState<DiscrepancyReport | null>(null);
  const [mining, setMining] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDiscrepancySummary()
      .then((r) => {
        const multi = r as { datasets?: DiscrepancyReport[] };
        if (multi.datasets?.length) setReport(multi.datasets[multi.datasets.length - 1]);
        else if ((r as DiscrepancyReport).dataset_id) setReport(r as DiscrepancyReport);
      })
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, []);

  const mine = useCallback(() => {
    setMining(true);
    setError(null);
    mineDiscrepancies(activeDatasetId ?? undefined)
      .then(setReport)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setMining(false));
  }, [activeDatasetId]);

  const totals = (report?.totals ?? null) as Totals | null;
  const byType = (report?.by_type ?? {}) as Record<string, number>;
  const cohorts = ((report?.cohorts ?? []) as unknown as CohortRow[]).slice(0, 20);
  const samples = ((report?.discrepancies ?? []) as unknown as DiscRow[]).slice(0, 15);
  const maxRate = useMemo(() => Math.max(...cohorts.map((c) => c.discrepancy_rate), 0.001), [cohorts]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}

      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="contained" startIcon={<Pickaxe size={16} />} disabled={mining} onClick={mine}>
          {mining ? 'Mining…' : 'Mine discrepancies'}
        </Button>
        <Typography variant="caption" sx={{ color: tokens.color.neutral, maxWidth: 620 }}>
          Runs a deterministic <strong>simulated online pass</strong> over{' '}
          {activeDatasetId ? <code>{activeDatasetId}</code> : 'the newest dataset'} and diffs it against the stored
          offline auto-labels over shared ground-truth ids. Critical discrepancies feed the rare-event store and the
          scenario database automatically.
        </Typography>
        {report ? (
          <Chip size="small" label={`dataset: ${report.dataset_id}`} sx={{ fontFamily: 'monospace', fontSize: 10.5, bgcolor: tokens.color.surfaceRaised }} />
        ) : null}
      </Box>

      {loading || mining ? <TileSkeleton n={4} /> : null}

      {!report && !loading && !mining ? (
        <SectionCard title="Discrepancy mining">
          <IllustratedEmpty
            art="search"
            title="No discrepancy report yet"
            message="Mine the newest dataset to compare the simulated online perception pass against the offline auto-labels — the disagreements are the continuous-learning loop's best training candidates."
            action={
              <Button variant="contained" size="small" startIcon={<Pickaxe size={15} />} onClick={mine}>
                Mine discrepancies
              </Button>
            }
          />
        </SectionCard>
      ) : null}

      {totals ? (
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          <MetricCard label="Objects compared" value={totals.objects.toLocaleString()} sub={`${totals.online_detections.toLocaleString()} online · ${totals.offline_annotations.toLocaleString()} offline`} info="Ground-truth objects present in both the offline auto-label pass and the simulated online pass." />
          <MetricCard label="Discrepancies" value={totals.discrepancies.toLocaleString()} accent={tokens.color.warn} info="Disagreements between the two passes: online misses, class flips, position drifts and online phantoms." />
          <MetricCard label="Discrepancy rate" value={fmtPct(totals.discrepancy_rate)} accent={totals.discrepancy_rate > 0.15 ? tokens.color.danger : tokens.color.warn} info="Discrepancies / objects compared. The online profile deliberately degrades at night and in rain — see the cohort table." />
        </Box>
      ) : null}

      {report ? (
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <SectionCard title="Discrepancies by type" sx={{ flex: '1 1 320px' }} help="online_miss = online failed to detect an object the offline pass labeled (the dangerous one); class_flip = same object, different class; position_drift = localization disagreement; online_phantom = online detection with no offline counterpart.">
            <Donut
              segments={Object.entries(byType).map(([t, n]) => ({ label: t, value: n, color: TYPE_COLORS[t] ?? tokens.color.neutral }))}
              centerLabel={totals ? totals.discrepancies.toLocaleString() : ''}
              centerSub="total"
            />
          </SectionCard>

          <SectionCard title="Worst cohorts" sx={{ flex: '2 1 480px' }} help="Cohort = class / weather / time-of-day. Sorted by discrepancy rate — the top rows tell you exactly where the online stack diverges from offline labeling (typically VRU classes at night and in rain).">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Cohort</TableCell>
                  <TableCell align="right"><HeadCell label="Objects" title="Objects" detail="Ground-truth objects in this cohort." /></TableCell>
                  <TableCell align="right">Discrepancies</TableCell>
                  <TableCell sx={{ width: '35%' }}>Rate</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {cohorts.map((c) => (
                  <TableRow key={c.cohort}>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>{c.cohort}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{c.objects.toLocaleString()}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{c.discrepancies.toLocaleString()}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ flex: 1, height: 7, bgcolor: tokens.color.border, borderRadius: 1, overflow: 'hidden' }}>
                          <Box sx={{ height: '100%', width: `${(c.discrepancy_rate / maxRate) * 100}%`, bgcolor: c.discrepancy_rate > 0.25 ? tokens.color.danger : c.discrepancy_rate > 0.12 ? '#ffa726' : tokens.color.warn, transition: `width ${tokens.motion.slow}` }} />
                        </Box>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace', width: 48 }}>{fmtPct(c.discrepancy_rate)}</Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </SectionCard>
        </Box>
      ) : null}

      {samples.length ? (
        <SectionCard title={`Sample discrepancies (first ${samples.length})`} help="Individual disagreements with their evidence. Safety-critical + critical-severity discrepancies are automatically pushed to the rare-event store for review prioritization.">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Type</TableCell>
                <TableCell>Class</TableCell>
                <TableCell>Conditions</TableCell>
                <TableCell>Severity</TableCell>
                <TableCell>Details</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {samples.map((d) => (
                <TableRow key={d.discrepancy_id}>
                  <TableCell>
                    <Chip size="small" label={d.type} sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: `${TYPE_COLORS[d.type] ?? tokens.color.neutral}22`, color: TYPE_COLORS[d.type] ?? tokens.color.neutral, border: `1px solid ${TYPE_COLORS[d.type] ?? tokens.color.neutral}` }} />
                  </TableCell>
                  <TableCell>
                    {d.class_name}
                    {d.safety_critical ? <Chip size="small" label="safety" sx={{ ml: 0.5, height: 16, fontSize: 9, bgcolor: tokens.color.dangerBg, color: tokens.color.danger }} /> : null}
                  </TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>{d.weather}/{d.time_of_day}</TableCell>
                  <TableCell><StatusChip status={d.severity} /></TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>
                    {Object.entries(d.details ?? {}).map(([k, v]) => `${k}=${typeof v === 'number' ? v : String(v)}`).join(' · ')}
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
