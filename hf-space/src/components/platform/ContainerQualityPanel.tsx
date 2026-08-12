/**
 * Container quality cards/table with verification rates (Phase 1 foundation).
 * Extends MegaEval ContainersTab data via /api/containers/quality.
 */
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { useContainerQuality } from '../../services/platform';
import { ErrorNote, LoadingBox, MetricCard, SectionCard, fmtPct } from '../labeleval/shared';

export default function ContainerQualityPanel({
  runId,
  refreshKey = 0,
}: {
  runId: string | null;
  refreshKey?: number;
}) {
  const poll = useContainerQuality(runId, refreshKey);

  if (!runId) {
    return (
      <SectionCard title="Container quality">
        <Typography variant="body2" color="text.secondary">
          Select a published evaluation run to load container quality profiles.
        </Typography>
      </SectionCard>
    );
  }
  if (poll.error && !poll.data) return <ErrorNote error={poll.error} />;
  if (poll.loading && !poll.data) return <LoadingBox label="Loading container quality…" />;
  const data = poll.data;
  if (!data) return null;

  const v = data.verification_headline || data.summary?.verification;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title="Container quality profile"
        action={
          <Typography variant="caption" color="text.secondary">
            {data.total} containers · {data.model_version}
          </Typography>
        }
      >
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
          <MetricCard label="Precision" value={fmtPct(data.summary.precision)} />
          <MetricCard label="Recall" value={fmtPct(data.summary.recall)} />
          <MetricCard label="F1" value={fmtPct(data.summary.f1)} />
          <MetricCard label="Mean IoU" value={fmtPct(data.summary.mean_iou)} />
          <MetricCard label="Verified" value={fmtPct(v?.verified_rate)} sub={`${v?.verified ?? 0} labels`} />
          <MetricCard label="HITL rate" value={fmtPct(v?.hitl_rate)} />
          <MetricCard label="Unverified" value={fmtPct(v?.unverified_rate)} />
          <MetricCard label="Disputed" value={fmtPct(v?.disputed_rate)} sub="Phase 2 hook" />
        </Box>

        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Container</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">P</TableCell>
              <TableCell align="right">R</TableCell>
              <TableCell align="right">F1</TableCell>
              <TableCell align="right">IoU</TableCell>
              <TableCell align="right">Verified</TableCell>
              <TableCell align="right">HITL</TableCell>
              <TableCell align="right">Risk</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.containers.slice(0, 20).map((c) => (
              <TableRow key={c.container_id} hover>
                <TableCell>#{c.container_id}</TableCell>
                <TableCell>{c.status ?? '—'}</TableCell>
                <TableCell align="right">{fmtPct(c.precision)}</TableCell>
                <TableCell align="right">{fmtPct(c.recall)}</TableCell>
                <TableCell align="right">{fmtPct(c.f1)}</TableCell>
                <TableCell align="right">{fmtPct(c.mean_iou)}</TableCell>
                <TableCell align="right">{fmtPct(c.verification?.verified_rate)}</TableCell>
                <TableCell align="right">{fmtPct(c.verification?.hitl_rate)}</TableCell>
                <TableCell align="right">{c.risk_score.toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionCard>
    </Box>
  );
}
