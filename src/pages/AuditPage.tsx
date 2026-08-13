import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { getProcessUnits, getAudit, usePoll } from '../services/labeleval';
import {
  SectionCard,
  MetricCard,
  HBar,
  LoadingBox,
  ErrorNote,
  fmtInt,
  fmtNum,
} from '../components/labeleval/shared';

export default function AuditPage() {
  const units = usePoll(getProcessUnits, 5000);
  const audit = usePoll(() => getAudit(200), 5000);

  const stages = units.data ? Object.entries(units.data.by_stage).sort((a, b) => b[1] - a[1]) : [];
  const maxStage = Math.max(1, ...stages.map(([, v]) => v));

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard title="Process Units — compute accounting" helpTerm="process_units">
        {units.error && !units.data ? <ErrorNote error={units.error} /> : null}
        {units.data ? (
          <>
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
              <MetricCard label="Total consumed" value={fmtInt(units.data.total)} term="process_units" />
              <MetricCard
                label="Per verified event"
                value={fmtNum(units.data.unit_economics.per_verified_event, 1)}
                sub="unit economics"
                info="Total process units divided by the number of verified labels — the compute cost of producing one trustworthy label. The number to drive down as automation improves."
              />
              <MetricCard
                label="Per million frames"
                value={fmtNum(units.data.unit_economics.per_million_frames, 1)}
                sub="unit economics"
                info="Process units normalized per million input frames — makes runs of different sizes comparable."
              />
              <MetricCard
                label="Per training dataset"
                value={fmtNum(units.data.unit_economics.per_training_dataset, 1)}
                sub="unit economics"
                info="Average process units consumed to produce one exported training dataset, end to end."
              />
            </Box>
            {stages.length > 0 ? (
              <>
                <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
                  BY STAGE
                </Typography>
                <Box sx={{ mt: 0.5, maxWidth: 560 }}>
                  {stages.map(([stage, v]) => (
                    <HBar key={stage} label={stage.replace(/_/g, ' ')} value={v} max={maxStage} color="#26a69a" />
                  ))}
                </Box>
              </>
            ) : (
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                No per-stage consumption yet.
              </Typography>
            )}
          </>
        ) : !units.error ? (
          <LoadingBox label="Loading process units…" />
        ) : null}
      </SectionCard>

      <SectionCard
        title={`Audit Trail (${audit.data?.events.length ?? 0} events)`}
        help="Append-only record of every state-changing action in the platform: who (actor — human reviewer or system component) did what (action) to which entity, when, and with what detail. This is the provenance layer that lineage and compliance queries rely on."
      >
        {audit.error && !audit.data ? <ErrorNote error={audit.error} /> : null}
        {audit.data && audit.data.events.length > 0 ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actor</TableCell>
                <TableCell>Action</TableCell>
                <TableCell>Entity</TableCell>
                <TableCell>Detail</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {audit.data.events.map((e) => (
                <TableRow key={e.event_id} hover>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12, whiteSpace: 'nowrap' }}>
                    {new Date(e.timestamp).toLocaleString()}
                  </TableCell>
                  <TableCell sx={{ fontSize: 12 }}>{e.actor}</TableCell>
                  <TableCell sx={{ fontSize: 12, fontWeight: 600 }}>{e.action}</TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {e.entity_type}/{e.entity_id}
                  </TableCell>
                  <TableCell sx={{ fontSize: 12, color: '#aab4be' }}>{e.detail}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : audit.data ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No audit events yet — every dataset generation, run, review action and training job is recorded here.
          </Typography>
        ) : null}
      </SectionCard>
    </Box>
  );
}
