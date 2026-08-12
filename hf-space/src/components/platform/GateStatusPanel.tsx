/**
 * Multi-gate readiness skeleton: Scenario / Coverage / Regression / Safety / Release
 * (+ quality / launch when wired).
 */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { CheckCircle2, CircleDashed, XCircle } from 'lucide-react';
import { useGateStatus } from '../../services/platform';
import { ErrorNote, LoadingBox, SectionCard } from '../labeleval/shared';

const DISPLAY_ORDER = ['scenario', 'coverage', 'regression', 'safety', 'quality', 'launch', 'release'];

function statusChip(passed: boolean | null, ready: boolean) {
  if (!ready) {
    return <Chip size="small" icon={<CircleDashed size={14} />} label="Skeleton" variant="outlined" />;
  }
  if (passed === true) {
    return <Chip size="small" color="success" icon={<CheckCircle2 size={14} />} label="Pass" />;
  }
  if (passed === false) {
    return <Chip size="small" color="error" icon={<XCircle size={14} />} label="Fail" />;
  }
  return <Chip size="small" label="Pending inputs" variant="outlined" />;
}

export default function GateStatusPanel({
  candidateRunId,
  baselineRunId,
  refreshKey = 0,
}: {
  candidateRunId: string | null;
  baselineRunId: string | null;
  refreshKey?: number;
}) {
  const poll = useGateStatus(candidateRunId, baselineRunId, refreshKey);

  if (poll.error && !poll.data) return <ErrorNote error={poll.error} />;
  if (poll.loading && !poll.data) return <LoadingBox label="Loading gate status…" />;
  const data = poll.data;
  if (!data) return null;

  const byType = Object.fromEntries(data.gates.map((g) => [g.gate_type, g]));

  return (
    <SectionCard
      title="Multi-gate readiness"
      action={
        <Typography variant="caption" color="text.secondary">
          {data.summary.release_ready
            ? 'Release ready'
            : `${data.summary.unwired_count} gates still Phase 2–6 stubs`}
        </Typography>
      }
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {DISPLAY_ORDER.map((type) => {
          const g = byType[type];
          if (!g) return null;
          return (
            <Box
              key={type}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1.5,
                py: 0.75,
                borderBottom: '1px solid #232a31',
              }}
            >
              <Typography sx={{ minWidth: 110, fontWeight: 600, textTransform: 'capitalize' }}>
                {type}
              </Typography>
              {statusChip(g.passed, g.ready)}
              <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
                {g.message}
              </Typography>
            </Box>
          );
        })}
      </Box>
    </SectionCard>
  );
}
