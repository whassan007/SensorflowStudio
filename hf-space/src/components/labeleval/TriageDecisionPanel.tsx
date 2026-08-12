import { useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { TriageDecision } from '../../types/labeleval';
import { SectionCard, StatusChip, GateLineList, HBar, fmtNum } from './shared';

export default function TriageDecisionPanel({ decisions }: { decisions: TriageDecision[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = decisions.find((d) => d.decision_id === selectedId) ?? null;

  const policy = decisions[0] ?? null;

  const reasonCounts = new Map<string, number>();
  for (const d of decisions) {
    for (const r of d.failure_reasons) reasonCounts.set(r, (reasonCounts.get(r) ?? 0) + 1);
  }
  const reasonRows = [...reasonCounts.entries()].sort((a, b) => b[1] - a[1]);
  const maxReason = Math.max(1, ...reasonRows.map(([, c]) => c));

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard title="Active Quality Policy">
        {policy ? (
          <>
            <Chip label={policy.policy_id} sx={{ bgcolor: '#12314a', color: '#90caf9', fontWeight: 700, mb: 1.5 }} />
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {Object.entries(policy.policy_values).map(([k, v]) => (
                <Chip
                  key={k}
                  size="small"
                  label={`${k} = ${fmtNum(v)}`}
                  sx={{ bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }}
                />
              ))}
            </Box>
          </>
        ) : (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No triage decisions yet — the policy appears after the first evaluation run.
          </Typography>
        )}
      </SectionCard>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <SectionCard title="Recent Triage Decisions" sx={{ flex: '2 1 460px' }}>
          {decisions.length === 0 ? (
            <Typography variant="body2" sx={{ color: '#8a949e' }}>
              No decisions to show.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Annotation</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Primary failure reason</TableCell>
                  <TableCell>Decided</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {decisions.map((d) => (
                  <TableRow
                    key={d.decision_id}
                    hover
                    selected={d.decision_id === selectedId}
                    onClick={() => setSelectedId(d.decision_id)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{d.annotation_id}</TableCell>
                    <TableCell>
                      <StatusChip status={d.status} />
                    </TableCell>
                    <TableCell sx={{ fontSize: 12, color: d.primary_failure_reason ? '#ef9a9a' : '#8a949e' }}>
                      {d.primary_failure_reason ? d.primary_failure_reason.replace(/_/g, ' ') : '—'}
                    </TableCell>
                    <TableCell sx={{ fontSize: 12 }}>{new Date(d.decided_at).toLocaleTimeString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </SectionCard>

        <Box sx={{ flex: '1 1 340px', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <SectionCard title="Gate Explainability">
            {selected ? (
              <>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selected.annotation_id}
                  </Typography>
                  <StatusChip status={selected.status} />
                </Box>
                {selected.primary_failure_reason ? (
                  <Typography variant="body2" sx={{ color: '#ef9a9a', mb: 1 }}>
                    Primary failure: <strong>{selected.primary_failure_reason.replace(/_/g, ' ')}</strong>
                  </Typography>
                ) : (
                  <Typography variant="body2" sx={{ color: '#a5d6a7', mb: 1 }}>
                    All applicable gates passed.
                  </Typography>
                )}
                <GateLineList checks={selected.gate_lines} />
              </>
            ) : (
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                Select a decision to see its per-gate ✓/✗ breakdown (actual vs. threshold).
              </Typography>
            )}
          </SectionCard>

          <SectionCard title="Failure Reason Distribution">
            {reasonRows.length === 0 ? (
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                No failures recorded.
              </Typography>
            ) : (
              reasonRows.map(([reason, count]) => (
                <HBar
                  key={reason}
                  label={reason.replace(/_/g, ' ')}
                  value={count}
                  max={maxReason}
                  color="#ef5350"
                />
              ))
            )}
          </SectionCard>
        </Box>
      </Box>
    </Box>
  );
}
