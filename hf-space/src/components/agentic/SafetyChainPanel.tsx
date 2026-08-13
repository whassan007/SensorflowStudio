/** Perception → tracking → prediction → planner → behavior chain with
 * explicit UNCERTAIN / UNAVAILABLE labeling. */
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { AgentResult } from '../../types/agentic';
import { EVIDENCE_STATUS_COLORS, KV, PanelTitle } from './common';

interface ChainLink {
  link: string;
  status: 'OBSERVED' | 'DERIVED' | 'HYPOTHESIS' | 'UNAVAILABLE';
  detail: string;
}

export default function SafetyChainPanel({ safety }: { safety: AgentResult }) {
  const chain = (safety.output.chain as ChainLink[] | undefined) ?? [];
  const kinematics = safety.output.kinematics as Record<string, unknown> | undefined;
  const worst = safety.output.worst_case as Record<string, unknown> | undefined;
  const uncertain = safety.output.behavioral_evidence === 'none';
  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
      <PanelTitle title="Safety Impact Chain" origin="ai" />
      {uncertain ? (
        <Typography
          variant="body2"
          sx={{ color: '#f9a825', fontWeight: 700, mb: 1 }}
        >
          {String(safety.output.assessment)}
        </Typography>
      ) : null}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'stretch', gap: 0.5, mb: 1 }}>
        {chain.map((c, i) => (
          <Box key={c.link} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Paper
              variant="outlined"
              sx={{
                p: 0.75,
                minWidth: 130,
                bgcolor: '#12171d',
                borderTop: `3px solid ${EVIDENCE_STATUS_COLORS[c.status]}`,
              }}
            >
              <Typography variant="caption" sx={{ fontWeight: 800, display: 'block' }}>
                {c.link.replace(/_/g, ' ')}
              </Typography>
              <Typography variant="caption" sx={{ color: EVIDENCE_STATUS_COLORS[c.status], fontWeight: 700 }}>
                {c.status === 'UNAVAILABLE' ? 'UNCERTAIN — no evidence' : c.status}
              </Typography>
              <Typography variant="caption" component="div" sx={{ color: '#8a949e', fontSize: 10.5 }}>
                {c.detail}
              </Typography>
            </Paper>
            {i < chain.length - 1 ? <Typography sx={{ color: '#5c6770' }}>→</Typography> : null}
          </Box>
        ))}
      </Box>
      {!uncertain && worst ? (
        <Box>
          <KV k="Worst replay min TTC" v={`${String(worst.min_ttc_s)} s`} mono />
          <KV k="Worst replay max DRAC" v={`${String(worst.max_drac)} m/s²`} mono />
          <KV k="Max collision probability" v={String(worst.max_p)} mono />
          {kinematics ? (
            <>
              <KV
                k="Stopping distance (nominal → delayed)"
                v={`${String(kinematics.stopping_distance_nominal_m)} m → ${String(kinematics.stopping_distance_delayed_m)} m`}
                mono
              />
              <KV
                k="Braking margin (nominal → delayed)"
                v={`${String(kinematics.braking_margin_nominal_m)} m → ${String(kinematics.braking_margin_delayed_m)} m`}
                mono
              />
              <KV k="Kinematics method" v={String(kinematics.method)} />
            </>
          ) : null}
          <Typography variant="caption" component="div" sx={{ color: '#5c6770', mt: 0.5 }}>
            {String(safety.output.coverage_note ?? '')}
          </Typography>
        </Box>
      ) : null}
    </Paper>
  );
}
