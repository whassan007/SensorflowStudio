/** Failure Evidence Graph rendered as a status-colored node panel. */
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { EvidenceGraph } from '../../types/agentic';
import { EVIDENCE_STATUS_COLORS, EvidenceStatusChip, PanelTitle } from './common';

const NODE_ORDER = [
  'Object',
  'Environment',
  'Sensors',
  'Prediction',
  'GroundTruth',
  'Tracking',
  'Planner',
  'HistoricalSimilarity',
  'Frequency',
  'SafetyConsequence',
];

export default function EvidenceGraphPanel({ graph }: { graph: EvidenceGraph }) {
  const byId = new Map(graph.nodes.map((n) => [n.node_id, n]));
  const nodes = [...graph.nodes].sort(
    (a, b) => NODE_ORDER.indexOf(a.node_type) - NODE_ORDER.indexOf(b.node_type)
  );
  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
      <PanelTitle
        title="Failure Evidence Graph"
        origin="deterministic"
        extra={
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            {(['OBSERVED', 'DERIVED', 'HYPOTHESIS', 'UNAVAILABLE'] as const).map((s) => (
              <EvidenceStatusChip key={s} status={s} />
            ))}
          </Box>
        }
      />
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))',
          gap: 1,
        }}
      >
        {nodes.map((n) => (
          <Paper
            key={n.node_id}
            variant="outlined"
            sx={{
              p: 1,
              bgcolor: '#12171d',
              borderLeft: `3px solid ${EVIDENCE_STATUS_COLORS[n.status]}`,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.5 }}>
              <Typography variant="caption" sx={{ fontWeight: 800, flex: 1 }}>
                {n.node_type}
              </Typography>
              <EvidenceStatusChip status={n.status} />
            </Box>
            <Typography variant="caption" component="div" sx={{ color: '#c3ccd4' }}>
              {n.summary}
            </Typography>
            {n.caveats.length > 0 ? (
              <Typography variant="caption" component="div" sx={{ color: '#f9a825', mt: 0.5 }}>
                {n.caveats.join(' · ')}
              </Typography>
            ) : null}
            <Typography variant="caption" component="div" sx={{ color: '#5c6770', mt: 0.5, fontSize: 10 }}>
              source: {n.source}
            </Typography>
          </Paper>
        ))}
      </Box>
      <Typography variant="caption" component="div" sx={{ color: '#5c6770', mt: 1 }}>
        Relations:{' '}
        {graph.edges
          .map(
            (e) =>
              `${byId.get(e.src)?.node_type ?? '?'} —${e.relation}→ ${byId.get(e.dst)?.node_type ?? '?'}`
          )
          .join('  ·  ')}
      </Typography>
    </Paper>
  );
}
