import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { ClassMetrics, QualityMetrics, ScenarioMetrics } from '../../types/labeleval';
import { SectionCard, MetricCard, fmtPct, fmtInt } from './shared';

function PrTable({ title, rows }: { title: string; rows: Array<ClassMetrics | ScenarioMetrics> }) {
  return (
    <Box sx={{ flex: '1 1 320px', minWidth: 300 }}>
      <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
        {title}
      </Typography>
      {rows.length === 0 ? (
        <Typography variant="body2" sx={{ color: '#8a949e', mt: 1 }}>
          No data yet.
        </Typography>
      ) : (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>{'class_name' in (rows[0] ?? {}) ? 'Class' : 'Scenario'}</TableCell>
              <TableCell align="right">Precision</TableCell>
              <TableCell align="right">Recall</TableCell>
              <TableCell align="right">F1</TableCell>
              <TableCell align="right">Support</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((r) => {
              const name = 'class_name' in r ? r.class_name : r.scenario;
              return (
                <TableRow key={name} hover>
                  <TableCell sx={{ fontWeight: 600 }}>{name.replace(/_/g, ' ')}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(r.precision)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(r.recall)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(r.f1)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtInt(r.support)}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </Box>
  );
}

export default function GroundTruthComparison({ metrics }: { metrics: QualityMetrics | null }) {
  const gtType = metrics?.gt_type ?? null;
  return (
    <SectionCard title="Ground Truth Comparison">
      {!metrics ? (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          No quality metrics yet — run the evaluation pipeline.
        </Typography>
      ) : (
        <>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', mb: 1.5 }}>
            <Chip
              size="small"
              label={metrics.gt_available ? 'GT available' : 'No GT reference'}
              sx={{
                bgcolor: metrics.gt_available ? '#1b3a24' : '#4a1f1f',
                color: metrics.gt_available ? '#a5d6a7' : '#ef9a9a',
                fontWeight: 700,
              }}
            />
            {gtType ? (
              <Chip
                size="small"
                label={gtType.replace(/_/g, ' ')}
                sx={{
                  bgcolor: gtType === 'PSEUDO_GROUND_TRUTH' ? '#4a2c00' : '#1b3a24',
                  color: gtType === 'PSEUDO_GROUND_TRUTH' ? '#ffcc80' : '#a5d6a7',
                  fontWeight: 700,
                }}
              />
            ) : null}
            <MetricCard label="Coverage" value={fmtPct(metrics.gt_coverage)} />
            <MetricCard label="Evaluation confidence" value={metrics.gt_available ? 'per-dataset' : 'none'} />
          </Box>

          {gtType === 'PSEUDO_GROUND_TRUTH' ? (
            <Alert severity="warning" variant="outlined" sx={{ mb: 1.5 }}>
              Pseudo-GT comparison, not gold-standard evaluation — reference labels were themselves machine-generated.
              Treat precision/recall below as directional signals, not certified quality figures.
            </Alert>
          ) : null}

          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <PrTable title="PER-CLASS PRECISION / RECALL" rows={metrics.per_class} />
            <PrTable title="PER-SCENARIO PRECISION / RECALL" rows={metrics.per_scenario} />
          </Box>
        </>
      )}
    </SectionCard>
  );
}
