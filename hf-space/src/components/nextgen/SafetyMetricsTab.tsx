/** Tab 3: safety-informed metrics — recall vs SCR divergence demo + risk-weighted breakdown. */
import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { TrendingDown, TrendingUp } from 'lucide-react';
import * as api from '../../services/nextgen';
import type { DivergenceDemo, SafetyReport } from '../../types/nextgen';
import { DataLabelChip, PANEL_SX } from './common';

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`;
}

function DeltaChip({ value }: { value: number | null }) {
  if (value === null) return <span>—</span>;
  const up = value >= 0;
  return (
    <Chip
      size="small"
      icon={up ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
      label={`${up ? '+' : ''}${(value * 100).toFixed(1)} pp`}
      sx={{
        height: 20,
        fontSize: 10.5,
        fontWeight: 800,
        bgcolor: up ? '#1b3a22' : '#3a1b1b',
        color: up ? '#81c784' : '#ef9a9a',
        '& .MuiChip-icon': { color: 'inherit' },
      }}
    />
  );
}

function Bars({ baseline, candidate, label }: { baseline: number | null | undefined; candidate: number | null | undefined; label: string }) {
  return (
    <Box sx={{ mb: 1 }}>
      <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 10.5 }}>
        {label}
      </Typography>
      {[
        { name: 'baseline', v: baseline, color: '#4fc3f7' },
        { name: 'candidate', v: candidate, color: '#ffb74d' },
      ].map((r) => (
        <Box key={r.name} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" sx={{ width: 62, color: '#c7ccd1', fontSize: 10 }}>
            {r.name}
          </Typography>
          <Box sx={{ flex: 1, height: 10, bgcolor: '#0d1117', borderRadius: 1, overflow: 'hidden' }}>
            <Box sx={{ width: `${((r.v ?? 0) * 100).toFixed(1)}%`, height: '100%', bgcolor: r.color }} />
          </Box>
          <Typography variant="caption" sx={{ width: 46, textAlign: 'right', fontFamily: 'monospace', fontSize: 10.5 }}>
            {pct(r.v)}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

function ByClassTable({ report, title }: { report: SafetyReport; title: string }) {
  return (
    <Paper sx={{ ...PANEL_SX, flex: '1 1 380px' }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
        {title}
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            {['class', 'n', 'recall', 'SCR', 'risk-weighted'].map((h) => (
              <TableCell key={h} sx={{ fontSize: 10.5, fontWeight: 800, color: '#8a949e' }}>
                {h}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {Object.entries(report.by_class).map(([cls, row]) => (
            <TableRow key={cls}>
              <TableCell sx={{ fontSize: 11 }}>{cls}</TableCell>
              <TableCell sx={{ fontSize: 11 }}>{row.n ?? row.n_objects ?? '—'}</TableCell>
              <TableCell sx={{ fontSize: 11, fontFamily: 'monospace' }}>{pct(row.recall)}</TableCell>
              <TableCell sx={{ fontSize: 11, fontFamily: 'monospace' }}>{pct(row.safety_critical_recall)}</TableCell>
              <TableCell sx={{ fontSize: 11, fontFamily: 'monospace' }}>{pct(row.risk_weighted_recall)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}

export default function SafetyMetricsTab() {
  const [demo, setDemo] = useState<DivergenceDemo | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.getDivergenceDemo().then(setDemo).catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!demo) return <Typography variant="body2" sx={{ color: '#8a949e' }}>Loading divergence demo…</Typography>;

  const d = demo.deltas;
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Paper sx={{ ...PANEL_SX, borderColor: '#e65100' }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mb: 0.5 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
            Recall ↑ / Safety-Critical Recall ↓ divergence demonstration
          </Typography>
          <DataLabelChip label={demo.data_label} />
        </Box>
        <Typography variant="body2" sx={{ color: '#c7ccd1', mb: 1 }}>
          {demo.headline}
        </Typography>
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
          {demo.construction}
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>overall recall</Typography>
            <DeltaChip value={d.overall_recall as number | null} />
          </Box>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>safety-critical recall</Typography>
            <DeltaChip value={d.safety_critical_recall as number | null} />
          </Box>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>risk-weighted recall</Typography>
            <DeltaChip value={d.risk_weighted_recall as number | null} />
          </Box>
        </Box>
      </Paper>

      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        <Paper sx={{ ...PANEL_SX, flex: '1 1 340px' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
            Open-loop vs safety-informed (both always reported)
          </Typography>
          <Bars label="Overall recall (open-loop)" baseline={demo.baseline.open_loop.recall} candidate={demo.candidate.open_loop.recall} />
          <Bars
            label="Safety-critical recall"
            baseline={demo.baseline.safety_informed.safety_critical_recall}
            candidate={demo.candidate.safety_informed.safety_critical_recall}
          />
          <Bars
            label="Risk-weighted recall"
            baseline={demo.baseline.safety_informed.risk_weighted_recall}
            candidate={demo.candidate.safety_informed.risk_weighted_recall}
          />
        </Paper>
        <Paper sx={{ ...PANEL_SX, flex: '1 1 340px' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
            Safety-critical region definition
          </Typography>
          <Typography variant="caption" sx={{ color: '#c7ccd1', display: 'block', mb: 1, lineHeight: 1.5 }}>
            {demo.baseline.region_definition}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
            {Object.entries(demo.baseline.region_params).map(([k, v]) => (
              <Chip key={k} size="small" label={`${k}=${v}`} sx={{ height: 20, fontSize: 10, fontFamily: 'monospace', bgcolor: '#12171d' }} />
            ))}
          </Box>
        </Paper>
      </Box>

      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        <ByClassTable report={demo.baseline} title="Baseline — per-class breakdown" />
        <ByClassTable report={demo.candidate} title="Candidate — per-class breakdown" />
      </Box>
    </Box>
  );
}
