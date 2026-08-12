import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import { FlaskConical } from 'lucide-react';
import type { BenchmarkResponse, BenchmarkRow } from '../../types/labeleval';
import { runBenchmark } from '../../services/labeleval';
import { SectionCard, fmtNum, fmtPct } from './shared';

type NumericKey = Exclude<keyof BenchmarkRow, 'technique'>;
type SortKey = keyof BenchmarkRow;

const COLUMNS: Array<{ key: SortKey; label: string; pct?: boolean }> = [
  { key: 'technique', label: 'Technique' },
  { key: 'precision', label: 'Precision', pct: true },
  { key: 'recall', label: 'Recall', pct: true },
  { key: 'rare_recall', label: 'Rare Recall', pct: true },
  { key: 'f1', label: 'F1', pct: true },
  { key: 'box_error_3d', label: '3D Box Error' },
  { key: 'id_swap_rate', label: 'ID Swap', pct: true },
  { key: 'consensus', label: 'Consensus', pct: true },
  { key: 'fp_rate', label: 'FP Rate', pct: true },
  { key: 'process_units', label: 'Process Units' },
];

const HIGHLIGHTS: Array<{ key: keyof BenchmarkResponse['highlights']; label: string; color: string }> = [
  { key: 'best_rare_recall', label: 'Best rare recall', color: '#ef5350' },
  { key: 'best_safety_recall', label: 'Best safety recall', color: '#ffa726' },
  { key: 'lowest_fp_rate', label: 'Lowest FP rate', color: '#ab47bc' },
  { key: 'lowest_process_units', label: 'Lowest process units', color: '#66bb6a' },
  { key: 'lowest_tracking_error', label: 'Lowest tracking error', color: '#42a5f5' },
];

export default function BenchmarkResults({
  benchmark,
  activeDatasetId,
  onRefresh,
}: {
  benchmark: BenchmarkResponse | null;
  activeDatasetId: string | null;
  onRefresh: () => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>('rare_recall');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [running, setRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const rows = useMemo(() => {
    const rows = [...(benchmark?.rows ?? [])];
    rows.sort((a, b) => {
      if (sortKey === 'technique') {
        return sortDir === 'asc' ? a.technique.localeCompare(b.technique) : b.technique.localeCompare(a.technique);
      }
      const av = a[sortKey as NumericKey];
      const bv = b[sortKey as NumericKey];
      return sortDir === 'asc' ? av - bv : bv - av;
    });
    return rows;
  }, [benchmark, sortKey, sortDir]);

  const run = async () => {
    setRunning(true);
    setErrorMsg(null);
    try {
      await runBenchmark(activeDatasetId);
      onRefresh();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else {
      setSortKey(key);
      setSortDir(key === 'technique' ? 'asc' : 'desc');
    }
  };

  return (
    <SectionCard
      title="Technique Benchmark — no single winner, each excels on a different axis"
      action={
        <Button
          size="small"
          variant="contained"
          startIcon={running ? <CircularProgress size={14} /> : <FlaskConical size={14} />}
          disabled={running}
          onClick={() => void run()}
        >
          Run Benchmark
        </Button>
      }
    >
      {errorMsg ? (
        <Alert severity="error" variant="outlined" onClose={() => setErrorMsg(null)} sx={{ mb: 1 }}>
          {errorMsg}
        </Alert>
      ) : null}

      {benchmark && rows.length > 0 ? (
        <>
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mb: 1.5 }}>
            {HIGHLIGHTS.map((h) => {
              const technique = benchmark.highlights[h.key];
              return technique ? (
                <Chip
                  key={h.key}
                  size="small"
                  label={`${h.label}: ${technique}`}
                  sx={{ bgcolor: '#12171d', border: `1px solid ${h.color}`, color: h.color, fontWeight: 600 }}
                />
              ) : null;
            })}
          </Box>
          <Table size="small">
            <TableHead>
              <TableRow>
                {COLUMNS.map((col) => (
                  <TableCell key={col.key} align={col.key === 'technique' ? 'left' : 'right'}>
                    <TableSortLabel
                      active={sortKey === col.key}
                      direction={sortKey === col.key ? sortDir : 'desc'}
                      onClick={() => toggleSort(col.key)}
                    >
                      {col.label}
                    </TableSortLabel>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.technique} hover>
                  <TableCell sx={{ fontWeight: 600 }}>{r.technique}</TableCell>
                  {COLUMNS.slice(1).map((col) => (
                    <TableCell key={col.key} align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                      {col.pct ? fmtPct(r[col.key as NumericKey]) : fmtNum(r[col.key as NumericKey], 2)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
            Benchmark {benchmark.benchmark_id} · {new Date(benchmark.created_at).toLocaleString()}
          </Typography>
        </>
      ) : (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          No benchmark results yet. Run the benchmark to compare detection techniques across quality, rare-event
          recall, tracking and compute-cost axes.
        </Typography>
      )}
    </SectionCard>
  );
}
