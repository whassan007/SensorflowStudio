/**
 * Live progress card for an in-flight (non-published) evaluation run.
 * Fed by the SSE stream when available, otherwise by 1.5s polling.
 */
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';
import type { RunProgress } from '../../types/megaeval';
import { fmtCompact } from '../../services/megaeval';
import { SectionCard } from '../labeleval/shared';
import { CompactStat, RunStatusChip } from './shared';

function fmtEta(s: number | null): string {
  if (s === null || Number.isNaN(s)) return '—';
  if (s < 60) return `${Math.max(0, Math.round(s))}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export default function RunProgressCard({ progress }: { progress: RunProgress }) {
  // Tolerate either 0..1 or 0..100 percent encodings from the backend.
  const pct = Math.max(0, Math.min(100, progress.percent <= 1 ? progress.percent * 100 : progress.percent));
  return (
    <SectionCard
      title={
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>
            Evaluation run {progress.run_id.slice(0, 8)} · {progress.model_version}
          </span>
          <RunStatusChip status={progress.status} />
        </Box>
      }
    >
      <LinearProgress
        variant="determinate"
        value={pct}
        sx={{ height: 10, borderRadius: 1, mb: 1.5, bgcolor: '#232a31' }}
      />
      <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
        <CompactStat label="Progress" value={`${pct.toFixed(1)}%`} />
        <CompactStat
          label="Objects"
          value={`${fmtCompact(progress.objects_processed)} / ${fmtCompact(progress.objects_total)}`}
        />
        <CompactStat
          label="Partitions"
          value={`${progress.partitions_done} / ${progress.partitions_total}`}
        />
        <CompactStat label="Workers" value={progress.workers} />
        <CompactStat label="Throughput" value={`${fmtCompact(progress.throughput_objs_per_s)} objs/s`} />
        <CompactStat label="ETA" value={fmtEta(progress.eta_s)} />
      </Box>
      {progress.error ? (
        <Typography variant="body2" sx={{ color: '#ef5350', mt: 1.5, fontFamily: 'monospace' }}>
          {progress.error}
        </Typography>
      ) : (
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1.5 }}>
          Map partitions across workers → reduce partial aggregates → materialize cube &amp; indexes → publish.
        </Typography>
      )}
    </SectionCard>
  );
}
