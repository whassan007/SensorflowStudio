import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import type { QueueStatus as QueueStatusType } from '../../types/labeleval';
import { getQueueStatus, usePoll } from '../../services/labeleval';
import { SectionCard, MetricCard, HBar, fmtInt, fmtNum, ErrorNote } from './shared';

export default function QueueStatus({ queue: queueProp }: { queue?: QueueStatusType | null }) {
  // Self-fetches unless a queue snapshot is supplied (e.g. from the SSE stream).
  const { data, error } = usePoll(getQueueStatus, queueProp === undefined ? 3000 : null, [
    queueProp === undefined,
  ]);
  const queue = queueProp !== undefined ? queueProp : data;

  const topics = queue ? Object.entries(queue.depth_by_topic) : [];
  const maxDepth = Math.max(1, ...topics.map(([, depth]) => depth));

  return (
    <SectionCard
      title="Queue Status"
      help="Live state of the message queue feeding the pipeline stages. Pending = waiting for a worker, Processing = in flight, Failed = errored messages that will be retried or dead-lettered. Depth-by-topic shows which stage is the current bottleneck."
      action={queue ? <Chip size="small" label={`backend: ${queue.backend}`} sx={{ bgcolor: '#232a31' }} /> : null}
    >
      {!queue && error ? <ErrorNote error={error} /> : null}
      {queue ? (
        <>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
            <MetricCard label="Pending" value={fmtInt(queue.pending)} accent="#ffa726" />
            <MetricCard label="Processing" value={fmtInt(queue.processing)} accent="#4fc3f7" />
            <MetricCard label="Completed" value={fmtInt(queue.completed)} accent="#66bb6a" />
            <MetricCard label="Failed" value={fmtInt(queue.failed)} accent="#ef5350" />
            <MetricCard label="Throughput" value={fmtNum(queue.throughput_per_s, 1)} sub="msgs / s" />
          </Box>
          {topics.length > 0 ? (
            <>
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                DEPTH BY TOPIC
              </Typography>
              <Box sx={{ mt: 0.5 }}>
                {topics.map(([topic, depth]) => (
                  <HBar key={topic} label={topic} value={depth} max={maxDepth} color="#7e57c2" />
                ))}
              </Box>
            </>
          ) : (
            <Typography variant="body2" sx={{ color: '#8a949e' }}>
              No topic depth data.
            </Typography>
          )}
        </>
      ) : !error ? (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          Waiting for queue data…
        </Typography>
      ) : null}
    </SectionCard>
  );
}
