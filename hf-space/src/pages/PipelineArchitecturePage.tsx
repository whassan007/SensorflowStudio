import { useState } from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import { X } from 'lucide-react';
import type { PipelineStateResponse, ServiceState, ServiceStatus } from '../types/labeleval';
import { getPipeline, usePoll } from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import QueueStatus from '../components/labeleval/QueueStatus';
import { SectionCard, StatusChip, LoadingBox, ErrorNote, fmtInt } from '../components/labeleval/shared';

// ---------------------------------------------------------------- graph layout

interface NodeDef {
  id: string;
  label: string;
  x: number;
  y: number;
  aliases: string[];
  desc: string;
}

const NODE_W = 116;
const NODE_H = 52;

const NODES: NodeDef[] = [
  { id: 'input', label: 'Input', x: 70, y: 235, aliases: ['input', 'ingest', 'sensor'], desc: 'Ingests raw sensor data: camera frames, LiDAR sweeps, radar and ego-pose. Everything downstream evaluates labels produced on these frames.' },
  { id: 'autolabels', label: 'Auto Labels', x: 215, y: 235, aliases: ['auto_label', 'auto-label', 'autolabel', 'labeling', 'labeler'], desc: 'The current model auto-labels each frame with 3D boxes, classes, confidences and track IDs. These labels are candidates, not truth — the rest of the pipeline exists to grade them.' },
  { id: 'queue', label: 'Queue', x: 360, y: 235, aliases: ['queue', 'broker'], desc: 'Message queue distributing labeled frames to the three evaluation engines. Depth per stage is visible in the Queue Status card below.' },
  { id: 'anomaly', label: 'Anomaly Detection', x: 505, y: 100, aliases: ['anomaly'], desc: 'Ensemble anomaly scoring: geometric plausibility, point-cloud statistics, temporal consistency. High scores flag physically implausible or novel labels.' },
  { id: 'regression', label: 'Regression', x: 505, y: 235, aliases: ['regression'], desc: 'Compares the labeling model against its registered baseline. Detected regressions block auto-verification for affected labels.' },
  { id: 'grader', label: 'Grader', x: 505, y: 370, aliases: ['grader', 'grading', 'consensus'], desc: 'Multiple independent grader models score each label; their agreement (consensus) is a confidence signal. Low consensus routes to human review.' },
  { id: 'validation', label: 'Quality Validation', x: 655, y: 235, aliases: ['validation', 'quality', 'gate'], desc: 'The quality gate: combines all engine evidence and applies the versioned quality policy thresholds, gate by gate, to produce a triage decision.' },
  { id: 'rare', label: 'Rare Events', x: 805, y: 100, aliases: ['rare'], desc: 'Mines the long tail: high-rarity, high-severity events are surfaced for review and prioritized for training data (they are worth the most).' },
  { id: 'autograded', label: 'Auto-Graded', x: 805, y: 235, aliases: ['auto_grad', 'auto-grad', 'autograd', 'triage'], desc: 'Labels that passed every gate with margin — verified automatically without human involvement. The automation-rate numerator.' },
  { id: 'review', label: 'Review (HITL)', x: 805, y: 370, aliases: ['review', 'hitl'], desc: 'Human-in-the-loop queue: flagged labels wait here for a reviewer verdict, ordered by severity. The badge shows the current queue depth.' },
  { id: 'relabel', label: 'Re-Label', x: 950, y: 370, aliases: ['relabel', 're-label', 're_label'], desc: 'Reviewer corrections are applied here, then fed back into Quality Validation — corrected labels are never trusted without re-passing the gates.' },
  { id: 'training', label: 'Training', x: 1095, y: 235, aliases: ['train'], desc: 'Verified labels become training data for the next model version, closing the flywheel. Only verified data ever reaches this node.' },
];

const EDGES: Array<{ from: string; to: string; label?: 'verified' }> = [
  { from: 'input', to: 'autolabels' },
  { from: 'autolabels', to: 'queue' },
  { from: 'queue', to: 'anomaly' },
  { from: 'queue', to: 'regression' },
  { from: 'queue', to: 'grader' },
  { from: 'anomaly', to: 'validation' },
  { from: 'regression', to: 'validation' },
  { from: 'grader', to: 'validation' },
  { from: 'validation', to: 'rare' },
  { from: 'validation', to: 'autograded' },
  { from: 'validation', to: 'review' },
  { from: 'review', to: 'relabel' },
  { from: 'relabel', to: 'validation' },
  { from: 'rare', to: 'review' },
  { from: 'autograded', to: 'training', label: 'verified' },
];

const STATE_COLORS: Record<ServiceState, string> = {
  HEALTHY: '#66bb6a',
  RUNNING: '#42a5f5',
  DEGRADED: '#f9a825',
  BLOCKED: '#ff9800',
  FAILED: '#ef5350',
  IDLE: '#5c6773',
};

function findService(pipeline: PipelineStateResponse | null, node: NodeDef): ServiceStatus | null {
  if (!pipeline) return null;
  return (
    pipeline.services.find((s) => {
      const name = s.service.toLowerCase();
      return node.aliases.some((a) => name.includes(a));
    }) ?? null
  );
}

function nodeById(id: string): NodeDef {
  const node = NODES.find((n) => n.id === id);
  if (!node) throw new Error(`Unknown pipeline node ${id}`);
  return node;
}

export default function PipelineArchitecturePage() {
  const { stream } = useLabelEval();
  const pipelinePoll = usePoll(getPipeline, stream ? null : 2000, [stream === null]);
  const pipeline = stream?.pipeline ?? pipelinePoll.data ?? null;
  const [selectedNode, setSelectedNode] = useState<NodeDef | null>(null);

  const running = pipeline?.running ?? false;
  const selectedService = selectedNode ? findService(pipeline, selectedNode) : null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {!pipeline && pipelinePoll.loading ? <LoadingBox label="Loading pipeline state…" /> : null}
      {!pipeline && pipelinePoll.error ? <ErrorNote error={pipelinePoll.error} /> : null}

      <SectionCard
        title="Pipeline Architecture"
        help="Live map of every service in the label-quality pipeline, left to right: sensor input → auto-labeling → fan-out to three parallel evaluation engines (anomaly, regression, grader) → quality gate → routing (auto-graded / human review / rare events) → re-label loop → training. Node border color is service health; the counter under each node is processed/total; click a node for its role, live status and process-unit consumption."
        action={
          pipeline ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <StatusChip status={running ? 'RUNNING' : 'IDLE'} />
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                stage: {pipeline.stage || '—'}
                {pipeline.last_run_id ? ` · run ${pipeline.last_run_id}` : ''}
              </Typography>
            </Box>
          ) : undefined
        }
      >
        <svg width="100%" viewBox="0 0 1180 460" style={{ minWidth: 720 }}>
          <defs>
            <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a5561" />
            </marker>
          </defs>

          {/* edges */}
          {EDGES.map((edge, i) => {
            const from = nodeById(edge.from);
            const to = nodeById(edge.to);
            const startX = from.x + NODE_W / 2;
            const endX = to.x - NODE_W / 2 - 4;
            // feedback edge relabel -> validation drawn below
            const isFeedback = edge.from === 'relabel' && edge.to === 'validation';
            const midX = (startX + endX) / 2;
            const d = isFeedback
              ? `M ${from.x} ${from.y + NODE_H / 2} C ${from.x} ${from.y + 75}, ${to.x} ${to.y + 210}, ${to.x} ${to.y + NODE_H / 2 + 4}`
              : `M ${startX} ${from.y} C ${midX} ${from.y}, ${midX} ${to.y}, ${endX} ${to.y}`;
            const verifiedCount = edge.label === 'verified' ? pipeline?.counters.verified ?? 0 : null;
            return (
              <g key={i}>
                <path
                  d={d}
                  fill="none"
                  stroke="#4a5561"
                  strokeWidth={1.5}
                  markerEnd="url(#arch-arrow)"
                  className={running ? 'pipeline-edge-active' : undefined}
                />
                {verifiedCount !== null ? (
                  <text x={midX} y={(from.y + to.y) / 2 - 8} textAnchor="middle" fill="#66bb6a" fontSize={11} fontWeight={700}>
                    {fmtInt(verifiedCount)} verified
                  </text>
                ) : null}
              </g>
            );
          })}

          {/* nodes */}
          {NODES.map((node) => {
            const service = findService(pipeline, node);
            const state: ServiceState = service?.state ?? 'IDLE';
            const color = STATE_COLORS[state];
            const flaggedBadge = node.id === 'review' ? pipeline?.review_queue_count ?? 0 : 0;
            const regressionDot = node.id === 'regression' && (pipeline?.regression_alert ?? false);
            return (
              <g
                key={node.id}
                onClick={() => setSelectedNode(node)}
                style={{ cursor: 'pointer' }}
                className={state === 'RUNNING' ? 'pipeline-node-running' : undefined}
              >
                <rect
                  x={node.x - NODE_W / 2}
                  y={node.y - NODE_H / 2}
                  width={NODE_W}
                  height={NODE_H}
                  rx={8}
                  fill="#161b21"
                  stroke={color}
                  strokeWidth={2}
                />
                <text x={node.x} y={node.y - 2} textAnchor="middle" fill="#e6e9ec" fontSize={11.5} fontWeight={700}>
                  {node.label}
                </text>
                <text x={node.x} y={node.y + 14} textAnchor="middle" fill={color} fontSize={9.5} fontWeight={600}>
                  {state}
                </text>
                <text x={node.x} y={node.y + NODE_H / 2 + 14} textAnchor="middle" fill="#8a949e" fontSize={10} fontFamily="monospace">
                  {service ? `${service.processed.toLocaleString()} / ${service.total.toLocaleString()}` : '— / —'}
                </text>
                {flaggedBadge > 0 ? (
                  <g>
                    <circle cx={node.x + NODE_W / 2 - 4} cy={node.y - NODE_H / 2 + 2} r={11} fill="#e65100" />
                    <text
                      x={node.x + NODE_W / 2 - 4}
                      y={node.y - NODE_H / 2 + 6}
                      textAnchor="middle"
                      fill="#fff"
                      fontSize={10}
                      fontWeight={800}
                    >
                      {flaggedBadge > 99 ? '99+' : flaggedBadge}
                    </text>
                  </g>
                ) : null}
                {regressionDot ? (
                  <circle cx={node.x - NODE_W / 2 + 4} cy={node.y - NODE_H / 2 + 2} r={6} fill="#ef5350">
                    <animate attributeName="opacity" values="1;0.3;1" dur="1s" repeatCount="indefinite" />
                  </circle>
                ) : null}
              </g>
            );
          })}
        </svg>
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          Click any node for its live service status and process-unit consumption. Edges animate while the pipeline is
          running.
        </Typography>
      </SectionCard>

      <QueueStatus queue={pipeline?.queue ?? undefined} />

      <Drawer
        anchor="right"
        open={selectedNode !== null}
        onClose={() => setSelectedNode(null)}
        PaperProps={{ sx: { width: 360, bgcolor: '#12171d' } }}
      >
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="h6">{selectedNode?.label}</Typography>
            <IconButton size="small" onClick={() => setSelectedNode(null)}>
              <X size={18} />
            </IconButton>
          </Box>
          {selectedNode ? (
            <Typography variant="body2" sx={{ color: '#aab4be', mb: 1.5 }}>
              {selectedNode.desc}
            </Typography>
          ) : null}
          {selectedService ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <StatusChip status={selectedService.state} />
              <Typography variant="body2" sx={{ color: '#aab4be' }}>
                Service: <strong>{selectedService.service}</strong>
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {selectedService.processed.toLocaleString()} / {selectedService.total.toLocaleString()} processed
              </Typography>
              <Typography variant="body2" sx={{ color: '#aab4be' }}>
                Process units consumed: <strong>{fmtInt(selectedService.process_units)}</strong>
              </Typography>
              {selectedService.detail ? (
                <Typography variant="body2" sx={{ color: '#8a949e' }}>
                  {selectedService.detail}
                </Typography>
              ) : null}
            </Box>
          ) : (
            <Typography variant="body2" sx={{ color: '#8a949e' }}>
              No live service is reporting for this node yet.
            </Typography>
          )}
        </Box>
      </Drawer>
    </Box>
  );
}
