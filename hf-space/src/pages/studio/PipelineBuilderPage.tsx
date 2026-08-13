/**
 * Pipeline Builder — direct-manipulation node-graph editor over the platform's
 * pipeline architecture. The as-built pipeline loads read-only (with live
 * service status where available); the user edits a clearly-labeled DRAFT:
 * drag stages, draw/remove edges, enable/disable stages, and read a live
 * "what will run" summary (topological order, orphans, cycles, diff vs
 * as-built). Nothing here executes — it is an explorer + what-if composer.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { Download, RotateCcw, Unlink } from 'lucide-react';
import NodeCanvas, { type GraphEdge, type GraphNode } from '../../components/visual/NodeCanvas';
import InspectorPanel, { type InspectorField } from '../../components/visual/InspectorPanel';
import { getPipeline } from '../../services/labeleval';
import type { PipelineStateResponse } from '../../types/labeleval';
import { loadLayout, saveLayout } from '../../services/studioux';
import { SectionCard } from '../../components/labeleval/shared';
import { useLabelEval } from '../../context/LabelEvalContext';
import { tokens } from '../../theme';

// ------------------------------------------------------- as-built definition
// Mirrors the read-only Pipeline Architecture page's graph (kept local so this
// page never touches that file).

interface StageDef {
  id: string;
  label: string;
  x: number;
  y: number;
  role: string;
  aliases: string[];
}

const AS_BUILT_STAGES: StageDef[] = [
  { id: 'input', label: 'Input', x: 90, y: 240, role: 'Ingests raw sensor data: camera frames, LiDAR sweeps, radar, ego-pose.', aliases: ['input', 'ingest', 'sensor'] },
  { id: 'autolabels', label: 'Auto Labels', x: 250, y: 240, role: 'The current model auto-labels each frame — candidates, not truth.', aliases: ['auto_label', 'autolabel', 'labeler'] },
  { id: 'queue', label: 'Queue', x: 410, y: 240, role: 'Message queue distributing labeled frames to the evaluation engines.', aliases: ['queue', 'broker'] },
  { id: 'anomaly', label: 'Anomaly Detection', x: 570, y: 100, role: 'Ensemble anomaly scoring: geometric plausibility, point statistics, temporal consistency.', aliases: ['anomaly'] },
  { id: 'regression', label: 'Regression', x: 570, y: 240, role: 'Compares the labeling model against its registered baseline.', aliases: ['regression'] },
  { id: 'grader', label: 'Grader', x: 570, y: 380, role: 'Independent grader models score each label; consensus is a confidence signal.', aliases: ['grader', 'consensus'] },
  { id: 'validation', label: 'Quality Validation', x: 740, y: 240, role: 'The quality gate: combines engine evidence, applies policy thresholds, produces triage decisions.', aliases: ['validation', 'quality', 'gate'] },
  { id: 'rare', label: 'Rare Events', x: 910, y: 100, role: 'Mines the long tail: high-rarity, high-severity events surfaced for review and training.', aliases: ['rare'] },
  { id: 'autograded', label: 'Auto-Graded', x: 910, y: 240, role: 'Labels that passed every gate with margin — verified without human involvement.', aliases: ['auto_grad', 'triage'] },
  { id: 'review', label: 'Review (HITL)', x: 910, y: 380, role: 'Human-in-the-loop queue: flagged labels wait for a reviewer verdict.', aliases: ['review', 'hitl'] },
  { id: 'relabel', label: 'Re-Label', x: 1070, y: 380, role: 'Reviewer corrections re-enter Quality Validation — never trusted without re-passing gates.', aliases: ['relabel'] },
  { id: 'training', label: 'Training', x: 1230, y: 240, role: 'Verified labels become training data for the next model version — the flywheel.', aliases: ['train'] },
];

const AS_BUILT_EDGES: Array<[string, string]> = [
  ['input', 'autolabels'],
  ['autolabels', 'queue'],
  ['queue', 'anomaly'],
  ['queue', 'regression'],
  ['queue', 'grader'],
  ['anomaly', 'validation'],
  ['regression', 'validation'],
  ['grader', 'validation'],
  ['validation', 'rare'],
  ['validation', 'autograded'],
  ['validation', 'review'],
  ['review', 'relabel'],
  ['relabel', 'validation'],
  ['rare', 'review'],
  ['autograded', 'training'],
];

const ROLE_COLORS: Record<string, string> = {
  input: '#4fc3f7',
  autolabels: '#4fc3f7',
  queue: '#8a949e',
  anomaly: '#ab47bc',
  regression: '#ab47bc',
  grader: '#ab47bc',
  validation: '#f9a825',
  rare: '#ff7043',
  autograded: '#66bb6a',
  review: '#ff7043',
  relabel: '#ff7043',
  training: '#66bb6a',
};

// ------------------------------------------------------------- draft state

interface DraftStage {
  id: string;
  x: number;
  y: number;
  enabled: boolean;
  note: string;
}

interface Draft {
  stages: DraftStage[];
  edges: Array<[string, string]>;
}

const LAYOUT_KEY = 'pipeline-builder-draft';

function asBuiltDraft(): Draft {
  return {
    stages: AS_BUILT_STAGES.map((s) => ({ id: s.id, x: s.x, y: s.y, enabled: true, note: '' })),
    edges: AS_BUILT_EDGES.map((e) => [...e] as [string, string]),
  };
}

/** Kahn topological sort over enabled stages; returns order + cycle members. */
function analyze(draft: Draft): { order: string[]; cycle: string[]; orphans: string[] } {
  const enabled = new Set(draft.stages.filter((s) => s.enabled).map((s) => s.id));
  const edges = draft.edges.filter(([f, t]) => enabled.has(f) && enabled.has(t));
  const indeg = new Map<string, number>();
  enabled.forEach((id) => indeg.set(id, 0));
  edges.forEach(([, t]) => indeg.set(t, (indeg.get(t) ?? 0) + 1));
  const queue = [...enabled].filter((id) => (indeg.get(id) ?? 0) === 0);
  const order: string[] = [];
  const deg = new Map(indeg);
  const q = [...queue];
  while (q.length) {
    const id = q.shift() as string;
    order.push(id);
    edges.filter(([f]) => f === id).forEach(([, t]) => {
      deg.set(t, (deg.get(t) ?? 0) - 1);
      if ((deg.get(t) ?? 0) === 0) q.push(t);
    });
  }
  const cycle = [...enabled].filter((id) => !order.includes(id));
  const touched = new Set(edges.flat());
  const orphans = [...enabled].filter((id) => !touched.has(id));
  return { order, cycle, orphans };
}

function diffVsAsBuilt(draft: Draft): string[] {
  const changes: string[] = [];
  const asKey = new Set(AS_BUILT_EDGES.map(([f, t]) => `${f}→${t}`));
  const drKey = new Set(draft.edges.map(([f, t]) => `${f}→${t}`));
  draft.stages.filter((s) => !s.enabled).forEach((s) => changes.push(`stage "${s.id}" disabled`));
  [...drKey].filter((k) => !asKey.has(k)).forEach((k) => changes.push(`edge ${k} added`));
  [...asKey].filter((k) => !drKey.has(k)).forEach((k) => changes.push(`edge ${k} removed`));
  return changes;
}

// ---------------------------------------------------------------- page

export default function PipelineBuilderPage() {
  const { stream, navigate } = useLabelEval();
  const [draft, setDraft] = useState<Draft>(asBuiltDraft);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStateResponse | null>(null);
  const loaded = useRef(false);

  useEffect(() => {
    loadLayout<Draft>(LAYOUT_KEY).then((saved) => {
      if (saved && Array.isArray(saved.stages) && saved.stages.length) setDraft(saved);
      loaded.current = true;
    });
    getPipeline().then(setPipeline).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (loaded.current) saveLayout(LAYOUT_KEY, draft);
  }, [draft]);

  const live = stream?.pipeline ?? pipeline;

  const stageDef = useCallback((id: string) => AS_BUILT_STAGES.find((s) => s.id === id), []);

  const liveState = useCallback(
    (id: string): string | null => {
      const def = stageDef(id);
      if (!def || !live) return null;
      const svc = live.services.find((s) => def.aliases.some((a) => s.service.toLowerCase().includes(a)));
      return svc?.state ?? null;
    },
    [live, stageDef]
  );

  const nodes: GraphNode[] = useMemo(
    () =>
      draft.stages.map((s) => {
        const def = stageDef(s.id);
        const state = liveState(s.id);
        return {
          id: s.id,
          x: s.x,
          y: s.y,
          label: def?.label ?? s.id,
          sublabel: s.enabled ? state ?? 'stage' : 'DISABLED',
          color: s.enabled ? ROLE_COLORS[s.id] ?? tokens.color.borderStrong : tokens.color.textFaint,
        };
      }),
    [draft.stages, stageDef, liveState]
  );

  const edges: GraphEdge[] = useMemo(() => {
    const enabled = new Set(draft.stages.filter((s) => s.enabled).map((s) => s.id));
    return draft.edges.map(([from, to]) => ({
      from,
      to,
      dashed: !enabled.has(from) || !enabled.has(to),
      color: !enabled.has(from) || !enabled.has(to) ? tokens.color.textFaint : undefined,
    }));
  }, [draft]);

  const summary = useMemo(() => analyze(draft), [draft]);
  const changes = useMemo(() => diffVsAsBuilt(draft), [draft]);

  const selected = draft.stages.find((s) => s.id === selectedId) ?? null;
  const selectedDef = selectedId ? stageDef(selectedId) : undefined;
  const selectedEdges = draft.edges.filter(([f, t]) => f === selectedId || t === selectedId);

  const inspectorFields: InspectorField[] = selected
    ? [
        { type: 'readonly', key: 'role', label: 'Role in the pipeline', value: selectedDef?.role ?? '' },
        ...(liveState(selected.id)
          ? [{ type: 'readonly', key: 'live', label: 'Live service state (as-built pipeline)', value: liveState(selected.id) } as InspectorField]
          : []),
        { type: 'toggle', key: 'enabled', label: 'Enabled in draft', value: selected.enabled, help: 'Disabled stages are excluded from the what-will-run order; edges through them go inert (dashed).' },
        { type: 'text', key: 'note', label: 'Draft note', value: selected.note, help: 'Free-form what-if annotation, stored with the draft.' },
      ]
    : [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Alert severity="info" variant="outlined" sx={{ py: 0.25 }}>
        <strong>DRAFT — nothing here executes.</strong> The real pipeline is read-only (see{' '}
        <Box component="span" onClick={() => navigate('pipeline')} sx={{ color: tokens.color.info, cursor: 'pointer', textDecoration: 'underline' }}>
          Pipeline Architecture
        </Box>
        {' '}for the as-built map). This is a visual explorer and what-if composer: your changes live in a local draft you can export.
      </Alert>

      <SectionCard
        title="Draft pipeline graph"
        help="Drag stages to rearrange. Drag from a stage's right-hand port onto another stage to draw a new edge. Click a stage to inspect and edit it (enable/disable, note). Live service states from the running pipeline are shown on nodes where available."
        action={
          <Box sx={{ display: 'flex', gap: 0.75 }}>
            <Button size="small" startIcon={<RotateCcw size={14} />} onClick={() => { setDraft(asBuiltDraft()); setSelectedId(null); }}>
              Reset to as-built
            </Button>
            <Button
              size="small"
              startIcon={<Download size={14} />}
              onClick={() => {
                const payload = { schema: 'sensorflow.pipeline-draft/v1', draft, analysis: summary, changes };
                const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'pipeline-draft.json';
                a.click();
                URL.revokeObjectURL(a.href);
              }}
            >
              Export draft
            </Button>
          </Box>
        }
      >
        <NodeCanvas
          nodes={nodes}
          edges={edges}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onMoveNode={(id, x, y) => setDraft((d) => ({ ...d, stages: d.stages.map((s) => (s.id === id ? { ...s, x, y } : s)) }))}
          onConnect={(from, to) =>
            setDraft((d) =>
              d.edges.some(([f, t]) => f === from && t === to) ? d : { ...d, edges: [...d.edges, [from, to]] }
            )
          }
          world={{ x: 0, y: 0, w: 1340, h: 500 }}
          height={430}
          ariaLabel="Pipeline draft node graph"
        />
      </SectionCard>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <Box sx={{ flex: '1 1 300px', minWidth: 280 }}>
          <InspectorPanel
            title={selectedDef?.label ?? 'Stage inspector'}
            subtitle={selected ? `stage id: ${selected.id}` : undefined}
            accent={selected ? ROLE_COLORS[selected.id] : undefined}
            fields={inspectorFields}
            emptyHint="Click a stage on the canvas to inspect its role, live status and draft settings."
            onChange={(key, value) => {
              if (!selectedId) return;
              setDraft((d) => ({
                ...d,
                stages: d.stages.map((s) => (s.id === selectedId ? { ...s, [key]: value } : s)),
              }));
            }}
            footer={
              selected && selectedEdges.length ? (
                <Box>
                  <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mb: 0.5 }}>
                    Connections — click to remove from the draft:
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {selectedEdges.map(([f, t]) => (
                      <Chip
                        key={`${f}->${t}`}
                        size="small"
                        icon={<Unlink size={12} />}
                        label={`${f} → ${t}`}
                        onClick={() => setDraft((d) => ({ ...d, edges: d.edges.filter(([ef, et]) => !(ef === f && et === t)) }))}
                        sx={{ height: 22, fontSize: 10.5, fontFamily: 'monospace', cursor: 'pointer', '&:hover': { bgcolor: tokens.color.dangerBg, color: tokens.color.danger } }}
                      />
                    ))}
                  </Box>
                </Box>
              ) : null
            }
          />
        </Box>

        <SectionCard
          title="What will run (live summary of the draft)"
          sx={{ flex: '2 1 420px' }}
          help="Recomputed on every edit: the topological execution order of enabled stages, any cycles (which would deadlock a real scheduler — the as-built re-label loop is broken by the review gate, so removing that edge and adding a direct loop shows up here), orphaned stages, and the diff against the as-built pipeline."
        >
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
            <Box>
              <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mb: 0.5 }}>
                Execution order (topological)
              </Typography>
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
                {summary.order.map((id, i) => (
                  <Box key={id} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Chip size="small" label={stageDef(id)?.label ?? id} sx={{ height: 22, fontSize: 11, bgcolor: `${ROLE_COLORS[id] ?? tokens.color.neutral}1c`, color: ROLE_COLORS[id] ?? tokens.color.text, fontWeight: 700 }} />
                    {i < summary.order.length - 1 ? <Typography variant="caption" sx={{ color: tokens.color.textFaint }}>→</Typography> : null}
                  </Box>
                ))}
                {summary.order.length === 0 ? <Typography variant="caption" sx={{ color: tokens.color.textFaint }}>nothing runnable</Typography> : null}
              </Box>
            </Box>

            {summary.cycle.length ? (
              <Alert severity="warning" variant="outlined" sx={{ py: 0 }}>
                Cycle detected involving: {summary.cycle.map((id) => stageDef(id)?.label ?? id).join(', ')} — these stages can
                never be scheduled in a strict order. (The as-built re-label feedback loop is the legitimate exception; a real
                scheduler breaks it at the review gate.)
              </Alert>
            ) : null}

            {summary.orphans.length ? (
              <Typography variant="caption" sx={{ color: tokens.color.warn }}>
                Orphaned (enabled but unconnected): {summary.orphans.map((id) => stageDef(id)?.label ?? id).join(', ')}
              </Typography>
            ) : null}

            <Box>
              <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mb: 0.5 }}>
                Draft vs as-built {changes.length === 0 ? '— identical' : `— ${changes.length} change${changes.length === 1 ? '' : 's'}`}
              </Typography>
              {changes.map((c) => (
                <Typography key={c} variant="caption" sx={{ display: 'block', fontFamily: 'monospace', color: tokens.color.info }}>
                  · {c}
                </Typography>
              ))}
            </Box>
          </Box>
        </SectionCard>
      </Box>
    </Box>
  );
}
