/**
 * My Dashboard — WYSIWYG dashboard composer.
 *
 * Users drag live widgets (seqeval verdict, ODD coverage mini-map, engine
 * comparison, review queue, pipeline counters, scenario DB, calibration
 * status) onto a 12-column grid: drag to move, grip to resize, × to remove,
 * with an edit/view mode toggle. The layout persists via localStorage plus
 * the optional /api/studio-ux backend (see services/studioux.ts).
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Switch from '@mui/material/Switch';
import Typography from '@mui/material/Typography';
import { ArrowUpRight, GripVertical, Plus, RotateCcw, X } from 'lucide-react';
import { useLabelEval, type PageId } from '../../context/LabelEvalContext';
import { loadLayout, saveLayout } from '../../services/studioux';
import { listSeqRuns, type SeqRunState } from '../../services/seqeval';
import { getBevReport, type BevReport } from '../../services/bevfusion';
import {
  getCalibrationStatus,
  getOddCoverage,
  searchScenarios,
  type CalibrationResult,
  type OddCoverageResponse,
  type ScenariosResponse,
} from '../../services/safety';
import { getRuns } from '../../services/megaeval';
import { PanelSkeleton, IllustratedEmpty } from '../../components/visual/Feedback';
import { fmtInt, fmtPct } from '../../components/labeleval/shared';
import { tokens, verdictColor } from '../../theme';

// ---------------------------------------------------------------- layout model

interface WidgetLayout {
  uid: string;
  type: WidgetType;
  x: number; // column 0..11
  y: number; // row
  w: number; // columns
  h: number; // rows
}

type WidgetType =
  | 'seqeval-verdict'
  | 'odd-mini'
  | 'engine-compare'
  | 'review-queue'
  | 'pipeline-counters'
  | 'scenario-db'
  | 'calibration-status';

const COLS = 12;
const ROW_H = 84;
const GAP = 10;
const LAYOUT_KEY = 'my-dashboard-layout';

interface WidgetMeta {
  title: string;
  source: PageId;
  sourceLabel: string;
  defaultW: number;
  defaultH: number;
  description: string;
}

const WIDGETS: Record<WidgetType, WidgetMeta> = {
  'seqeval-verdict': { title: 'Sequential regression verdict', source: 'seqeval', sourceLabel: 'Sequential Regression', defaultW: 4, defaultH: 2, description: 'Latest sequential run: decision, gate and stopping reason.' },
  'odd-mini': { title: 'ODD coverage', source: 'safety-odd', sourceLabel: 'ODD Coverage', defaultW: 4, defaultH: 3, description: 'Coverage rate + mini heatmap of class × condition cells.' },
  'engine-compare': { title: 'Perception engine comparison', source: 'bevfusion', sourceLabel: 'Perception Engines', defaultW: 4, defaultH: 2, description: 'BEV-fusion vs camera baseline: recommendation + headline deltas.' },
  'review-queue': { title: 'Review queue', source: 'review', sourceLabel: 'Human Review', defaultW: 2, defaultH: 2, description: 'Live human-review queue depth and alert count.' },
  'pipeline-counters': { title: 'Pipeline counters', source: 'overview', sourceLabel: 'Overview', defaultW: 4, defaultH: 2, description: 'Live processed / auto-labeled / verified counters.' },
  'scenario-db': { title: 'Scenario database', source: 'safety-scenarios', sourceLabel: 'Scenario DB', defaultW: 2, defaultH: 2, description: 'Stored test scenarios by severity.' },
  'calibration-status': { title: 'Calibration status', source: 'safety-calibration', sourceLabel: 'Calibration', defaultW: 3, defaultH: 2, description: 'Latest camera–LiDAR extrinsic validation verdict.' },
};

const STARTER_LAYOUT: WidgetLayout[] = [
  { uid: 'w1', type: 'seqeval-verdict', x: 0, y: 0, w: 4, h: 2 },
  { uid: 'w2', type: 'engine-compare', x: 4, y: 0, w: 4, h: 2 },
  { uid: 'w3', type: 'review-queue', x: 8, y: 0, w: 2, h: 2 },
  { uid: 'w4', type: 'odd-mini', x: 0, y: 2, w: 4, h: 3 },
  { uid: 'w5', type: 'pipeline-counters', x: 4, y: 2, w: 4, h: 2 },
];

let uidSeq = 0;
function newUid(): string {
  uidSeq += 1;
  return `w-${Date.now().toString(36)}-${uidSeq}`;
}

// ---------------------------------------------------------------- widgets

function Stat({ label, value, color }: { label: string; value: ReactNode; color?: string }) {
  return (
    <Box sx={{ minWidth: 76 }}>
      <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', fontSize: 10.5 }}>
        {label}
      </Typography>
      <Typography variant="h6" sx={{ fontWeight: 800, fontFamily: 'monospace', fontSize: 18, color: color ?? tokens.color.text }}>
        {value}
      </Typography>
    </Box>
  );
}

function useAsync<T>(fn: () => Promise<T>): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    let cancelled = false;
    ref
      .current()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return { data, error };
}

function WidgetError({ error }: { error: string }) {
  return (
    <Typography variant="caption" sx={{ color: tokens.color.warn }}>
      Data unavailable: {error.slice(0, 120)}
    </Typography>
  );
}

function SeqevalWidget() {
  const { data, error } = useAsync(async () => {
    const { runs } = await listSeqRuns();
    const done = runs.filter((r) => r.status === 'done').sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    return (done[0] ?? runs[0] ?? null) as SeqRunState | null;
  });
  if (error) return <WidgetError error={error} />;
  if (data === null) return <PanelSkeleton rows={2} header={false} />;
  if (!data) return <Typography variant="caption" sx={{ color: tokens.color.neutral }}>No sequential runs yet — launch one from the Sequential Regression page.</Typography>;
  const color = verdictColor(data.decision);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
      <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center', flexWrap: 'wrap' }}>
        <Chip size="small" label={(data.decision ?? data.status).toUpperCase()} sx={{ fontWeight: 900, bgcolor: `${color}22`, color, border: `1.5px solid ${color}` }} />
        {data.gate ? <Chip size="small" label={`gate: ${data.gate}`} sx={{ height: 20, fontSize: 10.5 }} /> : null}
      </Box>
      <Typography variant="caption" sx={{ color: tokens.color.textDim }}>
        {data.message ?? data.stopping_reason ?? `run ${data.run_id}`}
      </Typography>
      {data.budget ? (
        <Typography variant="caption" sx={{ color: tokens.color.neutral, fontFamily: 'monospace', fontSize: 10.5 }}>
          {fmtInt((data.budget as { used?: number }).used ?? 0)} samples used
        </Typography>
      ) : null}
    </Box>
  );
}

function OddMiniWidget() {
  const { data, error } = useAsync(async () => {
    const { runs } = await getRuns();
    const pub = runs.filter((r) => r.status === 'published').sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    if (!pub.length) return null;
    return getOddCoverage(pub[0].run_id, null, true);
  });
  if (error) return <WidgetError error={error} />;
  if (data === null) return <PanelSkeleton rows={3} header={false} />;
  if (!data) return <Typography variant="caption" sx={{ color: tokens.color.neutral }}>No published evaluation run yet.</Typography>;
  const d = data as OddCoverageResponse;
  const cells = d.cells ?? [];
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Box sx={{ display: 'flex', gap: 2 }}>
        <Stat label="coverage" value={fmtPct(d.summary.coverage_rate)} color={d.summary.coverage_rate >= 0.8 ? tokens.color.success : tokens.color.warn} />
        <Stat label="gaps" value={d.summary.gap_cells} color={d.summary.gap_cells > 0 ? tokens.color.danger : tokens.color.success} />
        <Stat label="recall" value={fmtPct(d.summary.overall_recall)} />
      </Box>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(14px, 1fr))', gap: '2px' }}>
        {cells.slice(0, 96).map((c) => (
          <Box
            key={c.cell_id}
            title={`${Object.values(c.cell).join(' × ')} — ${c.n} samples${c.recall !== null ? `, recall ${fmtPct(c.recall)}` : ''}`}
            sx={{
              height: 14,
              borderRadius: '2px',
              bgcolor: c.n === 0 ? tokens.color.surfaceRaised : c.is_gap ? tokens.color.danger : c.adequate ? tokens.color.success : tokens.color.warn,
              opacity: c.n === 0 ? 0.45 : 0.85,
            }}
          />
        ))}
      </Box>
    </Box>
  );
}

function EngineCompareWidget() {
  const { data, error } = useAsync(() => getBevReport().catch(() => null as BevReport | null));
  if (error) return <WidgetError error={error} />;
  if (data === null) return <PanelSkeleton rows={2} header={false} />;
  if (!data) return <Typography variant="caption" sx={{ color: tokens.color.neutral }}>No comparison yet — run one on the Perception Engines page.</Typography>;
  const color = verdictColor(data.recommendation);
  const top = data.headline_deltas.filter((h) => ['recall', 'safety_recall', 'position_error_m'].includes(h.metric));
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
      <Chip size="small" label={data.recommendation.replace(/_/g, ' ')} sx={{ alignSelf: 'flex-start', fontWeight: 900, bgcolor: `${color}22`, color, border: `1.5px solid ${color}` }} />
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {top.map((h) => (
          <Stat
            key={h.metric}
            label={h.metric.replace(/_/g, ' ')}
            value={`${h.delta >= 0 ? '+' : ''}${Math.abs(h.baseline) <= 1 && h.metric !== 'position_error_m' ? `${(h.delta * 100).toFixed(1)}pp` : h.delta.toFixed(2)}`}
            color={h.improved ? tokens.color.success : tokens.color.danger}
          />
        ))}
      </Box>
    </Box>
  );
}

function ReviewQueueWidget() {
  const { stream } = useLabelEval();
  const review = stream?.pipeline.review_queue_count ?? null;
  const alerts = stream?.alerts_count ?? null;
  if (review === null) return <PanelSkeleton rows={1} header={false} />;
  return (
    <Box sx={{ display: 'flex', gap: 2 }}>
      <Stat label="in queue" value={fmtInt(review)} color={review > 0 ? '#e65100' : tokens.color.success} />
      <Stat label="alerts" value={fmtInt(alerts ?? 0)} color={(alerts ?? 0) > 0 ? tokens.color.danger : tokens.color.success} />
    </Box>
  );
}

function PipelineCountersWidget() {
  const { stream } = useLabelEval();
  const c = stream?.pipeline.counters ?? null;
  if (!c) return <PanelSkeleton rows={1} header={false} />;
  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
      <Stat label="processed" value={fmtInt(c.frames_processed ?? 0)} />
      <Stat label="auto-labeled" value={fmtInt(c.auto_labeled ?? 0)} />
      <Stat label="verified" value={fmtInt(c.verified ?? 0)} color={tokens.color.success} />
      <Stat label="flagged" value={fmtInt(c.flagged ?? 0)} color={(c.flagged ?? 0) > 0 ? tokens.color.warn : undefined} />
    </Box>
  );
}

function ScenarioDbWidget() {
  const { data, error } = useAsync(() => searchScenarios({}) as Promise<ScenariosResponse>);
  if (error) return <WidgetError error={error} />;
  if (data === null) return <PanelSkeleton rows={1} header={false} />;
  const bySev = data.counts?.by_severity ?? {};
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      <Stat label="scenarios" value={fmtInt(data.counts.total)} />
      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
        {Object.entries(bySev).map(([sev, n]) => (
          <Chip key={sev} size="small" label={`${sev} ${n}`} sx={{ height: 18, fontSize: 9.5, bgcolor: tokens.color.surfaceRaised }} />
        ))}
      </Box>
    </Box>
  );
}

function CalibrationWidget() {
  const { data, error } = useAsync(() => getCalibrationStatus().catch(() => null as CalibrationResult | null));
  if (error) return <WidgetError error={error} />;
  if (data === null) return <PanelSkeleton rows={1} header={false} />;
  if (!data || !data.status) return <Typography variant="caption" sx={{ color: tokens.color.neutral }}>No validation run yet.</Typography>;
  const color = verdictColor(data.status);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
      <Chip size="small" label={data.status} sx={{ alignSelf: 'flex-start', fontWeight: 900, bgcolor: `${color}22`, color, border: `1.5px solid ${color}` }} />
      <Typography variant="caption" sx={{ color: tokens.color.textDim }}>
        {(data.checks ?? []).filter((c) => c.passed).length}/{(data.checks ?? []).length} checks passing
      </Typography>
    </Box>
  );
}

function WidgetBody({ type }: { type: WidgetType }) {
  switch (type) {
    case 'seqeval-verdict':
      return <SeqevalWidget />;
    case 'odd-mini':
      return <OddMiniWidget />;
    case 'engine-compare':
      return <EngineCompareWidget />;
    case 'review-queue':
      return <ReviewQueueWidget />;
    case 'pipeline-counters':
      return <PipelineCountersWidget />;
    case 'scenario-db':
      return <ScenarioDbWidget />;
    case 'calibration-status':
      return <CalibrationWidget />;
    default:
      return null;
  }
}

// ---------------------------------------------------------------- grid frame

interface DragState {
  uid: string;
  mode: 'move' | 'resize';
  startX: number;
  startY: number;
  orig: WidgetLayout;
}

export default function MyDashboardPage() {
  const { navigate } = useLabelEval();
  const [layout, setLayout] = useState<WidgetLayout[] | null>(null);
  const [edit, setEdit] = useState(false);
  const [drag, setDrag] = useState<DragState | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const loaded = useRef(false);

  useEffect(() => {
    loadLayout<WidgetLayout[]>(LAYOUT_KEY).then((saved) => {
      setLayout(saved && Array.isArray(saved) && saved.length ? saved : STARTER_LAYOUT);
      loaded.current = true;
    });
  }, []);

  useEffect(() => {
    if (loaded.current && layout) saveLayout(LAYOUT_KEY, layout);
  }, [layout]);

  const colW = useCallback((): number => {
    const el = gridRef.current;
    if (!el) return 90;
    return (el.clientWidth - GAP * (COLS - 1)) / COLS;
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!drag) return;
      const cw = colW();
      const dxCols = Math.round((e.clientX - drag.startX) / (cw + GAP));
      const dyRows = Math.round((e.clientY - drag.startY) / (ROW_H + GAP));
      setLayout((ls) =>
        (ls ?? []).map((wl) => {
          if (wl.uid !== drag.uid) return wl;
          if (drag.mode === 'move') {
            return {
              ...wl,
              x: Math.max(0, Math.min(COLS - wl.w, drag.orig.x + dxCols)),
              y: Math.max(0, drag.orig.y + dyRows),
            };
          }
          return {
            ...wl,
            w: Math.max(2, Math.min(COLS - wl.x, drag.orig.w + dxCols)),
            h: Math.max(1, drag.orig.h + dyRows),
          };
        })
      );
    },
    [drag, colW]
  );

  const startDrag = (e: React.PointerEvent, wl: WidgetLayout, mode: 'move' | 'resize') => {
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    setDrag({ uid: wl.uid, mode, startX: e.clientX, startY: e.clientY, orig: wl });
  };

  const addWidget = (type: WidgetType) => {
    const meta = WIDGETS[type];
    const maxY = (layout ?? []).reduce((m, wl) => Math.max(m, wl.y + wl.h), 0);
    setLayout((ls) => [...(ls ?? []), { uid: newUid(), type, x: 0, y: maxY, w: meta.defaultW, h: meta.defaultH }]);
  };

  const inUse = useMemo(() => new Set((layout ?? []).map((wl) => wl.type)), [layout]);

  if (!layout) return <PanelSkeleton rows={5} />;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Switch size="small" checked={edit} onChange={(e) => setEdit(e.target.checked)} inputProps={{ 'aria-label': 'Edit mode' }} />
          <Typography variant="body2" sx={{ fontWeight: 700, color: edit ? tokens.color.info : tokens.color.textDim }}>
            {edit ? 'Edit mode — drag, resize, remove' : 'View mode'}
          </Typography>
        </Box>
        {edit ? (
          <Button size="small" startIcon={<RotateCcw size={14} />} onClick={() => setLayout(STARTER_LAYOUT)}>
            Reset to starter layout
          </Button>
        ) : null}
        <Typography variant="caption" sx={{ color: tokens.color.neutral, ml: 'auto' }}>
          {layout.length} widget{layout.length === 1 ? '' : 's'} · layout saved automatically
        </Typography>
      </Box>

      {edit ? (
        <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', p: 1.25, border: `1px dashed ${tokens.color.borderStrong}`, borderRadius: 1, bgcolor: tokens.color.surfaceSunken, alignItems: 'center' }}>
          <Typography variant="caption" sx={{ color: tokens.color.neutral, mr: 0.5 }}>
            Widget palette:
          </Typography>
          {(Object.keys(WIDGETS) as WidgetType[]).map((t) => (
            <Chip
              key={t}
              size="small"
              icon={<Plus size={13} />}
              label={WIDGETS[t].title}
              onClick={() => addWidget(t)}
              title={WIDGETS[t].description}
              sx={{
                cursor: 'pointer',
                bgcolor: inUse.has(t) ? tokens.color.surfaceRaised : tokens.color.infoBg,
                color: inUse.has(t) ? tokens.color.textDim : tokens.color.info,
                border: `1px solid ${inUse.has(t) ? tokens.color.border : tokens.color.info}`,
              }}
            />
          ))}
        </Box>
      ) : null}

      {layout.length === 0 ? (
        <IllustratedEmpty
          art="canvas"
          title="An empty dashboard"
          message="Switch to edit mode and add widgets from the palette — every widget shows live data from its source page."
          action={
            <Button variant="outlined" size="small" onClick={() => setEdit(true)}>
              Enter edit mode
            </Button>
          }
        />
      ) : null}

      <Box
        ref={gridRef}
        onPointerMove={onPointerMove}
        onPointerUp={() => setDrag(null)}
        sx={{
          display: 'grid',
          gridTemplateColumns: `repeat(${COLS}, 1fr)`,
          gridAutoRows: `${ROW_H}px`,
          gap: `${GAP}px`,
          touchAction: drag ? 'none' : undefined,
        }}
      >
        {layout.map((wl) => {
          const meta = WIDGETS[wl.type];
          const dragging = drag?.uid === wl.uid;
          return (
            <Box
              key={wl.uid}
              sx={{
                gridColumn: `${wl.x + 1} / span ${wl.w}`,
                gridRow: `${wl.y + 1} / span ${wl.h}`,
                border: `1px solid ${dragging ? tokens.color.info : edit ? tokens.color.borderStrong : tokens.color.border}`,
                borderRadius: 1.5,
                bgcolor: tokens.color.surface,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                position: 'relative',
                boxShadow: dragging ? tokens.elevation.overlay : undefined,
                transition: dragging ? 'none' : `border-color ${tokens.motion.fast}, box-shadow ${tokens.motion.fast}`,
              }}
            >
              <Box
                onPointerDown={edit ? (e) => startDrag(e, wl, 'move') : undefined}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  px: 1,
                  py: 0.5,
                  borderBottom: `1px solid ${tokens.color.border}`,
                  bgcolor: tokens.color.surfaceSunken,
                  cursor: edit ? 'grab' : 'default',
                  userSelect: 'none',
                  flexShrink: 0,
                }}
              >
                {edit ? <GripVertical size={13} color={tokens.color.textFaint} /> : null}
                <Typography variant="caption" sx={{ fontWeight: 700, color: tokens.color.textDim, flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {meta.title}
                </Typography>
                {!edit ? (
                  <IconButton size="small" onClick={() => navigate(meta.source)} title={`Open ${meta.sourceLabel}`} sx={{ p: 0.25 }}>
                    <ArrowUpRight size={13} />
                  </IconButton>
                ) : (
                  <IconButton size="small" onClick={() => setLayout((ls) => (ls ?? []).filter((x) => x.uid !== wl.uid))} title="Remove widget" sx={{ p: 0.25, '&:hover': { color: tokens.color.danger } }}>
                    <X size={13} />
                  </IconButton>
                )}
              </Box>
              <Box sx={{ p: 1.25, flex: 1, minHeight: 0, overflow: 'auto' }}>
                <WidgetBody type={wl.type} />
              </Box>
              {edit ? (
                <Box
                  onPointerDown={(e) => startDrag(e, wl, 'resize')}
                  sx={{
                    position: 'absolute',
                    right: 0,
                    bottom: 0,
                    width: 18,
                    height: 18,
                    cursor: 'nwse-resize',
                    borderRight: `3px solid ${tokens.color.info}`,
                    borderBottom: `3px solid ${tokens.color.info}`,
                    borderBottomRightRadius: 6,
                    opacity: 0.7,
                    '&:hover': { opacity: 1 },
                  }}
                />
              ) : null}
            </Box>
          );
        })}
      </Box>
      {edit ? (
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
          Drag a widget by its header to move it on the grid; drag the blue corner grip to resize; × removes. Widgets show
          live data in both modes.
        </Typography>
      ) : null}
    </Box>
  );
}
