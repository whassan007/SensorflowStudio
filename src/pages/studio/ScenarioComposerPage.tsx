/**
 * Scenario Composer — WYSIWYG BEV scenario editor.
 *
 * Drag actors from the palette onto the top-down canvas, drag them (and their
 * trajectory waypoints) to shape the scene, set the environment with visual
 * toggles (the canvas re-renders the look), watch the JSON recipe write
 * itself, and run the composition through the surrogate-safety evaluation
 * (SSAM /api/safety/ssam/analyze accepts custom trajectories).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { CloudFog, CloudRain, Moon, Play, RotateCcw, Sun, Trash2 } from 'lucide-react';
import CanvasSurface, { snapTo, worldDrag, type CanvasApi } from '../../components/visual/canvas';
import InspectorPanel, { type InspectorField } from '../../components/visual/InspectorPanel';
import { analyzeSsam, type SsamAnalysis, type SsamTrajectory } from '../../services/safety';
import { getCapabilities, loadLayout, saveLayout } from '../../services/studioux';
import { SectionCard, fmtNum } from '../../components/labeleval/shared';
import { InfoDot } from '../../components/help/InfoTip';
import { tokens, verdictColor } from '../../theme';
import { useLabelEval } from '../../context/LabelEvalContext';

// ---------------------------------------------------------------- model

type ActorKind = 'vehicle' | 'truck' | 'pedestrian' | 'cyclist' | 'cone' | 'construction';

interface Waypoint {
  x: number;
  y: number;
}

interface Actor {
  id: string;
  kind: ActorKind;
  x: number;
  y: number;
  /** degrees, 0 = +x (ego forward) */
  heading: number;
  /** m/s along the waypoint path */
  speed: number;
  waypoints: Waypoint[];
}

interface Environment {
  timeOfDay: 'day' | 'night';
  weather: 'clear' | 'rain' | 'fog';
}

interface Composition {
  name: string;
  env: Environment;
  actors: Actor[];
}

const ACTOR_META: Record<ActorKind, { label: string; color: string; l: number; w: number; defaultSpeed: number; movable: boolean }> = {
  vehicle: { label: 'Vehicle', color: '#66bb6a', l: 4.5, w: 1.9, defaultSpeed: 8, movable: true },
  truck: { label: 'Truck', color: '#9ccc65', l: 8.5, w: 2.5, defaultSpeed: 6, movable: true },
  pedestrian: { label: 'Pedestrian', color: '#ff7043', l: 0.6, w: 0.6, defaultSpeed: 1.4, movable: true },
  cyclist: { label: 'Cyclist', color: '#ffca28', l: 1.8, w: 0.7, defaultSpeed: 4, movable: true },
  cone: { label: 'Cone', color: '#ffa726', l: 0.4, w: 0.4, defaultSpeed: 0, movable: false },
  construction: { label: 'Construction zone', color: '#ab47bc', l: 8, w: 4, defaultSpeed: 0, movable: false },
};

const WORLD = { x: -12, y: -28, w: 104, h: 56 };
const LAYOUT_KEY = 'scenario-composer';

const DEFAULT_COMPOSITION: Composition = {
  name: 'my-scenario',
  env: { timeOfDay: 'day', weather: 'clear' },
  actors: [],
};

let actorSeq = 0;
function newActorId(kind: ActorKind): string {
  actorSeq += 1;
  return `${kind}-${Date.now().toString(36)}-${actorSeq}`;
}

// ------------------------------------------------- recipe + trajectory compile

/** Machine-readable recipe — the live JSON preview and the export format. */
function toRecipe(c: Composition) {
  return {
    schema: 'sensorflow.scenario/v1',
    name: c.name,
    environment: { time_of_day: c.env.timeOfDay, weather: c.env.weather },
    ego: { x: 0, y: 0, heading_deg: 0 },
    actors: c.actors.map((a) => ({
      id: a.id,
      kind: a.kind,
      position: { x: round2(a.x), y: round2(a.y) },
      heading_deg: Math.round(a.heading),
      speed_mps: round2(a.speed),
      trajectory_waypoints: a.waypoints.map((w) => ({ x: round2(w.x), y: round2(w.y) })),
    })),
  };
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

/**
 * Compile the composition into SSAM trajectories: each actor walks its
 * waypoint path at its speed (dt = 0.5 s); static props stand still. The ego
 * vehicle drives straight ahead at 8 m/s so every scene has the AV in it.
 */
function compileTrajectories(c: Composition, durationS = 12): SsamTrajectory[] {
  const dt = 0.5;
  const steps = Math.round(durationS / dt);
  const out: SsamTrajectory[] = [];

  const ego: SsamTrajectory = {
    vehicle_id: 'ego',
    vehicle_type: 'car',
    length: 4.5,
    width: 1.9,
    states: Array.from({ length: steps + 1 }, (_, i) => ({ t: i * dt, x: 8 * i * dt, y: 0, speed: 8, heading: 0 })),
  };
  out.push(ego);

  c.actors.forEach((a) => {
    const meta = ACTOR_META[a.kind];
    const path: Waypoint[] = [{ x: a.x, y: a.y }, ...a.waypoints];
    const states: SsamTrajectory['states'] = [];
    if (!meta.movable || a.speed <= 0 || path.length < 2) {
      const heading = (a.heading * Math.PI) / 180;
      for (let i = 0; i <= steps; i += 1) states.push({ t: i * dt, x: a.x, y: a.y, speed: 0, heading });
    } else {
      // cumulative arc lengths
      const segLen: number[] = [];
      for (let i = 1; i < path.length; i += 1) segLen.push(Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y));
      const total = segLen.reduce((s, v) => s + v, 0);
      for (let i = 0; i <= steps; i += 1) {
        const dist = Math.min(a.speed * i * dt, total);
        let acc = 0;
        let seg = 0;
        while (seg < segLen.length - 1 && acc + segLen[seg] < dist) {
          acc += segLen[seg];
          seg += 1;
        }
        const p0 = path[seg];
        const p1 = path[seg + 1];
        const f = segLen[seg] > 0 ? (dist - acc) / segLen[seg] : 0;
        const x = p0.x + (p1.x - p0.x) * f;
        const y = p0.y + (p1.y - p0.y) * f;
        const heading = Math.atan2(p1.y - p0.y, p1.x - p0.x);
        const done = dist >= total;
        states.push({ t: i * dt, x, y, speed: done ? 0 : a.speed, heading });
      }
    }
    out.push({
      vehicle_id: a.id,
      vehicle_type: a.kind === 'vehicle' ? 'car' : a.kind,
      length: meta.l,
      width: meta.w,
      states,
    });
  });
  return out;
}

// ---------------------------------------------------------------- rendering

function ActorGlyph({ actor, selected, api, onSelect, onMove }: {
  actor: Actor;
  selected: boolean;
  api: CanvasApi;
  onSelect: () => void;
  onMove: (x: number, y: number) => void;
}) {
  const meta = ACTOR_META[actor.kind];
  const stroke = selected ? '#ffffff' : meta.color;
  const drag = worldDrag(api, {
    onStart: () => onSelect(),
    onMove: (w) => onMove(snapTo(w.x, 0.5), snapTo(w.y, 0.5)),
  });

  const common = { ...drag, style: { cursor: 'grab' as const } };
  const label = selected ? (
    <text x={0} y={-meta.w / 2 - 0.8} fontSize={1.6} fill="#fff" textAnchor="middle" fontWeight={700} pointerEvents="none">
      {meta.label}
    </text>
  ) : null;

  switch (actor.kind) {
    case 'pedestrian':
      return (
        <g transform={`translate(${actor.x}, ${-actor.y})`} {...common}>
          <circle r={0.5} fill={`${meta.color}55`} stroke={stroke} strokeWidth={selected ? 0.25 : 0.15} />
          <circle r={0.18} fill={stroke} />
          {label}
        </g>
      );
    case 'cone':
      return (
        <g transform={`translate(${actor.x}, ${-actor.y})`} {...common}>
          <polygon points="0,-0.55 0.45,0.45 -0.45,0.45" fill={`${meta.color}66`} stroke={stroke} strokeWidth={selected ? 0.2 : 0.12} />
          {label}
        </g>
      );
    case 'construction':
      return (
        <g transform={`translate(${actor.x}, ${-actor.y}) rotate(${-actor.heading})`} {...common}>
          <rect x={-meta.l / 2} y={-meta.w / 2} width={meta.l} height={meta.w} fill={`${meta.color}22`} stroke={stroke} strokeWidth={selected ? 0.28 : 0.18} strokeDasharray="0.8 0.5" rx={0.3} />
          <text x={0} y={0.35} fontSize={1.1} fill={stroke} textAnchor="middle" fontWeight={700} pointerEvents="none">WORK ZONE</text>
          {label}
        </g>
      );
    default:
      // vehicle / truck / cyclist: oriented box + heading tick
      return (
        <g transform={`translate(${actor.x}, ${-actor.y}) rotate(${-actor.heading})`} {...common}>
          <rect x={-meta.l / 2} y={-meta.w / 2} width={meta.l} height={meta.w} fill={`${meta.color}33`} stroke={stroke} strokeWidth={selected ? 0.28 : 0.18} rx={0.25} />
          <line x1={meta.l / 2} y1={0} x2={meta.l / 2 + 1} y2={0} stroke={stroke} strokeWidth={0.2} />
          {label}
        </g>
      );
  }
}

function EnvironmentDecor({ env, view }: { env: Environment; view: { x: number; y: number; w: number; h: number } }) {
  const items: React.ReactNode[] = [];
  if (env.weather === 'rain') {
    // deterministic rain streaks
    for (let i = 0; i < 60; i += 1) {
      const x = view.x + ((i * 37) % 100) / 100 * view.w;
      const y = view.y + ((i * 61) % 100) / 100 * view.h;
      items.push(<line key={`r${i}`} x1={x} y1={y} x2={x - 0.5} y2={y + 1.4} stroke="rgba(120,160,220,0.35)" strokeWidth={0.1} pointerEvents="none" />);
    }
  }
  if (env.weather === 'fog') {
    items.push(<rect key="fog" x={view.x} y={view.y} width={view.w} height={view.h} fill="rgba(200,210,220,0.13)" pointerEvents="none" />);
  }
  if (env.timeOfDay === 'night') {
    items.push(<rect key="night" x={view.x} y={view.y} width={view.w} height={view.h} fill="rgba(8,12,40,0.35)" pointerEvents="none" />);
  }
  return <>{items}</>;
}

// ---------------------------------------------------------------- page

export default function ScenarioComposerPage() {
  const { navigate } = useLabelEval();
  const [comp, setComp] = useState<Composition>(DEFAULT_COMPOSITION);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [canRun, setCanRun] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SsamAnalysis | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const loaded = useRef(false);

  useEffect(() => {
    loadLayout<Composition>(LAYOUT_KEY).then((saved) => {
      if (saved && Array.isArray(saved.actors)) setComp(saved);
      loaded.current = true;
    });
    getCapabilities().then((caps) => setCanRun(caps.ssamAnalyze));
  }, []);

  // autosave (skip until the initial load resolved to avoid clobbering)
  useEffect(() => {
    if (loaded.current) saveLayout(LAYOUT_KEY, comp);
  }, [comp]);

  const selected = comp.actors.find((a) => a.id === selectedId) ?? null;

  const updateActor = useCallback((id: string, patch: Partial<Actor>) => {
    setComp((c) => ({ ...c, actors: c.actors.map((a) => (a.id === id ? { ...a, ...patch } : a)) }));
  }, []);

  const addActor = useCallback((kind: ActorKind, x: number, y: number) => {
    const meta = ACTOR_META[kind];
    const actor: Actor = {
      id: newActorId(kind),
      kind,
      x: snapTo(x, 0.5),
      y: snapTo(y, 0.5),
      heading: 180,
      speed: meta.defaultSpeed,
      waypoints: [],
    };
    setComp((c) => ({ ...c, actors: [...c.actors, actor] }));
    setSelectedId(actor.id);
  }, []);

  const removeActor = useCallback((id: string) => {
    setComp((c) => ({ ...c, actors: c.actors.filter((a) => a.id !== id) }));
    setSelectedId((s) => (s === id ? null : s));
  }, []);

  const addWaypoint = useCallback((id: string) => {
    setComp((c) => ({
      ...c,
      actors: c.actors.map((a) => {
        if (a.id !== id) return a;
        const last = a.waypoints[a.waypoints.length - 1] ?? { x: a.x, y: a.y };
        const prev = a.waypoints.length > 1 ? a.waypoints[a.waypoints.length - 2] : { x: a.x, y: a.y };
        const dx = last.x - prev.x;
        const dy = last.y - prev.y;
        const n = Math.hypot(dx, dy) || 1;
        // extend along the current direction (or along heading for the first)
        const dir = a.waypoints.length === 0
          ? { x: Math.cos((a.heading * Math.PI) / 180), y: Math.sin((a.heading * Math.PI) / 180) }
          : { x: dx / n, y: dy / n };
        return { ...a, waypoints: [...a.waypoints, { x: snapTo(last.x + dir.x * 8, 0.5), y: snapTo(last.y + dir.y * 8, 0.5) }] };
      }),
    }));
  }, []);

  const recipe = useMemo(() => toRecipe(comp), [comp]);
  const recipeJson = useMemo(() => JSON.stringify(recipe, null, 2), [recipe]);

  const run = useCallback(() => {
    setRunning(true);
    setRunError(null);
    setResult(null);
    analyzeSsam({ trajectories: compileTrajectories(comp) })
      .then(setResult)
      .catch((e) => setRunError(e instanceof Error ? e.message : String(e)))
      .finally(() => setRunning(false));
  }, [comp]);

  const inspectorFields: InspectorField[] = selected
    ? [
        { type: 'readonly', key: 'id', label: 'Actor ID', value: selected.id },
        {
          type: 'select',
          key: 'kind',
          label: 'Kind',
          value: selected.kind,
          options: (Object.keys(ACTOR_META) as ActorKind[]).map((k) => ({ value: k, label: ACTOR_META[k].label })),
          help: 'Changing the kind swaps footprint and default dynamics; position and waypoints are kept.',
        },
        { type: 'readonly', key: 'pos', label: 'Position (ego frame)', value: `(${selected.x.toFixed(1)}, ${selected.y.toFixed(1)}) m` },
        { type: 'slider', key: 'heading', label: 'Heading', value: Math.round(selected.heading), min: -180, max: 180, step: 5, unit: '°', help: '0° faces the same way as the ego vehicle (+x).' },
        ...(ACTOR_META[selected.kind].movable
          ? [{ type: 'slider', key: 'speed', label: 'Speed', value: selected.speed, min: 0, max: 20, step: 0.5, unit: ' m/s', help: 'Speed along the waypoint path when the scenario runs.' } as InspectorField]
          : []),
        { type: 'readonly', key: 'wp', label: 'Trajectory waypoints', value: `${selected.waypoints.length} — drag the small circles on the canvas` },
      ]
    : [];

  const envBg = comp.env.timeOfDay === 'night' ? '#0a0e1c' : tokens.color.surfaceSunken;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title={
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
            Compose the scene
            <InfoDot
              title="How to compose"
              detail="Drag an actor chip from the palette and drop it anywhere on the canvas. Drag actors to reposition. Select an actor to edit its heading/speed and add trajectory waypoints (small circles — drag them too). The ego vehicle (blue triangle) always drives straight at 8 m/s. Environment toggles change the canvas look and are recorded in the recipe."
            />
          </Box>
        }
        action={
          <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center' }}>
            <Chip
              size="small"
              icon={comp.env.timeOfDay === 'day' ? <Sun size={13} /> : <Moon size={13} />}
              label={comp.env.timeOfDay}
              onClick={() => setComp((c) => ({ ...c, env: { ...c.env, timeOfDay: c.env.timeOfDay === 'day' ? 'night' : 'day' } }))}
              sx={{ cursor: 'pointer', bgcolor: comp.env.timeOfDay === 'night' ? '#1a237e' : 'rgba(249,168,37,0.18)', fontWeight: 700 }}
            />
            {(['clear', 'rain', 'fog'] as const).map((w) => (
              <Chip
                key={w}
                size="small"
                icon={w === 'rain' ? <CloudRain size={13} /> : w === 'fog' ? <CloudFog size={13} /> : <Sun size={13} />}
                label={w}
                onClick={() => setComp((c) => ({ ...c, env: { ...c.env, weather: w } }))}
                sx={{
                  cursor: 'pointer',
                  fontWeight: comp.env.weather === w ? 800 : 400,
                  bgcolor: comp.env.weather === w ? tokens.color.infoBg : 'transparent',
                  color: comp.env.weather === w ? tokens.color.info : tokens.color.textDim,
                  border: `1px solid ${comp.env.weather === w ? tokens.color.info : tokens.color.border}`,
                }}
              />
            ))}
            <Button
              size="small"
              startIcon={<RotateCcw size={14} />}
              onClick={() => {
                setComp(DEFAULT_COMPOSITION);
                setSelectedId(null);
                setResult(null);
              }}
            >
              Clear
            </Button>
          </Box>
        }
      >
        {/* palette */}
        <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mb: 1, alignItems: 'center' }}>
          <Typography variant="caption" sx={{ color: tokens.color.neutral, mr: 0.5 }}>
            Palette — drag onto the canvas:
          </Typography>
          {(Object.keys(ACTOR_META) as ActorKind[]).map((k) => (
            <Chip
              key={k}
              size="small"
              label={ACTOR_META[k].label}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('application/x-actor-kind', k);
                e.dataTransfer.effectAllowed = 'copy';
              }}
              sx={{
                cursor: 'grab',
                bgcolor: `${ACTOR_META[k].color}1f`,
                color: ACTOR_META[k].color,
                border: `1px solid ${ACTOR_META[k].color}66`,
                fontWeight: 700,
                '&:active': { cursor: 'grabbing' },
              }}
            />
          ))}
          <Typography variant="caption" sx={{ color: tokens.color.textFaint }}>
            {comp.actors.length} actor{comp.actors.length === 1 ? '' : 's'} placed
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'stretch' }}>
          <Box sx={{ flex: '3 1 560px', minWidth: 0 }}>
            <CanvasSurface
              world={WORLD}
              height={440}
              gridStep={10}
              background={envBg}
              ariaLabel="Scenario composition canvas"
              onBackgroundClick={() => setSelectedId(null)}
              onDropWorld={(w, e) => {
                const kind = e.dataTransfer.getData('application/x-actor-kind') as ActorKind;
                if (kind && ACTOR_META[kind]) addActor(kind, w.x, -w.y);
              }}
            >
              {(api) => (
                <>
                  <EnvironmentDecor env={comp.env} view={WORLD} />
                  {/* road hint: ego lane */}
                  <rect x={-12} y={-3.5} width={104} height={7} fill="rgba(255,255,255,0.03)" pointerEvents="none" />
                  <line x1={-12} y1={0} x2={92} y2={0} stroke="rgba(255,255,255,0.14)" strokeWidth={0.12} strokeDasharray="2.5 2.5" pointerEvents="none" />
                  {[20, 40, 60, 80].map((r) => (
                    <circle key={r} cx={0} cy={0} r={r} fill="none" stroke={tokens.color.border} strokeWidth={0.1} strokeDasharray="1 1.2" pointerEvents="none" />
                  ))}
                  {/* ego */}
                  <g pointerEvents="none">
                    <polygon points="2.4,0 -1.4,1.1 -1.4,-1.1" fill={tokens.color.info} stroke="#fff" strokeWidth={0.12} />
                    <text x={0} y={2.6} fontSize={1.5} fill={tokens.color.info} textAnchor="middle" fontWeight={700}>EGO → 8 m/s</text>
                  </g>

                  {/* trajectory paths + waypoints of the selected actor */}
                  {comp.actors.map((a) => {
                    if (a.waypoints.length === 0) return null;
                    const isSel = a.id === selectedId;
                    const pts = [{ x: a.x, y: a.y }, ...a.waypoints];
                    return (
                      <g key={`traj-${a.id}`}>
                        <polyline
                          points={pts.map((p) => `${p.x},${-p.y}`).join(' ')}
                          fill="none"
                          stroke={isSel ? '#fff' : `${ACTOR_META[a.kind].color}88`}
                          strokeWidth={isSel ? 0.18 : 0.12}
                          strokeDasharray="0.9 0.6"
                          pointerEvents="none"
                        />
                        {isSel
                          ? a.waypoints.map((wp, i) => (
                              <circle
                                key={i}
                                cx={wp.x}
                                cy={-wp.y}
                                r={0.8}
                                fill="rgba(255,255,255,0.25)"
                                stroke="#fff"
                                strokeWidth={0.15}
                                style={{ cursor: 'grab' }}
                                {...worldDrag(api, {
                                  onMove: (w) =>
                                    updateActor(a.id, {
                                      waypoints: a.waypoints.map((p, j) => (j === i ? { x: snapTo(w.x, 0.5), y: snapTo(-w.y, 0.5) } : p)),
                                    }),
                                })}
                              />
                            ))
                          : null}
                      </g>
                    );
                  })}

                  {comp.actors.map((a) => (
                    <ActorGlyph
                      key={a.id}
                      actor={a}
                      selected={a.id === selectedId}
                      api={api}
                      onSelect={() => setSelectedId(a.id)}
                      onMove={(x, y) => updateActor(a.id, { x, y: -y })}
                    />
                  ))}
                </>
              )}
            </CanvasSurface>
            <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mt: 0.5 }}>
              Ego frame: x forward (right), y left (up). Scroll to zoom, drag the background to pan, drop palette chips to place actors.
            </Typography>
          </Box>

          <Box sx={{ flex: '1 1 250px', display: 'flex', flexDirection: 'column', gap: 1.5, minWidth: 240 }}>
            <InspectorPanel
              title={selected ? ACTOR_META[selected.kind].label : 'Inspector'}
              subtitle={selected ? 'Live edits re-render the canvas' : undefined}
              accent={selected ? ACTOR_META[selected.kind].color : undefined}
              fields={inspectorFields}
              emptyHint="Select an actor on the canvas (or drop a new one from the palette) to edit its properties here."
              onChange={(key, value) => {
                if (!selected) return;
                if (key === 'kind') updateActor(selected.id, { kind: value as ActorKind, speed: ACTOR_META[value as ActorKind].defaultSpeed });
                else if (key === 'heading') updateActor(selected.id, { heading: value as number });
                else if (key === 'speed') updateActor(selected.id, { speed: value as number });
              }}
              footer={
                selected ? (
                  <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
                    {ACTOR_META[selected.kind].movable ? (
                      <Button size="small" variant="outlined" onClick={() => addWaypoint(selected.id)}>
                        + Waypoint
                      </Button>
                    ) : null}
                    {selected.waypoints.length > 0 ? (
                      <Button size="small" onClick={() => updateActor(selected.id, { waypoints: selected.waypoints.slice(0, -1) })}>
                        − Waypoint
                      </Button>
                    ) : null}
                    <Button size="small" color="error" startIcon={<Trash2 size={14} />} onClick={() => removeActor(selected.id)}>
                      Remove
                    </Button>
                  </Box>
                ) : null
              }
            />
          </Box>
        </Box>
      </SectionCard>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <SectionCard
          title="Live recipe (JSON)"
          sx={{ flex: '1 1 380px' }}
          help="The machine-readable scenario recipe, regenerated on every edit. This exact object is what gets compiled into trajectories for the evaluation run."
          action={
            <Button
              size="small"
              onClick={() => {
                const blob = new Blob([recipeJson], { type: 'application/json' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `${comp.name || 'scenario'}.json`;
                a.click();
                URL.revokeObjectURL(a.href);
              }}
            >
              Download
            </Button>
          }
        >
          <Box
            component="pre"
            sx={{
              m: 0,
              p: 1.25,
              bgcolor: tokens.color.surfaceSunken,
              border: `1px solid ${tokens.color.border}`,
              borderRadius: 1,
              fontSize: 11,
              fontFamily: 'monospace',
              maxHeight: 320,
              overflow: 'auto',
              color: tokens.color.textDim,
            }}
          >
            {recipeJson}
          </Box>
        </SectionCard>

        <SectionCard
          title="Run through evaluation"
          sx={{ flex: '1 1 380px' }}
          help="Compiles the composed actors into trajectories (each actor walks its waypoint path at its speed; the ego drives straight at 8 m/s) and runs the surrogate-safety (SSAM) analysis over them. Conflicts are scored with TTC / PET / DRAC / CSI — the same measures as the SSAM Conflicts page."
        >
          {!canRun ? (
            <Alert severity="info" variant="outlined">
              No scene-consuming evaluation API is reachable right now — the run action is hidden. Composition and recipe
              export still work.
            </Alert>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                <Button variant="contained" startIcon={<Play size={15} />} disabled={running || comp.actors.length === 0} onClick={run}>
                  {running ? 'Analyzing…' : 'Run surrogate-safety analysis'}
                </Button>
                <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                  {comp.actors.length === 0 ? 'Place at least one actor first.' : `ego + ${comp.actors.length} actors · 12 s @ 2 Hz`}
                </Typography>
              </Box>
              {runError ? <Alert severity="warning" variant="outlined">{runError}</Alert> : null}
              {result ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      size="small"
                      label={`${result.aggregate.num_conflicts} conflict${result.aggregate.num_conflicts === 1 ? '' : 's'}`}
                      sx={{
                        fontWeight: 900,
                        bgcolor: result.aggregate.num_conflicts > 0 ? tokens.color.dangerBg : tokens.color.successBg,
                        color: result.aggregate.num_conflicts > 0 ? tokens.color.danger : tokens.color.success,
                      }}
                    />
                    {result.aggregate.min_ttc_s !== null ? (
                      <Chip size="small" label={`min TTC ${fmtNum(result.aggregate.min_ttc_s, 2)} s`} sx={{ fontFamily: 'monospace' }} />
                    ) : null}
                    {result.aggregate.max_drac_mps2 !== null ? (
                      <Chip size="small" label={`max DRAC ${fmtNum(result.aggregate.max_drac_mps2, 2)} m/s²`} sx={{ fontFamily: 'monospace' }} />
                    ) : null}
                    <Chip size="small" label={`aggregate CSI ${fmtNum(result.aggregate.aggregate_csi)}`} sx={{ fontFamily: 'monospace' }} />
                  </Box>
                  {result.conflicts.slice(0, 6).map((c, i) => (
                    <Box key={i} sx={{ display: 'flex', gap: 1, alignItems: 'center', p: 0.75, border: `1px solid ${tokens.color.border}`, borderRadius: 1, bgcolor: tokens.color.surfaceSunken }}>
                      <Chip size="small" label={c.conflict_type} sx={{ height: 20, fontSize: 10.5, bgcolor: `${verdictColor('WARN')}22`, color: verdictColor('WARN') }} />
                      <Typography variant="caption" sx={{ fontFamily: 'monospace', color: tokens.color.textDim }}>
                        {c.vehicle_a} × {c.vehicle_b} · t {fmtNum(c.t_start_s)}–{fmtNum(c.t_end_s)} s
                        {c.min_ttc_s !== null ? ` · TTC ${fmtNum(c.min_ttc_s)} s` : ''} · CSI {fmtNum(c.csi)}
                      </Typography>
                    </Box>
                  ))}
                  {result.aggregate.num_conflicts === 0 ? (
                    <Typography variant="caption" sx={{ color: tokens.color.success }}>
                      No surrogate-safety conflicts detected in this composition — actors never got dangerously close to the
                      ego or each other.
                    </Typography>
                  ) : (
                    <Button size="small" onClick={() => navigate('safety-ssam')} sx={{ alignSelf: 'flex-start' }}>
                      Open the SSAM Conflicts page for the full toolkit →
                    </Button>
                  )}
                </Box>
              ) : null}
            </Box>
          )}
        </SectionCard>
      </Box>
    </Box>
  );
}
