/**
 * Direct-manipulation canvas foundation shared by the WYSIWYG surfaces
 * (Pipeline Builder node graph, Scenario Composer BEV editor, Perception
 * Engines BEV frame viewer).
 *
 * <CanvasSurface> renders an SVG with:
 *   - wheel zoom centered on the cursor,
 *   - background drag to pan,
 *   - a `CanvasApi` (toWorld / scale) passed to children via render prop so
 *     draggable elements can convert pointer events into world coordinates.
 *
 * `worldDrag(api, handlers)` returns pointer handlers implementing the
 * capture-move-release cycle in world coordinates — the single drag behavior
 * every draggable object on every canvas reuses.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import { tokens } from '../../theme';

export interface CanvasApi {
  /** Convert a client (pointer event) position into world coordinates. */
  toWorld: (clientX: number, clientY: number) => { x: number; y: number };
  /** Current zoom scale (world unit -> px multiplier vs the initial fit). */
  scale: number;
}

export interface WorldDragHandlers {
  onStart?: (world: { x: number; y: number }, e: React.PointerEvent) => void;
  onMove: (world: { x: number; y: number }, e: React.PointerEvent) => void;
  onEnd?: (world: { x: number; y: number }, e: React.PointerEvent) => void;
}

/**
 * Pointer handlers for dragging an element in world coordinates.
 * Attach the returned props to any SVG element inside a CanvasSurface.
 */
export function worldDrag(api: CanvasApi, handlers: WorldDragHandlers) {
  return {
    onPointerDown: (e: React.PointerEvent) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      (e.target as Element).setPointerCapture(e.pointerId);
      handlers.onStart?.(api.toWorld(e.clientX, e.clientY), e);
    },
    onPointerMove: (e: React.PointerEvent) => {
      if (!(e.target as Element).hasPointerCapture?.(e.pointerId)) return;
      handlers.onMove(api.toWorld(e.clientX, e.clientY), e);
    },
    onPointerUp: (e: React.PointerEvent) => {
      if (!(e.target as Element).hasPointerCapture?.(e.pointerId)) return;
      (e.target as Element).releasePointerCapture(e.pointerId);
      handlers.onEnd?.(api.toWorld(e.clientX, e.clientY), e);
    },
  };
}

export function snapTo(v: number, step: number): number {
  return step > 0 ? Math.round(v / step) * step : v;
}

interface CanvasSurfaceProps {
  /** World-space viewBox: the region visible at zoom 1. */
  world: { x: number; y: number; w: number; h: number };
  height: number | string;
  children: (api: CanvasApi) => ReactNode;
  /** Optional world-space grid rendering (line every `gridStep` units). */
  gridStep?: number;
  /** Background click (not drag) in world coords — e.g. deselect / place. */
  onBackgroundClick?: (world: { x: number; y: number }) => void;
  /** Native drop from an HTML drag source (palette) in world coords. */
  onDropWorld?: (world: { x: number; y: number }, e: React.DragEvent) => void;
  disablePanZoom?: boolean;
  background?: string;
  cursor?: string;
  ariaLabel?: string;
}

export default function CanvasSurface({
  world,
  height,
  children,
  gridStep,
  onBackgroundClick,
  onDropWorld,
  disablePanZoom = false,
  background = tokens.color.surfaceSunken,
  cursor,
  ariaLabel,
}: CanvasSurfaceProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  // view = the world-space rectangle currently mapped onto the svg viewport.
  const [view, setView] = useState(world);
  const panRef = useRef<{ startClient: { x: number; y: number }; startView: typeof world; moved: boolean } | null>(null);

  useEffect(() => {
    setView(world);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world.x, world.y, world.w, world.h]);

  const toWorld = useCallback(
    (clientX: number, clientY: number) => {
      const svg = svgRef.current;
      if (!svg) return { x: 0, y: 0 };
      const rect = svg.getBoundingClientRect();
      return {
        x: view.x + ((clientX - rect.left) / rect.width) * view.w,
        y: view.y + ((clientY - rect.top) / rect.height) * view.h,
      };
    },
    [view]
  );

  const api: CanvasApi = { toWorld, scale: world.w / view.w };

  const onWheel = useCallback(
    (e: React.WheelEvent) => {
      if (disablePanZoom) return;
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.12 : 1 / 1.12;
      setView((v) => {
        const nw = Math.min(world.w * 4, Math.max(world.w / 12, v.w * factor));
        const ratio = nw / v.w;
        const c = toWorld(e.clientX, e.clientY);
        return {
          x: c.x - (c.x - v.x) * ratio,
          y: c.y - (c.y - v.y) * ratio,
          w: nw,
          h: v.h * ratio,
        };
      });
    },
    [disablePanZoom, toWorld, world.w]
  );

  // Non-passive wheel listener so preventDefault actually stops page scroll.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || disablePanZoom) return;
    const stop = (e: WheelEvent) => e.preventDefault();
    svg.addEventListener('wheel', stop, { passive: false });
    return () => svg.removeEventListener('wheel', stop);
  }, [disablePanZoom]);

  const gridLines: ReactNode[] = [];
  if (gridStep && gridStep > 0) {
    const x0 = Math.floor(view.x / gridStep) * gridStep;
    const y0 = Math.floor(view.y / gridStep) * gridStep;
    for (let gx = x0; gx <= view.x + view.w; gx += gridStep) {
      gridLines.push(
        <line key={`v${gx}`} x1={gx} y1={view.y} x2={gx} y2={view.y + view.h} stroke={tokens.color.border} strokeWidth={view.w / 900} />
      );
    }
    for (let gy = y0; gy <= view.y + view.h; gy += gridStep) {
      gridLines.push(
        <line key={`h${gy}`} x1={view.x} y1={gy} x2={view.x + view.w} y2={gy} stroke={tokens.color.border} strokeWidth={view.w / 900} />
      );
    }
  }

  return (
    <Box
      sx={{
        border: `1px solid ${tokens.color.border}`,
        borderRadius: 1,
        overflow: 'hidden',
        bgcolor: background,
        height,
        transition: `border-color ${tokens.motion.normal}`,
        '&:focus-within': { borderColor: tokens.color.borderStrong },
      }}
      onDragOver={onDropWorld ? (e) => e.preventDefault() : undefined}
      onDrop={onDropWorld ? (e) => { e.preventDefault(); onDropWorld(toWorld(e.clientX, e.clientY), e); } : undefined}
    >
      <svg
        ref={svgRef}
        role="application"
        aria-label={ariaLabel}
        width="100%"
        height="100%"
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        style={{ display: 'block', cursor: cursor ?? (disablePanZoom ? 'default' : 'grab'), touchAction: 'none' }}
        onWheel={onWheel}
        onPointerDown={(e) => {
          if (e.button !== 0 || e.target !== svgRef.current) return;
          panRef.current = { startClient: { x: e.clientX, y: e.clientY }, startView: view, moved: false };
          (e.target as Element).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const pan = panRef.current;
          if (!pan) return;
          const svg = svgRef.current;
          if (!svg) return;
          const rect = svg.getBoundingClientRect();
          const dx = ((e.clientX - pan.startClient.x) / rect.width) * view.w;
          const dy = ((e.clientY - pan.startClient.y) / rect.height) * view.h;
          if (Math.abs(e.clientX - pan.startClient.x) + Math.abs(e.clientY - pan.startClient.y) > 3) pan.moved = true;
          if (!disablePanZoom && pan.moved) {
            setView({ ...view, x: pan.startView.x - dx, y: pan.startView.y - dy });
          }
        }}
        onPointerUp={(e) => {
          const pan = panRef.current;
          panRef.current = null;
          if (pan && !pan.moved && onBackgroundClick) onBackgroundClick(toWorld(e.clientX, e.clientY));
        }}
      >
        {gridLines}
        {children(api)}
      </svg>
    </Box>
  );
}
