/**
 * Draggable node-graph canvas (direct-manipulation primitive).
 *
 * Generic over node payloads: the caller owns the graph state and receives
 * `onMoveNode` / `onConnect` / `onSelect` callbacks; this component renders
 * nodes (default card rendering or a custom `renderNode`), bezier edges with
 * arrowheads, drag with snap-to-grid, and interactive edge drawing from a
 * node's output port to another node's input port.
 *
 * Used by the Pipeline Builder; reusable for any stage/flow editor.
 */
import { useState, type ReactNode } from 'react';
import CanvasSurface, { snapTo, worldDrag, type CanvasApi } from './canvas';
import { tokens } from '../../theme';

export interface GraphNode {
  id: string;
  x: number;
  y: number;
  label: string;
  sublabel?: string;
  color?: string;
  w?: number;
  h?: number;
  badge?: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
  color?: string;
  dashed?: boolean;
}

interface NodeCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  onMoveNode?: (id: string, x: number, y: number) => void;
  /** When provided, dragging from a node's output port draws a new edge. */
  onConnect?: (fromId: string, toId: string) => void;
  height?: number | string;
  world?: { x: number; y: number; w: number; h: number };
  snap?: number;
  nodeW?: number;
  nodeH?: number;
  renderNode?: (node: GraphNode, selected: boolean) => ReactNode;
  ariaLabel?: string;
}

const DEFAULT_W = 132;
const DEFAULT_H = 56;

function edgePath(from: GraphNode, to: GraphNode, w: number, h: number): string {
  const fw = from.w ?? w;
  const tw = to.w ?? w;
  const sx = from.x + fw / 2;
  const sy = from.y;
  const ex = to.x - tw / 2 - 4;
  const ey = to.y;
  if (ex < sx - 20) {
    // feedback edge: loop under both nodes
    const fh = from.h ?? h;
    const drop = Math.max(from.y, to.y) + fh + 40;
    return `M ${from.x} ${from.y + fh / 2} C ${from.x} ${drop}, ${to.x} ${drop}, ${to.x} ${to.y + (to.h ?? h) / 2 + 4}`;
  }
  const mx = (sx + ex) / 2;
  return `M ${sx} ${sy} C ${mx} ${sy}, ${mx} ${ey}, ${ex} ${ey}`;
}

export default function NodeCanvas({
  nodes,
  edges,
  selectedId,
  onSelect,
  onMoveNode,
  onConnect,
  height = 480,
  world = { x: 0, y: 0, w: 1200, h: 520 },
  snap = 20,
  nodeW = DEFAULT_W,
  nodeH = DEFAULT_H,
  renderNode,
  ariaLabel,
}: NodeCanvasProps) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const [dragOffset, setDragOffset] = useState<{ id: string; dx: number; dy: number } | null>(null);
  const [pendingEdge, setPendingEdge] = useState<{ fromId: string; x: number; y: number; overId: string | null } | null>(null);

  const nodeAt = (x: number, y: number): GraphNode | null =>
    nodes.find((n) => {
      const w2 = (n.w ?? nodeW) / 2;
      const h2 = (n.h ?? nodeH) / 2;
      return x >= n.x - w2 - 8 && x <= n.x + w2 + 8 && y >= n.y - h2 - 8 && y <= n.y + h2 + 8;
    }) ?? null;

  return (
    <CanvasSurface
      world={world}
      height={height}
      gridStep={snap * 2}
      onBackgroundClick={() => onSelect?.(null)}
      ariaLabel={ariaLabel ?? 'Node graph canvas'}
    >
      {(api: CanvasApi) => (
        <>
          <defs>
            <marker id="nc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a5561" />
            </marker>
            <marker id="nc-arrow-hot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={tokens.color.info} />
            </marker>
          </defs>

          {edges.map((edge, i) => {
            const from = byId.get(edge.from);
            const to = byId.get(edge.to);
            if (!from || !to) return null;
            const hot = selectedId === edge.from || selectedId === edge.to;
            return (
              <g key={`${edge.from}->${edge.to}-${i}`}>
                <path
                  d={edgePath(from, to, nodeW, nodeH)}
                  fill="none"
                  stroke={hot ? tokens.color.info : edge.color ?? '#4a5561'}
                  strokeWidth={hot ? 2 : 1.5}
                  strokeDasharray={edge.dashed ? '5 4' : undefined}
                  markerEnd={hot ? 'url(#nc-arrow-hot)' : 'url(#nc-arrow)'}
                  style={{ transition: `stroke ${tokens.motion.fast}` }}
                />
                {edge.label ? (
                  <text
                    x={(from.x + to.x) / 2}
                    y={(from.y + to.y) / 2 - 8}
                    textAnchor="middle"
                    fill={tokens.color.textDim}
                    fontSize={10.5}
                  >
                    {edge.label}
                  </text>
                ) : null}
              </g>
            );
          })}

          {/* edge being drawn */}
          {pendingEdge ? (() => {
            const from = byId.get(pendingEdge.fromId);
            if (!from) return null;
            return (
              <path
                d={`M ${from.x + (from.w ?? nodeW) / 2} ${from.y} L ${pendingEdge.x} ${pendingEdge.y}`}
                fill="none"
                stroke={tokens.color.info}
                strokeWidth={2}
                strokeDasharray="6 4"
                markerEnd="url(#nc-arrow-hot)"
                pointerEvents="none"
              />
            );
          })() : null}

          {nodes.map((node) => {
            const w = node.w ?? nodeW;
            const h = node.h ?? nodeH;
            const selected = selectedId === node.id;
            const targeted = pendingEdge?.overId === node.id && pendingEdge.fromId !== node.id;
            const drag = onMoveNode
              ? worldDrag(api, {
                  onStart: (p) => {
                    setDragOffset({ id: node.id, dx: node.x - p.x, dy: node.y - p.y });
                    onSelect?.(node.id);
                  },
                  onMove: (p) => {
                    const off = dragOffset && dragOffset.id === node.id ? dragOffset : { dx: 0, dy: 0 };
                    onMoveNode(node.id, snapTo(p.x + off.dx, snap), snapTo(p.y + off.dy, snap));
                  },
                  onEnd: () => setDragOffset(null),
                })
              : { onPointerDown: (e: React.PointerEvent) => { e.stopPropagation(); onSelect?.(node.id); } };
            return (
              <g key={node.id} {...drag} style={{ cursor: onMoveNode ? 'grab' : 'pointer' }}>
                {renderNode ? (
                  renderNode(node, selected)
                ) : (
                  <>
                    <rect
                      x={node.x - w / 2}
                      y={node.y - h / 2}
                      width={w}
                      height={h}
                      rx={8}
                      fill={selected ? tokens.color.surfaceRaised : tokens.color.surface}
                      stroke={targeted ? tokens.color.success : selected ? tokens.color.info : node.color ?? tokens.color.borderStrong}
                      strokeWidth={selected || targeted ? 2.5 : 2}
                      style={{ transition: `stroke ${tokens.motion.fast}` }}
                    />
                    <text x={node.x} y={node.y + (node.sublabel ? -2 : 4)} textAnchor="middle" fill={tokens.color.text} fontSize={12} fontWeight={700} pointerEvents="none">
                      {node.label}
                    </text>
                    {node.sublabel ? (
                      <text x={node.x} y={node.y + 14} textAnchor="middle" fill={node.color ?? tokens.color.textDim} fontSize={9.5} fontWeight={600} pointerEvents="none">
                        {node.sublabel}
                      </text>
                    ) : null}
                    {node.badge ? (
                      <g pointerEvents="none">
                        <rect x={node.x + w / 2 - 26} y={node.y - h / 2 - 9} width={30} height={16} rx={8} fill={tokens.color.dangerStrong} />
                        <text x={node.x + w / 2 - 11} y={node.y - h / 2 + 3} textAnchor="middle" fill="#fff" fontSize={9.5} fontWeight={800}>
                          {node.badge}
                        </text>
                      </g>
                    ) : null}
                  </>
                )}

                {/* connection ports */}
                {onConnect ? (
                  <>
                    {/* input port (left) */}
                    <circle cx={node.x - w / 2} cy={node.y} r={5} fill={targeted ? tokens.color.success : '#4a5561'} stroke={tokens.color.surfaceSunken} strokeWidth={1.5} pointerEvents="none" />
                    {/* output port (right) — drag to draw an edge */}
                    <circle
                      cx={node.x + w / 2}
                      cy={node.y}
                      r={6}
                      fill={pendingEdge?.fromId === node.id ? tokens.color.info : '#5c6873'}
                      stroke={tokens.color.surfaceSunken}
                      strokeWidth={1.5}
                      style={{ cursor: 'crosshair' }}
                      {...worldDrag(api, {
                        onStart: (p) => setPendingEdge({ fromId: node.id, x: p.x, y: p.y, overId: null }),
                        onMove: (p) => setPendingEdge({ fromId: node.id, x: p.x, y: p.y, overId: nodeAt(p.x, p.y)?.id ?? null }),
                        onEnd: (p) => {
                          const target = nodeAt(p.x, p.y);
                          if (target && target.id !== node.id) onConnect(node.id, target.id);
                          setPendingEdge(null);
                        },
                      })}
                    />
                  </>
                ) : null}
              </g>
            );
          })}
        </>
      )}
    </CanvasSurface>
  );
}
