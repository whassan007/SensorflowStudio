/**
 * Drag-handle list reordering (direct-manipulation primitive).
 *
 * Pointer-based (no DnD library): grab the handle, rows reorder live as you
 * drag, commit on release. Used for ordering query group-by chips, dashboard
 * widget stacking, and anywhere an ordered list needs rearranging.
 */
import { useRef, useState, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import { GripVertical } from 'lucide-react';
import { tokens } from '../../theme';

export interface DragListItem {
  id: string;
  content: ReactNode;
}

interface DragListProps {
  items: DragListItem[];
  onReorder: (orderedIds: string[]) => void;
  dense?: boolean;
}

export default function DragList({ items, onReorder, dense = false }: DragListProps) {
  const [order, setOrder] = useState<string[] | null>(null); // non-null while dragging
  const [dragId, setDragId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const ids = order ?? items.map((i) => i.id);
  const byId = new Map(items.map((i) => [i.id, i]));

  const indexFromPointer = (clientY: number): number => {
    const container = containerRef.current;
    if (!container) return 0;
    const rows = Array.from(container.querySelectorAll<HTMLElement>('[data-dragrow]'));
    for (let i = 0; i < rows.length; i += 1) {
      const r = rows[i].getBoundingClientRect();
      if (clientY < r.top + r.height / 2) return i;
    }
    return rows.length - 1;
  };

  return (
    <Box ref={containerRef} sx={{ display: 'flex', flexDirection: 'column', gap: dense ? 0.5 : 0.75 }}>
      {ids.map((id) => {
        const item = byId.get(id);
        if (!item) return null;
        const dragging = dragId === id;
        return (
          <Box
            key={id}
            data-dragrow
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              px: 1,
              py: dense ? 0.5 : 0.75,
              borderRadius: 1,
              border: `1px solid ${dragging ? tokens.color.info : tokens.color.border}`,
              bgcolor: dragging ? tokens.color.surfaceRaised : tokens.color.surface,
              boxShadow: dragging ? tokens.elevation.raised : 'none',
              transition: `border-color ${tokens.motion.fast}, box-shadow ${tokens.motion.fast}`,
              userSelect: 'none',
            }}
          >
            <Box
              component="span"
              sx={{
                display: 'inline-flex',
                color: tokens.color.textFaint,
                cursor: 'grab',
                touchAction: 'none',
                '&:hover': { color: tokens.color.info },
              }}
              onPointerDown={(e) => {
                e.preventDefault();
                (e.target as Element).setPointerCapture(e.pointerId);
                setDragId(id);
                setOrder(items.map((i) => i.id));
              }}
              onPointerMove={(e) => {
                if (dragId !== id || order === null) return;
                const target = indexFromPointer(e.clientY);
                const current = order.indexOf(id);
                if (target !== current && target >= 0) {
                  const next = [...order];
                  next.splice(current, 1);
                  next.splice(target, 0, id);
                  setOrder(next);
                }
              }}
              onPointerUp={(e) => {
                (e.target as Element).releasePointerCapture(e.pointerId);
                if (order) onReorder(order);
                setOrder(null);
                setDragId(null);
              }}
            >
              <GripVertical size={15} />
            </Box>
            <Box sx={{ flex: 1, minWidth: 0 }}>{item.content}</Box>
          </Box>
        );
      })}
    </Box>
  );
}
