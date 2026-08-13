/**
 * The explanation design system. Three primitives, used identically everywhere:
 *
 *   <Term k="precision">Precision</Term>   inline dotted-underline text w/ glossary tooltip
 *   <InfoDot term="wilson_ci" />           small (i) icon w/ glossary tooltip — for labels,
 *                                          column headers, card titles
 *   <InfoDot title="..." detail="..." />   same icon w/ ad-hoc content when no glossary key fits
 *
 * All tooltips render through one <GlossaryContent> so definitions look the
 * same on every screen. Definitions live in src/content/glossary.ts.
 */
import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { Info } from 'lucide-react';
import { GLOSSARY, type GlossaryEntry } from '../../content/glossary';

// ---------------------------------------------------------------- tooltip body

export function GlossaryContent({
  entry,
  compact = false,
}: {
  entry: Pick<GlossaryEntry, 'term' | 'short' | 'detail' | 'caveat'>;
  compact?: boolean;
}) {
  return (
    <Box sx={{ maxWidth: 340, py: 0.25 }}>
      <Typography variant="caption" sx={{ fontWeight: 700, color: '#4fc3f7', display: 'block', mb: 0.25 }}>
        {entry.term}
      </Typography>
      <Typography variant="caption" sx={{ color: '#e6e9ec', display: 'block' }}>
        {entry.short}
      </Typography>
      {!compact ? (
        <Typography variant="caption" sx={{ color: '#aab4be', display: 'block', mt: 0.5 }}>
          {entry.detail}
        </Typography>
      ) : null}
      {!compact && entry.caveat ? (
        <Typography variant="caption" sx={{ color: '#ffd54f', display: 'block', mt: 0.5 }}>
          Caveat: {entry.caveat}
        </Typography>
      ) : null}
    </Box>
  );
}

const TOOLTIP_SLOT_PROPS = {
  tooltip: {
    sx: {
      bgcolor: '#1d242c',
      border: '1px solid #2f3944',
      boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
      p: 1.25,
    },
  },
} as const;

function resolveEntry(
  term?: string,
  title?: ReactNode,
  detail?: ReactNode
): Pick<GlossaryEntry, 'term' | 'short' | 'detail' | 'caveat'> | null {
  if (term && GLOSSARY[term]) return GLOSSARY[term];
  if (title || detail) {
    return {
      term: typeof title === 'string' ? title : '',
      short: typeof title !== 'string' && title ? String(title) : (detail as string) ?? '',
      detail: title && detail ? (detail as string) : '',
    } as GlossaryEntry;
  }
  return null;
}

// ---------------------------------------------------------------- <ExplainTip>

/**
 * Wraps arbitrary children in a consistently-styled explanation tooltip.
 * Provide either a glossary `term` key or ad-hoc `title` + `detail`.
 */
export function ExplainTip({
  term,
  title,
  detail,
  compact = false,
  children,
}: {
  term?: string;
  title?: string;
  detail?: string;
  compact?: boolean;
  children: React.ReactElement;
}) {
  const entry = term && GLOSSARY[term] ? GLOSSARY[term] : title || detail ? { term: title ?? '', short: detail ?? '', detail: '' } : null;
  if (!entry) return children;
  return (
    <Tooltip title={<GlossaryContent entry={entry} compact={compact} />} slotProps={TOOLTIP_SLOT_PROPS} enterDelay={150}>
      {children}
    </Tooltip>
  );
}

// ---------------------------------------------------------------- <InfoDot>

/** Small info icon with a glossary (or ad-hoc) tooltip. */
export function InfoDot({
  term,
  title,
  detail,
  size = 12,
  sx,
}: {
  term?: string;
  title?: string;
  detail?: string;
  size?: number;
  sx?: object;
}) {
  const entry = resolveEntry(term, title, detail);
  if (!entry) return null;
  return (
    <Tooltip title={<GlossaryContent entry={entry} />} slotProps={TOOLTIP_SLOT_PROPS} enterDelay={100}>
      <Box
        component="span"
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          verticalAlign: 'middle',
          color: '#5c6873',
          cursor: 'help',
          ml: 0.5,
          '&:hover': { color: '#4fc3f7' },
          ...sx,
        }}
      >
        <Info size={size} />
      </Box>
    </Tooltip>
  );
}

// ---------------------------------------------------------------- <Term>

/** Inline term with dotted underline; hover reveals the glossary definition. */
export function Term({
  k,
  children,
  color,
}: {
  k: string;
  children?: ReactNode;
  color?: string;
}) {
  const entry = GLOSSARY[k];
  const text = children ?? entry?.term ?? k;
  if (!entry) return <span>{text}</span>;
  return (
    <Tooltip title={<GlossaryContent entry={entry} />} slotProps={TOOLTIP_SLOT_PROPS} enterDelay={150}>
      <Box
        component="span"
        sx={{
          borderBottom: '1px dotted #5c6873',
          cursor: 'help',
          color,
          '&:hover': { borderBottomColor: '#4fc3f7' },
        }}
      >
        {text}
      </Box>
    </Tooltip>
  );
}

// ---------------------------------------------------------------- <HeadCell> helper

/**
 * Column-header label + InfoDot in one span, for table heads:
 *   <TableCell><HeadCell label="Recall" term="recall" /></TableCell>
 */
export function HeadCell({ label, term, title, detail }: { label: string; term?: string; title?: string; detail?: string }) {
  return (
    <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', whiteSpace: 'nowrap' }}>
      {label}
      <InfoDot term={term} title={title} detail={detail} />
    </Box>
  );
}
