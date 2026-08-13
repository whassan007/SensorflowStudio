/**
 * Shared building blocks for the Safety & Compliance pages: published-run
 * selection (safety analyses require published megaeval runs) and a compact
 * markdown renderer for evidence packages.
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import type { EvaluationRunInfo } from '../../types/megaeval';
import { getRuns } from '../../services/megaeval';
import { tokens } from '../../theme';

// ---------------------------------------------------------------- runs hook

export function usePublishedRuns() {
  const [runs, setRuns] = useState<EvaluationRunInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    getRuns()
      .then((r) => {
        if (cancelled) return;
        const published = r.runs
          .filter((x) => x.status === 'published')
          .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
        setRuns(published);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return { runs, error };
}

export function RunSelect({
  label,
  value,
  onChange,
  runs,
  exclude,
}: {
  label: string;
  value: string | null;
  onChange: (id: string) => void;
  runs: EvaluationRunInfo[] | null;
  exclude?: string | null;
}) {
  const options = useMemo(() => (runs ?? []).filter((r) => r.run_id !== exclude), [runs, exclude]);
  return (
    <TextField
      select
      size="small"
      label={label}
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      sx={{ minWidth: 260 }}
    >
      {options.map((r) => (
        <MenuItem key={r.run_id} value={r.run_id}>
          <Box component="span" sx={{ fontFamily: 'monospace', fontSize: 12.5 }}>
            {r.run_id.slice(0, 14)}
          </Box>
          <Box component="span" sx={{ color: tokens.color.neutral, ml: 1, fontSize: 12 }}>
            {r.model_version}
          </Box>
        </MenuItem>
      ))}
    </TextField>
  );
}

// ---------------------------------------------------------------- markdown-lite

/** Minimal markdown renderer (headings, bold, code, lists, tables, hr) —
 * enough for the generated Safety Evidence Package; no external deps. */
export function MarkdownLite({ text }: { text: string }) {
  const blocks = useMemo(() => parseBlocks(text), [text]);
  return <Box sx={{ '& > *:first-of-type': { mt: 0 } }}>{blocks}</Box>;
}

function inline(text: string): ReactNode[] {
  // **bold** and `code`
  const parts: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith('**')) {
      parts.push(<strong key={key++}>{tok.slice(2, -2)}</strong>);
    } else {
      parts.push(
        <Box key={key++} component="code" sx={{ fontFamily: 'monospace', fontSize: '0.9em', bgcolor: tokens.color.surfaceRaised, px: 0.5, borderRadius: '3px' }}>
          {tok.slice(1, -1)}
        </Box>
      );
    }
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function parseBlocks(text: string): ReactNode[] {
  const lines = text.split('\n');
  const out: ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*$/.test(line)) {
      i += 1;
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      out.push(
        <Typography
          key={key++}
          variant={level === 1 ? 'h5' : level === 2 ? 'h6' : 'subtitle1'}
          sx={{ fontWeight: 800, mt: level === 1 ? 2 : 2.5, mb: 0.75, color: level >= 3 ? tokens.color.info : undefined }}
        >
          {inline(h[2])}
        </Typography>
      );
      i += 1;
      continue;
    }
    if (/^\s*(---|\*\*\*)\s*$/.test(line)) {
      out.push(<Box key={key++} sx={{ borderBottom: `1px solid ${tokens.color.border}`, my: 1.5 }} />);
      i += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
        i += 1;
      }
      out.push(
        <Box key={key++} component="ul" sx={{ my: 0.75, pl: 3 }}>
          {items.map((it, j) => (
            <Typography key={j} component="li" variant="body2" sx={{ color: tokens.color.textDim, mb: 0.25 }}>
              {inline(it)}
            </Typography>
          ))}
        </Box>
      );
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const rows: string[][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        const cells = lines[i].trim().slice(1, -1).split('|').map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i += 1;
      }
      if (rows.length) {
        const [head, ...body] = rows;
        out.push(
          <Box key={key++} component="table" sx={{ borderCollapse: 'collapse', my: 1, '& td, & th': { border: `1px solid ${tokens.color.border}`, px: 1, py: 0.4, fontSize: 12.5 }, '& th': { bgcolor: tokens.color.surfaceRaised, textAlign: 'left' } }}>
            <thead>
              <tr>{head.map((c, j) => <th key={j}>{inline(c)}</th>)}</tr>
            </thead>
            <tbody>
              {body.map((r, ri) => (
                <tr key={ri}>{r.map((c, j) => <td key={j}>{inline(c)}</td>)}</tr>
              ))}
            </tbody>
          </Box>
        );
      }
      continue;
    }
    // paragraph: gather consecutive plain lines
    const para: string[] = [line];
    i += 1;
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^(#{1,4})\s|^\s*[-*]\s|^\s*\|/.test(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    out.push(
      <Typography key={key++} variant="body2" sx={{ color: tokens.color.textDim, my: 0.75, lineHeight: 1.6 }}>
        {inline(para.join(' '))}
      </Typography>
    );
  }
  return out;
}

/** Trigger a client-side JSON download. */
export function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
