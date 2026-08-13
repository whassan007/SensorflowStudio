/**
 * Markdown renderer for the nextgen architecture docs.
 * Adapted from components/vitis/Markdown.tsx (which is read-only for this
 * feature) with GFM table support added — the architecture docs are
 * table-heavy comparison matrices.
 */
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';

function Inline({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**')) {
          return <strong key={i}>{p.slice(2, -2)}</strong>;
        }
        if (p.startsWith('`') && p.endsWith('`')) {
          return (
            <Box
              key={i}
              component="code"
              sx={{ bgcolor: '#232a31', px: 0.5, borderRadius: 0.5, fontFamily: 'monospace', fontSize: '0.9em' }}
            >
              {p.slice(1, -1)}
            </Box>
          );
        }
        return <span key={i}>{p}</span>;
      })}
    </>
  );
}

function splitRow(line: string): string[] {
  return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());
}

export default function Markdown({ source }: { source: string }) {
  const lines = source.split('\n');
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith('```')) {
      const code: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith('```')) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(
        <Box
          key={key++}
          component="pre"
          sx={{
            bgcolor: '#0d1117',
            border: '1px solid #232a31',
            borderRadius: 1,
            p: 1.5,
            overflowX: 'auto',
            fontSize: 11.5,
            fontFamily: 'monospace',
            lineHeight: 1.35,
            my: 1,
          }}
        >
          {code.join('\n')}
        </Box>
      );
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const header = splitRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      blocks.push(
        <Box key={key++} sx={{ overflowX: 'auto', my: 1, border: '1px solid #232a31', borderRadius: 1 }}>
          <Table size="small" sx={{ minWidth: 480 }}>
            <TableHead>
              <TableRow>
                {header.map((h, j) => (
                  <TableCell key={j} sx={{ fontWeight: 800, fontSize: 11, color: '#e0e3e7', bgcolor: '#12171d' }}>
                    <Inline text={h} />
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((r, ri) => (
                <TableRow key={ri}>
                  {r.map((c, ci) => (
                    <TableCell key={ci} sx={{ fontSize: 11, color: '#c7ccd1', verticalAlign: 'top' }}>
                      <Inline text={c} />
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      );
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      blocks.push(
        <Typography
          key={key++}
          variant={level === 1 ? 'h5' : level === 2 ? 'h6' : 'subtitle1'}
          sx={{ fontWeight: 800, mt: level === 1 ? 0.5 : 2, mb: 0.75 }}
        >
          <Inline text={h[2]} />
        </Typography>
      );
      i += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
        i += 1;
      }
      blocks.push(
        <Box key={key++} component="ul" sx={{ my: 0.5, pl: 3 }}>
          {items.map((it, j) => (
            <Typography key={j} component="li" variant="body2" sx={{ mb: 0.25, color: '#c7ccd1' }}>
              <Inline text={it} />
            </Typography>
          ))}
        </Box>
      );
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ''));
        i += 1;
      }
      blocks.push(
        <Box key={key++} component="ol" sx={{ my: 0.5, pl: 3 }}>
          {items.map((it, j) => (
            <Typography key={j} component="li" variant="body2" sx={{ mb: 0.25, color: '#c7ccd1' }}>
              <Inline text={it} />
            </Typography>
          ))}
        </Box>
      );
      continue;
    }
    if (line.trim() === '' || /^---+$/.test(line.trim())) {
      i += 1;
      continue;
    }
    const para: string[] = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^(#{1,4}\s|```|\s*[-*]\s|\s*\d+\.\s|\s*\|)/.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push(
      <Typography key={key++} variant="body2" sx={{ mb: 1, color: '#c7ccd1', lineHeight: 1.55 }}>
        <Inline text={para.join(' ')} />
      </Typography>
    );
  }
  return <Box>{blocks}</Box>;
}
