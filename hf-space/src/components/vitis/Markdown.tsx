/** Minimal markdown renderer for PRD docs (headings, lists, tables,
 * fenced code blocks, bold, inline code). No external dependencies. */
import Box from '@mui/material/Box';
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
    if (line.trim() === '') {
      i += 1;
      continue;
    }
    const para: string[] = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() !== '' && !/^(#{1,4}\s|```|\s*[-*]\s|\s*\d+\.\s)/.test(lines[i])) {
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
