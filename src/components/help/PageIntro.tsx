/**
 * Page-level explanation header: a 1–2 sentence subtitle under every page
 * title plus an expandable "About this page" panel (purpose / how to read /
 * actions / data flow). Open state persists per page in localStorage.
 * Content lives in src/content/pageHelp.ts.
 */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Collapse from '@mui/material/Collapse';
import Typography from '@mui/material/Typography';
import { ChevronDown, ChevronUp, CircleHelp } from 'lucide-react';
import type { PageId } from '../../context/LabelEvalContext';
import { PAGE_HELP } from '../../content/pageHelp';

const STORAGE_PREFIX = 'sf-help-open:';

function readOpen(page: PageId): boolean {
  try {
    return localStorage.getItem(STORAGE_PREFIX + page) === '1';
  } catch {
    return false;
  }
}

function writeOpen(page: PageId, open: boolean) {
  try {
    localStorage.setItem(STORAGE_PREFIX + page, open ? '1' : '0');
  } catch {
    /* private mode etc. — persistence is a nice-to-have */
  }
}

function HelpSection({ heading, body }: { heading: string; body: string }) {
  return (
    <Box sx={{ flex: '1 1 260px', minWidth: 240 }}>
      <Typography
        variant="caption"
        sx={{ color: '#4fc3f7', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.6, display: 'block', mb: 0.5 }}
      >
        {heading}
      </Typography>
      <Typography variant="body2" sx={{ color: '#aab4be', fontSize: 12.5, lineHeight: 1.55 }}>
        {body}
      </Typography>
    </Box>
  );
}

export default function PageIntro({ page, dense = false }: { page: PageId; dense?: boolean }) {
  const help = PAGE_HELP[page];
  const [open, setOpen] = useState(() => readOpen(page));

  const toggle = () => {
    setOpen((prev) => {
      writeOpen(page, !prev);
      return !prev;
    });
  };

  if (!help) return null;

  return (
    <Box sx={{ mb: dense ? 1 : 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
        <Typography variant="body2" sx={{ color: '#8a949e', flex: 1, maxWidth: 980, lineHeight: 1.5 }}>
          {help.subtitle}
        </Typography>
        <Button
          size="small"
          onClick={toggle}
          startIcon={<CircleHelp size={14} />}
          endIcon={open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          sx={{ color: open ? '#4fc3f7' : '#8a949e', flexShrink: 0, textTransform: 'none', fontSize: 12 }}
        >
          About this page
        </Button>
      </Box>
      <Collapse in={open} unmountOnExit>
        <Box
          sx={{
            mt: 1,
            p: 2,
            display: 'flex',
            gap: 3,
            flexWrap: 'wrap',
            bgcolor: '#141a20',
            border: '1px solid #232a31',
            borderLeft: '3px solid #4fc3f7',
            borderRadius: 1,
          }}
        >
          <HelpSection heading="Purpose" body={help.purpose} />
          <HelpSection heading="How to read it" body={help.reading} />
          <HelpSection heading="What you can do" body={help.actions} />
          <HelpSection heading="Data in / out" body={help.dataFlow} />
        </Box>
      </Collapse>
    </Box>
  );
}
