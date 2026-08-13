/**
 * Global help: the (?) button in the app bar. Opens a dialog with three tabs:
 *   How it works  — the pipeline story (input → engines → gate → triage → HITL → flywheel)
 *   Glossary      — searchable browser over every definition in src/content/glossary.ts
 *   Pages         — what each page does, deep-linking into the app
 */
import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import InputAdornment from '@mui/material/InputAdornment';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { ArrowRight, CircleHelp, Search, X } from 'lucide-react';
import { GLOSSARY, GLOSSARY_CATEGORIES, type GlossaryEntry } from '../../content/glossary';
import { PAGE_HELP } from '../../content/pageHelp';
import { useLabelEval, type PageId } from '../../context/LabelEvalContext';

// ---------------------------------------------------------------- how it works

const PIPELINE_STAGES: Array<{ name: string; page: PageId; text: string }> = [
  {
    name: '1 · Input',
    page: 'datasets',
    text: 'Datasets are ingested and versioned (Datasets). Raw frames flow into auto-labeling (Label Generation), which produces candidate labels at machine speed.',
  },
  {
    name: '2 · Evaluation engines',
    page: 'evaluation',
    text: 'Every candidate label is measured from multiple angles: geometry vs reference GT where it exists, GT-free structural validation (Quality Engine), camera–LiDAR consistency, anomaly ensembles (Rare Events), grader consensus, tracking quality, and model-version regression (Regression).',
  },
  {
    name: '3 · Quality gate',
    page: 'triage',
    text: 'The quality policy turns measurements into pass/fail gate lines: min IoU, max position/orientation error, min point support, min confidence, min consensus, anomaly threshold, tracking requirements.',
  },
  {
    name: '4 · Triage',
    page: 'triage',
    text: 'Gate results deterministically route each label: all-pass → AUTO_GRADED; failures → FLAGGED with explicit failure reasons; hopeless → REJECTED. Every decision records its policy ID for audit.',
  },
  {
    name: '5 · Human review (HITL)',
    page: 'review',
    text: 'Flagged labels and statistically-sampled labels reach human reviewers with full evidence (camera / LiDAR / BEV / temporal). Verdicts: verify, correct, or reject. Review of a stratified sample also puts confidence intervals on the automated headline metrics.',
  },
  {
    name: '6 · Flywheel',
    page: 'training',
    text: 'Verified labels become training datasets; new models train on them; every new model is evaluated back through the platform (Command Center evaluation runs, promote / do-not-promote decisions). Better labels → better models → better labels.',
  },
];

function HowItWorks({ goTo }: { goTo: (p: PageId) => void }) {
  return (
    <Box>
      <Typography variant="body2" sx={{ color: '#aab4be', mb: 2, lineHeight: 1.6 }}>
        Sensorflow Studio evaluates machine-generated perception labels at scale. The core loop: measure every label,
        gate it against a versioned quality policy, route the uncertain minority to humans, and feed verified results
        back into training. Aggregates lead, individual annotations are drill-down.
      </Typography>
      {PIPELINE_STAGES.map((s, i) => (
        <Box
          key={s.name}
          sx={{
            display: 'flex',
            gap: 1.5,
            p: 1.5,
            mb: 1,
            bgcolor: '#141a20',
            border: '1px solid #232a31',
            borderRadius: 1,
            alignItems: 'flex-start',
          }}
        >
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#4fc3f7', mb: 0.25 }}>
              {s.name}
            </Typography>
            <Typography variant="body2" sx={{ color: '#aab4be', fontSize: 12.5, lineHeight: 1.55 }}>
              {s.text}
            </Typography>
          </Box>
          <IconButton size="small" onClick={() => goTo(s.page)} title="Open related page" sx={{ color: '#5c6873' }}>
            <ArrowRight size={16} />
          </IconButton>
          {i < PIPELINE_STAGES.length ? null : null}
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------- glossary browser

function GlossaryBrowser() {
  const [query, setQuery] = useState('');
  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const entries = Object.values(GLOSSARY).filter(
      (e) =>
        !q ||
        e.term.toLowerCase().includes(q) ||
        e.short.toLowerCase().includes(q) ||
        e.detail.toLowerCase().includes(q)
    );
    const byCat = new Map<string, GlossaryEntry[]>();
    for (const cat of GLOSSARY_CATEGORIES) byCat.set(cat, []);
    for (const e of entries) byCat.get(e.category)?.push(e);
    return [...byCat.entries()].filter(([, list]) => list.length > 0);
  }, [query]);

  return (
    <Box>
      <TextField
        size="small"
        fullWidth
        placeholder="Search terms, definitions…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        sx={{ mb: 2 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Search size={15} color="#5c6873" />
            </InputAdornment>
          ),
        }}
      />
      {grouped.length === 0 ? (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          No terms match “{query}”.
        </Typography>
      ) : null}
      {grouped.map(([cat, entries]) => (
        <Box key={cat} sx={{ mb: 2 }}>
          <Typography
            variant="caption"
            sx={{ color: '#4fc3f7', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.6 }}
          >
            {cat}
          </Typography>
          {entries.map((e) => (
            <Box key={e.term} sx={{ py: 0.75, borderBottom: '1px solid #1d242c' }}>
              <Typography variant="body2" sx={{ fontWeight: 700, fontSize: 12.5 }}>
                {e.term}
              </Typography>
              <Typography variant="body2" sx={{ color: '#aab4be', fontSize: 12, lineHeight: 1.5 }}>
                {e.short} {e.detail}
              </Typography>
              {e.caveat ? (
                <Typography variant="body2" sx={{ color: '#ffd54f', fontSize: 12, lineHeight: 1.5, mt: 0.25 }}>
                  Caveat: {e.caveat}
                </Typography>
              ) : null}
            </Box>
          ))}
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------- page index

const PAGE_ORDER: PageId[] = [
  'command',
  'overview',
  'datasets',
  'label-generation',
  'rare-events',
  'raremine',
  'quality',
  'regression',
  'rca',
  'triage',
  'review',
  'training',
  'models',
  'evaluation',
  'audit',
  'pipeline',
  'ssam',
  'legacy',
];

const PAGE_NAMES: Record<PageId, string> = {
  command: 'Command Center',
  overview: 'Overview',
  datasets: 'Datasets',
  'label-generation': 'Label Generation',
  'rare-events': 'Rare Events',
  raremine: 'Rare-Event Miner',
  quality: 'Quality Engine',
  regression: 'Regression',
  rca: 'Root Cause Lab',
  triage: 'Triage',
  review: 'Human Review',
  training: 'Training',
  models: 'Models',
  evaluation: 'Evaluation Records',
  audit: 'Audit',
  pipeline: 'Pipeline Architecture',
  ssam: 'SSAM Safety',
  legacy: 'Legacy Studio',
};

function PageIndex({ goTo }: { goTo: (p: PageId) => void }) {
  return (
    <Box>
      {PAGE_ORDER.map((id) => (
        <Box
          key={id}
          onClick={() => goTo(id)}
          sx={{
            display: 'flex',
            gap: 1.5,
            p: 1.25,
            mb: 0.75,
            bgcolor: '#141a20',
            border: '1px solid #232a31',
            borderRadius: 1,
            cursor: 'pointer',
            alignItems: 'center',
            '&:hover': { borderColor: '#4fc3f7' },
          }}
        >
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 700, fontSize: 13 }}>
              {PAGE_NAMES[id]}
            </Typography>
            <Typography variant="body2" sx={{ color: '#8a949e', fontSize: 12, lineHeight: 1.45 }}>
              {PAGE_HELP[id].subtitle}
            </Typography>
          </Box>
          <ArrowRight size={15} color="#5c6873" />
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------- dialog shell

export default function HelpMenu() {
  const { navigate } = useLabelEval();
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<'how' | 'glossary' | 'pages'>('how');

  const goTo = (p: PageId) => {
    setOpen(false);
    navigate(p);
  };

  return (
    <>
      <Tooltip title="Help: how Sensorflow Studio works, glossary, page guide">
        <IconButton size="small" onClick={() => setOpen(true)} sx={{ ml: 1, color: '#8a949e' }} aria-label="Open help">
          <CircleHelp size={19} />
        </IconButton>
      </Tooltip>
      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{ sx: { bgcolor: '#12171d', border: '1px solid #232a31', height: '82vh' } }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 1.5, gap: 2 }}>
          <Typography variant="h6" sx={{ fontSize: 15, fontWeight: 700, flex: 1 }}>
            Sensorflow Studio Help
          </Typography>
          <IconButton size="small" onClick={() => setOpen(false)} aria-label="Close help">
            <X size={16} />
          </IconButton>
        </Box>
        <Tabs
          value={tab}
          onChange={(_, v: 'how' | 'glossary' | 'pages') => setTab(v)}
          sx={{ px: 2, minHeight: 36, '& .MuiTab-root': { minHeight: 36, fontSize: 12.5 } }}
        >
          <Tab value="how" label="How it works" />
          <Tab value="glossary" label="Glossary" />
          <Tab value="pages" label="Pages" />
        </Tabs>
        <DialogContent sx={{ pt: 2 }}>
          {tab === 'how' ? <HowItWorks goTo={goTo} /> : null}
          {tab === 'glossary' ? <GlossaryBrowser /> : null}
          {tab === 'pages' ? <PageIndex goTo={goTo} /> : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
