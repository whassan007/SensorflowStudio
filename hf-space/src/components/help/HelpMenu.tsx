/**
 * Global help: the (?) button in the app bar. Opens a dialog with tabs:
 *   How it works  — the pipeline story
 *   Glossary      — searchable glossary
 *   Pages         — what each page does (from pageGuides)
 *   Tips          — keyboard / nav tips
 *   Docs          — links into hf-space/docs/
 *   About         — current version and release notes
 */
import { useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import InputAdornment from '@mui/material/InputAdornment';
import Link from '@mui/material/Link';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { ArrowRight, CircleHelp, MessageCircle, Search, X } from 'lucide-react';
import { GLOSSARY, GLOSSARY_CATEGORIES, type GlossaryEntry } from '../../content/glossary';
import { PAGE_GUIDE_LIST } from '../../help/pageGuides';
import { useLabelEval, type PageId } from '../../context/LabelEvalContext';
import { AboutPanel } from './AboutDialog';
import { APP_VERSION } from '../../content/releases';

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
      {PIPELINE_STAGES.map((s) => (
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
          <IconButton size="small" onClick={() => goTo(s.page)} title="Open related page" aria-label={`Open ${s.page}`} sx={{ color: '#5c6873' }}>
            <ArrowRight size={16} />
          </IconButton>
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

function PageIndex({ goTo }: { goTo: (p: PageId) => void }) {
  return (
    <Box>
      {PAGE_GUIDE_LIST.map((g) => (
        <Box
          key={g.pageId}
          onClick={() => goTo(g.pageId)}
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
              {g.title}
            </Typography>
            <Typography variant="body2" sx={{ color: '#8a949e', fontSize: 12, lineHeight: 1.45 }}>
              {g.summary}
            </Typography>
          </Box>
          <ArrowRight size={15} color="#5c6873" />
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------- tips

const NAV_TIPS = [
  {
    title: 'Hash URLs',
    text: 'Every page is addressable as #/page or #/page/entityId. Browser back/forward and shareable deep links work.',
  },
  {
    title: 'Left drawer',
    text: 'Platform / Studio / Engines / Safety / Legacy sections list every major screen. Badges on Overview and Human Review show live alert and queue counts.',
  },
  {
    title: 'About this page',
    text: 'Under each page title, expand “About this page” for purpose, how to read the UI, actions, and data flow. Open state is remembered per page.',
  },
  {
    title: 'Tooltips & glossary',
    text: 'Hover dotted terms and (i) icons for definitions. The Glossary tab searches the full term bank.',
  },
  {
    title: 'Help chatbot',
    text: 'The floating chat bubble answers questions from the local FAQ + page-guide index (CPU-friendly). Ollama is optional enrichment when available.',
  },
  {
    title: 'Active dataset chip',
    text: 'The AppBar chip shows the dataset other pages follow. Change it on Datasets.',
  },
  {
    title: 'Version & About',
    text: 'The AppBar vX.Y.Z chip (and the matching chip at the bottom of the nav drawer) opens About: current version, links, and release notes. Help → About shows the same catalog.',
  },
];

function TipsPanel() {
  return (
    <Box>
      {NAV_TIPS.map((t) => (
        <Box key={t.title} sx={{ mb: 1.5, p: 1.5, bgcolor: '#141a20', border: '1px solid #232a31', borderRadius: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#4fc3f7', mb: 0.25 }}>
            {t.title}
          </Typography>
          <Typography variant="body2" sx={{ color: '#aab4be', fontSize: 12.5, lineHeight: 1.55 }}>
            {t.text}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------- docs

const DOC_LINKS: Array<{ label: string; path: string; note: string }> = [
  { label: 'Platform inventory', path: 'docs/PLATFORM_INVENTORY.md', note: 'What ships where in the Space.' },
  { label: 'Population-scale roadmap', path: 'docs/POPULATION_SCALE_ROADMAP.md', note: 'Megaeval aggregate-first design.' },
  { label: 'Hardening audit', path: 'docs/hardening/audit.md', note: 'Production-readiness findings.' },
  { label: 'ROTR architecture', path: 'docs/architecture/rotr-architecture.md', note: 'Right-of-the-road control plane.' },
  { label: 'Studio 2.0 review', path: 'docs/architecture/studio2-review.md', note: 'Governance / release gate.' },
  { label: 'Next-gen ADR', path: 'docs/architecture/nextgen-adr.md', note: 'Closed-loop lab decisions.' },
  { label: 'Vitis HIL PRD', path: 'docs/prd/vitis-hil-regression.md', note: 'Hardware acceleration demos.' },
  { label: 'Retro final report', path: 'docs/retro/final-report.md', note: 'Retrospective analyzer.' },
];

function DocsPanel() {
  return (
    <Box>
      <Typography variant="body2" sx={{ color: '#aab4be', mb: 2, lineHeight: 1.55 }}>
        In-repo guides under <code>hf-space/docs/</code>. On Hugging Face Spaces these are packaged with the backend;
        locally open them in the repo. The Production Readiness page also surfaces the hardening audit live.
      </Typography>
      {DOC_LINKS.map((d) => (
        <Box key={d.path} sx={{ mb: 1, p: 1.25, bgcolor: '#141a20', border: '1px solid #232a31', borderRadius: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 700, fontSize: 13 }}>
            {d.label}
          </Typography>
          <Typography variant="body2" sx={{ color: '#8a949e', fontSize: 12, mb: 0.5 }}>
            {d.note}
          </Typography>
          <Typography variant="caption" sx={{ color: '#5c6873', fontFamily: 'monospace' }}>
            {d.path}
          </Typography>
        </Box>
      ))}
      <Typography variant="body2" sx={{ color: '#8a949e', mt: 1, fontSize: 12 }}>
        Also see{' '}
        <Link href="#/production-readiness" underline="hover" sx={{ color: '#4fc3f7' }}>
          Production Readiness
        </Link>{' '}
        for the live audit browser.
      </Typography>
    </Box>
  );
}

// ---------------------------------------------------------------- dialog shell

type HelpTab = 'how' | 'glossary' | 'pages' | 'tips' | 'docs' | 'about';

export default function HelpMenu({ onOpenChat }: { onOpenChat?: () => void }) {
  const { navigate } = useLabelEval();
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<HelpTab>('how');

  const goTo = (p: PageId) => {
    setOpen(false);
    navigate(p);
  };

  return (
    <>
      <Tooltip title="Help: overview, glossary, page guides, tips, docs, about">
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
        <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 1.5, gap: 1 }}>
          <Typography variant="h6" sx={{ fontSize: 15, fontWeight: 700, flex: 1 }}>
            Sensorflow Studio Help
            <Typography component="span" sx={{ ml: 1, color: '#5c6873', fontSize: 12, fontFamily: 'monospace', fontWeight: 500 }}>
              v{APP_VERSION}
            </Typography>
          </Typography>
          {onOpenChat ? (
            <Button
              size="small"
              startIcon={<MessageCircle size={14} />}
              onClick={() => {
                setOpen(false);
                onOpenChat();
              }}
              sx={{ textTransform: 'none', color: '#4fc3f7' }}
              title="Open the help chatbot"
              aria-label="Open help chatbot"
            >
              Ask chatbot
            </Button>
          ) : null}
          <IconButton size="small" onClick={() => setOpen(false)} aria-label="Close help">
            <X size={16} />
          </IconButton>
        </Box>
        <Tabs
          value={tab}
          onChange={(_, v: HelpTab) => setTab(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ px: 2, minHeight: 36, '& .MuiTab-root': { minHeight: 36, fontSize: 12.5 } }}
        >
          <Tab value="how" label="How it works" />
          <Tab value="glossary" label="Glossary" />
          <Tab value="pages" label="Pages" />
          <Tab value="tips" label="Tips" />
          <Tab value="docs" label="Docs" />
          <Tab value="about" label="About" />
        </Tabs>
        <DialogContent sx={{ pt: 2 }}>
          {tab === 'how' ? <HowItWorks goTo={goTo} /> : null}
          {tab === 'glossary' ? <GlossaryBrowser /> : null}
          {tab === 'pages' ? <PageIndex goTo={goTo} /> : null}
          {tab === 'tips' ? <TipsPanel /> : null}
          {tab === 'docs' ? <DocsPanel /> : null}
          {tab === 'about' ? <AboutPanel /> : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
