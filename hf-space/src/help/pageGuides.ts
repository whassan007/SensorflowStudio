/**
 * Compact per-page guides used by Help menu jump links and the help chatbot.
 * Detailed copy for PageIntro lives in content/pageHelp.ts; this file is the
 * short "what / why / key actions" index.
 */
import type { PageId } from '../context/LabelEvalContext';
import { PAGE_HELP } from '../content/pageHelp';

export interface PageGuide {
  pageId: PageId;
  title: string;
  summary: string;
  keyActions: string[];
}

const PAGE_TITLES: Record<PageId, string> = {
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
  hillclimb: 'Hill Climbing EM',
  vitis: 'Hardware Acceleration',
  ssam: 'SSAM Safety',
  'safety-odd': 'ODD Coverage',
  'safety-gates': 'Release Gates',
  'safety-evidence': 'Evidence Package',
  'safety-ssam': 'SSAM Conflicts',
  'safety-calibration': 'Calibration',
  'safety-discrepancy': 'Discrepancy Mining',
  'safety-scenarios': 'Scenario DB',
  'safety-search': 'Semantic Search',
  seqeval: 'Sequential Regression',
  bevfusion: 'Perception Engines',
  'scenario-composer': 'Scenario Composer',
  'pipeline-builder': 'Pipeline Builder',
  'my-dashboard': 'My Dashboard',
  retro: 'Retrospective Analyzer',
  'closed-loop-lab': 'Closed-Loop Lab',
  'launch-readiness': 'Launch Readiness',
  studio2: 'Studio 2.0 Governance',
  legacy: 'Legacy Studio',
  'production-readiness': 'Production Readiness',
  rotr: 'ROTR Control Center',
};

/** Split an actions blurb into short bullet-like action phrases. */
function actionsFromBlurb(actions: string): string[] {
  return actions
    .split(/[.;]\s+/)
    .map((s) => s.replace(/^and\s+/i, '').trim())
    .filter((s) => s.length > 12)
    .slice(0, 5);
}

export const PAGE_GUIDES: Record<PageId, PageGuide> = Object.fromEntries(
  (Object.keys(PAGE_TITLES) as PageId[]).map((id) => {
    const help = PAGE_HELP[id];
    return [
      id,
      {
        pageId: id,
        title: PAGE_TITLES[id],
        summary: help.subtitle,
        keyActions: actionsFromBlurb(help.actions),
      } satisfies PageGuide,
    ];
  })
) as Record<PageId, PageGuide>;

export const PAGE_GUIDE_LIST: PageGuide[] = (Object.keys(PAGE_TITLES) as PageId[]).map(
  (id) => PAGE_GUIDES[id]
);

export function getPageGuide(id: PageId): PageGuide {
  return PAGE_GUIDES[id];
}
