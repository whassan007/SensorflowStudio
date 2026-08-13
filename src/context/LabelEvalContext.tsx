import { createContext, useContext } from 'react';
import type { StreamEvent } from '../types/labeleval';

export type PageId =
  | 'command'
  | 'overview'
  | 'datasets'
  | 'label-generation'
  | 'rare-events'
  | 'quality'
  | 'regression'
  | 'triage'
  | 'review'
  | 'training'
  | 'models'
  | 'evaluation'
  | 'audit'
  | 'pipeline'
  | 'rca'
  | 'raremine'
  | 'hillclimb'
  | 'vitis'
  | 'ssam'
  | 'legacy';

export const ALL_PAGE_IDS: PageId[] = [
  'command',
  'overview',
  'datasets',
  'label-generation',
  'rare-events',
  'quality',
  'regression',
  'triage',
  'review',
  'training',
  'models',
  'evaluation',
  'audit',
  'pipeline',
  'rca',
  'raremine',
  'hillclimb',
  'vitis',
  'ssam',
  'legacy',
];

/** Maps backend evidence_link.page strings (unknown vocabulary) onto our page ids. */
export function resolvePageId(page: string): PageId {
  const normalized = page.toLowerCase().replace(/_/g, '-');
  if ((ALL_PAGE_IDS as string[]).includes(normalized)) return normalized as PageId;
  const aliases: Record<string, PageId> = {
    'rare-event': 'rare-events',
    rareevents: 'rare-events',
    'human-review': 'review',
    hitl: 'review',
    'review-tasks': 'review',
    'quality-engine': 'quality',
    'label-eval': 'evaluation',
    evaluations: 'evaluation',
    model: 'models',
    dataset: 'datasets',
    train: 'training',
    'pipeline-architecture': 'pipeline',
  };
  return aliases[normalized] ?? 'overview';
}

export interface LabelEvalContextValue {
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;
  /** Latest event from the SSE stream (or the polling fallback). */
  stream: StreamEvent | null;
  page: PageId;
  /** Optional entity id carried by navigation (e.g. review task id). */
  entityId: string | null;
  navigate: (page: PageId, entityId?: string | null) => void;
}

export const LabelEvalContext = createContext<LabelEvalContextValue | null>(null);

export function useLabelEval(): LabelEvalContextValue {
  const ctx = useContext(LabelEvalContext);
  if (!ctx) throw new Error('useLabelEval must be used within the LabelEval provider');
  return ctx;
}
