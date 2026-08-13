/** Shared UI pieces for the Hill Climbing EM section. */

import { useCallback, useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { CheckCircle2, XCircle } from 'lucide-react';
import type { EvaluationResult, ReadinessStateId } from '../../types/hillclimb';

// ------------------------------------------------------------- async helper

export interface Async<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    fn()
      .then((d) => {
        if (alive) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: Error) => {
        if (alive) setError(e.message);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps]);
  return { data, loading, error, reload };
}

// ------------------------------------------------------------- score display

export const SCORE_COLORS = ['#ef5350', '#ff7043', '#f9a825', '#9ccc65', '#66bb6a'];

export function scoreColor(score: number): string {
  return SCORE_COLORS[Math.max(0, Math.min(4, Math.round(score) - 1))];
}

export function ScoreChip({ score, label }: { score: number; label?: string }) {
  return (
    <Chip
      size="small"
      label={`${label ?? 'Score'} ${score}/5`}
      sx={{ bgcolor: scoreColor(score), color: '#101418', fontWeight: 800, fontSize: 12 }}
    />
  );
}

// -------------------------------------------------------------- state chips

export const READINESS_COLORS: Record<ReadinessStateId, { bg: string; fg: string }> = {
  NOT_STARTED: { bg: '#37474f', fg: '#cfd8dc' },
  LEARNING: { bg: '#0d47a1', fg: '#90caf9' },
  PRACTICING: { bg: '#4527a0', fg: '#d1c4e9' },
  NEEDS_REVIEW: { bg: '#e65100', fg: '#ffe0b2' },
  COMPETENT: { bg: '#2e7d32', fg: '#c8e6c9' },
  STRONG: { bg: '#1b5e20', fg: '#a5d6a7' },
  INTERVIEW_READY: { bg: '#00695c', fg: '#80cbc4' },
};

export function ReadinessChip({ state }: { state: ReadinessStateId }) {
  const c = READINESS_COLORS[state] ?? READINESS_COLORS.NOT_STARTED;
  return (
    <Chip
      size="small"
      label={state.replace(/_/g, ' ')}
      sx={{ bgcolor: c.bg, color: c.fg, fontWeight: 700, fontSize: 10.5, height: 20 }}
    />
  );
}

export function PassChip({ passed, label }: { passed: boolean; label: string }) {
  return (
    <Chip
      size="small"
      icon={passed ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
      label={label}
      sx={{
        bgcolor: passed ? '#1b5e20' : '#5d1414',
        color: passed ? '#a5d6a7' : '#ffcdd2',
        fontWeight: 600,
        fontSize: 11,
        '& .MuiChip-icon': { color: 'inherit' },
      }}
    />
  );
}

// ------------------------------------------------------------- bullet lists

export function BulletList({ items, color = '#aab4be' }: { items: string[]; color?: string }) {
  if (!items.length) return null;
  return (
    <Box component="ul" sx={{ m: 0, pl: 2.25 }}>
      {items.map((s, i) => (
        <Typography key={i} component="li" variant="body2" sx={{ color, mb: 0.25 }}>
          {s}
        </Typography>
      ))}
    </Box>
  );
}

function LabeledBlock({ label, color, children }: { label: string; color: string; children: React.ReactNode }) {
  return (
    <Box sx={{ mb: 1.25 }}>
      <Typography
        variant="caption"
        sx={{ color, textTransform: 'uppercase', letterSpacing: 0.6, fontWeight: 700 }}
      >
        {label}
      </Typography>
      {children}
    </Box>
  );
}

// -------------------------------------------------------- evaluation panel

/** Structured coaching panel: score with evidence quotes, strengths /
 * weaknesses / missing evidence / misconceptions, and the follow-up probe. */
export function EvaluationPanel({
  evaluation,
  coaching,
}: {
  evaluation: EvaluationResult;
  coaching?: string;
}) {
  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.25, flexWrap: 'wrap' }}>
        <ScoreChip score={evaluation.score} />
        <Chip
          size="small"
          label={`confidence ${(evaluation.confidence * 100).toFixed(0)}%`}
          sx={{ bgcolor: '#232a31', color: '#aab4be', fontSize: 11 }}
        />
        <Chip
          size="small"
          label={evaluation.evaluator === 'llm' ? 'LLM + rubric floor' : 'deterministic rubric'}
          sx={{ bgcolor: '#232a31', color: '#8a949e', fontSize: 11 }}
        />
      </Box>

      {evaluation.evidence.length > 0 ? (
        <LabeledBlock label="Evidence — your own words" color="#4fc3f7">
          {evaluation.evidence.map((q, i) => (
            <Typography
              key={i}
              variant="body2"
              sx={{
                borderLeft: '3px solid #4fc3f7',
                pl: 1.25,
                py: 0.4,
                my: 0.5,
                color: '#cfd8dc',
                fontStyle: 'italic',
                bgcolor: '#12171d',
                borderRadius: '0 4px 4px 0',
              }}
            >
              “{q}”
            </Typography>
          ))}
        </LabeledBlock>
      ) : (
        <Typography variant="body2" sx={{ color: '#ef9a9a', mb: 1 }}>
          No quotable evidence found in your answer — scores above 1 require evidence, so this
          answer scores 1/5.
        </Typography>
      )}

      {evaluation.strengths.length > 0 ? (
        <LabeledBlock label="Strengths" color="#9ccc65">
          <BulletList items={evaluation.strengths} color="#c5e1a5" />
        </LabeledBlock>
      ) : null}
      {evaluation.weaknesses.length > 0 ? (
        <LabeledBlock label="Weaknesses" color="#ffb74d">
          <BulletList items={evaluation.weaknesses} color="#ffcc80" />
        </LabeledBlock>
      ) : null}
      {evaluation.missing_evidence.length > 0 ? (
        <LabeledBlock label="Missing evidence" color="#ef9a9a">
          <BulletList items={evaluation.missing_evidence} color="#ef9a9a" />
        </LabeledBlock>
      ) : null}
      {evaluation.misconceptions.length > 0 ? (
        <LabeledBlock label="Likely misconceptions" color="#ce93d8">
          <BulletList items={evaluation.misconceptions} color="#ce93d8" />
        </LabeledBlock>
      ) : null}

      {coaching ? (
        <LabeledBlock label="Coaching" color="#4fc3f7">
          <Typography variant="body2" sx={{ color: '#cfd8dc', whiteSpace: 'pre-line' }}>
            {coaching}
          </Typography>
        </LabeledBlock>
      ) : null}

      {evaluation.recommended_action ? (
        <Typography variant="body2" sx={{ color: '#8a949e', mt: 1 }}>
          <b>Next:</b> {evaluation.recommended_action}
        </Typography>
      ) : null}
    </Box>
  );
}
