/** Evidence library: every artifact backing the readiness matrix. */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Typography from '@mui/material/Typography';
import { EmptyState, ErrorNote, LoadingBox, SectionCard } from '../../components/labeleval/shared';
import { ScoreChip, useAsync } from '../../components/hillclimb/shared';
import { getEvidence } from '../../services/hillclimb';

const TYPE_LABELS: Record<string, string> = {
  exercise_attempt: 'Exercise',
  star_story: 'STAR story',
  design_submission: 'Design lab',
  simulation_debrief: 'Simulation',
  interview_transcript: 'Interview',
  diagnostic: 'Diagnostic',
};

export default function EvidenceView({ competencyFilter }: { competencyFilter: string | null }) {
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const evidence = useAsync(() => getEvidence(competencyFilter ?? undefined), [competencyFilter]);

  if (evidence.loading && !evidence.data) return <LoadingBox label="Loading evidence…" />;
  if (evidence.error && !evidence.data) return <ErrorNote error={evidence.error} />;

  const items = (evidence.data?.evidence ?? []).filter(
    (e) => typeFilter === 'all' || e.artifact_type === typeFilter
  );

  return (
    <SectionCard
      title={`Evidence library (${items.length})`}
      action={
        <ToggleButtonGroup size="small" exclusive value={typeFilter} onChange={(_e, v: string | null) => v && setTypeFilter(v)}>
          <ToggleButton value="all">All</ToggleButton>
          {Object.entries(TYPE_LABELS).map(([k, v]) => (
            <ToggleButton key={k} value={k}>
              {v}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      }
      help="Every score in the platform traces back to artifacts here: exercise attempts (with quoted answer evidence), diagnosed STAR stories, graded designs, simulation debriefs, and interview transcripts."
    >
      {items.length === 0 ? (
        <EmptyState
          title="No evidence yet"
          message="Evidence accumulates as you practice: run the diagnostic, submit exercises, diagnose STAR stories, grade designs, finish simulations and interviews."
        />
      ) : (
        items.map((e) => (
          <Box key={e.evidence_id} sx={{ py: 1.25, borderBottom: '1px solid #232a31' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              <Chip size="small" label={TYPE_LABELS[e.artifact_type] ?? e.artifact_type} sx={{ bgcolor: '#232a31', color: '#aab4be', fontSize: 10.5, height: 20 }} />
              {e.score > 0 ? <ScoreChip score={Math.round(e.score)} label="" /> : null}
              <Typography variant="caption" sx={{ color: '#5c6873' }}>
                {new Date(e.timestamp).toLocaleString()} · {e.source}
              </Typography>
            </Box>
            <Typography variant="body2" sx={{ color: '#e6e9ec', mt: 0.5 }}>
              {e.summary}
            </Typography>
            {e.quotes.slice(0, 2).map((q, i) => (
              <Typography key={i} variant="caption" sx={{ display: 'block', fontStyle: 'italic', color: '#8a949e' }}>
                “{q}”
              </Typography>
            ))}
            <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
              {e.competency_ids.map((c) => (
                <Chip key={c} size="small" label={c} sx={{ bgcolor: '#1a2027', color: '#5c6873', fontFamily: 'monospace', fontSize: 10, height: 18 }} />
              ))}
            </Box>
          </Box>
        ))
      )}
    </SectionCard>
  );
}
