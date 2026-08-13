/** Dashboard: readiness bars, bottleneck callout, ONE next-best-action card, journey state. */

import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { ArrowRight, Compass, Mountain } from 'lucide-react';
import { ErrorNote, HBar, LoadingBox, SectionCard } from '../../components/labeleval/shared';
import { useAsync } from '../../components/hillclimb/shared';
import { getJourney, getNextAction, getReadiness } from '../../services/hillclimb';
import type { Exercise, JourneyStateId } from '../../types/hillclimb';
import type { GoFn } from './HillClimbSection';

const JOURNEY_ORDER: JourneyStateId[] = [
  'NOT_STARTED', 'DIAGNOSTIC', 'LEARNING', 'PRACTICE', 'ASSESSMENT', 'REMEDIATION', 'REASSESS',
];

const DIM_COLORS: Record<string, string> = {
  'Technical Depth': '#4fc3f7',
  'System Design': '#9575cd',
  Execution: '#ffb74d',
  Leadership: '#66bb6a',
  Communication: '#f06292',
  'Safety/Risk': '#ef5350',
};

export default function DashboardView({ go }: { go: GoFn }) {
  const readiness = useAsync(getReadiness);
  const nba = useAsync(getNextAction);
  const journey = useAsync(getJourney);

  if (readiness.loading && !readiness.data) return <LoadingBox label="Computing readiness…" />;
  if (readiness.error && !readiness.data) return <ErrorNote error={readiness.error} />;

  const bn = readiness.data?.bottleneck ?? null;
  const dims = readiness.data?.dimensions ?? [];
  const j = journey.data;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {j ? (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: '#8a949e', mr: 0.5 }}>
            JOURNEY
          </Typography>
          {JOURNEY_ORDER.map((s) => (
            <Chip
              key={s}
              size="small"
              label={s.replace(/_/g, ' ')}
              sx={{
                height: 20,
                fontSize: 10.5,
                fontWeight: j.state === s ? 800 : 500,
                bgcolor: j.state === s ? '#0d47a1' : '#1a2027',
                color: j.state === s ? '#90caf9' : '#5c6873',
              }}
            />
          ))}
          {j.remediation_target ? (
            <Chip
              size="small"
              label={`remediating: ${j.remediation_target}`}
              sx={{ height: 20, fontSize: 10.5, bgcolor: '#e65100', color: '#ffe0b2' }}
            />
          ) : null}
        </Box>
      ) : null}

      {bn ? (
        <Alert
          severity="warning"
          variant="outlined"
          icon={<Mountain size={20} />}
          sx={{ '& .MuiAlert-message': { width: '100%' } }}
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
            Current bottleneck: {bn.name}{' '}
            <Chip
              size="small"
              label={`phase ${bn.phase} · ${bn.dimension}`}
              sx={{ ml: 0.5, height: 18, fontSize: 10, bgcolor: '#232a31', color: '#aab4be' }}
            />
          </Typography>
          <Typography variant="body2" sx={{ color: '#cfd8dc', mt: 0.5 }}>
            {bn.explanation}
          </Typography>
        </Alert>
      ) : readiness.data ? (
        <Alert severity="success" variant="outlined">
          No bottleneck: every competency is at COMPETENT or better. Keep polishing toward
          INTERVIEW_READY.
        </Alert>
      ) : null}

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <SectionCard
          title="Readiness by dimension"
          help="Aggregated readiness across the competency matrix, grouped into the six dimensions an EM interview loop actually probes. Bars show average score (1-5) over competencies with any signal; the fraction shows how many are COMPETENT or better."
          sx={{ flex: '2 1 420px' }}
        >
          {dims.map((d) => (
            <HBar
              key={d.dimension}
              label={`${d.dimension} (${d.competent}/${d.total} competent)`}
              value={d.avg_score}
              max={5}
              color={DIM_COLORS[d.dimension] ?? '#4fc3f7'}
              valueLabel={d.avg_score > 0 ? `${d.avg_score.toFixed(1)}/5` : 'no signal'}
            />
          ))}
          <Typography variant="caption" sx={{ color: '#5c6873' }}>
            Knowledge, application and evidence are tracked separately per competency — open the
            Competency Matrix for the full explainable breakdown.
          </Typography>
        </SectionCard>

        <SectionCard
          title={
            <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75 }}>
              <Compass size={16} /> Next best action
            </Box>
          }
          help="Exactly one concept, one exercise, one assessment — chosen by bottleneck analysis (the prerequisite blocking the most downstream competencies, not merely your lowest score)."
          sx={{ flex: '1 1 340px' }}
        >
          {nba.loading && !nba.data ? <LoadingBox label="Choosing your next move…" /> : null}
          {nba.error && !nba.data ? <ErrorNote error={nba.error} /> : null}
          {nba.data ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
              <Box>
                <Typography variant="caption" sx={{ color: '#4fc3f7', fontWeight: 700 }}>
                  1 · CONCEPT
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  {nba.data.concept.name}
                </Typography>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  Study: {nba.data.concept.study.join(', ') || nba.data.concept.description}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: '#9ccc65', fontWeight: 700 }}>
                  2 · EXERCISE
                </Typography>
                <Typography variant="body2" sx={{ color: '#cfd8dc' }}>
                  {nba.data.exercise.scenario.slice(0, 160)}…
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: '#ffb74d', fontWeight: 700 }}>
                  3 · ASSESSMENT
                </Typography>
                <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
                  {nba.data.assessment.description}
                </Typography>
              </Box>
              <Button
                variant="contained"
                size="small"
                endIcon={<ArrowRight size={15} />}
                onClick={() => go('practice', nba.data!.exercise as Exercise)}
              >
                Start this exercise
              </Button>
            </Box>
          ) : null}
        </SectionCard>
      </Box>

      {readiness.data && dims.every((d) => d.avg_score === 0) ? (
        <Alert
          severity="info"
          variant="outlined"
          action={
            <Button size="small" onClick={() => go('diagnostic')}>
              Run diagnostic
            </Button>
          }
        >
          No signal yet — run the adaptive diagnostic to seed your competency matrix.
        </Alert>
      ) : null}
    </Box>
  );
}
