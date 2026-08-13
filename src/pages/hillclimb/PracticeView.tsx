/** Practice workspace: scenario + free-text answer -> structured coaching panel. */

import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { RefreshCw } from 'lucide-react';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { EvaluationPanel, useAsync } from '../../components/hillclimb/shared';
import { generateExercise, getBlueprint, submitExercise } from '../../services/hillclimb';
import type { Exercise, SubmitExerciseResponse } from '../../types/hillclimb';
import type { GoFn } from './HillClimbSection';

export default function PracticeView({
  go,
  seedExercise,
  seedCompetency,
}: {
  go: GoFn;
  seedExercise: Exercise | null;
  seedCompetency: string | null;
}) {
  const blueprint = useAsync(getBlueprint);
  const [competencyId, setCompetencyId] = useState<string>(
    seedExercise?.competency_id ?? seedCompetency ?? 'p1.regression_detection'
  );
  const [difficulty, setDifficulty] = useState(2);
  const [exercise, setExercise] = useState<Exercise | null>(seedExercise);
  const [answer, setAnswer] = useState('');
  const [result, setResult] = useState<SubmitExerciseResponse | null>(null);
  const [asAssessment, setAsAssessment] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchExercise = (cid: string, diff: number) => {
    setBusy(true);
    setError(null);
    setResult(null);
    setAnswer('');
    generateExercise(cid, diff)
      .then(setExercise)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    if (!seedExercise) fetchExercise(competencyId, difficulty);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = () => {
    if (!exercise || !answer.trim()) return;
    setBusy(true);
    setError(null);
    submitExercise(exercise.exercise_id, answer, asAssessment)
      .then(setResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  const continueWithFollowUp = () => {
    if (!result || !exercise) return;
    const probe = result.evaluation.follow_up_question;
    setExercise({ ...exercise, scenario: probe, exercise_id: exercise.exercise_id });
    setAnswer('');
    setResult(null);
  };

  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <Box sx={{ flex: '3 1 480px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <SectionCard
          title="Exercise"
          action={
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <TextField
                select
                size="small"
                label="Competency"
                value={competencyId}
                onChange={(e) => {
                  setCompetencyId(e.target.value);
                  fetchExercise(e.target.value, difficulty);
                }}
                sx={{ minWidth: 240 }}
              >
                {(blueprint.data?.competencies ?? []).map((c) => (
                  <MenuItem key={c.id} value={c.id}>
                    P{c.phase} · {c.name}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                select
                size="small"
                label="Difficulty"
                value={difficulty}
                onChange={(e) => {
                  const d = Number(e.target.value);
                  setDifficulty(d);
                  fetchExercise(competencyId, d);
                }}
                sx={{ width: 110 }}
              >
                <MenuItem value={1}>Warmup</MenuItem>
                <MenuItem value={2}>Core</MenuItem>
                <MenuItem value={3}>Stretch</MenuItem>
              </TextField>
              <Button
                size="small"
                startIcon={<RefreshCw size={14} />}
                onClick={() => fetchExercise(competencyId, difficulty)}
                disabled={busy}
              >
                New variant
              </Button>
            </Box>
          }
          help="Every regeneration produces a structurally different scenario for the same competency (seeded template variation) — repetition can't be memorized. Answers are scored on concept coverage, tradeoffs, and quantified claims; never on length."
        >
          {error ? <ErrorNote error={error} /> : null}
          {exercise ? (
            <>
              <Box sx={{ display: 'flex', gap: 0.75, mb: 1.5, flexWrap: 'wrap' }}>
                <Chip size="small" label={exercise.competency_id} sx={{ bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }} />
                <Chip size="small" label={`family: ${exercise.family}`} sx={{ bgcolor: '#232a31', fontSize: 11 }} />
                <Chip size="small" label={exercise.template_id} sx={{ bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }} />
              </Box>
              <Typography variant="body2" sx={{ color: '#e6e9ec', lineHeight: 1.7, mb: 2 }}>
                {exercise.scenario}
              </Typography>
              {exercise.linked_tool ? (
                <Alert severity="info" variant="outlined" sx={{ mb: 2, py: 0.25 }}>
                  {exercise.linked_tool.label} — open the{' '}
                  <a href={`#${exercise.linked_tool.page}`} style={{ color: '#4fc3f7' }}>
                    RCA workbench
                  </a>{' '}
                  to run this diagnosis against live data.
                </Alert>
              ) : null}
              <TextField
                multiline
                minRows={6}
                fullWidth
                placeholder="Your answer — enumerate causes, mechanisms, numbers, tradeoffs, and a verification plan…"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                disabled={busy}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1.5 }}>
                <FormControlLabel
                  control={
                    <Switch size="small" checked={asAssessment} onChange={(e) => setAsAssessment(e.target.checked)} />
                  }
                  label={
                    <Typography variant="caption" sx={{ color: '#8a949e' }}>
                      Submit as assessment (4+/5 passes; failing routes remediation to the diagnosed prerequisite)
                    </Typography>
                  }
                />
                <Button variant="contained" onClick={submit} disabled={busy || !answer.trim()}>
                  {busy ? 'Evaluating…' : 'Submit for coaching'}
                </Button>
              </Box>
            </>
          ) : null}
        </SectionCard>

        {result ? (
          <SectionCard
            title="Coaching"
            action={
              result.evaluation.follow_up_question ? (
                <Button size="small" variant="outlined" onClick={continueWithFollowUp}>
                  Continue with follow-up
                </Button>
              ) : undefined
            }
          >
            <EvaluationPanel evaluation={result.evaluation} coaching={result.coaching} />
            {result.evaluation.follow_up_question ? (
              <Alert severity="info" variant="outlined" sx={{ mt: 2, py: 0.25 }}>
                <b>Follow-up:</b> {result.evaluation.follow_up_question}
              </Alert>
            ) : null}
            {result.journey && result.journey.state === 'REMEDIATION' ? (
              <Alert severity="warning" variant="outlined" sx={{ mt: 1.5, py: 0.25 }}>
                Assessment failed — remediation routed to prerequisite{' '}
                <b>{result.journey.remediation_target}</b>.{' '}
                <Button size="small" onClick={() => go('practice', null, result.journey!.remediation_target)}>
                  Practice it now
                </Button>
              </Alert>
            ) : null}
          </SectionCard>
        ) : null}
      </Box>

      <Box sx={{ flex: '1 1 260px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {result?.readiness_for_competency ? (
          <SectionCard title="Competency readiness" help="Updated after every attempt. Knowledge, application and evidence are tracked separately.">
            <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
              knowledge {result.readiness_for_competency.knowledge_score.toFixed(1)} · application{' '}
              {result.readiness_for_competency.application_score.toFixed(1)} · evidence{' '}
              {result.readiness_for_competency.evidence_score.toFixed(1)}
            </Typography>
            <Chip
              size="small"
              label={result.readiness_for_competency.readiness_state.replace(/_/g, ' ')}
              sx={{ mt: 1, bgcolor: '#232a31', color: '#aab4be', fontWeight: 700 }}
            />
          </SectionCard>
        ) : null}
        {exercise ? (
          <SectionCard title="Common failure modes" help="How answers to this exercise family typically go wrong.">
            {exercise.common_failure_modes.map((f, i) => (
              <Typography key={i} variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 0.75 }}>
                • {f}
              </Typography>
            ))}
          </SectionCard>
        ) : null}
      </Box>
    </Box>
  );
}
