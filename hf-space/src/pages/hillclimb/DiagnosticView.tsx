/** Adaptive diagnostic: a short question set that seeds the competency matrix. */

import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import LinearProgress from '@mui/material/LinearProgress';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { ScoreChip } from '../../components/hillclimb/shared';
import { answerDiagnostic, startDiagnostic } from '../../services/hillclimb';
import type { DiagnosticSession } from '../../types/hillclimb';
import type { GoFn } from './HillClimbSection';

export default function DiagnosticView({ go }: { go: GoFn }) {
  const [session, setSession] = useState<DiagnosticSession | null>(null);
  const [answer, setAnswer] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = () => {
    setBusy(true);
    setError(null);
    startDiagnostic(Math.floor(Math.random() * 100000))
      .then(setSession)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  const submit = () => {
    if (!session || !answer.trim()) return;
    setBusy(true);
    setError(null);
    answerDiagnostic(session.diagnostic_id, answer)
      .then((s) => {
        setSession(s);
        setAnswer('');
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 900 }}>
      {error ? <ErrorNote error={error} /> : null}

      {!session ? (
        <SectionCard
          title="Adaptive diagnostic"
          help="Six short scenario questions. The sequence adapts: a weak answer drills into prerequisites, a strong answer jumps ahead. Results seed your competency matrix and set the journey to LEARNING."
        >
          <Typography variant="body2" sx={{ color: '#aab4be', mb: 2 }}>
            Answer six scenario questions in your own words. There is no multiple choice and no
            memorizable answer key — each question is generated from a parameterized template, and
            your answers are scored against a concept rubric with quoted evidence.
          </Typography>
          <Button variant="contained" onClick={start} disabled={busy}>
            Start diagnostic
          </Button>
        </SectionCard>
      ) : null}

      {session && session.status === 'active' && session.current_question ? (
        <SectionCard
          title={`Question ${session.answered + 1} of ${session.total_questions}`}
          action={
            <Chip
              size="small"
              label={session.current_question.competency_id}
              sx={{ bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }}
            />
          }
        >
          <LinearProgress
            variant="determinate"
            value={(session.answered / session.total_questions) * 100}
            sx={{ mb: 2, height: 5, borderRadius: 2 }}
          />
          <Typography variant="body2" sx={{ color: '#e6e9ec', mb: 2, lineHeight: 1.65 }}>
            {session.current_question.scenario}
          </Typography>
          <TextField
            multiline
            minRows={5}
            fullWidth
            placeholder="Answer in your own words — mechanisms, numbers, tradeoffs…"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            disabled={busy}
          />
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1.5, gap: 1 }}>
            <Button variant="contained" onClick={submit} disabled={busy || !answer.trim()}>
              {busy ? 'Scoring…' : 'Submit answer'}
            </Button>
          </Box>
          {session.results.length > 0 ? (
            <Box sx={{ display: 'flex', gap: 0.75, mt: 2, flexWrap: 'wrap' }}>
              {session.results.map((r, i) => (
                <Chip
                  key={i}
                  size="small"
                  label={`${r.competency_id}: ${r.score}/5`}
                  sx={{ bgcolor: '#1a2027', color: '#8a949e', fontSize: 10.5, height: 20 }}
                />
              ))}
            </Box>
          ) : null}
        </SectionCard>
      ) : null}

      {session && session.status === 'complete' ? (
        <SectionCard title="Diagnostic complete — matrix seeded">
          <Alert severity="success" variant="outlined" sx={{ mb: 2 }}>
            Your competency matrix has been seeded from {session.results.length} scored answers and
            your journey advanced to LEARNING.
          </Alert>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 2 }}>
            {session.results.map((r, i) => (
              <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ScoreChip score={r.score} label="" />
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                  {r.competency_id}
                </Typography>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  {r.evaluation.weaknesses[0] ?? r.evaluation.strengths[0] ?? ''}
                </Typography>
              </Box>
            ))}
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button variant="contained" onClick={() => go('dashboard')}>
              See dashboard & next best action
            </Button>
            <Button onClick={start}>Retake (new questions)</Button>
          </Box>
        </SectionCard>
      ) : null}
    </Box>
  );
}
