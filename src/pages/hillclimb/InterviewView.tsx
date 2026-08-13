/** Interview room: chat-style adaptive session with per-answer evaluation chips. */

import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import TextField from '@mui/material/TextField';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Typography from '@mui/material/Typography';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { ScoreChip, scoreColor } from '../../components/hillclimb/shared';
import { answerInterview, endInterview, startInterview } from '../../services/hillclimb';
import type { InterviewSession, InterviewTurn } from '../../types/hillclimb';

const TYPE_LABEL: Record<InterviewTurn['question_type'], { label: string; color: string }> = {
  opening: { label: 'opening', color: '#4fc3f7' },
  probe: { label: 'weakness probe', color: '#ef5350' },
  depth_probe: { label: 'depth probe', color: '#ffb74d' },
  escalate: { label: 'escalation', color: '#66bb6a' },
  advance: { label: 'new competency', color: '#9575cd' },
};

export default function InterviewView() {
  const [mode, setMode] = useState('hybrid');
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [answer, setAnswer] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = () => {
    setBusy(true);
    setError(null);
    startInterview(mode, Math.floor(Math.random() * 100000))
      .then(setSession)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  const submit = () => {
    if (!session || !answer.trim()) return;
    setBusy(true);
    setError(null);
    answerInterview(session.session_id, answer)
      .then((s) => {
        setSession(s);
        setAnswer('');
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  const finish = () => {
    if (!session) return;
    setBusy(true);
    endInterview(session.session_id)
      .then(setSession)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 980 }}>
      {error ? <ErrorNote error={error} /> : null}

      {!session ? (
        <SectionCard
          title="AI interviewer"
          help="No fixed question list: the next question is chosen from your last evaluation — weakness → targeted probe, strong answer → escalation or dimension switch, polished-but-shallow → depth probe. The transcript is saved as evidence."
        >
          <Typography variant="body2" sx={{ color: '#aab4be', mb: 2 }}>
            Pick a mode and answer in your own words. The interviewer adapts after every answer.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_e, v: string | null) => v && setMode(v)}>
              <ToggleButton value="technical">Technical</ToggleButton>
              <ToggleButton value="management">Management</ToggleButton>
              <ToggleButton value="hybrid">Hybrid EM</ToggleButton>
            </ToggleButtonGroup>
            <Button variant="contained" onClick={start} disabled={busy}>
              Start interview
            </Button>
          </Box>
        </SectionCard>
      ) : null}

      {session ? (
        <SectionCard
          title={`Interview (${session.mode})`}
          action={
            session.status === 'active' ? (
              <Button size="small" color="warning" onClick={finish} disabled={busy}>
                End & save transcript
              </Button>
            ) : (
              <Chip size="small" label="transcript saved as evidence" sx={{ bgcolor: '#00695c', color: '#80cbc4' }} />
            )
          }
        >
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            {session.turns.map((t, i) => (
              <Box key={i}>
                <Box
                  sx={{
                    bgcolor: '#12171d',
                    border: '1px solid #232a31',
                    borderLeft: `3px solid ${TYPE_LABEL[t.question_type].color}`,
                    borderRadius: 1,
                    p: 1.25,
                    maxWidth: '85%',
                  }}
                >
                  <Typography variant="caption" sx={{ color: TYPE_LABEL[t.question_type].color, fontWeight: 700 }}>
                    INTERVIEWER · {TYPE_LABEL[t.question_type].label} · {t.competency_id} · difficulty {t.difficulty}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#e6e9ec', lineHeight: 1.6 }}>
                    {t.question}
                  </Typography>
                </Box>
                {t.answer ? (
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
                    <Box sx={{ bgcolor: '#1a2733', borderRadius: 1, p: 1.25, maxWidth: '85%' }}>
                      <Typography variant="body2" sx={{ color: '#cfd8dc' }}>
                        {t.answer}
                      </Typography>
                      {t.evaluation ? (
                        <Box sx={{ display: 'flex', gap: 0.75, mt: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                          <ScoreChip score={t.evaluation.score} />
                          {t.evaluation.weaknesses.slice(0, 2).map((w, j) => (
                            <Chip key={j} size="small" label={`gap: ${w.slice(0, 44)}`} sx={{ bgcolor: '#232a31', color: '#ffcc80', fontSize: 10, height: 18 }} />
                          ))}
                          {t.evaluation.evidence.length > 0 ? (
                            <Chip size="small" label={`${t.evaluation.evidence.length} evidence quote(s)`} sx={{ bgcolor: '#232a31', color: scoreColor(t.evaluation.score), fontSize: 10, height: 18 }} />
                          ) : null}
                        </Box>
                      ) : null}
                    </Box>
                  </Box>
                ) : null}
              </Box>
            ))}
          </Box>

          {session.status === 'active' ? (
            <Box sx={{ mt: 2 }}>
              <TextField
                multiline
                minRows={3}
                fullWidth
                placeholder="Your answer…"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                disabled={busy}
              />
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
                <Button variant="contained" onClick={submit} disabled={busy || !answer.trim()}>
                  {busy ? 'Evaluating…' : 'Answer'}
                </Button>
              </Box>
            </Box>
          ) : (
            <Alert severity="success" variant="outlined" sx={{ mt: 2 }}>
              Interview complete. Answers were folded into your readiness matrix and the transcript
              stored in the evidence library.
              <Button size="small" sx={{ ml: 1 }} onClick={start}>
                New interview
              </Button>
            </Alert>
          )}
        </SectionCard>
      ) : null}
    </Box>
  );
}
