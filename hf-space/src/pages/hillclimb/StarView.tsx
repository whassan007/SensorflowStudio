/** STAR Story Box: paste a raw story -> diagnosed S/T/A/R with claim-vs-evidence flags. */

import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { PassChip, ScoreChip } from '../../components/hillclimb/shared';
import { diagnoseStar } from '../../services/hillclimb';
import type { StarDiagnosis } from '../../types/hillclimb';

const COMPONENT_COLORS: Record<string, string> = {
  S: '#4fc3f7',
  T: '#9575cd',
  A: '#ffb74d',
  R: '#66bb6a',
};

export default function StarView() {
  const [text, setText] = useState('');
  const [saveEvidence, setSaveEvidence] = useState(true);
  const [diagnosis, setDiagnosis] = useState<StarDiagnosis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    diagnoseStar(text, saveEvidence)
      .then(setDiagnosis)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title="Your story, unstructured"
        help="Paste a real experience in any shape. The diagnoser (not a rewriter) segments it into Situation/Task/Action/Result, runs per-component checks, flags unquantified claims vs measurable evidence, and maps the story to the Phase-3 leadership competencies it actually evidences."
      >
        {error ? <ErrorNote error={error} /> : null}
        <TextField
          multiline
          minRows={6}
          fullWidth
          placeholder="e.g. 'When I joined, the team of 12 was missing every release date. I was asked to… I decided to… As a result, deploy failures dropped from 14% to 3%…'"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
        />
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1.5 }}>
          <FormControlLabel
            control={<Switch size="small" checked={saveEvidence} onChange={(e) => setSaveEvidence(e.target.checked)} />}
            label={
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                Save to evidence library on diagnose
              </Typography>
            }
          />
          <Button variant="contained" onClick={run} disabled={busy || !text.trim()}>
            {busy ? 'Diagnosing…' : 'Diagnose story'}
          </Button>
        </Box>
      </SectionCard>

      {diagnosis ? (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <ScoreChip score={diagnosis.overall_score} label="Story" />
            {diagnosis.checks.map((c) => (
              <PassChip key={c.check} passed={c.passed} label={c.label} />
            ))}
            {diagnosis.evidence_id ? (
              <Chip size="small" label="saved to evidence" sx={{ bgcolor: '#00695c', color: '#80cbc4', fontSize: 11 }} />
            ) : null}
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            {diagnosis.components.map((comp) => (
              <SectionCard
                key={comp.component}
                title={
                  <Box component="span" sx={{ color: COMPONENT_COLORS[comp.component] }}>
                    {comp.component} — {comp.label}
                  </Box>
                }
                sx={{ flex: '1 1 260px', borderTop: `2px solid ${COMPONENT_COLORS[comp.component]}` }}
              >
                {comp.present ? (
                  comp.sentences.map((s, i) => (
                    <Typography key={i} variant="body2" sx={{ color: '#cfd8dc', mb: 0.5 }}>
                      {s}
                    </Typography>
                  ))
                ) : (
                  <Typography variant="body2" sx={{ color: '#ef9a9a' }}>
                    {comp.issues[0] ?? 'Not detected.'}
                  </Typography>
                )}
              </SectionCard>
            ))}
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <SectionCard
              title="Claim vs evidence"
              help="Unquantified claims ('improved performance') are flagged; claims backed by numbers / before-after / named mechanisms count as measurable evidence. Unquantified claims cap the story score at 3."
              sx={{ flex: '2 1 380px' }}
            >
              {diagnosis.claim_flags.length === 0 ? (
                <Typography variant="body2" sx={{ color: '#8a949e' }}>
                  No outcome claims detected — a strong story should claim a measurable result.
                </Typography>
              ) : (
                diagnosis.claim_flags.map((f, i) => (
                  <Alert
                    key={i}
                    severity={f.kind === 'measurable_evidence' ? 'success' : 'warning'}
                    variant="outlined"
                    sx={{ mb: 1, py: 0.25 }}
                  >
                    <Typography variant="caption" sx={{ display: 'block', fontStyle: 'italic' }}>
                      “{f.sentence}”
                    </Typography>
                    <Typography variant="caption">{f.detail}</Typography>
                  </Alert>
                ))
              )}
            </SectionCard>

            <SectionCard
              title="Competencies evidenced"
              help="Phase-3 leadership competencies this story can serve as evidence for, with the matched cues."
              sx={{ flex: '1 1 280px' }}
            >
              {diagnosis.competencies.length === 0 ? (
                <Typography variant="body2" sx={{ color: '#8a949e' }}>
                  None detected yet — anchor the story to a leadership behavior.
                </Typography>
              ) : (
                diagnosis.competencies.map((c) => (
                  <Box key={c.competency_id} sx={{ mb: 1 }}>
                    <Chip
                      size="small"
                      label={c.competency_id}
                      sx={{ bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }}
                    />
                    <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
                      {c.reason}
                    </Typography>
                  </Box>
                ))
              )}
              {diagnosis.coaching.length > 0 ? (
                <>
                  <Typography variant="caption" sx={{ color: '#4fc3f7', fontWeight: 700, display: 'block', mt: 1 }}>
                    STRENGTHEN
                  </Typography>
                  {diagnosis.coaching.map((c, i) => (
                    <Typography key={i} variant="caption" sx={{ color: '#aab4be', display: 'block', mb: 0.5 }}>
                      • {c}
                    </Typography>
                  ))}
                </>
              ) : null}
            </SectionCard>
          </Box>
        </>
      ) : null}
    </Box>
  );
}
