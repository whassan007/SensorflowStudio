/** Leadership-facing retrospective scorecard with per-field provenance tags
 * and JSON export. */
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { Scorecard, ScorecardField } from '../../types/agentic';
import { OutcomeChip, PanelTitle } from './common';

const TAG_COLORS: Record<ScorecardField['tag'], { bg: string; fg: string }> = {
  OBSERVED: { bg: '#1b5e20', fg: '#a5d6a7' },
  PREDICTED: { bg: '#0d47a1', fg: '#90caf9' },
  HYPOTHETICAL: { bg: '#4a148c', fg: '#ce93d8' },
  REQUIRED_EVIDENCE: { bg: '#4e342e', fg: '#ffcc80' },
};

const FIELDS: { key: keyof Scorecard; label: string }[] = [
  { key: 'failure_summary', label: 'Failure' },
  { key: 'frequency', label: 'Frequency' },
  { key: 'exposure', label: 'Exposure' },
  { key: 'severity', label: 'Severity' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'novelty', label: 'Novelty' },
  { key: 'concentration', label: 'Concentration' },
  { key: 'downstream_impact', label: 'Downstream impact' },
  { key: 'mitigations', label: 'Mitigations' },
  { key: 'residual_risk', label: 'Residual risk' },
];

function renderValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export default function ScorecardView({ card }: { card: Scorecard }) {
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(card, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${card.scorecard_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
      <PanelTitle
        title={`Agentic Safety Scorecard — ${card.title}`}
        origin="deterministic"
        extra={
          <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center' }}>
            <Chip size="small" label={`evidence: ${card.evidence_quality}`} sx={{ height: 20, fontSize: 10.5, fontWeight: 700, bgcolor: '#232a31' }} />
            <OutcomeChip outcome={card.policy_outcome} />
            <Button size="small" variant="outlined" onClick={exportJson}>
              Export JSON
            </Button>
          </Box>
        }
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1 }}>
        {FIELDS.map(({ key, label }) => {
          const f = card[key] as ScorecardField;
          const tag = TAG_COLORS[f.tag];
          return (
            <Paper key={key} variant="outlined" sx={{ p: 1, bgcolor: '#12171d' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 800, flex: 1 }}>
                  {label}
                </Typography>
                <Chip size="small" label={f.tag.replace('_', ' ')} sx={{ height: 16, fontSize: 9, fontWeight: 800, bgcolor: tag.bg, color: tag.fg }} />
              </Box>
              <Typography variant="caption" component="div" sx={{ color: '#c3ccd4', wordBreak: 'break-word' }}>
                {renderValue(f.value)}
              </Typography>
              <Typography variant="caption" component="div" sx={{ color: '#5c6770', fontSize: 10, mt: 0.5 }}>
                evidence: {f.evidence_ref || '—'}
              </Typography>
            </Paper>
          );
        })}
      </Box>
      <Typography variant="caption" component="div" sx={{ color: '#5c6770', mt: 1 }}>
        {card.notes.join(' · ')} · recommended option: {card.recommended_option ?? '—'} · policy{' '}
        {card.policy_version} · generated {card.generated_at}
      </Typography>
    </Paper>
  );
}
