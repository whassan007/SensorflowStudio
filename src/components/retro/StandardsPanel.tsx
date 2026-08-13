/** Retrieved-standards citations with prominent SYNTHETIC_EXAMPLE badges. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { RetrievedStandard } from '../../types/retro';
import { TIER_META } from './tierTheme';

export default function StandardsPanel({ standards }: { standards: RetrievedStandard[] }) {
  if (!standards.length) {
    return (
      <Typography sx={{ fontSize: 13, color: '#8a949e' }}>
        No requirements retrieved — by the hard rule, nothing is cited.
      </Typography>
    );
  }
  const color = TIER_META.TIER3_RETRIEVED.color;
  return (
    <Box>
      <Typography sx={{ fontSize: 11.5, color: '#8a949e', mb: 1 }}>
        Every citation below is backed by an audited retrieval hit (no citation
        without retrieval). Synthetic demonstration rules are labeled; SOTIF
        entries are concept paraphrases, never standard text.
      </Typography>
      {standards.map((s) => (
        <Paper key={s.chunk_id} sx={{ p: 1.5, mb: 1, bgcolor: '#161c23', border: `1px solid ${color}44` }}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mb: 0.5 }}>
            <Chip size="small"
                  label={s.synthetic ? 'SYNTHETIC_EXAMPLE' : 'PARAPHRASE — NOT STANDARD TEXT'}
                  sx={{ bgcolor: s.synthetic ? '#e6510022' : '#4fc3f722',
                        color: s.synthetic ? '#ffab73' : '#4fc3f7', fontWeight: 800, fontSize: 10 }} />
            <Typography sx={{ fontSize: 13, fontWeight: 700 }}>
              {s.document} v{s.version}
            </Typography>
            <Chip size="small" label={`§ ${s.section}`} sx={{ bgcolor: '#232a31', fontSize: 10 }} />
            <Chip size="small" label={`relevance ${s.relevance_score.toFixed(2)}`}
                  sx={{ bgcolor: `${color}22`, color, fontSize: 10, fontWeight: 700 }} />
          </Box>
          <Typography sx={{ fontSize: 12.5, color: '#c3ccd4', fontStyle: 'italic' }}>
            “{s.retrieved_text.length > 380 ? `${s.retrieved_text.slice(0, 380)}…` : s.retrieved_text}”
          </Typography>
          <Typography sx={{ fontSize: 10.5, color: '#5c6770', mt: 0.5, fontFamily: 'monospace' }}>
            {s.doc_id} · {s.doc_type} · {s.source} · {s.jurisdiction} · effective {s.effective_date} · chunk {s.chunk_id}
          </Typography>
        </Paper>
      ))}
    </Box>
  );
}
