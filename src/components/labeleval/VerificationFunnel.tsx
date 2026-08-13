import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import type { FunnelResponse, FunnelStage } from '../../types/labeleval';
import { SectionCard, fmtInt, fmtPct } from './shared';

function StageBar({ stage, max, color }: { stage: FunnelStage; max: number; color: string }) {
  const width = max > 0 ? Math.max(2, (stage.count / max) * 100) : 2;
  return (
    <Box sx={{ mb: 0.75 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.25 }}>
        <Typography variant="caption" sx={{ color: '#aab4be', textTransform: 'uppercase', letterSpacing: 0.4 }}>
          {stage.stage.replace(/_/g, ' ')}
        </Typography>
        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
          {fmtInt(stage.count)} · {fmtPct(stage.pct_of_input)}
        </Typography>
      </Box>
      <Box sx={{ height: 16, bgcolor: '#232a31', borderRadius: 1, overflow: 'hidden' }}>
        <Box
          sx={{
            height: '100%',
            width: `${width}%`,
            background: `linear-gradient(90deg, ${color}, ${color}88)`,
            borderRadius: 1,
            transition: 'width 0.4s ease',
          }}
        />
      </Box>
    </Box>
  );
}

export default function VerificationFunnel({ funnel }: { funnel: FunnelResponse | null }) {
  if (!funnel || (funnel.main_path.length === 0 && funnel.side_path.length === 0)) {
    return (
      <SectionCard title="Verification Funnel">
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          No funnel data yet — run the evaluation pipeline to populate it.
        </Typography>
      </SectionCard>
    );
  }
  const maxMain = Math.max(1, ...funnel.main_path.map((s) => s.count));
  const maxSide = Math.max(1, ...funnel.side_path.map((s) => s.count));
  return (
    <SectionCard
      title="Verification Funnel"
      help="Where labels are in their lifecycle. Main path: frames → auto-labeled → evaluated → auto-graded → verified. Side branch: labels that failed a quality gate detour through human review, re-labeling and re-validation before re-joining as verified. Percentages are relative to the funnel's input stage."
    >
      <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
        <Box sx={{ flex: '2 1 380px' }}>
          <Typography variant="caption" sx={{ color: '#4fc3f7', fontWeight: 700 }}>
            MAIN PATH
          </Typography>
          <Box sx={{ mt: 1 }}>
            {funnel.main_path.map((s) => (
              <StageBar key={s.stage} stage={s} max={maxMain} color="#4fc3f7" />
            ))}
          </Box>
        </Box>
        <Box sx={{ flex: '1 1 300px' }}>
          <Typography variant="caption" sx={{ color: '#ffa726', fontWeight: 700 }}>
            SIDE BRANCH — FLAGGED → HITL → RE-LABEL → RE-VALIDATION → VERIFIED
          </Typography>
          <Box sx={{ mt: 1 }}>
            {funnel.side_path.map((s) => (
              <StageBar key={s.stage} stage={s} max={maxSide} color="#ffa726" />
            ))}
          </Box>
        </Box>
      </Box>
    </SectionCard>
  );
}
