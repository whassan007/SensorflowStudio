/**
 * The three SEPARATE miner confidences. They are never combined into a single
 * score anywhere in the system; this component renders them side by side to
 * make that separation visible.
 */
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import type { Candidate } from '../../types/raremine';

const GAUGES: { key: keyof Candidate; label: string; tip: string; color: string }[] = [
  {
    key: 'confidence_human_identity',
    label: 'Human identity',
    tip: 'Confidence that a real human is present (face/skin/limbs, gait, humanoid LiDAR shape). Independent of what they wear.',
    color: '#4fc3f7',
  },
  {
    key: 'confidence_costume',
    label: 'Costume present',
    tip: 'Confidence that a silhouette-distorting covering is present. A statue can score high here while human identity stays low.',
    color: '#ab47bc',
  },
  {
    key: 'confidence_rare_event',
    label: 'Rare event',
    tip: 'Confidence that this is a genuine costumed-pedestrian rare event — requires BOTH axes and is discounted by retained alternative hypotheses.',
    color: '#fb8c00',
  },
];

export default function ConfidenceGauges({ candidate }: { candidate: Candidate }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
      {GAUGES.map((g) => {
        const v = candidate[g.key] as number;
        return (
          <Tooltip key={g.key} title={g.tip} placement="top" arrow>
            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ color: '#8a949e', fontSize: 11 }}>
                  {g.label}
                </Typography>
                <Typography variant="caption" sx={{ color: g.color, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(v * 100).toFixed(0)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={v * 100}
                sx={{
                  height: 7,
                  borderRadius: 4,
                  bgcolor: '#1c242d',
                  '& .MuiLinearProgress-bar': { bgcolor: g.color, borderRadius: 4 },
                }}
              />
            </Box>
          </Tooltip>
        );
      })}
    </Box>
  );
}
