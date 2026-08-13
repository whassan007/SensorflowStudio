/**
 * Frame strip for a multi-frame track candidate: one tile per frame with the
 * representative frames marked (best evidence / worst case / model failure).
 */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { Award, CloudRainWind, ShieldAlert } from 'lucide-react';
import type { TrackView } from '../../types/raremine';
import { DIFFICULTY_COLORS, PriorityChip } from './shared';

export default function TrackStrip({ track }: { track: TrackView }) {
  const rf = track.representative_frames;
  const c = track.candidate;
  return (
    <Box sx={{ p: 1.5, bgcolor: '#141a20', border: '1px solid #232a31', borderRadius: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
        <PriorityChip value={c.curation_priority} />
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          {c.costume_type.join(' / ') || 'no confirmed edge case'} · {track.frame_count} frames
        </Typography>
        <Typography variant="caption" sx={{ color: '#5c6a76', fontFamily: 'monospace' }}>
          {track.sequence_id}
        </Typography>
        <Chip size="small" label={`max difficulty ${track.max_difficulty}`} sx={{ bgcolor: '#232a31', fontSize: 10.5, height: 20 }} />
        <Chip size="small" label={`max visibility ${track.max_visibility}`} sx={{ bgcolor: '#232a31', fontSize: 10.5, height: 20 }} />
        <Chip
          size="small"
          label={`minimal set: ${rf.minimal_set.length} frame(s)`}
          sx={{ bgcolor: '#0d47a1', color: '#90caf9', fontSize: 10.5, height: 20 }}
        />
      </Box>
      <Box sx={{ display: 'flex', gap: 0.5, overflowX: 'auto', pb: 0.5 }}>
        {track.frames.map((f) => {
          const markers = [];
          if (f.scene_id === rf.best_evidence) markers.push('best evidence');
          if (f.scene_id === rf.worst_case) markers.push('worst case');
          if (f.scene_id === rf.model_failure) markers.push('model failure');
          const marked = markers.length > 0;
          return (
            <Tooltip
              key={f.candidate_id}
              arrow
              title={`frame ${f.frame_index} · rare ${(f.confidence_rare_event * 100).toFixed(0)}% · ${f.perception_difficulty}${
                f.failure_observed ? ' · baseline failure observed' : ''
              }${marked ? ` · ${markers.join(', ')}` : ''}`}
            >
              <Box
                sx={{
                  minWidth: 34,
                  height: 46,
                  borderRadius: 0.75,
                  border: marked ? '2px solid #4fc3f7' : '1px solid #232a31',
                  bgcolor: '#0d1117',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  p: 0.4,
                  flexShrink: 0,
                }}
              >
                <Box
                  sx={{
                    width: '100%',
                    height: 7,
                    borderRadius: 0.5,
                    bgcolor: DIFFICULTY_COLORS[f.perception_difficulty] ?? '#37474f',
                    opacity: f.edge_case_detected ? 1 : 0.25,
                  }}
                />
                <Box sx={{ display: 'flex', gap: 0.2 }}>
                  {f.scene_id === rf.best_evidence ? <Award size={11} color="#66bb6a" /> : null}
                  {f.scene_id === rf.worst_case ? <CloudRainWind size={11} color="#fb8c00" /> : null}
                  {f.scene_id === rf.model_failure ? <ShieldAlert size={11} color="#e53935" /> : null}
                </Box>
                <Typography variant="caption" sx={{ fontSize: 8.5, color: '#5c6a76', lineHeight: 1 }}>
                  {f.frame_index}
                </Typography>
              </Box>
            </Tooltip>
          );
        })}
      </Box>
      <Typography variant="caption" sx={{ color: '#5c6a76' }}>
        Legend: <Award size={10} color="#66bb6a" /> best evidence · <CloudRainWind size={10} color="#fb8c00" /> worst
        case · <ShieldAlert size={10} color="#e53935" /> observed model failure. Bar color = per-frame difficulty. The
        recommended curation set is the marked frames only — never all {track.frame_count} near-identical frames.
      </Typography>
    </Box>
  );
}
