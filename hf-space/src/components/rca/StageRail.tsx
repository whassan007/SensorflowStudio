/** Guided left rail: the 13 ordered stages with status and enforced locking.
 * A stage is viewable only when every earlier stage is complete (or complete
 * with acknowledged unknowns) — plausible early explanations don't unlock
 * later stages; finishing the methodology does. */
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { CheckCircle2, CircleDot, Lock, TriangleAlert } from 'lucide-react';
import type { StageState } from '../../types/rca';
import { ACCENT, BG_PANEL, BORDER } from './common';

export function firstIncompleteIndex(stages: StageState[]): number {
  for (const s of stages) {
    if (s.status !== 'complete' && s.status !== 'complete_with_unknowns') return s.index;
  }
  return stages.length;
}

export default function StageRail({ stages, selected, onSelect }: {
  stages: StageState[];
  selected: number;
  onSelect: (index: number) => void;
}) {
  const frontier = firstIncompleteIndex(stages);
  const done = stages.filter(
    (s) => s.status === 'complete' || s.status === 'complete_with_unknowns'
  ).length;

  return (
    <Box
      sx={{
        width: 250,
        flexShrink: 0,
        borderRight: `1px solid ${BORDER}`,
        bgcolor: BG_PANEL,
        overflowY: 'auto',
      }}
    >
      <Box sx={{ px: 1.5, pt: 1.25, pb: 0.75 }}>
        <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700, letterSpacing: 0.8 }}>
          METHODOLOGY · {done}/{stages.length}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={(done / stages.length) * 100}
          sx={{ mt: 0.5, height: 5, borderRadius: 2, bgcolor: '#0d1116' }}
        />
      </Box>
      <List dense sx={{ py: 0 }}>
        {stages.map((s) => {
          const locked = s.index > frontier;
          const isSel = s.index === selected;
          const icon =
            s.status === 'complete' ? (
              <CheckCircle2 size={15} color="#66bb6a" />
            ) : s.status === 'complete_with_unknowns' ? (
              <TriangleAlert size={15} color="#ffb74d" />
            ) : locked ? (
              <Lock size={14} color="#5c666f" />
            ) : (
              <CircleDot size={15} color={ACCENT} />
            );
          const body = (
            <ListItemButton
              key={s.key}
              dense
              disabled={locked}
              selected={isSel}
              onClick={() => onSelect(s.index)}
              sx={{
                py: 0.5,
                px: 1.25,
                gap: 1,
                '&.Mui-selected': { bgcolor: 'rgba(79,195,247,0.12)' },
              }}
            >
              <Box sx={{ width: 18, display: 'flex', justifyContent: 'center', flexShrink: 0 }}>{icon}</Box>
              <Box sx={{ minWidth: 0 }}>
                <Typography
                  variant="body2"
                  sx={{
                    fontSize: 12.5,
                    fontWeight: isSel ? 700 : 500,
                    color: locked ? '#5c666f' : isSel ? ACCENT : '#cfd8e0',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {s.index}. {s.title}
                </Typography>
                {s.status === 'complete_with_unknowns' ? (
                  <Typography variant="caption" sx={{ color: '#ffb74d', fontSize: 10 }}>
                    proceeded with unknowns
                  </Typography>
                ) : null}
              </Box>
            </ListItemButton>
          );
          return locked ? (
            <Tooltip
              key={s.key}
              title="Locked: complete the earlier stages first. The methodology is ordered on purpose — an early plausible explanation is not a license to skip measurement-validity checks."
              placement="right"
              arrow
            >
              <span>{body}</span>
            </Tooltip>
          ) : (
            body
          );
        })}
      </List>
    </Box>
  );
}
