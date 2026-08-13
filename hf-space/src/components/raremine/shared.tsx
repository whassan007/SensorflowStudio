import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';

export const PRIORITY_COLORS: Record<string, string> = {
  CRITICAL: '#e53935',
  HIGH: '#fb8c00',
  MEDIUM: '#fdd835',
  LOW: '#78909c',
};

export const DIFFICULTY_COLORS: Record<string, string> = {
  EXTREME: '#e53935',
  HARD: '#fb8c00',
  MODERATE: '#fdd835',
  EASY: '#66bb6a',
};

export function PriorityChip({ value }: { value: string }) {
  return (
    <Chip
      size="small"
      label={value}
      sx={{
        bgcolor: PRIORITY_COLORS[value] ?? '#78909c',
        color: value === 'MEDIUM' ? '#1a1a1a' : '#fff',
        fontWeight: 700,
        fontSize: 10.5,
        height: 20,
      }}
    />
  );
}

export function DifficultyChip({ value }: { value: string }) {
  return (
    <Chip
      size="small"
      variant="outlined"
      label={`difficulty ${value}`}
      sx={{
        borderColor: DIFFICULTY_COLORS[value] ?? '#78909c',
        color: DIFFICULTY_COLORS[value] ?? '#78909c',
        fontWeight: 700,
        fontSize: 10.5,
        height: 20,
      }}
    />
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <Typography
      variant="caption"
      sx={{ color: '#4fc3f7', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.6, display: 'block', mb: 0.5 }}
    >
      {children}
    </Typography>
  );
}

/** Plain-language explainer used at the top of each view. */
export function Explainer({ children }: { children: ReactNode }) {
  return (
    <Box
      sx={{
        p: 1.5,
        bgcolor: '#141a20',
        border: '1px solid #232a31',
        borderLeft: '3px solid #4fc3f7',
        borderRadius: 1,
        mb: 2,
      }}
    >
      <Typography variant="body2" sx={{ color: '#aab4be', fontSize: 12.5, lineHeight: 1.6 }}>
        {children}
      </Typography>
    </Box>
  );
}

export function StatChip({ label, value, color }: { label: string; value: ReactNode; color?: string }) {
  return (
    <Chip
      size="small"
      label={
        <span>
          <b>{value}</b> {label}
        </span>
      }
      sx={{ bgcolor: '#232a31', color: color ?? '#e0e6ec', fontSize: 12, height: 24 }}
    />
  );
}
