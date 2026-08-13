/**
 * Loading / empty-state feedback components (Part 3 polish primitives).
 *
 * PanelSkeleton replaces spinner-style loading with content-shaped skeletons;
 * IllustratedEmpty renders a small inline-SVG illustration + a "how to
 * produce data" action so empty screens teach instead of dead-ending.
 */
import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import Skeleton from '@mui/material/Skeleton';
import Typography from '@mui/material/Typography';
import { tokens } from '../../theme';

// ---------------------------------------------------------------- skeletons

export function PanelSkeleton({ rows = 4, height = 22, header = true }: { rows?: number; height?: number; header?: boolean }) {
  return (
    <Box sx={{ p: 1 }} aria-busy="true" aria-label="Loading">
      {header ? <Skeleton variant="text" width="35%" height={26} sx={{ mb: 1 }} /> : null}
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} variant="rounded" height={height} sx={{ mb: 0.75, opacity: 1 - i * 0.12 }} />
      ))}
    </Box>
  );
}

export function ChartSkeleton({ height = 240 }: { height?: number }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1, height, p: 2 }} aria-busy="true" aria-label="Loading chart">
      {[0.5, 0.8, 0.35, 0.65, 0.9, 0.45, 0.7, 0.55].map((f, i) => (
        <Skeleton key={i} variant="rounded" width="100%" height={`${f * 100}%`} />
      ))}
    </Box>
  );
}

export function TileSkeleton({ n = 4 }: { n?: number }) {
  return (
    <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }} aria-busy="true">
      {Array.from({ length: n }, (_, i) => (
        <Skeleton key={i} variant="rounded" height={84} sx={{ flex: '1 1 150px' }} />
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------- empty states

type EmptyArt = 'data' | 'search' | 'canvas' | 'gauge' | 'map';

function Art({ variant }: { variant: EmptyArt }) {
  const stroke = tokens.color.textFaint;
  const accent = tokens.color.info;
  const common = { fill: 'none', strokeWidth: 1.6, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  switch (variant) {
    case 'search':
      return (
        <svg width="120" height="86" viewBox="0 0 120 86">
          <rect x="14" y="12" width="72" height="10" rx="5" stroke={stroke} {...common} />
          <rect x="14" y="30" width="52" height="10" rx="5" stroke={stroke} {...common} opacity={0.6} />
          <rect x="14" y="48" width="62" height="10" rx="5" stroke={stroke} {...common} opacity={0.35} />
          <circle cx="88" cy="52" r="16" stroke={accent} {...common} />
          <line x1="100" y1="64" x2="110" y2="74" stroke={accent} {...common} />
        </svg>
      );
    case 'canvas':
      return (
        <svg width="120" height="86" viewBox="0 0 120 86">
          <rect x="10" y="10" width="100" height="66" rx="6" stroke={stroke} {...common} strokeDasharray="5 4" />
          <rect x="26" y="26" width="26" height="16" rx="3" stroke={accent} {...common} />
          <circle cx="82" cy="52" r="9" stroke={stroke} {...common} />
          <path d="M 52 34 C 62 34, 66 52, 73 52" stroke={stroke} {...common} strokeDasharray="3 3" />
          <path d="M 62 66 l 6 -6 m -6 6 l 8 -2 m -8 2 l 2 -8" stroke={accent} {...common} />
        </svg>
      );
    case 'gauge':
      return (
        <svg width="120" height="86" viewBox="0 0 120 86">
          <path d="M 24 70 A 40 40 0 0 1 96 70" stroke={stroke} {...common} />
          <path d="M 24 70 A 40 40 0 0 1 52 33" stroke={accent} {...common} strokeWidth={3} />
          <line x1="60" y1="70" x2="44" y2="46" stroke={accent} {...common} />
          <circle cx="60" cy="70" r="4" fill={accent} stroke="none" />
        </svg>
      );
    case 'map':
      return (
        <svg width="120" height="86" viewBox="0 0 120 86">
          <path d="M 14 22 L 44 12 L 76 22 L 106 12 L 106 64 L 76 74 L 44 64 L 14 74 Z" stroke={stroke} {...common} />
          <line x1="44" y1="12" x2="44" y2="64" stroke={stroke} {...common} opacity={0.4} />
          <line x1="76" y1="22" x2="76" y2="74" stroke={stroke} {...common} opacity={0.4} />
          <circle cx="60" cy="40" r="6" stroke={accent} {...common} />
          <path d="M 60 46 L 60 56" stroke={accent} {...common} />
        </svg>
      );
    case 'data':
    default:
      return (
        <svg width="120" height="86" viewBox="0 0 120 86">
          <ellipse cx="60" cy="20" rx="34" ry="9" stroke={stroke} {...common} />
          <path d="M 26 20 V 62 C 26 67, 41 71, 60 71 C 79 71, 94 67, 94 62 V 20" stroke={stroke} {...common} />
          <path d="M 26 41 C 26 46, 41 50, 60 50 C 79 50, 94 46, 94 41" stroke={stroke} {...common} opacity={0.5} />
          <circle cx="92" cy="62" r="13" {...common} fill={tokens.color.canvas} stroke={accent} />
          <path d="M 92 56 v 12 m -6 -6 h 12" stroke={accent} {...common} />
        </svg>
      );
  }
}

export function IllustratedEmpty({
  art = 'data',
  title,
  message,
  action,
}: {
  art?: EmptyArt;
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <Box sx={{ textAlign: 'center', py: 5, px: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
      <Art variant={art} />
      <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
        {title}
      </Typography>
      <Typography variant="body2" sx={{ color: tokens.color.neutral, maxWidth: 520 }}>
        {message}
      </Typography>
      {action ? <Box sx={{ mt: 1 }}>{action}</Box> : null}
    </Box>
  );
}
