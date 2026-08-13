/**
 * Shared design tokens + theme factory for Sensorflow Studio.
 *
 * Semantic color tokens (success / warn / danger / info / neutral) are the
 * single source of truth for state coloring across pages — use these instead
 * of ad-hoc hex values in new code. Existing pages keep their local palettes;
 * the values here intentionally match the established dark-theme colors so
 * adoption is a no-op visually.
 */
import { createTheme, type Theme } from '@mui/material/styles';

export const tokens = {
  color: {
    success: '#66bb6a',
    successBg: 'rgba(102,187,106,0.12)',
    successStrong: '#1b5e20',
    warn: '#f9a825',
    warnBg: 'rgba(249,168,37,0.12)',
    danger: '#ef5350',
    dangerBg: 'rgba(239,83,80,0.14)',
    dangerStrong: '#b71c1c',
    info: '#4fc3f7',
    infoBg: 'rgba(79,195,247,0.12)',
    neutral: '#8a949e',
    neutralBg: 'rgba(138,148,158,0.12)',
    text: '#e6e9ec',
    textDim: '#aab4be',
    textFaint: '#5c6873',
    surface: '#161b21',
    surfaceRaised: '#1b222a',
    surfaceSunken: '#12171d',
    canvas: '#101418',
    border: '#232a31',
    borderStrong: '#2f3944',
  },
  motion: {
    fast: '120ms cubic-bezier(0.4, 0, 0.2, 1)',
    normal: '200ms cubic-bezier(0.4, 0, 0.2, 1)',
    slow: '320ms cubic-bezier(0.4, 0, 0.2, 1)',
  },
  radius: { sm: 4, md: 8, lg: 12 },
  elevation: {
    raised: '0 2px 10px rgba(0,0,0,0.35)',
    overlay: '0 6px 24px rgba(0,0,0,0.5)',
  },
} as const;

/** Color for a signed delta where positive = good (e.g. recall delta). */
export function deltaColor(delta: number, higherIsBetter = true, eps = 1e-9): string {
  if (Math.abs(delta) < eps) return tokens.color.neutral;
  const improved = higherIsBetter ? delta > 0 : delta < 0;
  return improved ? tokens.color.success : tokens.color.danger;
}

/** Traffic-light color for pass/block/unknown-style verdicts. */
export function verdictColor(v: string | null | undefined): string {
  const s = (v ?? '').toUpperCase();
  if (['PASS', 'PROMOTE', 'RELEASE_READY', 'CALIBRATED', 'ALLOW', 'OK'].includes(s)) return tokens.color.success;
  if (['BLOCK', 'BLOCKED', 'FAIL', 'REGRESSION', 'MISCALIBRATED', 'DO_NOT_PROMOTE', 'PERCEPTION_FAILURE'].includes(s))
    return tokens.color.danger;
  if (['WARN', 'INSUFFICIENT_EVIDENCE', 'DEGRADED', 'NEVER_RUN'].includes(s)) return tokens.color.warn;
  return tokens.color.neutral;
}

export function buildTheme(): Theme {
  return createTheme({
    palette: {
      mode: 'dark',
      background: { default: tokens.color.canvas, paper: tokens.color.surface },
      primary: { main: tokens.color.info },
      success: { main: tokens.color.success },
      warning: { main: tokens.color.warn },
      error: { main: tokens.color.danger },
      info: { main: tokens.color.info },
      divider: tokens.color.border,
    },
    typography: {
      fontFamily: "'Inter', 'Helvetica Neue', Arial, sans-serif",
    },
    components: {
      MuiButtonBase: {
        styleOverrides: {
          root: {
            // Keyboard-focus visibility: a clear outline that never shows on mouse click.
            '&.Mui-focusVisible': {
              outline: `2px solid ${tokens.color.info}`,
              outlineOffset: 1,
            },
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: { transition: `background-color ${tokens.motion.fast}, border-color ${tokens.motion.fast}, color ${tokens.motion.fast}` },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: { transition: `border-color ${tokens.motion.normal}, box-shadow ${tokens.motion.normal}` },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: { transition: `background-color ${tokens.motion.fast}` },
        },
      },
      MuiSkeleton: {
        defaultProps: { animation: 'wave' },
        styleOverrides: {
          root: { backgroundColor: 'rgba(138,148,158,0.09)' },
        },
      },
    },
  });
}
