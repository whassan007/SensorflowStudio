import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';

/** Card with a title and a visible caption subtitle (the vitis pages lean on
 * subtitles for honesty notes, so they must be visible, not tooltip-only). */
export function VitisSection({
  title,
  subtitle,
  action,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card variant="outlined" sx={{ bgcolor: '#161b21' }}>
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: subtitle ? 0.25 : 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, flex: 1 }}>
            {title}
          </Typography>
          {action}
        </Box>
        {subtitle ? (
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1.25, lineHeight: 1.45 }}>
            {subtitle}
          </Typography>
        ) : null}
        {children}
      </CardContent>
    </Card>
  );
}
