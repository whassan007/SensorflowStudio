/**
 * About: product name, current version, links, and chronological release notes.
 * Used as a Help menu tab and as a dialog opened from the version chip.
 */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import Link from '@mui/material/Link';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { ExternalLink, Info, X } from 'lucide-react';
import { ABOUT_CATALOG, APP_VERSION, type ReleaseNotes } from '../../content/releases';

const CARD_SX = {
  mb: 1.25,
  p: 1.5,
  bgcolor: '#141a20',
  border: '1px solid #232a31',
  borderRadius: 1,
};

function ReleaseCard({ release, current }: { release: ReleaseNotes; current: boolean }) {
  return (
    <Box sx={CARD_SX}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, flexWrap: 'wrap', mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#4fc3f7', fontFamily: 'monospace' }}>
          v{release.version}
        </Typography>
        <Typography variant="caption" sx={{ color: '#5c6873' }}>
          {release.date}
        </Typography>
        {current ? (
          <Chip
            size="small"
            label="current"
            sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: 'rgba(79,195,247,0.16)', color: '#4fc3f7' }}
          />
        ) : null}
      </Box>
      <Typography variant="body2" sx={{ fontWeight: 700, fontSize: 13, mb: 0.75 }}>
        {release.title}
      </Typography>
      <Box component="ul" sx={{ m: 0, pl: 2.25 }}>
        {release.highlights.map((h) => (
          <Typography
            key={h}
            component="li"
            variant="body2"
            sx={{ color: '#aab4be', fontSize: 12.5, lineHeight: 1.55, mb: 0.35 }}
          >
            {h}
          </Typography>
        ))}
      </Box>
    </Box>
  );
}

export function AboutPanel() {
  const { name, version, description, links, releases } = ABOUT_CATALOG;
  return (
    <Box>
      <Typography variant="h6" sx={{ fontSize: 16, fontWeight: 700, mb: 0.5 }}>
        {name}
      </Typography>
      <Chip
        size="small"
        label={`v${version}`}
        sx={{ mb: 1.5, bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }}
      />
      <Typography variant="body2" sx={{ color: '#aab4be', mb: 1.5, lineHeight: 1.6 }}>
        {description}
      </Typography>
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2.5 }}>
        <Link
          href={links.github}
          target="_blank"
          rel="noopener noreferrer"
          underline="hover"
          sx={{ color: '#4fc3f7', fontSize: 13, display: 'inline-flex', alignItems: 'center', gap: 0.5 }}
        >
          GitHub <ExternalLink size={12} />
        </Link>
        <Link
          href={links.hf_space}
          target="_blank"
          rel="noopener noreferrer"
          underline="hover"
          sx={{ color: '#4fc3f7', fontSize: 13, display: 'inline-flex', alignItems: 'center', gap: 0.5 }}
        >
          Hugging Face Space <ExternalLink size={12} />
        </Link>
      </Box>
      <Typography
        variant="caption"
        sx={{ color: '#4fc3f7', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.6, display: 'block', mb: 1 }}
      >
        Release notes
      </Typography>
      {releases.map((r) => (
        <ReleaseCard key={r.version} release={r} current={r.version === version} />
      ))}
    </Box>
  );
}

export function VersionChip({ onClick }: { onClick?: () => void }) {
  const chip = (
    <Chip
      size="small"
      label={`v${APP_VERSION}`}
      onClick={onClick}
      sx={{
        ml: 1,
        bgcolor: '#232a31',
        fontFamily: 'monospace',
        fontSize: 11,
        cursor: onClick ? 'pointer' : 'default',
      }}
      aria-label={`Sensorflow Studio version ${APP_VERSION}${onClick ? ', open About' : ''}`}
    />
  );
  if (!onClick) return chip;
  return <Tooltip title={`About Sensorflow Studio v${APP_VERSION}`}>{chip}</Tooltip>;
}

export default function AboutDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{ sx: { bgcolor: '#12171d', border: '1px solid #232a31', maxHeight: '82vh' } }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 1.5, gap: 1 }}>
        <Info size={16} color="#4fc3f7" />
        <Typography variant="h6" sx={{ fontSize: 15, fontWeight: 700, flex: 1 }}>
          About
        </Typography>
        <IconButton size="small" onClick={onClose} aria-label="Close about">
          <X size={16} />
        </IconButton>
      </Box>
      <DialogContent sx={{ pt: 1.5 }}>
        <AboutPanel />
      </DialogContent>
    </Dialog>
  );
}
