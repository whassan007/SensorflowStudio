import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import { ExternalLink } from 'lucide-react';

// In dev the vite server only proxies /api, so '/' would render the React app
// itself; in production the backend serves the legacy UI at the site root.
const LEGACY_URL = import.meta.env.DEV ? 'http://localhost:8000/' : '/';

export default function LegacyStudioPage() {
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          px: 2,
          py: 1,
          borderBottom: '1px solid #232a31',
          bgcolor: '#161b21',
        }}
      >
        <Typography variant="body2" sx={{ color: '#8a949e', flex: 1 }}>
          Legacy vanilla-JS studio served by the backend at {LEGACY_URL}. If the frame below is blank, make sure the
          backend is running on port 8000.
        </Typography>
        <Button
          size="small"
          variant="outlined"
          startIcon={<ExternalLink size={14} />}
          onClick={() => window.open(LEGACY_URL, '_blank', 'noopener')}
        >
          Open in new tab
        </Button>
      </Box>
      <iframe
        src={LEGACY_URL}
        title="Legacy Studio"
        style={{ flex: 1, width: '100%', border: 'none', background: '#101418' }}
      />
    </Box>
  );
}
