/** Tab 5: render the nextgen architecture decision documents. */
import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import * as api from '../../services/nextgen';
import type { ArchitectureDocs } from '../../types/nextgen';
import Markdown from './Markdown';
import { PANEL_SX } from './common';

const DOC_LABELS: Record<string, string> = {
  comparison: 'Three-Way Comparison',
  adr: 'ADR',
  rollout: 'Rollout Plan',
  worldmodel: 'World-Model Comparison',
};

export default function ArchitectureTab() {
  const [docs, setDocs] = useState<ArchitectureDocs | null>(null);
  const [active, setActive] = useState('comparison');
  const [error, setError] = useState('');

  useEffect(() => {
    api.getArchitectureDocs().then(setDocs).catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!docs) return <Typography variant="body2" sx={{ color: '#8a949e' }}>Loading architecture docs…</Typography>;

  const keys = Object.keys(docs.docs);
  const current = docs.docs[active];
  return (
    <Box>
      <Tabs
        value={active}
        onChange={(_, v) => setActive(v)}
        sx={{ mb: 1.5, minHeight: 34, '& .MuiTab-root': { minHeight: 34, fontSize: 12, textTransform: 'none' } }}
      >
        {keys.map((k) => (
          <Tab key={k} value={k} label={DOC_LABELS[k] ?? k} />
        ))}
      </Tabs>
      <Paper sx={{ ...PANEL_SX, p: 2.5, maxWidth: 980 }}>
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1, fontFamily: 'monospace' }}>
          {current?.file}
        </Typography>
        {current?.content ? (
          <Markdown source={current.content} />
        ) : (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            Document not found on disk.
          </Typography>
        )}
      </Paper>
    </Box>
  );
}
