import { useState } from 'react';
import Box from '@mui/material/Box';
import type { CopilotExplainRequest } from '../types/labeleval';
import { getRegression, usePoll } from '../services/labeleval';
import RegressionTracking from '../components/labeleval/RegressionTracking';
import CopilotDrawer from '../components/labeleval/CopilotDrawer';
import { LoadingBox, ErrorNote } from '../components/labeleval/shared';

export default function RegressionPage() {
  const regression = usePoll(getRegression, 5000);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotRequest, setCopilotRequest] = useState<CopilotExplainRequest | null>(null);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {regression.loading && !regression.data ? <LoadingBox label="Loading regression history…" /> : null}
      {regression.error && !regression.data ? <ErrorNote error={regression.error} /> : null}
      <RegressionTracking
        regression={regression.data}
        onAskCopilot={(request) => {
          setCopilotRequest(request);
          setCopilotOpen(true);
        }}
      />
      <CopilotDrawer open={copilotOpen} onClose={() => setCopilotOpen(false)} request={copilotRequest} />
    </Box>
  );
}
