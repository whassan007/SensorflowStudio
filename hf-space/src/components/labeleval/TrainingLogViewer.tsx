import { useEffect, useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function TrainingLogViewer({ logs }: { logs: string[] }) {
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = boxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  return (
    <Box>
      <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
        TRAINING LOG
      </Typography>
      <Box
        ref={boxRef}
        sx={{
          mt: 0.5,
          bgcolor: '#0a0e12',
          border: '1px solid #232a31',
          borderRadius: 1,
          p: 1.5,
          height: 240,
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: 12,
          lineHeight: 1.6,
          whiteSpace: 'pre-wrap',
        }}
      >
        {logs.length === 0 ? (
          <span style={{ color: '#5c6773' }}>No log output yet…</span>
        ) : (
          logs.map((line, i) => <div key={i}>{line}</div>)
        )}
      </Box>
    </Box>
  );
}
