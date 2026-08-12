import { useEffect, useState } from 'react';
import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import IconButton from '@mui/material/IconButton';
import { Sparkles, X } from 'lucide-react';
import type { CopilotExplainRequest, CopilotExplainResponse } from '../../types/labeleval';
import { copilotExplain } from '../../services/labeleval';

export default function CopilotDrawer({
  open,
  onClose,
  request,
}: {
  open: boolean;
  onClose: () => void;
  request: CopilotExplainRequest | null;
}) {
  const [result, setResult] = useState<CopilotExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Reset when a new context arrives.
    setResult(null);
    setError(null);
  }, [request]);

  const analyze = async () => {
    if (!request) return;
    setLoading(true);
    setError(null);
    try {
      const res = await copilotExplain(request);
      setResult(res);
    } catch (err) {
      // A 503 may still carry an offline-deterministic analysis in its body.
      const message = err instanceof Error ? err.message : String(err);
      const jsonStart = message.indexOf('{');
      if (jsonStart >= 0) {
        try {
          const body = JSON.parse(message.slice(jsonStart)) as Partial<CopilotExplainResponse>;
          if (typeof body.analysis === 'string' && body.analysis) {
            setResult({
              status: body.status ?? 'offline',
              provider: body.provider ?? 'offline_deterministic',
              analysis: body.analysis,
              structured: body.structured ?? null,
            });
            setLoading(false);
            return;
          }
        } catch {
          /* body was not JSON */
        }
      }
      setError(`Copilot unavailable: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const structured = result?.structured ?? null;

  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: 420, bgcolor: '#12171d' } }}>
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1.5, height: '100%', overflowY: 'auto' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Sparkles size={18} color="#4fc3f7" />
            <Typography variant="h6">Copilot</Typography>
          </Box>
          <IconButton size="small" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </Box>

        <Alert severity="info" variant="outlined" sx={{ fontSize: 12 }}>
          Advisory only — the Copilot never changes metrics or overrides gates.
        </Alert>

        {request ? (
          <Box>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              CONTEXT
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
              <Chip size="small" label={request.context_type} sx={{ bgcolor: '#232a31' }} />
              {request.annotation_id ? <Chip size="small" label={request.annotation_id} sx={{ bgcolor: '#232a31' }} /> : null}
              {request.event_id ? <Chip size="small" label={request.event_id} sx={{ bgcolor: '#232a31' }} /> : null}
              {request.model_version ? <Chip size="small" label={request.model_version} sx={{ bgcolor: '#232a31' }} /> : null}
            </Box>
          </Box>
        ) : (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No context selected.
          </Typography>
        )}

        <Button
          variant="contained"
          onClick={() => void analyze()}
          disabled={!request || loading}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <Sparkles size={16} />}
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </Button>

        {error ? (
          <Alert severity="warning" variant="outlined">
            {error} The Copilot service may be offline — evidence and gate results above remain authoritative.
          </Alert>
        ) : null}

        {result ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              Provider: {result.provider} · Status: {result.status}
            </Typography>

            {structured ? (
              <>
                <Box>
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    FAILURE CLASSIFICATION
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>
                    {structured.failure_classification}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    OBSERVED EVIDENCE
                  </Typography>
                  <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
                    {structured.observed_evidence.map((e, i) => (
                      <li key={i}>
                        <Typography variant="body2">{e}</Typography>
                      </li>
                    ))}
                  </ul>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    LIKELY CAUSE
                  </Typography>
                  <Typography variant="body2">{structured.likely_cause}</Typography>
                </Box>
                {structured.contributing_factors.length > 0 ? (
                  <Box>
                    <Typography variant="caption" sx={{ color: '#8a949e' }}>
                      CONTRIBUTING FACTORS
                    </Typography>
                    <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
                      {structured.contributing_factors.map((f, i) => (
                        <li key={i}>
                          <Typography variant="body2">{f}</Typography>
                        </li>
                      ))}
                    </ul>
                  </Box>
                ) : null}
                <Box sx={{ border: '1px dashed #4fc3f7', borderRadius: 1, p: 1 }}>
                  <Typography variant="caption" sx={{ color: '#4fc3f7', fontWeight: 700 }}>
                    Hypothesis — not a conclusion
                  </Typography>
                  <Typography variant="body2">{structured.hypothesis}</Typography>
                </Box>
                {structured.recommended_investigation.length > 0 ? (
                  <Box>
                    <Typography variant="caption" sx={{ color: '#8a949e' }}>
                      RECOMMENDED INVESTIGATION
                    </Typography>
                    <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
                      {structured.recommended_investigation.map((r, i) => (
                        <li key={i}>
                          <Typography variant="body2">{r}</Typography>
                        </li>
                      ))}
                    </ul>
                  </Box>
                ) : null}
                <Box>
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    CONFIDENCE {(structured.confidence * 100).toFixed(0)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, structured.confidence * 100)}
                    sx={{ height: 8, borderRadius: 1, mt: 0.5 }}
                  />
                </Box>
                <Divider />
              </>
            ) : null}

            {result.analysis ? (
              <Box>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  ANALYSIS
                </Typography>
                <Box
                  component="pre"
                  sx={{
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'monospace',
                    fontSize: 12,
                    bgcolor: '#101418',
                    border: '1px solid #232a31',
                    borderRadius: 1,
                    p: 1.5,
                    m: 0,
                  }}
                >
                  {result.analysis}
                </Box>
              </Box>
            ) : null}
          </Box>
        ) : null}
      </Box>
    </Drawer>
  );
}
