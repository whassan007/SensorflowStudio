/** Hardware / compatibility status card — honest platform reporting. */
import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import { Cpu } from 'lucide-react';
import { getRetroCompat, getRetroEnv } from '../../services/retro';
import type { CompatReport, EnvironmentReport } from '../../types/retro';

const CHECK_COLORS: Record<string, string> = {
  PASS: '#66bb6a',
  FAIL: '#ef5350',
  SKIPPED: '#8a949e',
};

export default function HardwareCard() {
  const [env, setEnv] = useState<EnvironmentReport | null>(null);
  const [compat, setCompat] = useState<CompatReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRetroEnv().then((r) => setEnv(r.environment)).catch((e: Error) => setError(e.message));
    getRetroCompat().then((r) => setCompat(r.report)).catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return <Typography sx={{ fontSize: 12, color: '#ef5350' }}>hardware report unavailable: {error}</Typography>;
  }
  if (!env || !compat) {
    return <Typography sx={{ fontSize: 12, color: '#8a949e' }}>Detecting environment…</Typography>;
  }

  return (
    <Paper sx={{ p: 1.5, bgcolor: '#161c23', border: '1px solid #232a31' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Cpu size={16} color="#4fc3f7" />
        <Typography sx={{ fontSize: 13, fontWeight: 700 }}>Hardware & vLLM compatibility</Typography>
        <Chip size="small"
              label={compat.vllm_supported ? 'vLLM SUPPORTED' : 'vLLM UNSUPPORTED HERE'}
              sx={{ bgcolor: compat.vllm_supported ? '#66bb6a22' : '#ef535022',
                    color: compat.vllm_supported ? '#66bb6a' : '#ef5350', fontWeight: 800, fontSize: 10 }} />
      </Box>
      <Typography sx={{ fontSize: 12, color: '#aab4be', mb: 1 }}>{compat.platform_summary}</Typography>
      <Typography sx={{ fontSize: 11.5, fontFamily: 'monospace', color: '#8a949e', mb: 1 }}>
        {env.os_name} {env.os_version} · {env.machine_arch}
        {env.is_apple_silicon ? ' (Apple Silicon)' : ''} · Python {env.python_version}
        {env.gpus.length ? ` · ${env.gpus.map((g) => `${g.vendor} ${g.model}`).join(', ')}` : ' · no GPU detected'}
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        {compat.checks.map((c) => (
          <Box key={c.link} sx={{ display: 'flex', gap: 1, alignItems: 'baseline' }}>
            <Chip size="small" label={c.status}
                  sx={{ bgcolor: `${CHECK_COLORS[c.status]}22`, color: CHECK_COLORS[c.status],
                        fontWeight: 700, fontSize: 9.5, width: 70 }} />
            <Typography sx={{ fontSize: 11.5, fontFamily: 'monospace', minWidth: 110 }}>{c.link}</Typography>
            <Typography sx={{ fontSize: 11.5, color: '#8a949e' }}>{c.reason}</Typography>
          </Box>
        ))}
      </Box>
      {env.ollama_endpoint ? (
        <Typography sx={{ fontSize: 11.5, color: '#66bb6a', mt: 1 }}>
          Ollama reachable at {env.ollama_endpoint} ({env.ollama_models.length} models) — real local
          inference available, labeled Ollama-on-CPU/Metal (not vLLM).
        </Typography>
      ) : (
        <Typography sx={{ fontSize: 11.5, color: '#8a949e', mt: 1 }}>
          Ollama not reachable — deterministic mock backend is the only runnable option here.
        </Typography>
      )}
    </Paper>
  );
}
