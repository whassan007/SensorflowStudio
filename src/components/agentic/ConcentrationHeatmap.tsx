/** Uniform-vs-concentrated analysis as a relative-risk heatmap per stratum. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import type { ConcentrationAnalysis } from '../../types/agentic';
import { fmtCi, fmtRate, PanelTitle } from './common';

function riskColor(rr: number | null): string {
  if (rr === null) return '#37474f';
  if (rr >= 10) return '#b71c1c';
  if (rr >= 3) return '#d84315';
  if (rr >= 1.5) return '#ef6c00';
  if (rr >= 0.75) return '#37474f';
  return '#1b5e20';
}

export default function ConcentrationHeatmap({ conc }: { conc: ConcentrationAnalysis }) {
  const byDim = new Map<string, typeof conc.strata>();
  for (const s of conc.strata) {
    byDim.set(s.dimension, [...(byDim.get(s.dimension) ?? []), s]);
  }
  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
      <PanelTitle
        title="Distribution & Concentration"
        origin="deterministic"
        extra={
          <Chip
            size="small"
            label={conc.determination.toUpperCase()}
            sx={{
              height: 18,
              fontSize: 10,
              fontWeight: 800,
              bgcolor: conc.determination === 'concentrated' ? '#e65100' : '#37474f',
              color: '#fff',
            }}
          />
        }
      />
      {conc.concentrated_dimensions.length > 0 ? (
        <Typography variant="caption" component="div" sx={{ mb: 1, color: '#ffcc80' }}>
          Concentrated in: {conc.concentrated_dimensions.join(', ')}
        </Typography>
      ) : null}
      {[...byDim.entries()].map(([dim, strata]) => (
        <Box key={dim} sx={{ mb: 1 }}>
          <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
            {dim}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.25 }}>
            {strata.map((s) => (
              <Tooltip
                key={`${s.dimension}:${s.stratum}`}
                title={
                  <Box sx={{ fontSize: 11 }}>
                    <div>rate {fmtRate(s.stratum_rate)} vs population {fmtRate(s.baseline_rate)}</div>
                    <div>events {s.events} / exposure {s.exposure.toLocaleString()} ({(s.exposure_share * 100).toFixed(1)}%)</div>
                    <div>odds ratio {s.odds_ratio ?? '—'} · risk diff {fmtRate(s.risk_difference)}</div>
                    <div>rate CI {fmtCi(s.rate_wilson_ci)}</div>
                    {s.small_sample_flag ? <div>⚠ small-sample: estimate unstable</div> : null}
                  </Box>
                }
              >
                <Paper
                  variant="outlined"
                  sx={{
                    px: 1,
                    py: 0.5,
                    bgcolor: riskColor(s.relative_risk),
                    color: '#fff',
                    minWidth: 110,
                    cursor: 'default',
                    border: s.small_sample_flag ? '1px dashed #ffcc80' : undefined,
                  }}
                >
                  <Typography variant="caption" sx={{ fontWeight: 800, display: 'block' }}>
                    {s.stratum}
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                    RR {s.relative_risk === null ? '—' : s.relative_risk.toFixed(1)}
                    {s.small_sample_flag ? ' ⚠' : ''}
                  </Typography>
                </Paper>
              </Tooltip>
            ))}
          </Box>
        </Box>
      ))}
      <Typography variant="caption" component="div" sx={{ color: '#5c6770' }}>
        {conc.method}
      </Typography>
    </Paper>
  );
}
