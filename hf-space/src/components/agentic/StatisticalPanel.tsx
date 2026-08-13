/** Deterministic statistics: rates, Wilson CIs, exact-binomial significance,
 * seqeval sequential decision, power/MDE and small-sample flags. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { StatisticalAssessment } from '../../types/agentic';
import { fmtCi, fmtRate, KV, PanelTitle } from './common';

export default function StatisticalPanel({ stat }: { stat: StatisticalAssessment }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
      <PanelTitle
        title="Statistical Assessment"
        origin="deterministic"
        extra={
          <Chip
            size="small"
            label={stat.significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT'}
            sx={{
              height: 18,
              fontSize: 10,
              fontWeight: 800,
              bgcolor: stat.significant ? '#b71c1c' : '#37474f',
              color: '#fff',
            }}
          />
        }
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1 }}>
        <Box>
          <KV
            k="Candidate rate"
            v={`${fmtRate(stat.candidate.rate)} (${stat.candidate.events}/${stat.candidate.denominator.toLocaleString()})`}
            mono
          />
          <KV k="Candidate Wilson 95% CI" v={fmtCi(stat.candidate.wilson_ci)} mono />
          <KV
            k="Baseline rate"
            v={`${fmtRate(stat.baseline.rate)} (${stat.baseline.events}/${stat.baseline.denominator.toLocaleString()})`}
            mono
          />
          <KV k="Baseline Wilson 95% CI" v={fmtCi(stat.baseline.wilson_ci)} mono />
          <KV k="Absolute delta" v={fmtRate(stat.absolute_delta)} mono />
          <KV
            k="Relative delta"
            v={stat.relative_delta === null ? '—' : `${stat.relative_delta}×`}
            mono
          />
          <KV
            k="Exact binomial p"
            v={stat.exact_binomial_p === null ? '—' : stat.exact_binomial_p.toExponential(3)}
            mono
          />
        </Box>
        <Box>
          <KV k="seqeval decision" v={stat.seqeval.decision} mono />
          <KV k="seqeval delegated to" v={stat.seqeval.delegated_to} mono />
          <KV k="Clusters fed" v={String(stat.seqeval.clusters_fed)} mono />
          <KV
            k="MDE (abs) at this n"
            v={stat.power_mde.mde_abs === null ? '—' : fmtRate(stat.power_mde.mde_abs)}
            mono
          />
          <KV k="Power assessment" v={stat.power_mde.assessment} />
          <KV k="Significance method" v={stat.significance_method} />
        </Box>
      </Box>
      {stat.small_sample_flags.length > 0 ? (
        <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          {stat.small_sample_flags.map((f) => (
            <Chip
              key={f}
              size="small"
              label={f}
              sx={{ height: 20, fontSize: 10.5, bgcolor: '#4e342e', color: '#ffcc80' }}
            />
          ))}
        </Box>
      ) : null}
      <Typography variant="caption" component="div" sx={{ color: '#5c6770', mt: 1 }}>
        {stat.rare_event_handling}
      </Typography>
    </Paper>
  );
}
