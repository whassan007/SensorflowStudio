import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import type { QualityMetrics } from '../../types/labeleval';
import { SectionCard, HBar, fmtPct, pctFraction } from './shared';
import { Term } from '../help/InfoTip';

function ConsensusGauge({ value }: { value: number | null }) {
  const frac = pctFraction(value);
  const size = 140;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const circumference = Math.PI * r; // half circle
  const color = frac >= 0.8 ? '#66bb6a' : frac >= 0.6 ? '#ffa726' : '#ef5350';
  return (
    <Box sx={{ textAlign: 'center' }}>
      <svg width={size} height={size / 2 + stroke} viewBox={`0 0 ${size} ${size / 2 + stroke}`}>
        <path
          d={`M ${stroke / 2} ${size / 2 + stroke / 2} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${size / 2 + stroke / 2}`}
          fill="none"
          stroke="#232a31"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d={`M ${stroke / 2} ${size / 2 + stroke / 2} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${size / 2 + stroke / 2}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${circumference * frac} ${circumference}`}
        />
        <text x={size / 2} y={size / 2 + 2} textAnchor="middle" fill="#e6e9ec" fontSize={22} fontWeight={700}>
          {value === null ? '—' : fmtPct(value)}
        </text>
      </svg>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
        Canonical grader_consensus_score
      </Typography>
    </Box>
  );
}

export default function GraderDisagreement({
  metrics,
  graderCount = 3,
  disagreementTypes = [],
}: {
  metrics: QualityMetrics | null;
  graderCount?: number;
  disagreementTypes?: string[];
}) {
  const consensus = metrics?.global.grader_consensus ?? null;
  const frac = pctFraction(consensus);
  return (
    <SectionCard
      title={`Grader Disagreement — ${graderCount} graders`}
      helpTerm="grader_consensus"
    >
      <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <ConsensusGauge value={consensus} />
        <Box sx={{ flex: '1 1 280px', minWidth: 260 }}>
          <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
            AGREEMENT DIMENSIONS (dataset aggregate)
          </Typography>
          <Box sx={{ mt: 1 }}>
            <HBar label="Class agreement" value={frac} max={1} color="#4fc3f7" valueLabel={fmtPct(consensus)} />
            <HBar label="Spatial agreement" value={frac} max={1} color="#7e57c2" valueLabel={fmtPct(consensus)} />
            <HBar label="Temporal agreement" value={frac} max={1} color="#26a69a" valueLabel={fmtPct(consensus)} />
          </Box>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Aggregate approximated by the consensus score; the per-annotation class/spatial/temporal breakdown is shown
            in each evidence panel.
          </Typography>
          {disagreementTypes.length > 0 ? (
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 1 }}>
              {disagreementTypes.map((d) => (
                <Chip key={d} size="small" label={d} sx={{ bgcolor: '#4a2c00', color: '#ffcc80' }} />
              ))}
            </Box>
          ) : null}
        </Box>
      </Box>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1.5 }}>
        Agreement statistic: <Term k="cohens_kappa">Cohen&apos;s κ</Term> for 2 graders,{' '}
        <Term k="fleiss_kappa">Fleiss&apos; κ</Term> for a fixed panel of 3+,{' '}
        <Term k="krippendorff_alpha">Krippendorff&apos;s α</Term> when graders have missing ratings. Low consensus is
        evidence for the Quality Gate — it never directly rejects a label.
      </Typography>
    </SectionCard>
  );
}
