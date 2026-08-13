/**
 * Coverage & curator-quality tab: diversity coverage matrix, confidence
 * calibration reliability chart, precision/recall/yield/model-value cards and
 * the recurring-miss improvement report.
 */
import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { CuratorReport, DiversityReport, ImprovementReport, QuantvalReport } from '../../types/raremine';
import {
  getCuratorReport,
  getDiversityReport,
  getImprovementReport,
  getQuantvalReport,
} from '../../services/raremine';
import { Explainer, SectionLabel } from './shared';

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card sx={{ bgcolor: '#141a20', border: '1px solid #232a31', flex: '1 1 160px', minWidth: 150 }}>
      <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          {label}
        </Typography>
        <Typography variant="h5" sx={{ fontWeight: 800, color: '#4fc3f7' }}>
          {value}
        </Typography>
        {sub ? (
          <Typography variant="caption" sx={{ color: '#5c6a76' }}>
            {sub}
          </Typography>
        ) : null}
      </CardContent>
    </Card>
  );
}

function CalibrationChart({ report }: { report: CuratorReport }) {
  const bins = report.calibration;
  const W = 320;
  const H = 180;
  const pad = 34;
  const bw = (W - pad - 10) / bins.length;
  return (
    <svg width={W} height={H} style={{ background: '#0d1117', borderRadius: 6, border: '1px solid #232a31' }}>
      {/* perfect-calibration diagonal */}
      <line x1={pad} y1={H - pad} x2={W - 10} y2={10} stroke="#37474f" strokeDasharray="4 4" />
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <text x={4} y={H - pad - v * (H - pad - 10) + 4} fill="#5c6a76" fontSize={9}>
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      {bins.map((b, i) => {
        const x = pad + i * bw;
        if (!b.n) {
          return (
            <text key={b.bin} x={x + bw / 2} y={H - pad + 12} fill="#37474f" fontSize={9} textAnchor="middle">
              {b.bin}
            </text>
          );
        }
        const h = (b.observed_rate ?? 0) * (H - pad - 10);
        return (
          <g key={b.bin}>
            <rect x={x + 3} y={H - pad - h} width={bw - 6} height={h} fill="#4fc3f7" opacity={0.75} rx={2} />
            <text x={x + bw / 2} y={H - pad - h - 4} fill="#c3ccd5" fontSize={9} textAnchor="middle">
              {((b.observed_rate ?? 0) * 100).toFixed(0)}% (n={b.n})
            </text>
            <text x={x + bw / 2} y={H - pad + 12} fill="#8a949e" fontSize={9} textAnchor="middle">
              {b.bin}
            </text>
          </g>
        );
      })}
      <text x={W / 2} y={H - 6} fill="#5c6a76" fontSize={9} textAnchor="middle">
        stated rare-event confidence → observed true rate (vs planted truth)
      </text>
    </svg>
  );
}

function CoverageMatrix({ report }: { report: DiversityReport }) {
  const costumes = Object.keys(report.coverage_matrix).sort();
  const conditions = Array.from(
    new Set(costumes.flatMap((c) => Object.keys(report.coverage_matrix[c])))
  ).sort();
  if (!costumes.length) {
    return (
      <Typography variant="caption" sx={{ color: '#5c6a76' }}>
        No diversity selection yet — run mining first.
      </Typography>
    );
  }
  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Table size="small" sx={{ '& td, & th': { borderColor: '#232a31', fontSize: 11, py: 0.4 } }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ color: '#8a949e' }}>costume ↓ / condition →</TableCell>
            {conditions.map((cond) => (
              <TableCell key={cond} sx={{ color: '#8a949e', whiteSpace: 'nowrap' }}>
                {cond.replace(':', ': ')}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {costumes.map((cost) => (
            <TableRow key={cost}>
              <TableCell sx={{ color: '#c3ccd5', fontWeight: 700 }}>{cost}</TableCell>
              {conditions.map((cond) => {
                const n = report.coverage_matrix[cost][cond] ?? 0;
                return (
                  <TableCell key={cond} sx={{ bgcolor: n ? 'rgba(79,195,247,0.14)' : undefined, color: n ? '#4fc3f7' : '#37474f' }}>
                    {n || '·'}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

export default function CoverageQualityTab({ refreshKey }: { refreshKey: number }) {
  const [curator, setCurator] = useState<CuratorReport | null>(null);
  const [diversity, setDiversity] = useState<DiversityReport | null>(null);
  const [quant, setQuant] = useState<QuantvalReport | null>(null);
  const [improvement, setImprovement] = useState<ImprovementReport | null>(null);

  useEffect(() => {
    getCuratorReport().then(setCurator).catch(() => setCurator(null));
    getDiversityReport().then(setDiversity).catch(() => setDiversity(null));
    getQuantvalReport().then(setQuant).catch(() => setQuant(null));
    getImprovementReport().then(setImprovement).catch(() => setImprovement(null));
  }, [refreshKey]);

  const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Explainer>
        This tab measures the MINER, not the scenes: because rare events are planted synthetically, ground truth is
        known and mining precision/recall are exact. Calibration compares the miner&apos;s stated confidence with how
        often it was actually right — a well-calibrated 80% bin should be right about 80% of the time. None of these
        statistics change any candidate; they feed the improvement loop for the next run.
      </Explainer>

      {curator ? (
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          <MetricCard label="Mining precision" value={pct(curator.mining_precision)} sub={`FDR ${pct(curator.false_discovery_rate)}`} />
          <MetricCard
            label="Mining recall"
            value={pct(curator.mining_recall)}
            sub={`${curator.confusion.tp}/${curator.planted_positives} planted events found`}
          />
          <MetricCard
            label="Curation yield"
            value={pct(curator.curation_yield.yield)}
            sub={`${curator.curation_yield.approved}/${curator.curation_yield.reviewed} reviewed approved`}
          />
          <MetricCard
            label="Model value"
            value={pct(curator.model_value.fraction)}
            sub={`${curator.model_value.expose_model_failure}/${curator.model_value.curated} curated expose a real failure`}
          />
          {quant ? (
            <MetricCard
              label="Difficulty agreement"
              value={pct(quant.within_one_level_agreement)}
              sub={`predicted vs observed, ${quant.with_model_outputs} tracks with model outputs`}
            />
          ) : null}
        </Box>
      ) : null}

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Box sx={{ flex: '1 1 340px' }}>
          <SectionLabel>Confidence calibration (reliability)</SectionLabel>
          {curator ? <CalibrationChart report={curator} /> : null}
        </Box>
        <Box sx={{ flex: '2 1 480px' }}>
          <SectionLabel>Diversity coverage matrix (selected examples)</SectionLabel>
          {diversity ? (
            <>
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                <Chip size="small" label={`coverage (diverse): ${pct(diversity.coverage_selected)}`} sx={{ bgcolor: '#1b5e20', color: '#fff' }} />
                <Chip size="small" label={`coverage (naive top-k): ${pct(diversity.coverage_naive_topk)}`} sx={{ bgcolor: '#37474f', color: '#fff' }} />
                <Chip size="small" label={`budget ${diversity.budget} of ${diversity.pool_size}`} sx={{ bgcolor: '#232a31' }} />
              </Box>
              <CoverageMatrix report={diversity} />
            </>
          ) : null}
        </Box>
      </Box>

      {quant ? (
        <Box>
          <SectionLabel>Predicted vs observed difficulty (agreement matrix)</SectionLabel>
          <Table size="small" sx={{ maxWidth: 480, '& td, & th': { borderColor: '#232a31', fontSize: 11.5, py: 0.4 } }}>
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#8a949e' }}>predicted ↓ / observed →</TableCell>
                {['EASY', 'MODERATE', 'HARD', 'EXTREME'].map((d) => (
                  <TableCell key={d} sx={{ color: '#8a949e' }}>
                    {d}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(quant.agreement_matrix).map(([pred, row]) => (
                <TableRow key={pred}>
                  <TableCell sx={{ color: '#c3ccd5', fontWeight: 700 }}>{pred}</TableCell>
                  {['EASY', 'MODERATE', 'HARD', 'EXTREME'].map((obs) => (
                    <TableCell
                      key={obs}
                      sx={{
                        color: row[obs] ? '#4fc3f7' : '#37474f',
                        bgcolor: pred === obs && row[obs] ? 'rgba(102,187,106,0.12)' : undefined,
                      }}
                    >
                      {row[obs] || '·'}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      ) : null}

      {improvement ? (
        <Box sx={{ p: 1.5, bgcolor: '#141a20', border: '1px solid #232a31', borderRadius: 1 }}>
          <SectionLabel>Recurring-miss improvement report (feeds the next run)</SectionLabel>
          <Typography variant="body2" sx={{ color: '#c3ccd5', fontSize: 12.5, mb: 0.5 }}>
            {improvement.recurring_misses.total} planted event(s) missed —{' '}
            {Object.entries(improvement.recurring_misses.by_costume_type)
              .map(([k, v]) => `${k}: ${v}`)
              .join(', ') || 'none'}
            . Over-fires on confounders: {improvement.over_fires.total}.
          </Typography>
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 0.5 }}>
            Suggested next-run sensitivity boosts:{' '}
            {Object.entries(improvement.next_run_config.sensitivity_boost)
              .map(([k, v]) => `${k} +${v}`)
              .join(', ') || 'none'}
            {improvement.next_run_config.rare_event_threshold
              ? ` · tighten rare-event threshold to ${improvement.next_run_config.rare_event_threshold}`
              : ''}
          </Typography>
          <Typography variant="caption" sx={{ color: '#5c6a76' }}>
            Use “Re-mine with improvements” on the Mining Run panel to apply this configuration.
          </Typography>
        </Box>
      ) : null}
    </Box>
  );
}
