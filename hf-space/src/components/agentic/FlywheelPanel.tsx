/** Evaluation flywheel: suites created from validated failures (governance
 * fields + contamination-guard badges) and the seqeval regression hook. */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { EvaluationSuite, RegressionSuiteResult } from '../../types/agentic';
import { runRegression } from '../../services/agentic';
import { fmtCi, KV, PanelTitle } from './common';

const DECISION_COLORS: Record<string, string> = {
  PASS: '#1b5e20',
  REGRESSION: '#b71c1c',
  INSUFFICIENT_EVIDENCE: '#4e342e',
};

export default function FlywheelPanel({ suites }: { suites: EvaluationSuite[] }) {
  const [regression, setRegression] = useState<RegressionSuiteResult[] | null>(null);
  const [running, setRunning] = useState(false);

  const evaluate = async () => {
    setRunning(true);
    try {
      const res = await runRegression();
      setRegression(res.suites);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
        <PanelTitle title="Evaluation Suites from Validated Failures" origin="deterministic" />
        {suites.length === 0 ? (
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            No suites yet — suites are created by the LEARNING FLYWHEEL stage after a failure is
            human-validated.
          </Typography>
        ) : (
          suites.map((s) => (
            <Paper key={s.suite_id} variant="outlined" sx={{ p: 1, mb: 1, bgcolor: '#12171d' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 800, flex: 1 }}>
                  {s.name} <span style={{ color: '#8a949e' }}>v{s.version}</span>
                </Typography>
                <Chip size="small" label={s.approval_status.toUpperCase()} sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: s.approval_status === 'approved' ? '#1b5e20' : '#37474f', color: '#fff' }} />
                <Chip size="small" label="CONTAMINATION GUARD" sx={{ height: 18, fontSize: 10, fontWeight: 800, bgcolor: '#4a148c', color: '#ce93d8' }} />
                {s.governance_overrides.length > 0 ? (
                  <Chip size="small" label={`${s.governance_overrides.length} recorded override(s)`} sx={{ height: 18, fontSize: 10, bgcolor: '#4e342e', color: '#ffcc80' }} />
                ) : null}
              </Box>
              <KV k="Creation reason" v={s.creation_reason} />
              <KV k="Source failures" v={s.source_failures.join(', ')} mono />
              <KV k="Taxonomy tags" v={s.taxonomy_tags.join(', ')} mono />
              <KV k="Sampling policy" v={s.sampling_policy} />
              <KV k="Coverage" v={JSON.stringify(s.coverage)} mono />
              <KV k="Known limitations" v={s.known_limitations.join('; ')} />
              <KV
                k="Members"
                v={`${s.members.length} (training-eligible: ${s.members.filter((m) => m.training_eligible).length} — requires explicit recorded override)`}
              />
            </Paper>
          ))
        )}
      </Paper>

      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
        <PanelTitle
          title="Regression Hook — candidate vs baseline per suite (stats: seqeval)"
          origin="deterministic"
          extra={
            <Button size="small" variant="contained" disabled={running} onClick={evaluate}>
              {running ? 'Evaluating…' : 'Evaluate candidate'}
            </Button>
          }
        />
        {regression ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                {['Suite', 'n', 'Baseline', 'Candidate', 'Δ', 'Δ CI (anytime-valid)', 'Decision'].map((h) => (
                  <TableCell key={h} sx={{ fontSize: 11, fontWeight: 800, color: '#8a949e' }}>
                    {h}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {regression.map((r) => (
                <TableRow key={r.suite}>
                  <TableCell sx={{ fontSize: 11.5 }}>{r.suite}</TableCell>
                  <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>{r.n}</TableCell>
                  <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>{r.baseline_rate ?? '—'}</TableCell>
                  <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>{r.candidate_rate ?? '—'}</TableCell>
                  <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>{r.delta ?? '—'}</TableCell>
                  <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>{fmtCi(r.delta_ci)}</TableCell>
                  <TableCell>
                    <Chip size="small" label={r.decision} sx={{ height: 18, fontSize: 10, fontWeight: 800, bgcolor: DECISION_COLORS[r.decision] ?? '#37474f', color: '#fff' }} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Run the evaluation to compare the candidate model against the general,
            historical-regression, rare-event, safety-critical and failure-derived suites.
            INSUFFICIENT_EVIDENCE is reported as-is — it is never a pass.
          </Typography>
        )}
      </Paper>
    </Box>
  );
}
