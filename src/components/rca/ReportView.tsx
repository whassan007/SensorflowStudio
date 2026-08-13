/** Stage 12: recommended experiments, minimum-additional-evidence answer,
 * rendered final report with remediation tiers, JSON/markdown export, and the
 * training-mode reveal. */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import Typography from '@mui/material/Typography';
import { Download, Eye, FileJson } from 'lucide-react';
import type { Experiments, Investigation, RcaReport, RevealResponse } from '../../types/rca';
import { BORDER, CONFIDENCE_COLORS, Explainer, SectionCard, StatusPill, tableSx } from './common';

function download(filename: string, text: string, mime: string) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const COST_COLORS: Record<string, string> = { low: '#66bb6a', medium: '#ffb74d', high: '#ef5350' };

export function ExperimentsView({ exp }: { exp: Experiments }) {
  return (
    <>
      <Explainer text={exp.explainer} />
      <SectionCard title="Minimum additional evidence required">
        <Typography variant="body2" sx={{ fontSize: 12.5, color: '#e3eaf0' }}>
          {exp.minimum_additional_evidence}
        </Typography>
        <Box sx={{ display: 'flex', gap: 3, mt: 1 }}>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Current effective n: <b>{exp.power.effective_n.toLocaleString()}</b>
          </Typography>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Needed for the {exp.power.practical_margin_pp.toFixed(1)}pp margin:{' '}
            <b>{exp.power.needed_effective_n.toLocaleString()}</b>
          </Typography>
        </Box>
      </SectionCard>
      <SectionCard title="Recommended experiments — ranked by information gain / cost">
        <Box component="table" sx={tableSx}>
          <thead>
            <tr>
              <th>#</th><th>Design</th><th>Instantiated for this investigation</th>
              <th>Discriminates</th><th>Cost</th><th>Gain</th><th>Priority</th>
            </tr>
          </thead>
          <tbody>
            {exp.experiments.map((d) => (
              <tr key={d.id}>
                <td style={{ fontWeight: 800 }}>{d.rank}</td>
                <td style={{ fontWeight: 700, whiteSpace: 'nowrap' }}>{d.design}</td>
                <td><Typography variant="caption" sx={{ color: '#cfd8e0' }}>{d.description}</Typography></td>
                <td>
                  {d.discriminates.map((h) => (
                    <Chip key={h} size="small" label={h} sx={{ height: 17, fontSize: 9.5, mr: 0.25, mb: 0.25, bgcolor: '#232a31' }} />
                  ))}
                </td>
                <td>
                  <Chip size="small" label={`${d.cost} · ~${d.expected_days}d`} sx={{
                    height: 18, fontSize: 10, fontWeight: 700,
                    bgcolor: `${COST_COLORS[d.cost]}22`, color: COST_COLORS[d.cost],
                  }} />
                </td>
                <td style={{ fontFamily: 'monospace' }}>{d.information_gain.toFixed(1)}</td>
                <td style={{ fontFamily: 'monospace', fontWeight: 800 }}>{d.priority.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}

export function ReportView({ report, inv, reveal, onReveal }: {
  report: RcaReport;
  inv: Investigation;
  reveal: RevealResponse | null;
  onReveal: () => void;
}) {
  const [showMd, setShowMd] = useState(false);
  const ef = report.executive_finding;
  const confColor = CONFIDENCE_COLORS[ef.confidence];
  return (
    <>
      <SectionCard
        title="Executive finding"
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button size="small" startIcon={<FileJson size={14} />} onClick={() => download(`${report.investigation_id}-report.json`, JSON.stringify(report, null, 2), 'application/json')}>
              Export JSON
            </Button>
            <Button size="small" startIcon={<Download size={14} />} onClick={() => download(`${report.investigation_id}-report.md`, report.markdown, 'text/markdown')}>
              Export MD
            </Button>
          </Box>
        }
      >
        <Typography variant="h6" sx={{ fontWeight: 800, fontSize: 17 }}>
          {ef.conclusion ?? ef.top_hypothesis}
          <Chip size="small" label={`confidence: ${ef.confidence}`} sx={{ ml: 1.5, fontWeight: 800, bgcolor: `${confColor}22`, color: confColor, border: `1px solid ${confColor}55` }} />
        </Typography>
        <Typography variant="body2" sx={{ color: '#aab4be', mt: 0.5 }}>
          {ef.label} — score {ef.score} (gap to runner-up {ef.score_gap_to_runner_up}). Claims under
          investigation: offline {report.claims.offline_delta_pp >= 0 ? '+' : ''}{report.claims.offline_delta_pp}pp vs shadow{' '}
          {report.claims.shadow_delta_pp}pp on {report.claims.metric}.
        </Typography>
      </SectionCard>

      <SectionCard title="Per-stage summary">
        <Box component="table" sx={tableSx}>
          <thead>
            <tr><th>Stage</th><th>Question</th><th>Verdict</th><th>Mismatches / unknowns</th></tr>
          </thead>
          <tbody>
            {report.stage_summaries.map((s) => (
              <tr key={s.stage}>
                <td style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>{s.stage}</td>
                <td><Typography variant="caption" sx={{ color: '#8a949e' }}>{s.headline}</Typography></td>
                <td><StatusPill status={s.verdict === 'clean' ? 'PASS' : s.verdict === 'mismatch' ? 'MISMATCH' : 'UNKNOWN'} /></td>
                <td>
                  <Typography variant="caption">
                    {[...s.mismatches, ...s.unknowns].join('; ') || '—'}
                  </Typography>
                </td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>

      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2 }}>
        {(['containment', 'short_term', 'long_term'] as const).map((tier) => (
          <SectionCard key={tier} title={tier === 'containment' ? 'Containment (now)' : tier === 'short_term' ? 'Short term' : 'Long term'}>
            {report.remediation[tier].map((x) => (
              <Typography key={x} variant="body2" sx={{ fontSize: 12, py: 0.25, color: '#cfd8e0' }}>
                • {x}
              </Typography>
            ))}
          </SectionCard>
        ))}
      </Box>

      {report.acknowledged_unknowns.length ? (
        <SectionCard title="Unknowns explicitly acknowledged during this investigation">
          {report.acknowledged_unknowns.map((e, i) => (
            <Typography key={i} variant="caption" sx={{ display: 'block', color: '#ffb74d' }}>
              ⚠ {e.message} — note: {String((e.data as any)?.note ?? '')}
            </Typography>
          ))}
        </SectionCard>
      ) : null}

      <SectionCard
        title="Raw markdown"
        action={<Button size="small" onClick={() => setShowMd((s) => !s)}>{showMd ? 'hide' : 'show'}</Button>}
      >
        <Collapse in={showMd}>
          <Box component="pre" sx={{ fontSize: 11, fontFamily: 'monospace', whiteSpace: 'pre-wrap', color: '#aab4be', m: 0, maxHeight: 420, overflowY: 'auto' }}>
            {report.markdown}
          </Box>
        </Collapse>
      </SectionCard>

      {inv.training_mode ? (
        <SectionCard title="Training mode — the answer key">
          {reveal ? (
            <Box sx={{ p: 1.25, borderRadius: 1, border: `1px solid ${BORDER}`, bgcolor: '#0d1116' }}>
              <Typography variant="body2" sx={{ fontWeight: 800, color: reveal.cause === ef.top_hypothesis ? '#66bb6a' : '#ef5350' }}>
                Planted cause: {reveal.cause}{' '}
                {reveal.cause === ef.top_hypothesis ? '— your board got it right ✓' : `— your board's top pick was ${ef.top_hypothesis} ✗`}
              </Typography>
              <Typography variant="caption" sx={{ color: '#aab4be' }}>{reveal.explanation}</Typography>
            </Box>
          ) : (
            <Button variant="outlined" size="small" startIcon={<Eye size={14} />} onClick={onReveal}>
              Reveal the planted root cause
            </Button>
          )}
        </SectionCard>
      ) : null}
    </>
  );
}
