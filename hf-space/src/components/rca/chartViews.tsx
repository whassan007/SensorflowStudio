/** Visualization-heavy stage views: distribution shift, conditional
 * performance heatmap (Simpson's paradox surface), paired transition matrix,
 * significance CI plot, feature parity ranked deltas. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import {
  BORDER,
  DeltaBar,
  Explainer,
  KV,
  PsiBadge,
  SectionCard,
  StatusPill,
  tableSx,
} from './common';

// ------------------------------------------------ stage 3: distribution shift

export function ShiftView({ data }: { data: any }) {
  if (data.volume_ok === false) {
    return (
      <>
        <Explainer text={data.explainer} />
        <SectionCard title="Distribution shift">
          <Typography variant="body2" sx={{ color: '#ffb74d' }}>
            Too little data to assess shift: PSI estimates are upward-biased at
            this volume, so shift can be neither confirmed nor excluded. This
            is itself a finding — record it and proceed with the unknown
            acknowledged.
          </Typography>
        </SectionCard>
      </>
    );
  }
  const renderRows = (rows: any[]) => (
    <Box component="table" sx={tableSx}>
      <thead>
        <tr>
          <th>Dimension</th>
          <th>PSI (practical effect)</th>
          <th>JS div</th>
          <th>Test</th>
          <th>p-value (see caveat)</th>
          <th>Means (off → sh)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r: any) => (
          <tr key={r.dimension}>
            <td style={{ fontWeight: 600 }}>{r.dimension}</td>
            <td><PsiBadge psi={r.psi} /></td>
            <td style={{ fontFamily: 'monospace' }}>{r.js.toFixed(3)}</td>
            <td style={{ fontFamily: 'monospace' }}>{r.test}={r.stat.toFixed(2)}</td>
            <td style={{ fontFamily: 'monospace', color: r.p_value < 0.05 ? '#ffb74d' : '#8a949e' }}>
              {r.p_value < 0.001 ? '<0.001' : r.p_value.toFixed(3)}
            </td>
            <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
              {r.offline_mean !== undefined
                ? `${r.offline_mean.toFixed(1)} → ${r.shadow_mean.toFixed(1)}`
                : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </Box>
  );
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard title="Population mix (segment dimensions)">
        {renderRows(data.segments)}
      </SectionCard>
      <SectionCard title="Feature distributions" subtitle="An isolated large shift in ONE feature while segment mixes stay stable points at a pipeline artifact, not a population change — cross-check with Feature Parity (stage 7).">
        {renderRows(data.features)}
      </SectionCard>
      <Typography variant="caption" sx={{ color: '#ffb74d' }}>
        ⚠ {data.caveat}
      </Typography>
    </>
  );
}

// ------------------------------------------ stage 4: conditional performance

function heatColor(v: number, cap = 12): string {
  const t = Math.max(-1, Math.min(1, v / cap));
  if (t >= 0) {
    const a = Math.round(40 + t * 150);
    return `rgba(102, 187, 106, ${a / 255})`;
  }
  const a = Math.round(40 + -t * 170);
  return `rgba(239, 83, 80, ${a / 255})`;
}

export function ConditionalHeatmapView({ data }: { data: any }) {
  const rows: any[] = data.rows;
  const scenes = Array.from(new Set(rows.map((r) => r.scene))).sort();
  const times = Array.from(new Set(rows.map((r) => r.time_of_day))).sort();
  const cell = (scene: string, tod: string) => rows.find((r) => r.scene === scene && r.time_of_day === tod);

  const grid = (which: 'offline_delta_pp' | 'shadow_delta_pp', title: string) => (
    <Box>
      <Typography variant="caption" sx={{ fontWeight: 800, color: '#aab4be', display: 'block', mb: 0.5 }}>
        {title}
      </Typography>
      <Box component="table" sx={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th />
            {times.map((t) => (
              <Box component="th" key={t} sx={{ fontSize: 11, color: '#8a949e', px: 1, pb: 0.5 }}>{t}</Box>
            ))}
          </tr>
        </thead>
        <tbody>
          {scenes.map((s) => (
            <tr key={s}>
              <Box component="td" sx={{ fontSize: 11, color: '#8a949e', pr: 1, textAlign: 'right' }}>{s}</Box>
              {times.map((t) => {
                const c = cell(s, t);
                const v = c ? c[which] : 0;
                return (
                  <td key={t}>
                    <Tooltip
                      arrow
                      title={c ? `${s}/${t}: offline ${c.offline_delta_pp.toFixed(1)}pp · shadow ${c.shadow_delta_pp.toFixed(1)}pp · n=${c.shadow_n} · ${c.interpretation}` : ''}
                    >
                      <Box
                        sx={{
                          width: 86,
                          height: 40,
                          m: 0.25,
                          borderRadius: 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: heatColor(v),
                          border: c?.interpretation === 'sign_flip' ? '2px dashed #ffb74d' : `1px solid ${BORDER}`,
                          fontFamily: 'monospace',
                          fontSize: 12,
                          fontWeight: 700,
                        }}
                      >
                        {v >= 0 ? '+' : ''}{v.toFixed(1)}
                      </Box>
                    </Tooltip>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </Box>
    </Box>
  );

  const agg = data.aggregate;
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard
        title="Segment performance heatmap (B − A, pp)"
        subtitle="Dashed amber border = sign flip between environments. If per-segment colors AGREE while the aggregates disagree, the mix — not the model — flipped the sign (Simpson's paradox)."
      >
        <Box sx={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
          {grid('offline_delta_pp', `OFFLINE (aggregate ${agg.offline_delta_pp >= 0 ? '+' : ''}${agg.offline_delta_pp.toFixed(1)}pp)`)}
          {grid('shadow_delta_pp', `SHADOW (aggregate ${agg.shadow_delta_pp >= 0 ? '+' : ''}${agg.shadow_delta_pp.toFixed(1)}pp)`)}
        </Box>
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
          Within-segment sign consistency: {(agg.sign_consistency * 100).toFixed(0)}% of scored volume.
        </Typography>
      </SectionCard>
      <SectionCard title="Segment detail">
        <Box component="table" sx={tableSx}>
          <thead>
            <tr>
              <th>Segment</th><th>Offline Δ</th><th>Shadow Δ</th><th>Vol (off/sh)</th><th>Shift ×</th><th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.segment}>
                <td style={{ fontWeight: 600 }}>{r.segment}</td>
                <td><DeltaBar value={r.offline_delta_pp} max={18} width={100} /></td>
                <td><DeltaBar value={r.shadow_delta_pp} max={18} width={100} /></td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{r.offline_n}/{r.shadow_n}</td>
                <td style={{ fontFamily: 'monospace' }}>{r.shift_ratio.toFixed(2)}</td>
                <td>
                  <Chip size="small" label={r.interpretation} sx={{
                    height: 18, fontSize: 10,
                    bgcolor: r.interpretation === 'sign_flip' ? '#ffb74d22' : '#232a31',
                    color: r.interpretation === 'sign_flip' ? '#ffb74d' : '#aab4be',
                  }} />
                </td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}

// ------------------------------------------------ stage 5: paired comparison

function TransitionMatrix({ m, title }: { m: any; title: string }) {
  const cellSx = (bg: string, color: string) => ({
    width: 130, height: 62, borderRadius: 1, m: 0.25, p: 1,
    bgcolor: bg, border: `1px solid ${BORDER}`,
    display: 'flex', flexDirection: 'column', justifyContent: 'center', color,
  });
  return (
    <Box>
      <Typography variant="caption" sx={{ fontWeight: 800, color: '#aab4be' }}>{title} (n={m.n.toLocaleString()})</Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'auto auto auto', alignItems: 'center', mt: 0.5 }}>
        <Box />
        <Typography variant="caption" sx={{ textAlign: 'center', color: '#8a949e' }}>B correct</Typography>
        <Typography variant="caption" sx={{ textAlign: 'center', color: '#8a949e' }}>B incorrect</Typography>
        <Typography variant="caption" sx={{ color: '#8a949e', pr: 0.5 }}>A correct</Typography>
        <Box sx={cellSx('#20302144', '#a5d6a7')}>
          <Typography variant="caption">both correct</Typography>
          <Typography sx={{ fontFamily: 'monospace', fontWeight: 800 }}>{m.both_correct.toLocaleString()}</Typography>
        </Box>
        <Box sx={cellSx('#ef535022', '#ef9a9a')}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>REGRESSIONS ↓</Typography>
          <Typography sx={{ fontFamily: 'monospace', fontWeight: 800 }}>{m.regressions.toLocaleString()}</Typography>
        </Box>
        <Typography variant="caption" sx={{ color: '#8a949e', pr: 0.5 }}>A incorrect</Typography>
        <Box sx={cellSx('#66bb6a22', '#a5d6a7')}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>improvements ↑</Typography>
          <Typography sx={{ fontFamily: 'monospace', fontWeight: 800 }}>{m.improvements.toLocaleString()}</Typography>
        </Box>
        <Box sx={cellSx('#232a3155', '#90a4ae')}>
          <Typography variant="caption">both wrong</Typography>
          <Typography sx={{ fontFamily: 'monospace', fontWeight: 800 }}>{m.both_wrong.toLocaleString()}</Typography>
        </Box>
      </Box>
      <Typography variant="caption" sx={{ color: '#8a949e' }}>net {m.net_pp >= 0 ? '+' : ''}{m.net_pp.toFixed(1)}pp</Typography>
    </Box>
  );
}

export function PairedTransitionView({ data }: { data: any }) {
  const drill = (rows: any[], label: string) => (
    <SectionCard title={`Transitions by ${label}`}>
      <Box component="table" sx={tableSx}>
        <thead>
          <tr>
            <th>{label}</th><th>Regressions (A✓→B✗)</th><th>Improvements (A✗→B✓)</th>
            {rows[0]?.lift !== undefined ? <th>Unit share</th> : null}
            {rows[0]?.lift !== undefined ? <th>Regression lift</th> : null}
            <th>Net</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.segment}>
              <td style={{ fontWeight: 600, fontFamily: r.segment.startsWith('[') ? 'monospace' : undefined }}>{r.segment}</td>
              <td style={{ fontFamily: 'monospace', color: '#ef9a9a' }}>{r.regressions}</td>
              <td style={{ fontFamily: 'monospace', color: '#a5d6a7' }}>{r.improvements}</td>
              {r.lift !== undefined ? <td style={{ fontFamily: 'monospace' }}>{(r.unit_share * 100).toFixed(0)}%</td> : null}
              {r.lift !== undefined ? (
                <td>
                  <Chip size="small" label={`${r.lift.toFixed(1)}×`} sx={{
                    height: 18, fontSize: 10, fontWeight: 700,
                    bgcolor: r.lift >= 1.5 ? '#ef535022' : '#232a31',
                    color: r.lift >= 1.5 ? '#ef5350' : '#aab4be',
                  }} />
                </td>
              ) : null}
              <td><DeltaBar value={r.net_pp} max={15} width={90} /></td>
            </tr>
          ))}
        </tbody>
      </Box>
    </SectionCard>
  );
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard title="Error-transition matrix — same units, both models">
        <Box sx={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <TransitionMatrix m={data.shadow} title="SHADOW (scored sample)" />
          <TransitionMatrix m={data.offline} title="OFFLINE (eval set)" />
        </Box>
      </SectionCard>
      {drill(data.by_band, `serving-confidence band (${data.conf_col})`)}
      {drill(data.by_segment, 'segment')}
      {drill(data.by_class, 'object class')}
    </>
  );
}

// ---------------------------------------------------- stage 6: significance

function CiPlot({ res, marginPp }: { res: any; marginPp: number }) {
  // Scale: pp values mapped onto an SVG x-axis.
  const w = 560; const h = 90;
  const lo = Math.min(res.ci_low * 100, -marginPp) - 1.2;
  const hi = Math.max(res.ci_high * 100, marginPp) + 1.2;
  const x = (v: number) => ((v - lo) / (hi - lo)) * (w - 40) + 20;
  const y = 38;
  return (
    <svg width={w} height={h}>
      {/* practical-regression zone */}
      <rect x={20} y={y - 16} width={x(-marginPp) - 20} height={32} fill="#ef5350" opacity={0.10} />
      <line x1={x(-marginPp)} y1={y - 22} x2={x(-marginPp)} y2={y + 22} stroke="#ef5350" strokeDasharray="4 3" />
      <text x={x(-marginPp)} y={y - 26} fill="#ef9a9a" fontSize={10} textAnchor="middle">
        −{marginPp.toFixed(1)}pp practical margin
      </text>
      <line x1={x(0)} y1={y - 22} x2={x(0)} y2={y + 22} stroke="#5c666f" />
      <text x={x(0)} y={y + 34} fill="#8a949e" fontSize={10} textAnchor="middle">0</text>
      {/* CI */}
      <line x1={x(res.ci_low * 100)} y1={y} x2={x(res.ci_high * 100)} y2={y} stroke="#4fc3f7" strokeWidth={4} strokeLinecap="round" />
      <circle cx={x(res.delta * 100)} cy={y} r={5.5} fill="#4fc3f7" />
      <text x={x(res.delta * 100)} y={y + 20} fill="#4fc3f7" fontSize={11} textAnchor="middle" fontFamily="monospace">
        {res.delta * 100 >= 0 ? '+' : ''}{(res.delta * 100).toFixed(2)}pp
      </text>
      <text x={x(res.ci_low * 100)} y={y - 10} fill="#8a949e" fontSize={10} textAnchor="middle" fontFamily="monospace">
        {(res.ci_low * 100).toFixed(2)}
      </text>
      <text x={x(res.ci_high * 100)} y={y - 10} fill="#8a949e" fontSize={10} textAnchor="middle" fontFamily="monospace">
        {(res.ci_high * 100).toFixed(2)}
      </text>
    </svg>
  );
}

const OUTCOME_LABEL: Record<string, { label: string; color: string }> = {
  significant_regression: { label: 'SIGNIFICANT REGRESSION', color: '#ef5350' },
  no_significant_difference: { label: 'NO SIGNIFICANT DIFFERENCE', color: '#66bb6a' },
  insufficient_evidence: { label: 'INSUFFICIENT EVIDENCE', color: '#ffb74d' },
};

export function SignificanceView({ data }: { data: any }) {
  const o = OUTCOME_LABEL[data.outcome];
  const sp = data.shadow_paired;
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard
        title="Paired shadow delta vs practical-significance margin"
        action={<Chip size="small" label={o.label} sx={{ fontWeight: 800, bgcolor: `${o.color}22`, color: o.color, border: `1px solid ${o.color}66` }} />}
      >
        <CiPlot res={sp} marginPp={data.practical_margin_pp} />
        <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap', mt: 1 }}>
          <KV k="Units / clusters" v={`${sp.n.toLocaleString()} / ${sp.n_clusters.toLocaleString()}`} />
          <KV k="ICC (paired diffs)" v={sp.icc.toFixed(3)} />
          <KV k="Design effect" v={sp.design_effect.toFixed(2)} />
          <KV k="Effective n" v={sp.effective_n.toLocaleString()} />
        </Box>
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 0.5 }}>
          Engine: {data.engine} — {data.engine_note}
        </Typography>
      </SectionCard>
      {data.seqeval ? (
        <SectionCard title="Anytime-valid complement (sensorflow.seqeval)">
          <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <KV k="Sequential decision" v={<StatusPill status={data.seqeval.decision === 'REGRESSION' ? 'MISMATCH' : data.seqeval.decision === 'PASS' ? 'PASS' : 'UNKNOWN'} />} />
            <KV k="Confidence sequence" v={`[${(data.seqeval.delta_ci[0] * 100).toFixed(2)}, ${(data.seqeval.delta_ci[1] * 100).toFixed(2)}]pp`} />
            <KV k="e-value (regression)" v={data.seqeval.e_regression} />
            <KV k="P(regression | data)" v={data.seqeval.bayes_p_regression ?? '—'} />
          </Box>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>{data.seqeval.note}</Typography>
        </SectionCard>
      ) : null}
      <SectionCard title="Per-model accuracy (Wilson 95% CI)">
        <Box component="table" sx={tableSx}>
          <thead>
            <tr><th>Measurement</th><th>Rate</th><th>95% CI</th><th>n</th></tr>
          </thead>
          <tbody>
            {Object.entries(data.wilson as Record<string, any>).map(([k, v]) => (
              <tr key={k}>
                <td style={{ fontWeight: 600 }}>{k.replace('_', ' · model ')}</td>
                <td style={{ fontFamily: 'monospace' }}>{(v.rate * 100).toFixed(2)}%</td>
                <td style={{ fontFamily: 'monospace' }}>[{(v.ci_low * 100).toFixed(2)}, {(v.ci_high * 100).toFixed(2)}]</td>
                <td style={{ fontFamily: 'monospace' }}>{v.n.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}

// --------------------------------------------------- stage 7: feature parity

export function FeatureParityView({ data }: { data: any }) {
  const maxSmd = Math.max(...data.rows.map((r: any) => r.within_segment_smd), 1);
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard
        title="Feature parity — ranked by within-segment SMD"
        subtitle="Within-segment standardized mean difference controls for population mix: a value that still differs INSIDE matched segments is a pipeline difference, not a world change."
      >
        <Box component="table" sx={tableSx}>
          <thead>
            <tr>
              <th>Feature</th><th>Within-segment SMD</th><th>Raw SMD</th>
              <th>Mean (off → sh)</th><th>Median ratio</th><th>Missing (off / sh)</th><th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r: any) => (
              <tr key={r.feature} style={r.skew_flag ? { background: '#ef535011' } : undefined}>
                <td style={{ fontWeight: 700 }}>{r.feature}</td>
                <td>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                    <Box sx={{ width: 130, height: 9, bgcolor: '#0d1116', borderRadius: 1, border: `1px solid ${BORDER}` }}>
                      <Box sx={{
                        width: `${Math.min(100, (r.within_segment_smd / maxSmd) * 100)}%`,
                        height: '100%',
                        bgcolor: r.skew_flag ? '#ef5350' : '#4fc3f7',
                        borderRadius: 1,
                      }} />
                    </Box>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>{r.within_segment_smd.toFixed(2)}</Typography>
                  </Box>
                </td>
                <td style={{ fontFamily: 'monospace' }}>{r.smd.toFixed(2)}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{r.offline_mean.toFixed(1)} → {r.shadow_mean.toFixed(1)}</td>
                <td style={{ fontFamily: 'monospace', fontWeight: r.skew_flag ? 800 : 400, color: r.skew_flag ? '#ef5350' : undefined }}>
                  ×{r.median_ratio.toFixed(2)}
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
                  {(r.offline_missing * 100).toFixed(1)}% / {(r.shadow_missing * 100).toFixed(1)}%
                </td>
                <td><StatusPill status={r.skew_flag ? 'MISMATCH' : 'PASS'} /></td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}
