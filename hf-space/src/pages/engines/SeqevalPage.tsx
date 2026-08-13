/**
 * Sequential Regression dashboard (/api/seqeval) — decision-first:
 * verdict banner, per-node decision tables (delta + CI, n vs n_eff, decision
 * chips, safety-primary badges), the regression-map heatmap
 * (class × condition), the sequential-evidence chart (log e-value vs n
 * against stopping boundaries — showing exactly when and why the run
 * stopped), the budget funnel, and a run launcher.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import MenuItem from '@mui/material/MenuItem';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ChevronDown, ChevronUp, PlayCircle, ShieldAlert } from 'lucide-react';
import {
  getSeqAttribution,
  getSeqRun,
  listSeqRuns,
  startSeqRun,
  type SeqAttribution,
  type SeqNode,
  type SeqRunState,
} from '../../services/seqeval';
import { getPopulations } from '../../services/megaeval';
import type { PopulationMeta } from '../../types/megaeval';
import { HeatmapGrid, SeriesChart, type HeatCell, type Series } from '../../components/visual/charts';
import { IllustratedEmpty, PanelSkeleton, TileSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, MetricCard, SectionCard, fmtPct } from '../../components/labeleval/shared';
import { HeadCell, InfoDot, Term } from '../../components/help/InfoTip';
import { tokens, verdictColor } from '../../theme';

const DECISION_HELP: Record<string, string> = {
  REGRESSION: 'Anytime-valid evidence of a drop beyond the practical margin in at least one tested node — the gate blocks.',
  PASS: 'Overall + all pre-registered safety primaries proven within the margin (equivalence-style claim) — the gate allows.',
  INSUFFICIENT_EVIDENCE: 'NOT proven equivalent. Expand the budget or report — never treated as a pass.',
};

const EFFECT_PRESETS: Array<{ label: string; effects: Record<string, number> }> = [
  { label: 'clean candidate (no planted effect)', effects: {} },
  { label: 'pedestrian@night −8% (safety primary)', effects: { 'pedestrian|night': -0.08 } },
  { label: 'global −1% (subtle broad drop)', effects: { __global__: -0.01 } },
  { label: 'cyclist@night −5% + global +0.5%', effects: { 'cyclist|night': -0.05, __global__: 0.005 } },
];

function nodeGroup(key: string): string {
  if (key === 'overall') return 'overall';
  const prefix = key.split(':')[0];
  return prefix;
}

function prettyNode(key: string): string {
  return key.replace(/^(class|stratum|difficulty|safety):/, '');
}

// ------------------------------------------------------------ evidence chart

function EvidenceChart({ run, nodeKey }: { run: SeqRunState; nodeKey: string }) {
  const traj = run.trajectories?.[nodeKey];
  if (!traj || !traj.points.length) {
    return (
      <Typography variant="body2" sx={{ color: tokens.color.neutral, p: 2 }}>
        No trajectory recorded for this node (it may have had no samples).
      </Typography>
    );
  }
  const pts = traj.points;
  const last = pts[pts.length - 1];
  const stopped = last.decision !== 'INSUFFICIENT_EVIDENCE';

  const series: Series[] = [
    {
      id: 'regression evidence',
      color: tokens.color.danger,
      width: 2.2,
      points: pts.map((p) => ({ x: p.n, y: p.log_e_regression, meta: p })),
    },
    {
      id: 'pass evidence',
      color: tokens.color.success,
      width: 2.2,
      points: pts.map((p) => ({ x: p.n, y: p.log_e_pass, meta: p })),
    },
  ];

  return (
    <Box>
      <SeriesChart
        series={series}
        height={320}
        xLabel="samples analyzed (n)"
        yLabel="log e-value (accumulated evidence)"
        refLinesY={[
          { y: traj.boundaries.log_e_regression, label: `REGRESSION boundary (${traj.boundaries.log_e_regression.toFixed(2)})`, color: tokens.color.danger },
          { y: traj.boundaries.log_e_pass, label: `PASS boundary (${traj.boundaries.log_e_pass.toFixed(2)})`, color: tokens.color.success },
          { y: 0, label: '', color: tokens.color.borderStrong, dashed: false },
        ]}
        markerX={stopped ? { x: last.n, label: `stopped @ n=${last.n} → ${last.decision}`, color: verdictColor(last.decision) } : null}
      />
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap', mt: 0.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 16, height: 3, bgcolor: tokens.color.danger }} />
          <Typography variant="caption" sx={{ color: tokens.color.textDim }}>evidence FOR a regression (log e)</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 16, height: 3, bgcolor: tokens.color.success }} />
          <Typography variant="caption" sx={{ color: tokens.color.textDim }}>evidence FOR equivalence (pass)</Typography>
        </Box>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
          Whichever line crosses its boundary first ends the test for this node — crossing red blocks, crossing green
          proves equivalence, and running out of budget with neither = insufficient evidence.
        </Typography>
      </Box>
    </Box>
  );
}

// ------------------------------------------------------------ budget funnel

function BudgetFunnel({ run }: { run: SeqRunState }) {
  const b = run.budget;
  if (!b) return null;
  const stages = [
    { label: 'Full population', value: b.full_population, color: tokens.color.textFaint, note: 'every inference unit available' },
    { label: 'Frozen sampling plan', value: b.planned_total, color: tokens.color.info, note: 'allocated before any candidate outcome was seen (hashed)' },
    { label: 'Actually consumed', value: b.samples_used, color: verdictColor(run.decision), note: `stopped early: ${run.stopping_reason ?? '—'}` },
  ];
  if (b.escalation_used > 0) {
    stages.push({ label: 'Escalation', value: b.escalation_used, color: tokens.color.warn, note: 'extra targeted samples for suspect strata' });
  }
  const max = Math.max(...stages.map((s) => s.value), 1);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {stages.map((s) => (
        <Box key={s.label}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" sx={{ color: tokens.color.textDim }}>{s.label}</Typography>
            <Typography variant="caption" sx={{ fontFamily: 'monospace', color: tokens.color.text }}>
              {s.value.toLocaleString()} <span style={{ color: tokens.color.neutral }}>({fmtPct(s.value / max)})</span>
            </Typography>
          </Box>
          <Box sx={{ height: 14, bgcolor: tokens.color.border, borderRadius: 1, overflow: 'hidden' }}>
            <Box sx={{ height: '100%', width: `${Math.max(1.5, (s.value / max) * 100)}%`, bgcolor: s.color, transition: `width ${tokens.motion.slow}` }} />
          </Box>
          <Typography variant="caption" sx={{ color: tokens.color.textFaint, fontSize: 10.5 }}>{s.note}</Typography>
        </Box>
      ))}
      <Typography variant="caption" sx={{ color: tokens.color.neutral, mt: 0.5 }}>
        The run answered with <strong>{fmtPct(b.samples_used / Math.max(b.full_population, 1))}</strong> of the
        population — that saving is the point of sequential testing.
      </Typography>
    </Box>
  );
}

// ------------------------------------------------------------ decision table

function DecisionTable({ nodes, title, help }: { nodes: SeqNode[]; title: string; help: string }) {
  if (!nodes.length) return null;
  return (
    <SectionCard title={title} help={help}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Node</TableCell>
            <TableCell align="right"><HeadCell label="Δ recall" title="Delta estimate" detail="Candidate minus baseline recall in this node, estimated from paired outcomes on identical samples." /></TableCell>
            <TableCell><HeadCell label="95% CS" title="Confidence sequence" detail="Anytime-valid confidence interval — valid at every look, not just the final one. Wide intervals mean the node is under-sampled." /></TableCell>
            <TableCell align="right"><HeadCell label="n / clusters" title="Samples vs effective clusters" detail="Objects analyzed vs container clusters they came from — inference runs on clusters, so n_eff (clusters) is the honest sample size." /></TableCell>
            <TableCell align="right"><HeadCell label="e-value (reg / pass)" title="Evidence values" detail="Accumulated betting-style evidence for regression and for equivalence, each against its stopping threshold." /></TableCell>
            <TableCell align="center">Decision</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {nodes.map((n) => {
            const color = verdictColor(n.decision);
            return (
              <TableRow key={n.node} sx={{ bgcolor: n.decision === 'REGRESSION' ? tokens.color.dangerBg : undefined }}>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>
                  {prettyNode(n.node)}
                  {n.safety_primary ? (
                    <Chip size="small" icon={<ShieldAlert size={11} />} label="safety primary" sx={{ ml: 0.75, height: 17, fontSize: 9.5, bgcolor: 'rgba(239,83,80,0.15)', color: '#ff8a80', fontWeight: 700 }} />
                  ) : null}
                  {n.suspect ? (
                    <Chip size="small" label="suspect" title="Screening flagged this node as suspicious — it received escalated attention." sx={{ ml: 0.5, height: 17, fontSize: 9.5, bgcolor: tokens.color.warnBg, color: tokens.color.warn }} />
                  ) : null}
                </TableCell>
                <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700, color: n.delta_estimate < 0 ? tokens.color.danger : tokens.color.success }}>
                  {n.delta_estimate >= 0 ? '+' : ''}{(n.delta_estimate * 100).toFixed(2)}%
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5, color: tokens.color.textDim, minWidth: 130 }}>
                  [{(n.delta_ci[0] * 100).toFixed(1)}%, {(n.delta_ci[1] * 100).toFixed(1)}%]
                </TableCell>
                <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                  {n.n.toLocaleString()} / {n.n_clusters.toLocaleString()}
                </TableCell>
                <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 11.5, color: tokens.color.textDim }}>
                  {n.e_regression.toFixed(2)}<span style={{ color: tokens.color.textFaint }}>/{n.e_threshold_regression.toFixed(0)}</span>
                  {' · '}
                  {n.e_pass.toFixed(2)}<span style={{ color: tokens.color.textFaint }}>/{n.e_threshold_pass.toFixed(0)}</span>
                </TableCell>
                <TableCell align="center">
                  <Chip size="small" label={n.decision.replace(/_/g, ' ')} title={DECISION_HELP[n.decision]} sx={{ height: 20, fontSize: 10, fontWeight: 800, bgcolor: `${color}22`, color, border: `1px solid ${color}` }} />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </SectionCard>
  );
}

// ------------------------------------------------------------ run launcher

function RunLauncher({ onStarted }: { onStarted: (run: SeqRunState) => void }) {
  const [open, setOpen] = useState(false);
  const [populations, setPopulations] = useState<PopulationMeta[]>([]);
  const [populationId, setPopulationId] = useState('');
  const [preset, setPreset] = useState(1);
  const [targetN, setTargetN] = useState(3000);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPopulations()
      .then((r) => {
        const sorted = [...r.populations].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
        setPopulations(sorted);
        if (sorted.length) setPopulationId((p) => p || sorted[0].population_id);
      })
      .catch(() => undefined);
  }, []);

  const start = () => {
    setStarting(true);
    setError(null);
    startSeqRun({
      population_id: populationId,
      baseline: { model_version: 'baseline-v1', effects: {} },
      candidate: { model_version: `candidate-${EFFECT_PRESETS[preset].effects && Object.keys(EFFECT_PRESETS[preset].effects).length ? 'planted' : 'clean'}`, effects: EFFECT_PRESETS[preset].effects },
      policy: { target_n: targetN },
      sync: true,
    })
      .then(onStarted)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setStarting(false));
  };

  return (
    <SectionCard
      title="Run launcher"
      help="Starts a new sequential evaluation: a sampling plan is frozen (hashed) over the population, then baseline and candidate are compared on identical samples batch by batch until the evidence crosses a boundary or the budget runs out. The synthetic candidate harness lets you plant a known per-stratum effect to watch the engine find it."
      action={
        <Button size="small" endIcon={open ? <ChevronUp size={14} /> : <ChevronDown size={14} />} onClick={() => setOpen((o) => !o)}>
          {open ? 'Hide' : 'New run'}
        </Button>
      }
    >
      <Collapse in={open} unmountOnExit>
        {error ? <ErrorNote error={error} /> : null}
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField select size="small" label="Population" value={populationId} onChange={(e) => setPopulationId(e.target.value)} sx={{ minWidth: 240 }}>
            {populations.map((p) => (
              <MenuItem key={p.population_id} value={p.population_id}>
                <span style={{ fontFamily: 'monospace', fontSize: 12.5 }}>{p.population_id.slice(0, 14)}</span>
                <span style={{ color: tokens.color.neutral, marginLeft: 8, fontSize: 12 }}>{p.num_objects.toLocaleString()} objects</span>
              </MenuItem>
            ))}
          </TextField>
          <TextField select size="small" label="Planted candidate effect" value={preset} onChange={(e) => setPreset(Number(e.target.value))} sx={{ minWidth: 280 }}>
            {EFFECT_PRESETS.map((p, i) => (
              <MenuItem key={p.label} value={i}>{p.label}</MenuItem>
            ))}
          </TextField>
          <TextField size="small" label="Target n" type="number" value={targetN} onChange={(e) => setTargetN(Math.max(500, Number(e.target.value) || 3000))} sx={{ width: 110 }} />
          <Button variant="contained" startIcon={<PlayCircle size={16} />} disabled={!populationId || starting} onClick={start}>
            {starting ? 'Running (sequential)…' : 'Start run'}
          </Button>
        </Box>
        <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mt: 1 }}>
          Runs synchronously and typically finishes in a few seconds — early stopping is the whole point.
        </Typography>
      </Collapse>
      {!open ? (
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
          Launch a new sequential comparison against a megaeval population, optionally planting a known regression to see the engine catch it.
        </Typography>
      ) : null}
    </SectionCard>
  );
}

// ------------------------------------------------------------ page

export default function SeqevalPage() {
  const [runsList, setRunsList] = useState<SeqRunState[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<SeqRunState | null>(null);
  const [attribution, setAttribution] = useState<SeqAttribution | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartNode, setChartNode] = useState('overall');

  const refreshList = useCallback(() => {
    listSeqRuns()
      .then((r) => {
        const sorted = [...r.runs].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
        setRunsList(sorted);
        setRunId((id) => id ?? sorted[0]?.run_id ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(refreshList, [refreshList]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    setRun(null);
    setAttribution(null);
    getSeqRun(runId)
      .then((r) => {
        if (!cancelled) {
          setRun(r);
          setChartNode('overall');
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    getSeqAttribution(runId)
      .then((a) => {
        if (!cancelled) setAttribution(a);
      })
      .catch(() => undefined); // 409 when no attribution — heatmap falls back to nodes
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const onStarted = useCallback((r: SeqRunState) => {
    setRunId(r.run_id);
    refreshList();
  }, [refreshList]);

  // ---------------- regression map (class × condition)
  const heatmap = useMemo(() => {
    const rows: Array<{ cls: string; cond: string; delta: number | null; decision: string; safety: boolean; n: number }> = [];
    const attRows = (attribution?.all_nodes ?? []) as Array<Record<string, unknown>>;
    const source = attRows.length
      ? attRows.map((r) => ({ node: String(r.node), delta: (r.abs_delta as number | null), decision: String(r.decision), safety: Boolean(r.safety_primary), n: Number(r.n ?? 0) }))
      : (run?.nodes ?? []).map((n) => ({ node: n.node, delta: n.delta_estimate, decision: n.decision, safety: n.safety_primary, n: n.n }));
    source.forEach((r) => {
      const m = r.node.match(/^stratum:([^|]+)\|(.+)$/);
      if (!m) return;
      rows.push({ cls: m[1], cond: m[2], delta: r.delta, decision: r.decision, safety: r.safety, n: r.n });
    });
    if (!rows.length) return null;
    const classes = [...new Set(rows.map((r) => r.cls))];
    const conds = [...new Set(rows.map((r) => r.cond))];
    const cells: HeatCell[] = rows.map((r) => {
      const d = r.delta ?? 0;
      const mag = Math.min(1, Math.abs(d) / 0.08);
      const color =
        r.decision === 'REGRESSION'
          ? '#a02824'
          : d < -0.002
            ? `rgba(239,83,80,${0.18 + mag * 0.5})`
            : d > 0.002
              ? `rgba(102,187,106,${0.18 + mag * 0.45})`
              : 'rgba(138,148,158,0.15)';
      return {
        row: r.cls,
        col: r.cond,
        value: d,
        color,
        label: r.delta === null ? '—' : `${d >= 0 ? '+' : ''}${(d * 100).toFixed(1)}${r.decision === 'REGRESSION' ? ' ✕' : ''}`,
        tooltip: (
          <Box>
            <Typography variant="caption" sx={{ fontWeight: 700, color: tokens.color.info, display: 'block' }}>
              {r.cls} × {r.cond} {r.safety ? '· safety primary' : ''}
            </Typography>
            <Typography variant="caption" sx={{ display: 'block' }}>
              Δ recall {r.delta === null ? '—' : `${(d * 100).toFixed(2)}%`} · n={r.n.toLocaleString()} · decision: {r.decision}
            </Typography>
          </Box>
        ),
      };
    });
    return { classes, conds, cells };
  }, [attribution, run]);

  const trajNodes = useMemo(() => Object.keys(run?.trajectories ?? {}), [run]);
  const nodes = run?.nodes ?? [];
  const grouped = useMemo(() => {
    const overall = nodes.filter((n) => nodeGroup(n.node) === 'overall');
    const classes = nodes.filter((n) => nodeGroup(n.node) === 'class');
    const strata = [...nodes.filter((n) => nodeGroup(n.node) === 'stratum')].sort(
      (a, b) => Number(b.safety_primary) - Number(a.safety_primary) || a.delta_estimate - b.delta_estimate
    );
    const other = nodes.filter((n) => !['overall', 'class', 'stratum'].includes(nodeGroup(n.node)));
    return { overall, classes, strata, other };
  }, [nodes]);

  const decisionColor = verdictColor(run?.decision);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}

      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <TextField select size="small" label="Run" value={runId ?? ''} onChange={(e) => setRunId(e.target.value)} sx={{ minWidth: 300 }}>
          {(runsList ?? []).map((r) => (
            <MenuItem key={r.run_id} value={r.run_id}>
              <span style={{ fontFamily: 'monospace', fontSize: 12.5 }}>{r.run_id}</span>
              <Chip size="small" label={r.decision ?? r.status} sx={{ ml: 1, height: 17, fontSize: 9.5, bgcolor: `${verdictColor(r.decision ?? r.status)}22`, color: verdictColor(r.decision ?? r.status) }} />
            </MenuItem>
          ))}
        </TextField>
        {run ? (
          <Typography variant="caption" sx={{ color: tokens.color.neutral, fontFamily: 'monospace' }}>
            {run.baseline.model_version} → {run.candidate.model_version} · pop {run.population_id.slice(0, 12)} · metric {run.policy.metric} · margin ±{(run.policy.delta_margin * 100).toFixed(1)}%
          </Typography>
        ) : null}
      </Box>

      {loading ? <TileSkeleton n={3} /> : null}

      {runsList !== null && runsList.length === 0 && !loading ? (
        <SectionCard title="Sequential regression detection">
          <IllustratedEmpty
            art="gauge"
            title="No sequential runs yet"
            message="Launch one below: the engine freezes a stratified sampling plan over a megaeval population, then compares candidate vs baseline on identical samples until anytime-valid evidence settles the question — usually with a fraction of the population."
          />
        </SectionCard>
      ) : null}

      {run ? (
        <>
          {/* verdict banner */}
          <Alert
            severity={run.decision === 'PASS' ? 'success' : run.decision === 'REGRESSION' ? 'error' : 'warning'}
            variant="outlined"
            sx={{ borderWidth: 2, '& .MuiAlert-message': { width: '100%' } }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
              <Typography variant="h6" sx={{ fontWeight: 900, color: decisionColor }}>
                {(run.decision ?? run.status).replace(/_/g, ' ')}
              </Typography>
              <Chip size="small" label={`gate: ${run.gate ?? '—'}`} sx={{ fontWeight: 700, bgcolor: tokens.color.surfaceRaised }} />
              <Typography variant="body2" sx={{ color: tokens.color.textDim }}>
                {run.message ?? ''} {run.stopping_reason ? `(stopping reason: ${run.stopping_reason.replace(/_/g, ' ')})` : ''}
              </Typography>
              <InfoDot title={run.decision ?? 'decision'} detail={DECISION_HELP[run.decision ?? ''] ?? 'Run still in progress.'} />
            </Box>
          </Alert>

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <MetricCard label="Samples used" value={run.budget?.samples_used.toLocaleString() ?? '—'} sub={`of ${run.budget?.planned_total.toLocaleString()} planned`} accent={tokens.color.info} info="Objects actually consumed before the evidence crossed a stopping boundary." />
            <MetricCard label="Population fraction" value={run.budget ? fmtPct(run.budget.fraction_of_population) : '—'} sub={`${run.budget?.full_population.toLocaleString()} total`} info="How little of the full population the sequential design needed to reach its decision." />
            <MetricCard label="Plan hash" value={<span style={{ fontSize: 15, fontFamily: 'monospace' }}>{run.plan?.plan_hash.slice(0, 12) ?? '—'}</span>} sub="frozen before outcomes" info="The sampling plan is hashed before any candidate outcome is observed, so allocations cannot chase noise." />
            <MetricCard label="Safety primaries" value={run.policy.safety_primaries.length} sub={run.policy.safety_primaries.join(', ')} info="Pre-registered strata that must individually prove equivalence for a PASS — a global average cannot hide them." />
          </Box>

          {/* THE key visual */}
          <SectionCard
            title={
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
                Sequential evidence — when and why the run stopped
                <InfoDot title="Sequential evidence chart" detail="Each point is one analysis batch. The red line accumulates evidence FOR a regression (log e-value), the green line FOR equivalence. Dashed horizontal lines are the pre-committed stopping boundaries (derived from the alpha allocated to this node). The moment a line crosses its boundary, the decision for this node is final — valid despite the repeated looks (anytime-valid inference)." />
              </Box>
            }
          >
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
              {trajNodes.map((k) => (
                <Chip
                  key={k}
                  size="small"
                  label={prettyNode(k)}
                  onClick={() => setChartNode(k)}
                  sx={{
                    height: 22,
                    fontSize: 11,
                    cursor: 'pointer',
                    fontFamily: 'monospace',
                    bgcolor: chartNode === k ? tokens.color.infoBg : tokens.color.surfaceRaised,
                    color: chartNode === k ? tokens.color.info : tokens.color.textDim,
                    border: `1px solid ${chartNode === k ? tokens.color.info : tokens.color.border}`,
                  }}
                />
              ))}
            </Box>
            <EvidenceChart run={run} nodeKey={chartNode} />
          </SectionCard>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <SectionCard
              title="Regression map — class × condition"
              sx={{ flex: '2 1 480px' }}
              help="Per-stratum recall delta (candidate − baseline). Red = drop, green = gain; ✕ marks strata with a confirmed anytime-valid regression. This is where 'the model regressed' becomes 'pedestrians at night regressed 8%'."
            >
              {heatmap ? (
                <HeatmapGrid rows={heatmap.classes} cols={heatmap.conds} cells={heatmap.cells} cellH={38} />
              ) : (
                <Typography variant="body2" sx={{ color: tokens.color.neutral }}>
                  No per-stratum attribution available for this run.
                </Typography>
              )}
            </SectionCard>

            <SectionCard
              title="Budget funnel"
              sx={{ flex: '1 1 320px' }}
              help="Planned vs consumed sampling budget against the full population. The frozen plan bounds worst-case cost; early stopping usually spends far less."
            >
              <BudgetFunnel run={run} />
            </SectionCard>
          </Box>

          <DecisionTable
            nodes={[...grouped.overall, ...grouped.classes]}
            title="Decisions — overall & per class"
            help="Level-1 (overall) and level-2 (class) nodes with their paired delta estimates, anytime-valid confidence sequences, and e-value evidence against thresholds. Alpha is split across the hierarchy so all claims hold simultaneously."
          />
          <DecisionTable
            nodes={grouped.strata}
            title="Decisions — strata (class × condition)"
            help="The finest tested granularity. Safety primaries are pre-registered and must individually prove equivalence for the run to PASS; they are sorted first."
          />
          {run.sanity ? (
            <Typography variant="caption" sx={{ color: tokens.color.textFaint, fontFamily: 'monospace' }}>
              sanity: plan {String((run.sanity as Record<string, unknown>).plan_hash ?? '').slice(0, 12)} · dataset {String((run.sanity as Record<string, unknown>).dataset_fingerprint ?? '')} · smoke discordance {String((run.sanity as Record<string, unknown>).smoke_discordance ?? '—')} · <Term k="seqeval_anytime_valid">anytime-valid</Term> inference on container clusters
            </Typography>
          ) : null}
        </>
      ) : null}

      {runId && !run && !loading ? <PanelSkeleton rows={6} /> : null}

      <RunLauncher onStarted={onStarted} />
    </Box>
  );
}
