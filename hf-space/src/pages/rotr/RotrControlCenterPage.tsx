/**
 * ROTR Control Center — right-of-the-road violation detection, triage,
 * evaluation & training. Four audiences, four views:
 *   Executive      — headline metrics, gate state, regression status
 *   Engineering    — taxonomy distributions, attribution heatmap, clusters,
 *                    model comparison, consequence replay (BEV schematic)
 *   Data           — coverage, rare patterns, HITL queue, training
 *                    candidates with role badges + contamination guard,
 *                    regression suite
 *   Infrastructure — measured throughput + delegation stats; unmeasured
 *                    figures are flagged honestly, never invented
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Select from '@mui/material/Select';
import Tab from '@mui/material/Tab';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import RotrAttributionHeatmap from '../../components/rotr/RotrAttributionHeatmap';
import RotrBevReplay from '../../components/rotr/RotrBevReplay';
import RotrQueryBuilder from '../../components/rotr/RotrQueryBuilder';
import * as api from '../../services/rotr';
import type {
  RotrAttributionMatrix,
  RotrConsequenceDetail,
  RotrHITLReview,
  RotrQueryResponse,
  RotrRegressionResult,
  RotrRunListItem,
  RotrRunSummary,
  RotrStopshipPolicy,
  RotrSuite,
  RotrTrainingCandidate,
} from '../../types/rotr';

const MODEL_VERSIONS = ['stack-v1', 'stack-v2-improved', 'stack-v3-planning-regression'];

const CONSEQUENCE_COLOR: Record<string, string> = {
  SAFETY_CRITICAL: '#b71c1c',
  PLANNER_INTERVENTION: '#e65100',
  DEGRADED_COMFORT: '#f9a825',
  NO_MATERIAL_CONSEQUENCE: '#37474f',
};

const ROLE_COLOR: Record<string, string> = {
  REGRESSION: '#6a1b9a',
  LAUNCH: '#4527a0',
  TRAIN: '#1565c0',
  VALIDATION: '#00695c',
  TEST: '#5d4037',
  MONITORING: '#37474f',
};

function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`;
}

function SectionCard({ title, children, subtitle }: { title: string; subtitle?: string; children: ReactNode }) {
  return (
    <Paper sx={{ p: 2, bgcolor: '#161b22', border: '1px solid #232a31', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: subtitle ? 0 : 1.25 }}>
        {title}
      </Typography>
      {subtitle ? (
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1.25 }}>
          {subtitle}
        </Typography>
      ) : null}
      {children}
    </Paper>
  );
}

function Tile({ label, value, hint, color }: { label: string; value: string; hint?: string; color?: string }) {
  return (
    <Paper sx={{ p: 1.5, bgcolor: '#161b22', border: '1px solid #232a31', minWidth: 148, flex: 1 }}>
      <Typography variant="caption" sx={{ color: '#8a949e' }}>
        {label}
      </Typography>
      <Typography variant="h5" sx={{ fontWeight: 800, color: color ?? '#e6edf3' }}>
        {value}
      </Typography>
      {hint ? (
        <Typography variant="caption" sx={{ color: '#6b7681' }}>
          {hint}
        </Typography>
      ) : null}
    </Paper>
  );
}

function DistBar({ counts, total }: { counts: Record<string, number>; total: number }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return (
    <Box>
      {entries.map(([k, n]) => (
        <Box key={k} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.4 }}>
          <Typography variant="caption" sx={{ width: 190, color: '#c9d1d9', fontFamily: 'monospace', fontSize: 11 }}>
            {k}
          </Typography>
          <Box sx={{ flex: 1, bgcolor: '#0d1117', borderRadius: 1, height: 12 }}>
            <Box sx={{ width: `${(n / Math.max(total, 1)) * 100}%`, bgcolor: '#4fc3f7', height: 12, borderRadius: 1 }} />
          </Box>
          <Typography variant="caption" sx={{ width: 26, textAlign: 'right', color: '#8a949e' }}>
            {n}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

export default function RotrControlCenterPage() {
  const [tab, setTab] = useState(0);
  const [runs, setRuns] = useState<RotrRunListItem[]>([]);
  const [runId, setRunId] = useState<string>('');
  const [run, setRun] = useState<RotrRunSummary | null>(null);
  const [matrix, setMatrix] = useState<RotrAttributionMatrix | null>(null);
  const [allViolations, setAllViolations] = useState<RotrQueryResponse | null>(null);
  const [selectedVid, setSelectedVid] = useState<string | null>(null);
  const [consequence, setConsequence] = useState<RotrConsequenceDetail | null>(null);
  const [queryResult, setQueryResult] = useState<RotrQueryResponse | null>(null);
  const [regressions, setRegressions] = useState<RotrRegressionResult[]>([]);
  const [baselineId, setBaselineId] = useState<string>('');
  const [hitl, setHitl] = useState<RotrHITLReview[]>([]);
  const [candidates, setCandidates] = useState<RotrTrainingCandidate[]>([]);
  const [suite, setSuite] = useState<RotrSuite | null>(null);
  const [policy, setPolicy] = useState<RotrStopshipPolicy | null>(null);
  const [overrideReason, setOverrideReason] = useState('');
  const [nScenarios, setNScenarios] = useState(28);
  const [seed, setSeed] = useState(7);
  const [modelVersion, setModelVersion] = useState('stack-v1');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [lastRunWallMs, setLastRunWallMs] = useState<number | null>(null);

  const loadRun = useCallback(async (id: string) => {
    const [summary, m, all, queue] = await Promise.all([
      api.getRun(id),
      api.getAttributionMatrix(id),
      api.structuredQuery({ run_id: id }),
      api.getHitlQueue(id),
    ]);
    setRun(summary);
    setMatrix(m);
    setAllViolations(all);
    setHitl(queue);
    setSelectedVid(null);
    setConsequence(null);
    setQueryResult(null);
  }, []);

  const refreshGlobal = useCallback(async () => {
    const [rs, regs, cands, s, pol] = await Promise.all([
      api.listRuns(),
      api.listRegressions(),
      api.listCandidates(),
      api.getSuite(),
      api.getStopshipPolicy(),
    ]);
    setRuns(rs);
    setRegressions(regs);
    setCandidates(cands);
    setSuite(s);
    setPolicy(pol);
    return rs;
  }, []);

  useEffect(() => {
    refreshGlobal()
      .then((rs) => {
        if (rs.length && !runId) {
          setRunId(rs[rs.length - 1].run_id);
          return loadRun(rs[rs.length - 1].run_id);
        }
        return undefined;
      })
      .catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generateAndRun = async () => {
    setBusy(true);
    setError(null);
    try {
      const start = performance.now();
      const summary = await api.createRun({ n_scenarios: nScenarios, seed, model_version: modelVersion });
      setLastRunWallMs(performance.now() - start);
      await refreshGlobal();
      setRunId(summary.run_id);
      await loadRun(summary.run_id);
      setNotice(`run ${summary.run_id} complete: ${summary.n_violations} violations on ${summary.n_scenarios} scenarios`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const selectRun = async (id: string) => {
    setRunId(id);
    setBusy(true);
    try {
      await loadRun(id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const selectViolation = async (vid: string) => {
    setSelectedVid(vid);
    try {
      setConsequence(await api.getConsequence(runId, vid));
    } catch (e) {
      setError(String(e));
    }
  };

  const runQuery = async (text: string | undefined, filters: Record<string, string>) => {
    setBusy(true);
    try {
      setQueryResult(await api.structuredQuery({ run_id: runId, text, filters }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const doRegression = async () => {
    if (!baselineId || !runId) return;
    setBusy(true);
    try {
      await api.runRegression({ baseline_run_id: baselineId, candidate_run_id: runId });
      setRegressions(await api.listRegressions());
      setNotice('regression comparison complete (statistics delegated to seqeval)');
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const act = async (review: RotrHITLReview, action: 'VALIDATE' | 'REJECT') => {
    try {
      await api.hitlAction({
        run_id: runId,
        review_id: review.review_id,
        action,
        actor: 'studio-reviewer',
        notes: action === 'VALIDATE' ? 'confirmed via control center' : 'rejected via control center',
      });
      setHitl(await api.getHitlQueue(runId));
      setCandidates(await api.listCandidates());
      setSuite(await api.getSuite());
      setNotice(action === 'VALIDATE' ? 'validated → regression artifact created (role REGRESSION, protected)' : 'review rejected');
    } catch (e) {
      setError(String(e));
    }
  };

  const promote = async (c: RotrTrainingCandidate) => {
    try {
      await api.promoteCandidate({ candidate_id: c.candidate_id, actor: 'studio-reviewer' });
      setCandidates(await api.listCandidates());
      setNotice(`candidate ${c.candidate_id} promoted to training`);
    } catch (e) {
      setError(`contamination guard: ${String(e)}`);
    }
  };

  const override = async (c: RotrTrainingCandidate) => {
    try {
      await api.overrideCandidate({ candidate_id: c.candidate_id, actor: 'safety-lead', reason: overrideReason });
      setCandidates(await api.listCandidates());
      setNotice('governance override recorded (audited)');
    } catch (e) {
      setError(String(e));
    }
  };

  const distributions = useMemo(() => {
    const axes = ['legality', 'vulnerability', 'visibility', 'behavior', 'traffic_control'] as const;
    const out: Record<string, Record<string, number>> = {};
    for (const axis of axes) out[axis] = {};
    const layerCounts: Record<string, number> = {};
    const consCounts: Record<string, number> = {};
    for (const v of allViolations?.results ?? []) {
      for (const axis of axes) {
        const val = v.taxonomy[axis] ?? '?';
        out[axis][val] = (out[axis][val] ?? 0) + 1;
      }
      const l = v.primary_layer ?? 'unattributed (HITL)';
      layerCounts[l] = (layerCounts[l] ?? 0) + 1;
      const c = v.consequence_class ?? 'UNCLASSIFIED';
      consCounts[c] = (consCounts[c] ?? 0) + 1;
    }
    return { axes: out, layers: layerCounts, consequences: consCounts };
  }, [allViolations]);

  const scCount = distributions.consequences['SAFETY_CRITICAL'] ?? 0;
  const latestReg = regressions.length ? regressions[regressions.length - 1] : null;
  const metrics = run?.metrics;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* controls */}
      <Paper sx={{ p: 1.5, bgcolor: '#161b22', border: '1px solid #232a31', display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <FormControl size="small" sx={{ minWidth: 230 }}>
          <InputLabel>stack under evaluation</InputLabel>
          <Select label="stack under evaluation" value={modelVersion} onChange={(e) => setModelVersion(e.target.value)}>
            {MODEL_VERSIONS.map((m) => (
              <MenuItem key={m} value={m}>
                {m}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <TextField size="small" type="number" label="scenarios" value={nScenarios} onChange={(e) => setNScenarios(Number(e.target.value))} sx={{ width: 100 }} />
        <TextField size="small" type="number" label="seed" value={seed} onChange={(e) => setSeed(Number(e.target.value))} sx={{ width: 90 }} />
        <Button variant="contained" onClick={generateAndRun} disabled={busy}>
          Generate bank &amp; run detection
        </Button>
        {busy ? <CircularProgress size={20} /> : null}
        <Box sx={{ flex: 1 }} />
        <FormControl size="small" sx={{ minWidth: 260 }}>
          <InputLabel>inspect run</InputLabel>
          <Select label="inspect run" value={runId} onChange={(e) => selectRun(e.target.value)}>
            {runs.map((r) => (
              <MenuItem key={r.run_id} value={r.run_id}>
                {r.run_id} · {r.model_version} · {r.n_violations} violations
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {error ? (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      ) : null}
      {notice ? (
        <Alert severity="info" onClose={() => setNotice(null)}>
          {notice}
        </Alert>
      ) : null}

      {!run ? (
        <Alert severity="info">No runs yet — generate a scenario bank and run detection to populate the control center.</Alert>
      ) : (
        <>
          {/* gate banner */}
          <Alert
            severity={run.gate.outcome === 'NO_GO' ? 'error' : 'success'}
            sx={{ fontWeight: 700, alignItems: 'center' }}
            action={
              <Chip size="small" label={`stop-ship policy ${run.gate.policy_version}`} sx={{ fontFamily: 'monospace', fontSize: 10 }} />
            }
          >
            RELEASE GATE: {run.gate.outcome}
            {run.gate.outcome === 'NO_GO'
              ? ` — ${run.gate.events.filter((e) => e.fired).length} catastrophic conjunction(s) fired (VRU + missed detection + safety-critical exposure). Deterministic policy, not LLM-driven.`
              : ' — no catastrophic conjunction observed on this run.'}
          </Alert>

          <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ borderBottom: '1px solid #232a31' }}>
            <Tab label="Executive" />
            <Tab label="Engineering" />
            <Tab label="Data" />
            <Tab label="Infrastructure" />
          </Tabs>

          {/* ------------------------------------------------ EXECUTIVE */}
          {tab === 0 && metrics ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
                <Tile label="violations detected" value={String(run.n_violations)} hint={`${metrics.n_committed_violations} planted · recall ${pct(metrics.rotr_recall)}`} />
                <Tile label="safety-critical" value={String(scCount)} color="#ef5350" hint="counterfactual consequence class" />
                <Tile label="SC-ROTR recall" value={pct(metrics.sc_rotr_recall)} hint={`exposure-derived weights ${metrics.weight_calibration.version}`} />
                <Tile label="false accusations" value={pct(metrics.false_accusation_rate)} hint={`${metrics.n_planted_non_violations} planted non-violations`} />
                <Tile label="BCR" value={pct(metrics.bcr)} hint="behavioral consequence rate" />
                <Tile label="CFR" value={pct(metrics.cfr)} hint={`95% CI [${pct(metrics.cfr_wilson_95[0])}, ${pct(metrics.cfr_wilson_95[1])}]`} />
              </Box>
              <SectionCard title="Regression status" subtitle="claims are delegated to seqeval's sequential engine; every comparison lands in the six-outcome distinction">
                {latestReg ? (
                  <Box>
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mb: 1 }}>
                      <Chip
                        label={latestReg.primary_outcome}
                        sx={{
                          fontWeight: 800,
                          bgcolor: latestReg.primary_outcome.includes('REGRESSION') ? '#b71c1c' : '#1b5e20',
                          color: '#fff',
                        }}
                      />
                      <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#8a949e' }}>
                        {latestReg.baseline_model} → {latestReg.candidate_model}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {Object.entries(latestReg.six_outcomes).map(([k, v]) => (
                        <Chip key={k} size="small" label={k.replace(/_/g, ' ')} sx={{ height: 20, fontSize: 10, bgcolor: v ? '#37474f' : '#161b22', color: v ? '#fff' : '#5c6670', border: '1px solid #232a31' }} />
                      ))}
                    </Box>
                    <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
                      {latestReg.distribution_note}
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    No comparisons yet — pick a baseline run under Engineering → Model comparison.
                  </Typography>
                )}
              </SectionCard>
              <SectionCard title="Runs">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>run</TableCell>
                      <TableCell>stack</TableCell>
                      <TableCell align="right">scenarios</TableCell>
                      <TableCell align="right">violations</TableCell>
                      <TableCell align="right">ROTR recall</TableCell>
                      <TableCell>gate</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {runs.map((r) => (
                      <TableRow key={r.run_id} hover selected={r.run_id === runId} onClick={() => selectRun(r.run_id)} sx={{ cursor: 'pointer' }}>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{r.run_id}</TableCell>
                        <TableCell>{r.model_version}</TableCell>
                        <TableCell align="right">{r.n_scenarios}</TableCell>
                        <TableCell align="right">{r.n_violations}</TableCell>
                        <TableCell align="right">{pct(r.rotr_recall)}</TableCell>
                        <TableCell>
                          <Chip size="small" label={r.gate_outcome} sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: r.gate_outcome === 'NO_GO' ? '#b71c1c' : '#1b5e20', color: '#fff' }} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </SectionCard>
            </Box>
          ) : null}

          {/* ------------------------------------------------ ENGINEERING */}
          {tab === 1 ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' }, gap: 2 }}>
                <SectionCard title="Violations by legality axis">
                  <DistBar counts={distributions.axes.legality ?? {}} total={run.n_violations} />
                </SectionCard>
                <SectionCard title="By attributed causal layer">
                  <DistBar counts={distributions.layers} total={run.n_violations} />
                </SectionCard>
                <SectionCard title="Consequence breakdown">
                  {Object.entries(distributions.consequences).map(([k, n]) => (
                    <Box key={k} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Chip size="small" label={k} sx={{ bgcolor: CONSEQUENCE_COLOR[k] ?? '#37474f', color: '#fff', height: 20, fontSize: 10, minWidth: 200 }} />
                      <Typography variant="body2" sx={{ fontWeight: 700 }}>
                        {n}
                      </Typography>
                    </Box>
                  ))}
                </SectionCard>
              </Box>

              <SectionCard title="Attribution matrix" subtitle="a violation is NEVER auto-attributed to perception — every layer needs its own positive evidence; click a row to open its consequence replay">
                {matrix ? <RotrAttributionHeatmap matrix={matrix} selectedViolationId={selectedVid} onSelect={selectViolation} /> : <CircularProgress size={20} />}
              </SectionCard>

              {consequence ? (
                <SectionCard title={`Consequence replay — ${consequence.violation_id}`} subtitle="observed trajectory vs corrected-input counterfactual; scrub time with the slider">
                  <RotrBevReplay detail={consequence} />
                </SectionCard>
              ) : (
                <Alert severity="info">Select a violation in the attribution matrix to open its BEV consequence replay.</Alert>
              )}

              <SectionCard title="Cluster browser" subtitle="recurring patterns grouped by structured signature (legality · actor · control · behavior · layer)">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>cluster</TableCell>
                      <TableCell>signature key</TableCell>
                      <TableCell align="right">members</TableCell>
                      <TableCell>environments</TableCell>
                      <TableCell>consequences</TableCell>
                      <TableCell />
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {run.clusters.map((c) => (
                      <TableRow key={c.cluster_id} hover>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{c.cluster_id}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{c.key}</TableCell>
                        <TableCell align="right">{c.count}</TableCell>
                        <TableCell sx={{ fontSize: 11 }}>{c.environment_spread.join(', ')}</TableCell>
                        <TableCell sx={{ fontSize: 11 }}>
                          {Object.entries(c.consequence_distribution)
                            .map(([k, n]) => `${k}:${n}`)
                            .join(' ')}
                        </TableCell>
                        <TableCell>
                          <Button size="small" onClick={() => selectViolation(c.exemplar_violation_id)}>
                            open exemplar
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </SectionCard>

              <SectionCard title="Model comparison" subtitle="baseline vs the currently inspected run; statistics delegated to seqeval (rare-event aware sequential testing)">
                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', mb: 1.5, flexWrap: 'wrap' }}>
                  <FormControl size="small" sx={{ minWidth: 260 }}>
                    <InputLabel>baseline run</InputLabel>
                    <Select label="baseline run" value={baselineId} onChange={(e) => setBaselineId(e.target.value)}>
                      {runs
                        .filter((r) => r.run_id !== runId)
                        .map((r) => (
                          <MenuItem key={r.run_id} value={r.run_id}>
                            {r.run_id} · {r.model_version}
                          </MenuItem>
                        ))}
                    </Select>
                  </FormControl>
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    candidate = {runId} ({run.model_version})
                  </Typography>
                  <Button variant="outlined" disabled={!baselineId || busy} onClick={doRegression}>
                    Run seqeval-gated comparison
                  </Button>
                </Box>
                {latestReg ? (
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>metric</TableCell>
                        <TableCell align="right">delta (candidate − baseline)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(latestReg.metric_deltas).map(([k, v]) => (
                        <TableRow key={k}>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{k}</TableCell>
                          <TableCell align="right" sx={{ color: v > 0 && k.startsWith('violation_rate') ? '#ef5350' : v < 0 && k.startsWith('violation_rate') ? '#66bb6a' : '#c9d1d9', fontFamily: 'monospace' }}>
                            {v > 0 ? '+' : ''}
                            {v.toFixed(4)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : null}
              </SectionCard>

              <SectionCard title="Taxonomy mining — structured query" subtitle="six-axis taxonomy (+ road geometry / traffic control / visibility extensions); text parses through a deterministic keyword map">
                <RotrQueryBuilder onRun={runQuery} busy={busy} />
                {queryResult ? (
                  <Box sx={{ mt: 1.5 }}>
                    <Typography variant="caption" sx={{ color: '#8a949e' }}>
                      parsed query:{' '}
                      {Object.entries(queryResult.query)
                        .filter(([k, v]) => v != null && k !== 'text')
                        .map(([k, v]) => `${k}=${v}`)
                        .join(' · ') || '(no filters)'}{' '}
                      → {queryResult.n_results} result(s)
                    </Typography>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>violation</TableCell>
                          <TableCell>rule</TableCell>
                          <TableCell>layer</TableCell>
                          <TableCell>consequence</TableCell>
                          <TableCell>environment</TableCell>
                          <TableCell>cluster</TableCell>
                          <TableCell />
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {queryResult.results.map((r) => (
                          <TableRow key={r.violation_id} hover>
                            <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{r.violation_id.replace(/^bank-[0-9a-f]+-/, '')}</TableCell>
                            <TableCell sx={{ fontSize: 11 }}>{r.rule_id}</TableCell>
                            <TableCell sx={{ fontSize: 11 }}>{r.primary_layer ?? '—'}</TableCell>
                            <TableCell>
                              <Chip size="small" label={r.consequence_class ?? '—'} sx={{ height: 18, fontSize: 9, bgcolor: CONSEQUENCE_COLOR[r.consequence_class ?? ''] ?? '#37474f', color: '#fff' }} />
                            </TableCell>
                            <TableCell sx={{ fontSize: 11 }}>
                              {r.environment.visibility}/{r.environment.lighting}/{r.environment.weather}
                            </TableCell>
                            <TableCell sx={{ fontFamily: 'monospace', fontSize: 10 }}>{r.cluster_id ?? '—'}</TableCell>
                            <TableCell>
                              <Button size="small" onClick={() => selectViolation(r.violation_id)}>
                                replay
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Box>
                ) : null}
              </SectionCard>
            </Box>
          ) : null}

          {/* ------------------------------------------------ DATA */}
          {tab === 2 ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <SectionCard title="Scenario coverage" subtitle="planted opportunity mix of the inspected bank (kind × environment), with commit outcomes for this stack">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>kind</TableCell>
                      <TableCell align="right">scenarios</TableCell>
                      <TableCell align="right">opportunities</TableCell>
                      <TableCell align="right">committed</TableCell>
                      <TableCell>environments</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(
                      run.scenario_summaries.reduce<Record<string, { n: number; opp: number; com: number; envs: Set<string> }>>((acc, s) => {
                        const e = acc[s.kind] ?? { n: 0, opp: 0, com: 0, envs: new Set<string>() };
                        e.n += 1;
                        if (s.is_violation_opportunity) e.opp += 1;
                        if (s.committed) e.com += 1;
                        e.envs.add(`${s.visibility}/${s.lighting}`);
                        acc[s.kind] = e;
                        return acc;
                      }, {})
                    ).map(([kind, e]) => (
                      <TableRow key={kind}>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{kind}</TableCell>
                        <TableCell align="right">{e.n}</TableCell>
                        <TableCell align="right">{e.opp}</TableCell>
                        <TableCell align="right" sx={{ color: e.com ? '#ef5350' : '#66bb6a' }}>
                          {e.com}
                        </TableCell>
                        <TableCell sx={{ fontSize: 11 }}>{[...e.envs].join(', ')}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </SectionCard>

              <SectionCard title="Rare patterns" subtitle="singleton clusters — signatures observed exactly once on this run (raremine-style rarity lens)">
                {run.clusters.filter((c) => c.count === 1).length ? (
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {run.clusters
                      .filter((c) => c.count === 1)
                      .map((c) => (
                        <Chip key={c.cluster_id} size="small" label={c.key} onClick={() => selectViolation(c.exemplar_violation_id)} sx={{ fontFamily: 'monospace', fontSize: 10 }} />
                      ))}
                  </Box>
                ) : (
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    No singleton signatures on this run.
                  </Typography>
                )}
              </SectionCard>

              <SectionCard title="HITL validation queue" subtitle="confirmed violation → failure signature → cluster → regression-test artifact; validation is the ONLY path into the suite">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>review</TableCell>
                      <TableCell>violation</TableCell>
                      <TableCell>cluster</TableCell>
                      <TableCell>status</TableCell>
                      <TableCell />
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {hitl.map((r) => (
                      <TableRow key={r.review_id} hover>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{r.review_id.replace(/^rev-bank-[0-9a-f]+-/, '')}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{r.violation_id.replace(/^bank-[0-9a-f]+-/, '')}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 10 }}>{r.cluster_id ?? '—'}</TableCell>
                        <TableCell>
                          <Chip size="small" label={r.status} sx={{ height: 18, fontSize: 10, bgcolor: r.status === 'PENDING' ? '#e65100' : r.status === 'VALIDATED' ? '#1b5e20' : '#37474f', color: '#fff' }} />
                        </TableCell>
                        <TableCell>
                          {r.status === 'PENDING' ? (
                            <>
                              <Button size="small" color="success" onClick={() => act(r, 'VALIDATE')}>
                                validate
                              </Button>
                              <Button size="small" color="inherit" onClick={() => act(r, 'REJECT')}>
                                reject
                              </Button>
                            </>
                          ) : null}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </SectionCard>

              <SectionCard title="Training candidates" subtitle="immutable dataset roles; REGRESSION/LAUNCH members are never training-eligible without a recorded governance override (contamination guard)">
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
                  <TextField size="small" fullWidth label="governance override reason (required before promoting a protected member)" value={overrideReason} onChange={(e) => setOverrideReason(e.target.value)} />
                </Box>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>candidate</TableCell>
                      <TableCell>violation</TableCell>
                      <TableCell>role</TableCell>
                      <TableCell>guard</TableCell>
                      <TableCell>training-eligible</TableCell>
                      <TableCell />
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {candidates.map((c) => (
                      <TableRow key={c.candidate_id} hover>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{c.candidate_id}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{c.violation_id.replace(/^bank-[0-9a-f]+-/, '')}</TableCell>
                        <TableCell>
                          <Chip size="small" label={c.dataset_role} sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: ROLE_COLOR[c.dataset_role] ?? '#37474f', color: '#fff' }} />
                        </TableCell>
                        <TableCell>
                          <Chip size="small" label={c.guard_state} sx={{ height: 18, fontSize: 10, bgcolor: c.guard_state === 'PROTECTED' ? '#b71c1c' : c.guard_state === 'OVERRIDDEN' ? '#e65100' : '#1b5e20', color: '#fff' }} />
                        </TableCell>
                        <TableCell sx={{ fontSize: 11 }}>{c.training_eligible ? 'yes' : 'no'}</TableCell>
                        <TableCell>
                          <Button size="small" onClick={() => promote(c)}>
                            promote to training
                          </Button>
                          <Button size="small" color="warning" disabled={!overrideReason.trim()} onClick={() => override(c)}>
                            record override
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </SectionCard>

              <SectionCard title="Regression suite" subtitle={`suite members re-run against every candidate · registry backend: ${suite?.registry_backend ?? '—'}`}>
                {suite?.members.length ? (
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>candidate</TableCell>
                        <TableCell>violation</TableCell>
                        <TableCell>role</TableCell>
                        <TableCell>added by</TableCell>
                        <TableCell>added at</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {suite.members.map((m) => (
                        <TableRow key={m.candidate_id}>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{m.candidate_id}</TableCell>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{m.violation_id.replace(/^bank-[0-9a-f]+-/, '')}</TableCell>
                          <TableCell>
                            <Chip size="small" label={m.role} sx={{ height: 18, fontSize: 10, bgcolor: ROLE_COLOR[m.role] ?? '#37474f', color: '#fff' }} />
                          </TableCell>
                          <TableCell sx={{ fontSize: 11 }}>{m.added_by ?? '—'}</TableCell>
                          <TableCell sx={{ fontSize: 11 }}>{m.added_at}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <Typography variant="caption" sx={{ color: '#8a949e' }}>
                    Empty — validate a HITL item to create the first regression artifact.
                  </Typography>
                )}
              </SectionCard>
            </Box>
          ) : null}

          {/* ------------------------------------------------ INFRASTRUCTURE */}
          {tab === 3 && metrics ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
                <Tile
                  label="last run wall time (measured in this browser)"
                  value={lastRunWallMs != null ? `${(lastRunWallMs / 1000).toFixed(1)}s` : 'not measured this session'}
                  hint={lastRunWallMs != null ? `${(run.n_scenarios / (lastRunWallMs / 1000)).toFixed(1)} scenarios/s incl. attribution + counterfactual replay` : 'generate a run to measure'}
                />
                <Tile label="scenarios in run" value={String(run.n_scenarios)} hint={`bank ${run.bank_id}`} />
                <Tile
                  label="seqeval samples used (last comparison)"
                  value={latestReg?.seqeval && typeof latestReg.seqeval.samples_used === 'number' ? String(latestReg.seqeval.samples_used) : '—'}
                  hint={latestReg?.seqeval && typeof latestReg.seqeval.full_population === 'number' ? `of ${latestReg.seqeval.full_population} population objects (early stopping)` : 'run a comparison first'}
                />
                <Tile label="evaluation cost" value="NOT MEASURED" color="#8a949e" hint="no cost meter exists in this deployment; a production system could attach one per run" />
                <Tile label="storage footprint" value="NOT MEASURED" color="#8a949e" hint="runs persist as JSON under runs/rotr/; no quota accounting is implemented" />
              </Box>
              <SectionCard title="Deterministic versions" subtitle="everything that must be pinned for a byte-identical re-run">
                <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
                  <Chip size="small" label={`ruleset ${run.ruleset_version}`} sx={{ fontFamily: 'monospace' }} />
                  <Chip size="small" label={`weights ${metrics.weight_calibration.version}`} sx={{ fontFamily: 'monospace' }} />
                  <Chip size="small" label={`stop-ship policy ${policy?.policy_version ?? '—'}`} sx={{ fontFamily: 'monospace' }} />
                  <Chip size="small" label={`stack ${run.model_version}`} sx={{ fontFamily: 'monospace' }} />
                  <Chip size="small" label={`bank ${run.bank_id} (content-addressed)`} sx={{ fontFamily: 'monospace' }} />
                </Box>
                <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
                  {metrics.weight_calibration.method}
                </Typography>
              </SectionCard>
              <SectionCard title="Honesty ledger" subtitle="what is real vs illustrative on this page">
                <Typography variant="body2" sx={{ color: '#c9d1d9', fontSize: 13 }}>
                  Real: rule verdicts, attribution evidence, counterfactual trajectories, seqeval statistics, contamination-guard enforcement, all determinism guarantees. Illustrative
                  (synthetic substrate): every physical threshold, the scenario kinematics, the exposure-derived weights (calibrated on the bank itself). Surrogate caveat: {metrics.surrogate_caveat}
                </Typography>
              </SectionCard>
            </Box>
          ) : null}
        </>
      )}
    </Box>
  );
}
