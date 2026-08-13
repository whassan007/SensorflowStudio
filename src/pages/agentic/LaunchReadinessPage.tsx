/** Launch Readiness — agentic misclassification triage.
 *
 * Failure queue → five-layer investigation (evidence graph, agent analyses,
 * statistics, safety chain, concentration) → deterministic launch decision +
 * human review → scorecard → learning flywheel.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Tab from '@mui/material/Tab';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { PlayCircle, Radar } from 'lucide-react';
import type {
  AuditRecord,
  EvaluationSuite,
  EvidenceGraph,
  FailureEvent,
  HumanReviewDecision,
  PipelineState,
  Scorecard,
  Stage,
} from '../../types/agentic';
import { STAGES } from '../../types/agentic';
import {
  detectFailures,
  getAudit,
  getEvidence,
  getFailureQueue,
  getHumanReviews,
  getScorecard,
  getState,
  getWorkedExample,
  listSuites,
  runStage,
} from '../../services/agentic';
import AgentOutputsPanel from '../../components/agentic/AgentOutputsPanel';
import ConcentrationHeatmap from '../../components/agentic/ConcentrationHeatmap';
import DecisionPanel from '../../components/agentic/DecisionPanel';
import EvidenceGraphPanel from '../../components/agentic/EvidenceGraphPanel';
import FlywheelPanel from '../../components/agentic/FlywheelPanel';
import SafetyChainPanel from '../../components/agentic/SafetyChainPanel';
import ScorecardView from '../../components/agentic/ScorecardView';
import StatisticalPanel from '../../components/agentic/StatisticalPanel';
import { OutcomeChip, SeverityChip } from '../../components/agentic/common';

const STAGE_LABELS: Record<Stage, string> = {
  FAILURE_DETECTION: '1 · Detection',
  EVIDENCE_AGGREGATION: '2 · Evidence',
  FAILURE_ANALYSIS: '3 · Analysis',
  LAUNCH_DECISION: '4 · Decision',
  LEARNING_FLYWHEEL: '5 · Flywheel',
};

const STAGE_STATUS_COLORS: Record<string, string> = {
  pending: '#37474f',
  running: '#0d47a1',
  complete: '#1b5e20',
  failed: '#b71c1c',
  blocked: '#4e342e',
};

export default function LaunchReadinessPage() {
  const [topTab, setTopTab] = useState<'queue' | 'flywheel'>('queue');
  const [failures, setFailures] = useState<FailureEvent[]>([]);
  const [suites, setSuites] = useState<EvaluationSuite[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [state, setState] = useState<PipelineState | null>(null);
  const [graph, setGraph] = useState<EvidenceGraph | null>(null);
  const [decisions, setDecisions] = useState<HumanReviewDecision[]>([]);
  const [audit, setAudit] = useState<AuditRecord[]>([]);
  const [auditValid, setAuditValid] = useState<boolean | null>(null);
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [detailTab, setDetailTab] = useState<'investigation' | 'decision' | 'scorecard'>('investigation');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterOutcome, setFilterOutcome] = useState('all');
  const [filterStage, setFilterStage] = useState('all');

  const refreshQueue = useCallback(async () => {
    const res = await getFailureQueue();
    setFailures(res.failures);
  }, []);

  const refreshSuites = useCallback(async () => {
    const res = await listSuites();
    setSuites(res.suites);
  }, []);

  const refreshSelected = useCallback(async (id: string) => {
    const st = await getState(id).then((r) => r.state).catch(() => null);
    setState(st);
    setGraph(await getEvidence(id).then((r) => r.graph).catch(() => null));
    setDecisions(await getHumanReviews(id).then((r) => r.decisions).catch(() => []));
    const auditRes = await getAudit(id).catch(() => null);
    setAudit(auditRes?.records ?? []);
    setAuditValid(auditRes?.chain.valid ?? null);
    if (st?.scorecard_id) {
      setScorecard(await getScorecard(st.scorecard_id).then((r) => r.scorecard).catch(() => null));
    } else {
      setScorecard(null);
    }
  }, []);

  useEffect(() => {
    refreshQueue().catch((e) => setError((e as Error).message));
    refreshSuites().catch(() => undefined);
  }, [refreshQueue, refreshSuites]);

  useEffect(() => {
    if (selectedId) refreshSelected(selectedId).catch((e) => setError((e as Error).message));
  }, [selectedId, refreshSelected]);

  const withBusy = async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const scan = () =>
    withBusy('scan', async () => {
      await detectFailures();
      await refreshQueue();
    });

  const workedExample = () =>
    withBusy('worked-example', async () => {
      const res = await getWorkedExample();
      await refreshQueue();
      await refreshSuites();
      setSelectedId(res.walkthrough.failure_id);
      setDetailTab('decision');
    });

  const advance = (stage?: Stage) =>
    withBusy(stage ?? 'next', async () => {
      if (!selectedId) return;
      await runStage(selectedId, stage);
      await refreshSelected(selectedId);
      await refreshQueue();
      await refreshSuites();
    });

  const filtered = useMemo(
    () =>
      failures.filter((f) => {
        if (filterSeverity !== 'all' && f.severity !== filterSeverity) return false;
        if (filterOutcome !== 'all' && f.policy_outcome !== filterOutcome) return false;
        if (filterStage !== 'all' && f.status !== filterStage) return false;
        return true;
      }),
    [failures, filterSeverity, filterOutcome, filterStage]
  );

  const selected = failures.find((f) => f.failure_id === selectedId) ?? null;
  const agents = state ? Object.values(state.agent_results) : [];
  const safety = state?.agent_results['safety_impact'];
  const nextStage = state?.stages.find((s) => s.status === 'pending')?.stage;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {/* header actions */}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Tabs value={topTab} onChange={(_, v) => setTopTab(v)} sx={{ minHeight: 34, flex: 1 }}>
          <Tab value="queue" label="Failure Queue & Investigation" sx={{ minHeight: 34, py: 0 }} />
          <Tab value="flywheel" label={`Evaluation Flywheel (${suites.length})`} sx={{ minHeight: 34, py: 0 }} />
        </Tabs>
        <Button size="small" variant="outlined" startIcon={<Radar size={15} />} disabled={busy !== null} onClick={scan}>
          Scan for failures
        </Button>
        <Button size="small" variant="contained" startIcon={<PlayCircle size={15} />} disabled={busy !== null} onClick={workedExample}>
          Run the pedestrian→cone walkthrough
        </Button>
        {busy ? <CircularProgress size={18} /> : null}
      </Box>
      {error ? (
        <Alert severity="error" variant="outlined" onClose={() => setError(null)}>
          {error}
        </Alert>
      ) : null}

      {topTab === 'flywheel' ? (
        <FlywheelPanel suites={suites} />
      ) : (
        <>
          {/* filters + queue */}
          <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
            <Box sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, flex: 1 }}>
                Detected Failure Events ({filtered.length})
              </Typography>
              <TextField size="small" select label="Severity" value={filterSeverity} onChange={(e) => setFilterSeverity(e.target.value)} sx={{ width: 120 }}>
                <MenuItem value="all">all</MenuItem>
                {['S0', 'S1', 'S2', 'S3', 'S4', 'S5'].map((s) => (
                  <MenuItem key={s} value={s}>{s}</MenuItem>
                ))}
              </TextField>
              <TextField size="small" select label="Policy outcome" value={filterOutcome} onChange={(e) => setFilterOutcome(e.target.value)} sx={{ width: 210 }}>
                <MenuItem value="all">all</MenuItem>
                {['AUTOMATIC_STOP_SHIP', 'LAUNCH_REVIEW_REQUIRED', 'CONTINUE_INVESTIGATION', 'NO_LAUNCH_IMPACT', 'INDETERMINATE'].map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </TextField>
              <TextField size="small" select label="Status" value={filterStage} onChange={(e) => setFilterStage(e.target.value)} sx={{ width: 150 }}>
                <MenuItem value="all">all</MenuItem>
                {['detected', 'investigating', 'decided', 'closed'].map((s) => (
                  <MenuItem key={s} value={s}>{s}</MenuItem>
                ))}
              </TextField>
            </Box>
            {failures.length === 0 ? (
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                No failures detected yet — run a scan or launch the worked example.
              </Typography>
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {['Failure', 'Kind', 'Candidate rate', 'Severity', 'Policy outcome', 'Status', 'Validated'].map((h) => (
                      <TableCell key={h} sx={{ fontSize: 11, fontWeight: 800, color: '#8a949e' }}>{h}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filtered.map((f) => (
                    <TableRow
                      key={f.failure_id}
                      hover
                      selected={f.failure_id === selectedId}
                      onClick={() => setSelectedId(f.failure_id)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell sx={{ fontSize: 12 }}>{f.title}</TableCell>
                      <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>{f.kind}</TableCell>
                      <TableCell sx={{ fontSize: 11.5, fontFamily: 'monospace' }}>
                        {f.detection_basis.candidate_events}/{f.detection_basis.denominator.toLocaleString()} ({(f.detection_basis.candidate_rate * 100).toFixed(4)}%)
                      </TableCell>
                      <TableCell><SeverityChip severity={f.severity} /></TableCell>
                      <TableCell><OutcomeChip outcome={f.policy_outcome} /></TableCell>
                      <TableCell sx={{ fontSize: 11.5 }}>{f.status}</TableCell>
                      <TableCell sx={{ fontSize: 11.5 }}>{f.validated ? '✓ human-validated' : '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </Paper>

          {/* investigation */}
          {selected && state ? (
            <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                {selected.title}{' '}
                <span style={{ color: '#5c6770', fontFamily: 'monospace', fontSize: 11 }}>{selected.failure_id}</span>
              </Typography>

              {/* five-layer stage progress */}
              <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', flexWrap: 'wrap', mb: 1.5 }}>
                {STAGES.map((s, i) => {
                  const rec = state.stages.find((r) => r.stage === s);
                  return (
                    <Box key={s} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Chip
                        size="small"
                        label={`${STAGE_LABELS[s]} · ${rec?.status ?? '?'}`}
                        title={rec?.detail}
                        sx={{
                          height: 22,
                          fontSize: 10.5,
                          fontWeight: 700,
                          bgcolor: STAGE_STATUS_COLORS[rec?.status ?? 'pending'],
                          color: '#fff',
                        }}
                      />
                      {i < STAGES.length - 1 ? <Typography sx={{ color: '#5c6770' }}>→</Typography> : null}
                    </Box>
                  );
                })}
                {nextStage ? (
                  <Button size="small" variant="outlined" disabled={busy !== null} onClick={() => advance(nextStage)} sx={{ ml: 1 }}>
                    Run {STAGE_LABELS[nextStage]}
                  </Button>
                ) : null}
              </Box>

              <Tabs value={detailTab} onChange={(_, v) => setDetailTab(v)} sx={{ minHeight: 34, mb: 1.5 }}>
                <Tab value="investigation" label="Investigation" sx={{ minHeight: 34, py: 0 }} />
                <Tab value="decision" label="Launch Decision" sx={{ minHeight: 34, py: 0 }} disabled={!state.policy_evaluation} />
                <Tab value="scorecard" label="Scorecard" sx={{ minHeight: 34, py: 0 }} disabled={!scorecard} />
              </Tabs>

              {detailTab === 'investigation' ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                  {graph ? (
                    <EvidenceGraphPanel graph={graph} />
                  ) : (
                    <Alert severity="info" variant="outlined">
                      Evidence graph not built yet — run the EVIDENCE AGGREGATION stage.
                    </Alert>
                  )}
                  {state.statistical ? <StatisticalPanel stat={state.statistical} /> : null}
                  {safety ? <SafetyChainPanel safety={safety} /> : null}
                  {state.concentration ? <ConcentrationHeatmap conc={state.concentration} /> : null}
                  {agents.length > 0 ? <AgentOutputsPanel agents={agents} /> : null}
                </Box>
              ) : null}

              {detailTab === 'decision' && state.policy_evaluation ? (
                <DecisionPanel
                  failureId={selected.failure_id}
                  evaluation={state.policy_evaluation}
                  decisions={decisions}
                  audit={audit}
                  auditValid={auditValid}
                  onReviewed={() => refreshSelected(selected.failure_id).then(refreshQueue)}
                />
              ) : null}

              {detailTab === 'scorecard' && scorecard ? <ScorecardView card={scorecard} /> : null}
            </Paper>
          ) : (
            <Alert severity="info" variant="outlined">
              Select a failure from the queue to open its five-layer investigation, or run the
              worked example for the full pedestrian→cone walkthrough.
            </Alert>
          )}
        </>
      )}
    </Box>
  );
}
