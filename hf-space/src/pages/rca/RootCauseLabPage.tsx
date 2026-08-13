/** Root Cause Lab — a guided, skeptical workbench for "offline says +5%,
 * shadow says −2%": 13 enforced stages from comparison validity through
 * root-cause scoring to recommended experiments. */
import { type ComponentType, useCallback, useEffect, useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ArrowLeft, CheckCheck, FlaskConical, Plus, StickyNote } from 'lucide-react';
import * as api from '../../services/rca';
import type {
  Confidence,
  DecisionTree,
  DiagnosticsResponse,
  Experiments,
  Investigation,
  InvestigationSummary,
  RcaReport,
  RevealResponse,
  RootCause,
  Scoreboard,
} from '../../types/rca';
import {
  BG_PANEL,
  BORDER,
  Explainer,
  FindingChips,
  SectionCard,
  tableSx,
} from '../../components/rca/common';
import StageRail, { firstIncompleteIndex } from '../../components/rca/StageRail';
import {
  ComparisonValidityView,
  LabelIntegrityView,
  OfflineAuditView,
  PopulationView,
  ServingParityView,
  TrafficAuditView,
} from '../../components/rca/stageViews';
import {
  ConditionalHeatmapView,
  FeatureParityView,
  PairedTransitionView,
  ShiftView,
  SignificanceView,
} from '../../components/rca/chartViews';
import { DecisionTreeView, ScoreBoardView } from '../../components/rca/boardViews';
import { ExperimentsView, ReportView } from '../../components/rca/ReportView';
import { AcknowledgeUnknownsDialog, RecordFindingDialog } from '../../components/rca/dialogs';

const STAGE_VIEWS: Record<string, ComponentType<{ data: any }>> = {
  comparison_validity: ComparisonValidityView,
  offline_audit: OfflineAuditView,
  population_validation: PopulationView,
  distribution_shift: ShiftView,
  conditional_performance: ConditionalHeatmapView,
  paired_comparison: PairedTransitionView,
  statistical_significance: SignificanceView,
  feature_parity: FeatureParityView,
  serving_parity: ServingParityView,
  shadow_traffic: TrafficAuditView,
  label_integrity: LabelIntegrityView,
};

// ----------------------------------------------------------- creation panel

function CreatePanel({ onCreated }: { onCreated: (inv: Investigation) => void }) {
  const [causes, setCauses] = useState<{ id: RootCause; label: string }[]>([]);
  const [baseline, setBaseline] = useState('detr3d-a-v41');
  const [candidate, setCandidate] = useState('detr3d-b-v42');
  const [cause, setCause] = useState<string>('DEMO');
  const [seed, setSeed] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    api.getCauses().then((r) => setCauses(r.causes)).catch(() => undefined);
  }, []);

  const create = async () => {
    setBusy(true);
    setErr('');
    try {
      const inv = await api.createInvestigation({
        baseline_model: baseline,
        candidate_model: candidate,
        cause: cause === 'DEMO' ? null : (cause as RootCause),
        seed: seed ? Number(seed) : null,
        training_mode: cause === 'DEMO',
      });
      onCreated(inv);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <SectionCard
      title="New investigation"
      subtitle="Generates a full offline + shadow evaluation pair with a planted root cause. In demo mode the cause is hidden — work the methodology, then reveal the answer at the end."
    >
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
        <TextField label="Baseline model (A)" size="small" value={baseline} onChange={(e) => setBaseline(e.target.value)} sx={{ width: 180 }} />
        <TextField label="Candidate model (B)" size="small" value={candidate} onChange={(e) => setCandidate(e.target.value)} sx={{ width: 180 }} />
        <TextField select label="Scenario" size="small" value={cause} onChange={(e) => setCause(e.target.value)} sx={{ width: 320 }}>
          <MenuItem value="DEMO">
            <em>Demo investigation — hidden random root cause (training mode)</em>
          </MenuItem>
          {causes.map((c) => (
            <MenuItem key={c.id} value={c.id}>{c.id} — {c.label}</MenuItem>
          ))}
        </TextField>
        <TextField label="Seed (optional)" size="small" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ''))} sx={{ width: 130 }} />
        <Button variant="contained" size="small" startIcon={busy ? <CircularProgress size={14} /> : <Plus size={15} />} disabled={busy} onClick={create}>
          Create
        </Button>
      </Box>
      {err ? <Alert severity="error" sx={{ mt: 1 }}>{err}</Alert> : null}
    </SectionCard>
  );
}

// -------------------------------------------------------- investigation list

function InvestigationList({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<InvestigationSummary[]>([]);
  useEffect(() => {
    api.listInvestigations().then((r) => setRows(r.investigations)).catch(() => undefined);
  }, []);
  return (
    <SectionCard title="Investigations">
      {rows.length === 0 ? (
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          None yet — create one above.
        </Typography>
      ) : (
        <Box component="table" sx={tableSx}>
          <thead>
            <tr><th>Name</th><th>Models</th><th>Claims</th><th>Progress</th><th>Mode</th><th /></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td style={{ fontWeight: 600 }}>{r.name}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{r.candidate_model} vs {r.baseline_model}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
                  <span style={{ color: '#a5d6a7' }}>+{r.claims.offline_delta_pp}pp off</span>{' / '}
                  <span style={{ color: '#ef9a9a' }}>{r.claims.shadow_delta_pp}pp sh</span>
                </td>
                <td>{r.stages_complete}/{r.stages_total} stages</td>
                <td>
                  {r.training_mode ? (
                    <Chip size="small" label={r.revealed ? `revealed: ${r.scenario_cause}` : 'training (hidden cause)'} sx={{ height: 18, fontSize: 10, bgcolor: '#232a31' }} />
                  ) : (
                    <Chip size="small" label={r.scenario_cause} sx={{ height: 18, fontSize: 10, bgcolor: '#232a31' }} />
                  )}
                </td>
                <td><Button size="small" onClick={() => onOpen(r.id)}>Open</Button></td>
              </tr>
            ))}
          </tbody>
        </Box>
      )}
    </SectionCard>
  );
}

// ----------------------------------------------------------- hypothesis banner

function HypothesisBanner({ inv, workingSet }: { inv: Investigation; workingSet: RootCause[] }) {
  return (
    <Paper variant="outlined" sx={{ px: 1.5, py: 0.75, mb: 1.5, bgcolor: BG_PANEL, borderColor: BORDER, display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
      <Typography variant="caption" sx={{ fontWeight: 800, color: '#8a949e', letterSpacing: 0.6 }}>
        CLAIMS
      </Typography>
      <Chip size="small" label={`offline +${inv.claims.offline_delta_pp}pp (n=${inv.claims.offline_n.toLocaleString()})`} sx={{ height: 20, fontSize: 11, bgcolor: '#66bb6a22', color: '#a5d6a7', fontFamily: 'monospace' }} />
      <Chip size="small" label={`shadow ${inv.claims.shadow_delta_pp}pp (n=${inv.claims.shadow_scored_n.toLocaleString()})`} sx={{ height: 20, fontSize: 11, bgcolor: '#ef535022', color: '#ef9a9a', fontFamily: 'monospace' }} />
      <Box sx={{ width: 1, height: 20, bgcolor: BORDER }} />
      <Typography variant="caption" sx={{ fontWeight: 800, color: '#8a949e', letterSpacing: 0.6 }}>
        WORKING HYPOTHESES ({workingSet.length})
      </Typography>
      {workingSet.map((h) => (
        <Chip key={h} size="small" label={h} sx={{ height: 20, fontSize: 10, bgcolor: '#232a31', color: '#cfd8e0' }} />
      ))}
      <Typography variant="caption" sx={{ color: '#5c666f', ml: 'auto' }}>
        narrowed by recorded evidence only — never a premature single conclusion
      </Typography>
    </Paper>
  );
}

// -------------------------------------------------------- investigation view

function InvestigationWorkspace({ invId, onBack }: { invId: string; onBack: () => void }) {
  const [inv, setInv] = useState<Investigation | null>(null);
  const [stageIdx, setStageIdx] = useState(0);
  const [diag, setDiag] = useState<Record<string, DiagnosticsResponse>>({});
  const [workingSet, setWorkingSet] = useState<RootCause[]>([]);
  const [board, setBoard] = useState<Scoreboard | null>(null);
  const [tree, setTree] = useState<DecisionTree | null>(null);
  const [experiments, setExperiments] = useState<Experiments | null>(null);
  const [report, setReport] = useState<RcaReport | null>(null);
  const [reveal, setReveal] = useState<RevealResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [findingOpen, setFindingOpen] = useState(false);
  const [ackOpen, setAckOpen] = useState(false);

  const stage = inv?.stages[stageIdx];
  const stageKey = stage?.key ?? '';

  const refreshWorkingSet = useCallback((id: string) => {
    api.getScoreboard(id, true)
      .then((b) => setWorkingSet(b.working_hypothesis_set))
      .catch(() => undefined);
  }, []);

  const load = useCallback(async () => {
    const got = await api.getInvestigation(invId);
    setInv(got);
    setStageIdx((cur) => Math.min(cur, firstIncompleteIndex(got.stages)));
    refreshWorkingSet(invId);
    return got;
  }, [invId, refreshWorkingSet]);

  useEffect(() => {
    load().catch((e) => setError(String(e)));
  }, [load]);

  // Load whatever the selected stage needs.
  useEffect(() => {
    if (!inv || !stage) return;
    setError('');
    if (STAGE_VIEWS[stage.key] && !diag[stage.key]) {
      setLoading(true);
      api.getDiagnostics(inv.id, stage.key)
        .then((d) => {
          setDiag((m) => ({ ...m, [stage.key]: d }));
          refreshWorkingSet(inv.id);
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setLoading(false));
    }
    if (stage.key === 'root_cause_scoring' && (!board || !tree)) {
      setLoading(true);
      Promise.all([api.getScoreboard(inv.id), api.getDecisionTree(inv.id)])
        .then(([b, t]) => { setBoard(b); setTree(t); })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setLoading(false));
    }
    if (stage.key === 'recommendations_report' && (!experiments || !report)) {
      setLoading(true);
      Promise.all([api.getExperiments(inv.id), api.getReport(inv.id)])
        .then(([x, r]) => { setExperiments(x); setReport(r); })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setLoading(false));
    }
  }, [inv, stage, diag, board, tree, experiments, report, refreshWorkingSet]);

  const complete = async (ack = false, note = '') => {
    if (!inv || !stage) return;
    try {
      const res = await api.completeStage(inv.id, stage.index, {
        acknowledge_unknowns: ack,
        note,
      });
      setInv(res.investigation);
      setAckOpen(false);
      if (stage.index + 1 < res.investigation.stages.length) setStageIdx(stage.index + 1);
      refreshWorkingSet(inv.id);
    } catch (e) {
      const err = e as Error & { status?: number };
      if (err.status === 409 && /UNKNOWN/i.test(err.message)) {
        setAckOpen(true);
      } else {
        setError(err.message);
      }
    }
  };

  const submitFinding = async (f: { title: string; status: any; severity: any; detail: string }) => {
    if (!inv || !stage) return;
    await api.recordFinding(inv.id, { stage: stage.key, ...f });
    setFindingOpen(false);
    await load();
    // Board scores include human findings — invalidate cached views.
    setBoard(null);
    setReport(null);
  };

  const assess = async (hypothesis: RootCause, confidence: Confidence, note: string) => {
    if (!inv) return;
    const b = await api.assessHypothesis(inv.id, { hypothesis, confidence, note });
    setBoard(b);
    setReport(null);
  };

  const doReveal = async () => {
    if (!inv) return;
    const r = await api.revealCause(inv.id);
    setReveal(r);
    await load();
  };

  const jumpToStage = useCallback((sk: string) => {
    if (!inv) return;
    const target = inv.stages.find((s) => s.key === sk);
    if (target && target.index <= firstIncompleteIndex(inv.stages)) setStageIdx(target.index);
  }, [inv]);

  const stageFindings = useMemo(
    () => (inv && stage ? inv.findings.filter((f) => f.stage === stage.key) : []),
    [inv, stage]
  );
  const unknownTitles = stageFindings
    .filter((f) => f.status === 'UNKNOWN' && f.severity === 'CRITICAL')
    .map((f) => f.title);

  if (!inv || !stage) {
    return error ? <Alert severity="error">{error}</Alert> : <CircularProgress size={22} sx={{ m: 3 }} />;
  }

  const isDone = stage.status === 'complete' || stage.status === 'complete_with_unknowns';
  const View = STAGE_VIEWS[stage.key];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Button size="small" startIcon={<ArrowLeft size={14} />} onClick={onBack}>
          All investigations
        </Button>
        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>{inv.name}</Typography>
        <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#8a949e' }}>
          {inv.candidate_model} vs {inv.baseline_model} · seed {inv.seed}
        </Typography>
        {inv.training_mode ? (
          <Chip size="small" icon={<FlaskConical size={12} />} label="training mode" sx={{ height: 20, fontSize: 10, bgcolor: '#232a31' }} />
        ) : null}
      </Box>
      <HypothesisBanner inv={inv} workingSet={workingSet} />
      <Box sx={{ display: 'flex', flex: 1, minHeight: 0, border: `1px solid ${BORDER}`, borderRadius: 1, overflow: 'hidden' }}>
        <StageRail stages={inv.stages} selected={stageIdx} onSelect={setStageIdx} />
        <Box sx={{ flex: 1, minWidth: 0, overflowY: 'auto', p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.75 }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" sx={{ fontWeight: 800, fontSize: 16 }}>
                Stage {stage.index}: {stage.title}
              </Typography>
            </Box>
            {STAGE_VIEWS[stage.key] || stage.key === 'root_cause_scoring' ? (
              <Button size="small" variant="outlined" startIcon={<StickyNote size={13} />} onClick={() => setFindingOpen(true)}>
                Record finding
              </Button>
            ) : null}
            {!isDone ? (
              <Button size="small" variant="contained" startIcon={<CheckCheck size={14} />} onClick={() => complete()}>
                Mark stage complete
              </Button>
            ) : (
              <Chip size="small" label={stage.status === 'complete' ? 'complete' : 'complete with unknowns'} sx={{
                height: 24, fontWeight: 700, fontSize: 11,
                bgcolor: stage.status === 'complete' ? '#66bb6a22' : '#ffb74d22',
                color: stage.status === 'complete' ? '#66bb6a' : '#ffb74d',
              }} />
            )}
          </Box>
          {stageFindings.length ? (
            <Box sx={{ mb: 1 }}>
              <FindingChips findings={stageFindings} />
            </Box>
          ) : null}
          {error ? <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert> : null}
          {loading ? <CircularProgress size={20} sx={{ my: 2 }} /> : null}

          {View && diag[stage.key] ? <View data={diag[stage.key].data} /> : null}

          {stage.key === 'root_cause_scoring' && board && tree ? (
            <>
              <ScoreBoardView board={board} onJumpToStage={jumpToStage} onAssess={assess} />
              <DecisionTreeView tree={tree} />
            </>
          ) : null}

          {stage.key === 'recommendations_report' && experiments && report ? (
            <>
              <ExperimentsView exp={experiments} />
              <ReportView report={report} inv={inv} reveal={reveal} onReveal={doReveal} />
            </>
          ) : null}

          {stage.key === 'root_cause_scoring' && !board && !loading ? (
            <Explainer text="Scoring runs once stages 0–10 are complete; every point on the board traces back to a recorded stage finding." />
          ) : null}
        </Box>
      </Box>

      <RecordFindingDialog
        open={findingOpen}
        stageTitle={stage.title}
        onClose={() => setFindingOpen(false)}
        onSubmit={submitFinding}
      />
      <AcknowledgeUnknownsDialog
        open={ackOpen}
        stageTitle={stage.title}
        unknownTitles={unknownTitles}
        onClose={() => setAckOpen(false)}
        onAcknowledge={(note) => complete(true, note)}
      />
    </Box>
  );
}

// -------------------------------------------------------------------- page

export default function RootCauseLabPage() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  if (activeId) {
    return (
      <Box sx={{ height: 'calc(100vh - 170px)', minHeight: 480 }}>
        <InvestigationWorkspace invId={activeId} onBack={() => { setActiveId(null); setRefreshKey((k) => k + 1); }} />
      </Box>
    );
  }
  return (
    <Box key={refreshKey}>
      <Explainer text="Offline evaluation says the candidate is +5% better; shadow production says −2% worse. This lab refuses to assume either number is real: it walks a 13-stage skeptical methodology — comparison validity, offline audit, population, shift, conditional performance, paired transitions, significance, feature/serving parity, traffic, labels — before scoring 8 root-cause hypotheses against the evidence." />
      <CreatePanel onCreated={(inv) => setActiveId(inv.id)} />
      <InvestigationList onOpen={setActiveId} />
    </Box>
  );
}
