/**
 * Release Gates — candidate/baseline pickers, a visual gate pipeline
 * (5 connected stages with PASS/BLOCK states, expandable actual-vs-threshold
 * checks), and the persisted release-policy editor.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import Switch from '@mui/material/Switch';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { CheckCircle2, ChevronDown, ChevronUp, FileText, MinusCircle, PlayCircle, XCircle } from 'lucide-react';
import {
  evaluateGates,
  getGatePolicy,
  getGateResult,
  setGatePolicy,
  type GateEvaluation,
  type GateResult,
} from '../../services/safety';
import { RunSelect, usePublishedRuns } from '../../components/safety/shared';
import { IllustratedEmpty, PanelSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { InfoDot } from '../../components/help/InfoTip';
import { useLabelEval } from '../../context/LabelEvalContext';
import { tokens, verdictColor } from '../../theme';

const GATE_ORDER = ['scenario_quality', 'coverage', 'regression', 'safety', 'evidence'];
const GATE_SHORT: Record<string, string> = {
  scenario_quality: 'Scenario Quality',
  coverage: 'ODD Coverage',
  regression: 'Regression',
  safety: 'Surrogate Safety',
  evidence: 'Evidence Package',
};

function gateStatusIcon(status: string, size = 16) {
  if (status === 'PASS') return <CheckCircle2 size={size} color={tokens.color.success} />;
  if (status === 'BLOCK') return <XCircle size={size} color={tokens.color.danger} />;
  return <MinusCircle size={size} color={tokens.color.neutral} />;
}

// ------------------------------------------------------------ gate pipeline

function GatePipeline({ gates, expanded, onToggle }: { gates: GateResult[]; expanded: string | null; onToggle: (g: string) => void }) {
  const ordered = useMemo(() => {
    const byGate = new Map(gates.map((g) => [g.gate, g]));
    const known = GATE_ORDER.filter((g) => byGate.has(g)).map((g) => byGate.get(g)!);
    const rest = gates.filter((g) => !GATE_ORDER.includes(g.gate));
    return [...known, ...rest];
  }, [gates]);

  const W = 168;
  const H = 74;
  const GAPX = 42;
  const total = ordered.length * W + (ordered.length - 1) * GAPX + 20;

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <svg width="100%" viewBox={`0 0 ${total} 110`} style={{ minWidth: Math.min(total, 900), display: 'block' }}>
        <defs>
          <marker id="gate-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a5561" />
          </marker>
        </defs>
        {ordered.map((g, i) => {
          const x = 10 + i * (W + GAPX);
          const color = verdictColor(g.status);
          const failing = g.checks.filter((c) => !c.passed).length;
          const isOpen = expanded === g.gate;
          return (
            <g key={g.gate}>
              {i < ordered.length - 1 ? (
                <line x1={x + W} y1={55} x2={x + W + GAPX - 6} y2={55} stroke="#4a5561" strokeWidth={2} markerEnd="url(#gate-arrow)" />
              ) : null}
              <g onClick={() => onToggle(g.gate)} style={{ cursor: 'pointer' }}>
                <rect
                  x={x}
                  y={18}
                  width={W}
                  height={H}
                  rx={10}
                  fill={isOpen ? tokens.color.surfaceRaised : tokens.color.surface}
                  stroke={color}
                  strokeWidth={isOpen ? 3 : 2}
                  style={{ transition: `stroke-width 120ms` }}
                />
                <text x={x + W / 2} y={40} textAnchor="middle" fill={tokens.color.text} fontSize={12.5} fontWeight={700}>
                  {GATE_SHORT[g.gate] ?? g.name}
                </text>
                <text x={x + W / 2} y={58} textAnchor="middle" fill={color} fontSize={12} fontWeight={800}>
                  {g.status}
                </text>
                <text x={x + W / 2} y={74} textAnchor="middle" fill={tokens.color.neutral} fontSize={9.5}>
                  {g.checks.length} checks{failing ? ` · ${failing} failing` : ''} · click to expand
                </text>
                <text x={x + W / 2} y={104} textAnchor="middle" fill={tokens.color.textFaint} fontSize={8.5}>
                  {(g.standard_refs[0] ?? '').slice(0, 34)}
                </text>
              </g>
            </g>
          );
        })}
      </svg>
    </Box>
  );
}

function fmtVal(v: number | string | boolean | null): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return String(v);
}

function GateDetail({ gate }: { gate: GateResult }) {
  return (
    <Box sx={{ p: 1.5, border: `1px solid ${tokens.color.border}`, borderLeft: `3px solid ${verdictColor(gate.status)}`, borderRadius: 1, bgcolor: tokens.color.surfaceSunken }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        {gateStatusIcon(gate.status, 18)}
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          {gate.name}
        </Typography>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
          {gate.standard_refs.join(' · ')}
        </Typography>
      </Box>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Check</TableCell>
            <TableCell align="right">Actual</TableCell>
            <TableCell align="center">vs</TableCell>
            <TableCell align="left">Threshold</TableCell>
            <TableCell align="center">Result</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {gate.checks.map((c, i) => (
            <TableRow key={`${c.check}-${i}`}>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{c.check}</TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700, color: c.passed ? tokens.color.text : tokens.color.danger }}>
                {fmtVal(c.actual)}
              </TableCell>
              <TableCell align="center" sx={{ fontFamily: 'monospace', fontSize: 12, color: tokens.color.neutral }}>{c.direction}</TableCell>
              <TableCell align="left" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{fmtVal(c.threshold)}</TableCell>
              <TableCell align="center">{c.passed ? <CheckCircle2 size={15} color={tokens.color.success} /> : <XCircle size={15} color={tokens.color.danger} />}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {gate.notes ? (
        <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mt: 1 }}>
          {gate.notes}
        </Typography>
      ) : null}
    </Box>
  );
}

// ------------------------------------------------------------ policy editor

function PolicyEditor({ onSaved }: { onSaved: () => void }) {
  const [policy, setPolicy] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [draft, setDraft] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getGatePolicy()
      .then((p) => {
        setPolicy(p.policy);
        setDraft(JSON.parse(JSON.stringify(p.policy)));
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const dirty = useMemo(() => JSON.stringify(policy) !== JSON.stringify(draft), [policy, draft]);

  const setField = (section: string, key: string, value: unknown) => {
    setDraft((d) => (d ? { ...d, [section]: { ...d[section], [key]: value } } : d));
    setSaved(false);
  };

  const save = () => {
    if (!draft) return;
    setSaving(true);
    setError(null);
    setGatePolicy(draft)
      .then((p) => {
        setPolicy(p.policy);
        setDraft(JSON.parse(JSON.stringify(p.policy)));
        setSaved(true);
        onSaved();
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSaving(false));
  };

  if (error) return <ErrorNote error={error} />;
  if (!draft) return <PanelSkeleton rows={3} />;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {Object.entries(draft).map(([section, fields]) => (
          <Box key={section} sx={{ flex: '1 1 260px', border: `1px solid ${tokens.color.border}`, borderRadius: 1, p: 1.5, bgcolor: tokens.color.surfaceSunken }}>
            <Typography variant="caption" sx={{ fontWeight: 800, color: tokens.color.info, textTransform: 'uppercase', letterSpacing: 0.6 }}>
              {GATE_SHORT[section] ?? section}
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1 }}>
              {Object.entries(fields).map(([k, v]) => {
                if (typeof v === 'boolean') {
                  return (
                    <Box key={k} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Typography variant="caption" sx={{ color: tokens.color.textDim, fontFamily: 'monospace' }}>{k}</Typography>
                      <Switch size="small" checked={v} onChange={(e) => setField(section, k, e.target.checked)} />
                    </Box>
                  );
                }
                if (typeof v === 'number' || v === null) {
                  return (
                    <TextField
                      key={k}
                      size="small"
                      label={k}
                      type="number"
                      value={v ?? ''}
                      inputProps={{ step: 'any' }}
                      onChange={(e) => setField(section, k, e.target.value === '' ? null : Number(e.target.value))}
                    />
                  );
                }
                if (Array.isArray(v)) {
                  return (
                    <TextField
                      key={k}
                      size="small"
                      label={`${k} (comma-separated)`}
                      value={(v as unknown[]).join(',')}
                      onChange={(e) => setField(section, k, e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
                    />
                  );
                }
                if (typeof v === 'object') {
                  return (
                    <Typography key={k} variant="caption" sx={{ color: tokens.color.textFaint, fontFamily: 'monospace' }}>
                      {k}: {JSON.stringify(v)}
                    </Typography>
                  );
                }
                return (
                  <TextField key={k} size="small" label={k} value={String(v)} onChange={(e) => setField(section, k, e.target.value)} />
                );
              })}
            </Box>
          </Box>
        ))}
      </Box>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
        <Button variant="contained" size="small" disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save policy'}
        </Button>
        <Button size="small" disabled={!dirty} onClick={() => setDraft(JSON.parse(JSON.stringify(policy)))}>
          Discard changes
        </Button>
        {saved ? <Chip size="small" label="saved — applies to the next evaluation" sx={{ bgcolor: tokens.color.successBg, color: tokens.color.success }} /> : null}
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
          The policy is persisted server-side and versioned into every evidence package.
        </Typography>
      </Box>
    </Box>
  );
}

// ------------------------------------------------------------ page

export default function ReleaseGatesPage() {
  const { navigate } = useLabelEval();
  const { runs, error: runsError } = usePublishedRuns();
  const [candidate, setCandidate] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<GateEvaluation | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [showPolicy, setShowPolicy] = useState(false);

  useEffect(() => {
    if (runs?.length && !candidate) {
      setCandidate(runs[0].run_id);
      if (runs.length > 1) setBaseline(runs[1].run_id);
    }
  }, [runs, candidate]);

  // Show the last recorded result for the selected candidate, if any.
  useEffect(() => {
    if (!candidate) return;
    let cancelled = false;
    getGateResult(candidate)
      .then((r) => {
        if (!cancelled) {
          setEvaluation(r);
          if (r.baseline_run_id) setBaseline((b) => b ?? r.baseline_run_id);
        }
      })
      .catch(() => {
        if (!cancelled) setEvaluation(null);
      });
    return () => {
      cancelled = true;
    };
  }, [candidate]);

  const run = useCallback(() => {
    if (!candidate || !baseline) return;
    setRunning(true);
    setError(null);
    evaluateGates(candidate, baseline)
      .then((r) => {
        setEvaluation(r);
        setExpanded(r.blocking_gates[0] ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setRunning(false));
  }, [candidate, baseline]);

  const decisionColor = verdictColor(evaluation?.decision);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {runsError ? <ErrorNote error={runsError} /> : null}
      {error ? <ErrorNote error={error} /> : null}

      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <RunSelect label="Candidate run" value={candidate} onChange={setCandidate} runs={runs} />
        <RunSelect label="Baseline run" value={baseline} onChange={setBaseline} runs={runs} exclude={candidate} />
        <Button variant="contained" startIcon={<PlayCircle size={16} />} disabled={!candidate || !baseline || running} onClick={run}>
          {running ? 'Evaluating gates…' : 'Evaluate gates'}
        </Button>
        <Button size="small" startIcon={<FileText size={14} />} disabled={!evaluation} onClick={() => navigate('safety-evidence', evaluation?.candidate_run_id)}>
          Open evidence package
        </Button>
      </Box>

      {evaluation ? (
        <Alert
          severity={evaluation.decision === 'RELEASE_READY' ? 'success' : 'error'}
          variant="outlined"
          sx={{ borderWidth: 2, '& .MuiAlert-message': { width: '100%' } }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
            <Typography variant="h6" sx={{ fontWeight: 800, color: decisionColor }}>
              {evaluation.decision.replace(/_/g, ' ')}
            </Typography>
            <Typography variant="body2" sx={{ color: tokens.color.textDim }}>
              candidate <code>{evaluation.candidate_run_id.slice(0, 14)}</code> vs baseline{' '}
              <code>{evaluation.baseline_run_id.slice(0, 14)}</code>
              {evaluation.blocking_gates.length ? ` — blocked by: ${evaluation.blocking_gates.join(', ')}` : ' — all gates passed'}
            </Typography>
            <Chip size="small" label={`evidence: ${evaluation.evidence_package_id}`} sx={{ fontFamily: 'monospace', fontSize: 10.5, bgcolor: tokens.color.surfaceRaised }} />
          </Box>
        </Alert>
      ) : null}

      {evaluation ? (
        <SectionCard
          title="Gate pipeline"
          help="A release decision is the conjunction of five deterministic gates, each grounded in a safety-standard practice (ISO 26262 verification, ISO 34503 ODD coverage, regression policy, FHWA SSAM surrogate safety, UL 4600-style evidence). A single BLOCK anywhere blocks the release. Click a stage to expand its actual-vs-threshold checks."
        >
          <GatePipeline gates={evaluation.gates} expanded={expanded} onToggle={(g) => setExpanded((e) => (e === g ? null : g))} />
          <Collapse in={expanded !== null} unmountOnExit>
            {expanded ? (() => {
              const g = evaluation.gates.find((x) => x.gate === expanded);
              return g ? <GateDetail gate={g} /> : null;
            })() : null}
          </Collapse>
        </SectionCard>
      ) : (
        <SectionCard title="Gate pipeline">
          <IllustratedEmpty
            art="gauge"
            title="No gate evaluation yet"
            message="Pick a candidate and baseline run above and evaluate the gates. The pipeline (scenario quality → ODD coverage → regression → surrogate safety → evidence) renders here with per-check pass/fail detail."
            action={
              <Button variant="contained" size="small" startIcon={<PlayCircle size={15} />} disabled={!candidate || !baseline || running} onClick={run}>
                {running ? 'Evaluating…' : 'Evaluate gates'}
              </Button>
            }
          />
        </SectionCard>
      )}

      <SectionCard
        title={
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
            Release policy
            <InfoDot title="Release policy" detail="The thresholds every gate checks against. Edits persist server-side and are stamped (with values) into each Safety Evidence Package, so a decision is always reproducible against the exact policy used." />
          </Box>
        }
        action={
          <Button size="small" endIcon={showPolicy ? <ChevronUp size={14} /> : <ChevronDown size={14} />} onClick={() => setShowPolicy((s) => !s)}>
            {showPolicy ? 'Hide editor' : 'Edit policy'}
          </Button>
        }
      >
        <Collapse in={showPolicy} unmountOnExit>
          <PolicyEditor onSaved={() => undefined} />
        </Collapse>
        {!showPolicy ? (
          <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
            Thresholds for all five gates (geometric pass rate, coverage rate, regression margins, CSI increase ratio…). Open the editor to tune them; changes apply to the next evaluation.
          </Typography>
        ) : null}
      </SectionCard>
    </Box>
  );
}
