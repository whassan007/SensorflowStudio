/**
 * Studio 2.0 Governance — the unified control plane UI.
 *
 * Tabs:
 *   Release Board  decisions history + evaluate-candidate + human-approval
 *                  flow (deliberately separated from GO)
 *   Control Plane  entity browser (models / datasets-with-role-badges /
 *                  policies / runs-with-reproducibility-badges / ...)
 *   Hardware Gates combination × metric matrix with pass/fail/insufficient
 *   Funnel         evaluation funnel + safety/compute panels with honest
 *                  UNAVAILABLE flags
 *   Architecture   renders docs/architecture/studio2-review.md
 */
import { useCallback, useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Divider from '@mui/material/Divider';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Tab from '@mui/material/Tab';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import Markdown from '../../components/nextgen/Markdown';
import { fmtNum, fmtPct } from '../../components/labeleval/shared';
import * as api from '../../services/studio2';
import type {
  DemoResult,
  Funnel,
  FunnelPanel,
  HardwareMatrix,
  RegistryEntity,
  ReleaseDecision,
} from '../../services/studio2';

const STATUS_COLORS: Record<string, { bg: string; fg: string }> = {
  GO: { bg: '#1b5e20', fg: '#a5d6a7' },
  REVIEW: { bg: '#e65100', fg: '#ffcc80' },
  NO_GO: { bg: '#b71c1c', fg: '#ef9a9a' },
};

const ROLE_COLORS: Record<string, string> = {
  TRAINING: '#5c6bc0',
  VALIDATION: '#26a69a',
  TEST: '#8d6e63',
  REGRESSION: '#ef5350',
  LAUNCH: '#ab47bc',
  MONITORING: '#78909c',
};

function StatusChip({ status }: { status: string }) {
  const c = STATUS_COLORS[status] ?? { bg: '#37474f', fg: '#cfd8dc' };
  return (
    <Chip
      size="small"
      label={status.replace('_', '-')}
      sx={{ bgcolor: c.bg, color: c.fg, fontWeight: 800, fontSize: 11 }}
    />
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <Box component="span" sx={{ fontFamily: 'monospace', fontSize: 11.5 }}>
      {children}
    </Box>
  );
}

function Unavailable({ reason }: { reason?: string }) {
  return (
    <Chip
      size="small"
      label="UNAVAILABLE"
      title={reason}
      sx={{ bgcolor: '#37474f', color: '#ffcc80', fontWeight: 700, fontSize: 10 }}
    />
  );
}

// ================================================================ Release Board

function EvidenceTuple({ decision }: { decision: ReleaseDecision }) {
  const t = decision.evidence_tuple || {};
  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#0d1117' }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
        Evidence tuple
      </Typography>
      <Table size="small">
        <TableBody>
          {Object.entries(t).map(([k, v]) => (
            <TableRow key={k}>
              <TableCell sx={{ fontSize: 11, color: '#8a949e', width: 190, verticalAlign: 'top' }}>{k}</TableCell>
              <TableCell sx={{ fontSize: 11 }}>
                {v === null ? (
                  <Unavailable reason="subsystem input missing at evaluation time" />
                ) : (
                  <Mono>{typeof v === 'string' ? v : JSON.stringify(v)}</Mono>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}

function DecisionDetail({
  decision,
  onApproved,
}: {
  decision: ReleaseDecision;
  onApproved: () => void;
}) {
  const [approver, setApprover] = useState('');
  const [rationale, setRationale] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const c = STATUS_COLORS[decision.status] ?? STATUS_COLORS.REVIEW;

  const approve = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.approveDecision(decision.entity_id, approver, rationale);
      onApproved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Stack spacing={1.5}>
      <Paper sx={{ p: 2, bgcolor: c.bg, color: c.fg }}>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <Typography variant="h5" sx={{ fontWeight: 900 }}>
            {decision.status.replace('_', '-')}
          </Typography>
          <Chip size="small" label={`confidence ${decision.confidence}`} sx={{ bgcolor: 'rgba(0,0,0,0.25)', color: 'inherit' }} />
          <Chip
            size="small"
            label={`evidence completeness ${fmtPct(decision.evidence_completeness)}`}
            sx={{ bgcolor: 'rgba(0,0,0,0.25)', color: 'inherit' }}
          />
          <Chip size="small" label={`policy ${decision.policy_version}`} sx={{ bgcolor: 'rgba(0,0,0,0.25)', color: 'inherit', fontFamily: 'monospace' }} />
          <Mono>{decision.entity_id}</Mono>
        </Stack>
        <Typography variant="body2" sx={{ mt: 1, opacity: 0.9 }}>
          {decision.status === 'GO'
            ? 'GO is a recommendation, not a deployment. Deployment requires the separate human approval below.'
            : decision.status === 'REVIEW'
              ? 'Evidence is incomplete or raises questions — human review required before this candidate can be re-evaluated for GO.'
              : 'Blocking conditions present — the candidate cannot ship until they are resolved and re-evaluated.'}
        </Typography>
      </Paper>

      {decision.blocking_conditions.length > 0 && (
        <Alert severity="error" variant="outlined">
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Blocking conditions</Typography>
          {decision.blocking_conditions.map((b, i) => (
            <Typography key={i} variant="body2">• {b}</Typography>
          ))}
        </Alert>
      )}
      {decision.degraded_inputs.length > 0 && (
        <Alert severity="warning" variant="outlined">
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Degraded inputs (named gaps — never a silent GO)</Typography>
          {decision.degraded_inputs.map((g, i) => (
            <Typography key={i} variant="body2">• {g}</Typography>
          ))}
        </Alert>
      )}
      {decision.unresolved_questions.length > 0 && (
        <Alert severity="info" variant="outlined">
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Unresolved questions</Typography>
          {decision.unresolved_questions.map((q, i) => (
            <Typography key={i} variant="body2">• {q}</Typography>
          ))}
        </Alert>
      )}

      <EvidenceTuple decision={decision} />

      <Paper variant="outlined" sx={{ p: 1.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
          Human deployment approval
        </Typography>
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
          Separate recorded action — a GO decision never auto-authorizes deployment. Only GO decisions can be
          approved; the approval records who and why in the audit trail.
        </Typography>
        {decision.deployment_authorized && decision.approval ? (
          <Alert severity="success" variant="outlined">
            Deployment authorized by <strong>{decision.approval.approver}</strong> at {decision.approval.approved_at}:
            &nbsp;“{decision.approval.rationale}”
          </Alert>
        ) : (
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <TextField size="small" label="Approver" value={approver} onChange={(e) => setApprover(e.target.value)} sx={{ width: 180 }} />
            <TextField size="small" label="Rationale" value={rationale} onChange={(e) => setRationale(e.target.value)} sx={{ flex: 1, minWidth: 240 }} />
            <Tooltip title={decision.status !== 'GO' ? 'Only GO decisions can be approved for deployment' : ''}>
              <span>
                <Button
                  variant="contained"
                  color="success"
                  disabled={busy || decision.status !== 'GO' || !approver.trim() || !rationale.trim()}
                  onClick={approve}
                >
                  Authorize deployment
                </Button>
              </span>
            </Tooltip>
          </Stack>
        )}
        {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
      </Paper>
    </Stack>
  );
}

function ReleaseBoard() {
  const [decisions, setDecisions] = useState<ReleaseDecision[]>([]);
  const [selected, setSelected] = useState<ReleaseDecision | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState<DemoResult | null>(null);

  const refresh = useCallback(async () => {
    const res = await api.listDecisions();
    setDecisions(res.decisions);
    setSelected((cur) => {
      if (!cur) return res.decisions[0] ?? null;
      return res.decisions.find((d) => d.entity_id === cur.entity_id) ?? res.decisions[0] ?? null;
    });
  }, []);

  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
    api.getLatestDemo().then(setDemo).catch(() => undefined);
  }, [refresh]);

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <Stack spacing={1.5}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Button
          variant="contained"
          disabled={busy !== null}
          onClick={() => act('evaluate', () => api.evaluateRelease())}
        >
          {busy === 'evaluate' ? 'Evaluating…' : 'Evaluate candidate (live sources)'}
        </Button>
        <Button
          variant="outlined"
          disabled={busy !== null}
          onClick={() => act('demo', async () => setDemo(await api.runDemo()))}
        >
          {busy === 'demo' ? 'Running…' : 'Run closed-loop demo'}
        </Button>
        {busy && <CircularProgress size={18} />}
      </Stack>
      {error && <Alert severity="error">{error}</Alert>}

      {demo && (
        <Paper variant="outlined" sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
            Latest closed-loop demo <Mono>{demo.demo_id}</Mono> (seed {demo.seed}) →{' '}
            <StatusChip status={demo.decision.status} />
          </Typography>
          <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
            {demo.steps.map((s) => (
              <Tooltip
                key={s.step}
                title={s.available ? JSON.stringify(s, null, 1) : String(s.reason ?? '')}
              >
                <Chip
                  size="small"
                  label={s.step}
                  sx={{
                    bgcolor: s.available ? '#1b5e20' : '#37474f',
                    color: s.available ? '#a5d6a7' : '#ffcc80',
                    fontSize: 10.5,
                  }}
                />
              </Tooltip>
            ))}
          </Stack>
          {demo.regression_dataset && (
            <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 0.5 }}>
              Flywheel: failure registered as protected REGRESSION dataset{' '}
              <Mono>{demo.regression_dataset.entity_id}</Mono>
            </Typography>
          )}
        </Paper>
      )}

      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={1.5} alignItems="flex-start">
        <Paper variant="outlined" sx={{ width: { xs: '100%', lg: 380 }, flexShrink: 0 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 800, p: 1.25, pb: 0.5 }}>
            Decision history ({decisions.length})
          </Typography>
          <Divider />
          <Box sx={{ maxHeight: 480, overflowY: 'auto' }}>
            {decisions.length === 0 && (
              <Typography variant="body2" sx={{ p: 1.5, color: '#8a949e' }}>
                No release decisions yet — evaluate a candidate or run the demo.
              </Typography>
            )}
            {decisions.map((d) => (
              <Box
                key={d.entity_id}
                onClick={() => setSelected(d)}
                sx={{
                  px: 1.25,
                  py: 0.75,
                  cursor: 'pointer',
                  bgcolor: selected?.entity_id === d.entity_id ? 'rgba(79,195,247,0.10)' : 'transparent',
                  borderBottom: '1px solid #232a31',
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' },
                }}
              >
                <Stack direction="row" spacing={1} alignItems="center">
                  <StatusChip status={d.status} />
                  {d.deployment_authorized && (
                    <Chip size="small" label="DEPLOY AUTHORIZED" sx={{ bgcolor: '#0d47a1', color: '#90caf9', fontSize: 9.5, fontWeight: 800 }} />
                  )}
                  <Mono>{d.entity_id}</Mono>
                </Stack>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  {d.evaluated_at} · completeness {fmtPct(d.evidence_completeness)}
                </Typography>
              </Box>
            ))}
          </Box>
        </Paper>
        <Box sx={{ flex: 1, minWidth: 0, width: '100%' }}>
          {selected ? (
            <DecisionDetail decision={selected} onApproved={refresh} />
          ) : (
            <Typography variant="body2" sx={{ color: '#8a949e', p: 2 }}>
              Select a decision to inspect its evidence tuple and approval state.
            </Typography>
          )}
        </Box>
      </Stack>
    </Stack>
  );
}

// ================================================================ Control Plane

const BROWSER_KINDS = ['models', 'datasets', 'scenarios', 'policies', 'experiments', 'runs', 'safety_cases', 'approvals'];

function ControlPlane() {
  const [kind, setKind] = useState('datasets');
  const [entities, setEntities] = useState<RegistryEntity[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (k: string) => {
    const res = await api.listEntities(k);
    setEntities(res.entities);
  }, []);

  const refreshCounts = useCallback(async () => {
    const s = await api.getStatus();
    setCounts(s.registry_counts);
  }, []);

  useEffect(() => {
    load(kind).catch((e) => setError(String(e)));
    refreshCounts().catch(() => undefined);
  }, [kind, load, refreshCounts]);

  const ingest = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.runIngest();
      await load(kind);
      await refreshCounts();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Stack spacing={1.5}>
      <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
        {BROWSER_KINDS.map((k) => (
          <Chip
            key={k}
            label={`${k} (${counts[k] ?? 0})`}
            size="small"
            onClick={() => setKind(k)}
            sx={{
              bgcolor: kind === k ? 'rgba(79,195,247,0.18)' : '#232a31',
              color: kind === k ? '#4fc3f7' : '#c7ccd1',
              fontWeight: kind === k ? 800 : 500,
            }}
          />
        ))}
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="outlined" disabled={busy} onClick={ingest}>
          {busy ? 'Ingesting…' : 'Auto-ingest existing stores'}
        </Button>
      </Stack>
      {error && <Alert severity="error">{error}</Alert>}
      <Paper variant="outlined" sx={{ overflowX: 'auto' }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 800 }}>Entity</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Name</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Badges</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Provenance</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Detail</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {entities.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} sx={{ color: '#8a949e' }}>
                  No {kind} registered yet. Auto-ingest scans megaeval / seqeval / safety / agentic /
                  bevfusion stores retroactively.
                </TableCell>
              </TableRow>
            )}
            {entities.map((e) => {
              const role = typeof e.role === 'string' ? e.role : null;
              const repro = typeof e.reproducibility === 'string' ? (e.reproducibility as string) : null;
              const missing = Array.isArray(e.missing_components) ? (e.missing_components as string[]) : [];
              const overrides = Array.isArray(e.governance_overrides) ? e.governance_overrides.length : 0;
              return (
                <TableRow key={e.entity_id} hover>
                  <TableCell><Mono>{e.entity_id}</Mono></TableCell>
                  <TableCell sx={{ fontSize: 12 }}>{String(e.name ?? e.policy_version ?? '—')}</TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                      {role && (
                        <Chip size="small" label={role} sx={{ bgcolor: ROLE_COLORS[role] ?? '#37474f', color: '#fff', fontWeight: 800, fontSize: 10 }} />
                      )}
                      {e.protected_evaluation === true && (
                        <Tooltip title="Contamination guard: cannot move to TRAINING without an audited governance override">
                          <Chip size="small" label="PROTECTED" sx={{ bgcolor: '#4a148c', color: '#ce93d8', fontSize: 10, fontWeight: 800 }} />
                        </Tooltip>
                      )}
                      {overrides > 0 && (
                        <Chip size="small" label={`${overrides} override${overrides > 1 ? 's' : ''}`} sx={{ bgcolor: '#e65100', color: '#ffcc80', fontSize: 10 }} />
                      )}
                      {repro && (
                        <Tooltip title={missing.length ? `missing: ${missing.join(', ')}` : 'full model/dataset/scenario/config/calibration/seed/policy tuple'}>
                          <Chip
                            size="small"
                            label={repro === 'REPRODUCIBLE' ? 'REPRODUCIBLE' : 'NON-REPRODUCIBLE'}
                            sx={{
                              bgcolor: repro === 'REPRODUCIBLE' ? '#1b5e20' : '#b71c1c',
                              color: repro === 'REPRODUCIBLE' ? '#a5d6a7' : '#ef9a9a',
                              fontSize: 10,
                              fontWeight: 800,
                            }}
                          />
                        </Tooltip>
                      )}
                      {typeof e.engine === 'string' && (
                        <Chip size="small" label={e.engine as string} sx={{ bgcolor: '#232a31', fontSize: 10 }} />
                      )}
                      {typeof e.case_kind === 'string' && (
                        <Chip size="small" label={e.case_kind as string} sx={{ bgcolor: '#232a31', fontSize: 10 }} />
                      )}
                    </Stack>
                  </TableCell>
                  <TableCell sx={{ fontSize: 11, color: '#8a949e' }}>
                    {String((e.provenance as Record<string, unknown>)?.source_package ?? '—')}
                  </TableCell>
                  <TableCell sx={{ fontSize: 11, color: '#8a949e', maxWidth: 340 }}>
                    {missing.length > 0
                      ? `missing: ${missing.join(', ')}`
                      : typeof e.status === 'string'
                        ? String(e.status)
                        : String(e.created_at ?? '')}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>
    </Stack>
  );
}

// ================================================================ Hardware Gates

const CELL_COLORS: Record<string, { bg: string; fg: string }> = {
  PASS: { bg: 'rgba(46,125,50,0.25)', fg: '#a5d6a7' },
  FAIL: { bg: 'rgba(183,28,28,0.35)', fg: '#ef9a9a' },
  INSUFFICIENT: { bg: 'rgba(230,81,0,0.25)', fg: '#ffcc80' },
};

function HardwareGates() {
  const [matrix, setMatrix] = useState<HardwareMatrix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async (refresh: boolean) => {
    setBusy(true);
    setError(null);
    try {
      setMatrix(await api.getHardwareMatrix(refresh));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    load(false);
  }, [load]);

  if (error) return <Alert severity="warning">Hardware matrix unavailable: {error}</Alert>;
  if (!matrix) return <CircularProgress size={22} />;

  const metricNames = Object.keys(matrix.rows.find((r) => r.metrics)?.metrics ?? {});

  return (
    <Stack spacing={1.5}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <StatusChip status={matrix.status === 'PASS' ? 'GO' : matrix.status === 'FAIL_CRITICAL' ? 'NO_GO' : 'REVIEW'} />
        <Typography variant="body2">
          matrix <Mono>{matrix.matrix_id}</Mono> · {matrix.n_pass} pass / {matrix.n_fail} fail /{' '}
          {matrix.n_insufficient} insufficient of {matrix.n_combinations} combinations
        </Typography>
        <Chip size="small" label={`min support n=${matrix.min_support.n}`} title={matrix.min_support.method} sx={{ bgcolor: '#232a31', fontSize: 10.5 }} />
        {matrix.source_run_id && <Chip size="small" label={`source ${matrix.source_run_id}`} sx={{ bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 10.5 }} />}
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="outlined" disabled={busy} onClick={() => load(true)}>
          {busy ? 'Recomputing…' : 'Recompute'}
        </Button>
      </Stack>
      {matrix.global_vs_matrix_note && <Alert severity="error">{matrix.global_vs_matrix_note}</Alert>}
      {matrix.global_metrics && (
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          Global aggregate: {Object.entries(matrix.global_metrics).map(([k, v]) => `${k}=${typeof v === 'number' ? fmtNum(v) : v}`).join(' · ')} — a
          global pass never overrides a failing critical combination.
        </Typography>
      )}
      <Paper variant="outlined" sx={{ overflowX: 'auto' }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 800 }}>Combination (region × platform × sensor gen)</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>n</TableCell>
              {metricNames.map((m) => (
                <TableCell key={m} sx={{ fontWeight: 800 }}>{m}</TableCell>
              ))}
              <TableCell sx={{ fontWeight: 800 }}>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {matrix.rows.map((r) => {
              const c = CELL_COLORS[r.status];
              return (
                <TableRow key={r.combination_label} sx={{ outline: r.critical ? '1px solid rgba(171,71,188,0.5)' : undefined }}>
                  <TableCell sx={{ fontSize: 11.5 }}>
                    {r.combination_label}
                    {r.critical && (
                      <Chip size="small" label="CRITICAL" sx={{ ml: 0.75, bgcolor: '#4a148c', color: '#ce93d8', fontSize: 9, height: 16 }} />
                    )}
                    <Typography variant="caption" sx={{ display: 'block', color: '#657078' }}>
                      calib {r.combination.calibration_version} · fw {r.combination.firmware}
                    </Typography>
                  </TableCell>
                  <TableCell sx={{ fontSize: 11 }}>{r.n.toLocaleString()}</TableCell>
                  {metricNames.map((m) => (
                    <TableCell key={m} sx={{ fontSize: 11, bgcolor: c.bg, color: c.fg, fontFamily: 'monospace' }}>
                      {r.metrics?.[m] !== undefined && r.metrics?.[m] !== null ? fmtNum(r.metrics[m] as number) : '—'}
                    </TableCell>
                  ))}
                  <TableCell>
                    <Tooltip title={r.failed_checks?.join('; ') ?? r.reason ?? r.derivation?.metrics_source ?? ''}>
                      <Chip size="small" label={r.status} sx={{ bgcolor: c.bg, color: c.fg, fontWeight: 800, fontSize: 10 }} />
                    </Tooltip>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>
      {matrix.insufficient.length > 0 && (
        <Alert severity="warning" variant="outlined">
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Combinations lacking sufficient evidence</Typography>
          {matrix.insufficient.map((iRow, idx) => (
            <Typography key={idx} variant="body2">
              • {iRow.combination_label}: {iRow.reason}
            </Typography>
          ))}
        </Alert>
      )}
    </Stack>
  );
}

// ================================================================ Funnel

function PanelCard({ title, panel }: { title: string; panel: FunnelPanel }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, flex: 1, minWidth: 260 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.75 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, flex: 1 }}>{title}</Typography>
        {!panel.available && <Unavailable reason={panel.reason} />}
      </Stack>
      {panel.available ? (
        <Box component="pre" sx={{ m: 0, fontSize: 10.5, fontFamily: 'monospace', whiteSpace: 'pre-wrap', color: '#c7ccd1', maxHeight: 260, overflowY: 'auto' }}>
          {JSON.stringify(panel.data, null, 1)}
        </Box>
      ) : (
        <Typography variant="caption" sx={{ color: '#8a949e' }}>{panel.reason}</Typography>
      )}
      <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: '#657078' }}>
        source: {panel.source}
      </Typography>
    </Paper>
  );
}

function FunnelTab() {
  const [funnel, setFunnel] = useState<Funnel | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getFunnel().then(setFunnel).catch((e) => setError(String(e)));
  }, []);

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!funnel) return <CircularProgress size={22} />;

  const numericOf = (s: (typeof funnel.stages)[number]): number | null => {
    if (!s.available || !s.data) return null;
    const vals = Object.values(s.data).filter((v): v is number => typeof v === 'number');
    return vals.length ? Math.max(...vals) : null;
  };
  const maxVal = Math.max(1, ...funnel.stages.map((s) => numericOf(s) ?? 0));

  return (
    <Stack spacing={1.5}>
      <Paper variant="outlined" sx={{ p: 1.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
          Evaluation funnel — raw → selected → simulated → evaluated → failed → HITL → regression suite
        </Typography>
        <Stack spacing={0.75}>
          {funnel.stages.map((s) => {
            const v = numericOf(s);
            const width = v !== null ? Math.max(2, (Math.log10(v + 1) / Math.log10(maxVal + 1)) * 100) : 0;
            return (
              <Stack key={s.stage} direction="row" spacing={1} alignItems="center">
                <Typography variant="caption" sx={{ width: 210, flexShrink: 0, color: '#c7ccd1' }}>
                  {s.label}
                </Typography>
                {s.available ? (
                  <>
                    <Box sx={{ height: 16, width: `${width}%`, bgcolor: '#0d47a1', borderRadius: 0.5, minWidth: 4 }} />
                    <Tooltip title={JSON.stringify(s.data)}>
                      <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#90caf9' }}>
                        {v !== null ? v.toLocaleString() : '—'}
                      </Typography>
                    </Tooltip>
                  </>
                ) : (
                  <Unavailable reason={s.reason} />
                )}
                <Typography variant="caption" sx={{ color: '#657078', ml: 'auto !important', flexShrink: 0 }}>
                  {s.source}
                </Typography>
              </Stack>
            );
          })}
        </Stack>
      </Paper>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} useFlexGap flexWrap="wrap">
        <PanelCard title="Safety (SCR · VRU miss rate · TTC)" panel={funnel.safety} />
        <PanelCard title="Model comparison" panel={funnel.model_comparison} />
      </Stack>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} useFlexGap flexWrap="wrap">
        <PanelCard title="Distribution drift" panel={funnel.drift} />
        <PanelCard title="Compute (cache · gauntlet · throughput)" panel={funnel.compute} />
      </Stack>
    </Stack>
  );
}

// ================================================================ Architecture

function ArchitectureTab() {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getDoc('studio2-review.md')
      .then((d) => setContent(d.content))
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <Alert severity="error">{error}</Alert>;
  if (content === null) return <CircularProgress size={22} />;
  return (
    <Paper variant="outlined" sx={{ p: 2.5, maxWidth: 1100 }}>
      <Markdown source={content} />
    </Paper>
  );
}

// ================================================================ page shell

const TABS = ['Release Board', 'Control Plane', 'Hardware Gates', 'Funnel', 'Architecture'];

export default function Studio2GovernancePage() {
  const [tab, setTab] = useState(0);
  return (
    <Box>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 1.5, minHeight: 36 }}>
        {TABS.map((t) => (
          <Tab key={t} label={t} sx={{ minHeight: 36, fontSize: 12.5, fontWeight: 700 }} />
        ))}
      </Tabs>
      {tab === 0 && <ReleaseBoard />}
      {tab === 1 && <ControlPlane />}
      {tab === 2 && <HardwareGates />}
      {tab === 3 && <FunnelTab />}
      {tab === 4 && <ArchitectureTab />}
    </Box>
  );
}
