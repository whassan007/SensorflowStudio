/** Launch decision panel: policy outcome banner, the deterministic matrix row
 * that fired, option cards with expected loss + hard-constraint feasibility,
 * the human-review form, and the audit trail viewer. */
import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import type {
  AuditRecord,
  HumanReviewDecision,
  PolicyEvaluation,
} from '../../types/agentic';
import { submitHumanReview } from '../../services/agentic';
import { fmtRate, KV, OUTCOME_STYLE, PanelTitle, SeverityChip } from './common';

const OPTION_TITLES: Record<string, string> = {
  STOP_SHIP: 'Stop Ship',
  OPTION_A_DELAY: 'Option A — Delay Launch',
  OPTION_B_MITIGATION: 'Option B — Ship w/ Mitigation',
  OPTION_C_REDUCED_ODD: 'Option C — Reduced ODD',
  EXPAND_EVALUATION: 'Expand Evaluation',
  HUMAN_SAFETY_REVIEW: 'Human Safety Review',
  PROCEED: 'Proceed',
};

const DECISIONS = [
  'confirm_failure',
  'reject_failure',
  'approve_option',
  'approve_launch',
  'block_launch',
  'request_more_evidence',
];

export default function DecisionPanel({
  failureId,
  evaluation,
  decisions,
  audit,
  auditValid,
  onReviewed,
}: {
  failureId: string;
  evaluation: PolicyEvaluation;
  decisions: HumanReviewDecision[];
  audit: AuditRecord[];
  auditValid: boolean | null;
  onReviewed: () => void;
}) {
  const [reviewer, setReviewer] = useState('');
  const [decision, setDecision] = useState('confirm_failure');
  const [rationale, setRationale] = useState('');
  const [overrideReason, setOverrideReason] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAudit, setShowAudit] = useState(false);

  const style = OUTCOME_STYLE[evaluation.outcome];
  const row = evaluation.matrix_row_fired;

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await submitHumanReview(failureId, {
        reviewer,
        decision,
        rationale,
        approved_option: decision === 'approve_option' ? evaluation.recommended_option : null,
        evidence_reviewed: ['evidence_graph', 'statistical_assessment', 'policy_evaluation'],
        override_reason: overrideReason || null,
      });
      setRationale('');
      onReviewed();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {/* outcome banner */}
      <Paper sx={{ p: 1.5, bgcolor: style.bg, color: style.fg }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ fontWeight: 900, fontSize: 17 }}>
            {style.label}
          </Typography>
          <SeverityChip severity={evaluation.severity} />
          <Chip
            size="small"
            label={`policy ${evaluation.policy_version}`}
            sx={{ height: 20, fontSize: 10.5, fontFamily: 'monospace', bgcolor: 'rgba(0,0,0,0.3)', color: style.fg }}
          />
          <Chip
            size="small"
            label="DETERMINISTIC POLICY ENGINE"
            sx={{ height: 20, fontSize: 10, fontWeight: 800, bgcolor: 'rgba(0,0,0,0.3)', color: style.fg }}
          />
        </Box>
        <Typography variant="caption" component="div" sx={{ mt: 0.5, opacity: 0.9 }}>
          {(evaluation.severity_assignment?.taxonomy_description ?? '') +
            (evaluation.automatic_stop_ship_condition
              ? ` · pre-authorized stop-ship condition ${evaluation.automatic_stop_ship_condition.condition_id} fired`
              : '')}
        </Typography>
        {evaluation.indeterminate_reasons?.length ? (
          <Typography variant="caption" component="div" sx={{ mt: 0.5, fontWeight: 700 }}>
            Fail-safe reasons: {evaluation.indeterminate_reasons.join('; ')}
          </Typography>
        ) : null}
      </Paper>

      {/* matrix row */}
      {row ? (
        <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
          <PanelTitle title="Option Selection Matrix — row that fired" origin="deterministic" />
          <KV k={`Row ${row.row}`} v={`${row.condition} → ${OPTION_TITLES[row.option] ?? row.option}`} mono />
          <KV k="Description" v={row.description} />
        </Paper>
      ) : null}

      {/* option cards */}
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
        <PanelTitle title="Options — expected loss & residual risk" origin="deterministic" />
        <Typography variant="caption" component="div" sx={{ color: '#8a949e', mb: 1 }}>
          Expected loss is reported for every option, but SAFETY IS A HARD CONSTRAINT: options whose
          residual failure rate violates the policy limit are excluded from selection no matter how
          cheap they are.
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(215px, 1fr))', gap: 1 }}>
          {evaluation.expected_loss_table.map((o) => {
            const recommended = o.option === evaluation.recommended_option;
            return (
              <Paper
                key={o.option}
                variant="outlined"
                sx={{
                  p: 1,
                  bgcolor: '#12171d',
                  opacity: o.feasible ? 1 : 0.65,
                  border: recommended ? '2px solid #4fc3f7' : undefined,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5, flexWrap: 'wrap' }}>
                  <Typography variant="caption" sx={{ fontWeight: 800, flex: 1 }}>
                    {OPTION_TITLES[o.option] ?? o.option}
                  </Typography>
                  {recommended ? (
                    <Chip size="small" label="RECOMMENDED" sx={{ height: 16, fontSize: 9, fontWeight: 800, bgcolor: '#0d47a1', color: '#90caf9' }} />
                  ) : null}
                  {!o.feasible ? (
                    <Chip size="small" label="INFEASIBLE (SAFETY)" sx={{ height: 16, fontSize: 9, fontWeight: 800, bgcolor: '#b71c1c', color: '#fff' }} />
                  ) : null}
                </Box>
                <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace' }}>
                  expected loss {o.expected_loss.toLocaleString()}
                </Typography>
                <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace' }}>
                  residual rate {fmtRate(o.residual_failure_rate)}
                </Typography>
                <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', color: '#8a949e' }}>
                  business cost {o.business_cost.toLocaleString()}
                </Typography>
                {o.infeasible_reason ? (
                  <Typography variant="caption" component="div" sx={{ color: '#ef9a9a', mt: 0.5 }}>
                    {o.infeasible_reason}
                  </Typography>
                ) : null}
              </Paper>
            );
          })}
        </Box>
        {evaluation.option_c_evaluation && !evaluation.option_c_evaluation.feasible ? (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
              Option C (reduced ODD) rejected checks:
            </Typography>
            {evaluation.option_c_evaluation.checks
              .filter((c) => !c.passed)
              .map((c) => (
                <Typography key={c.check} variant="caption" component="div" sx={{ color: '#ef9a9a' }}>
                  ✗ {c.check} — {c.detail}
                </Typography>
              ))}
          </Box>
        ) : null}
      </Paper>

      {/* mandatory triggers */}
      {evaluation.mandatory_review_triggers?.some((t) => t.fired) ? (
        <Alert severity="warning" variant="outlined" sx={{ py: 0.5 }}>
          Mandatory human review triggers fired:{' '}
          {evaluation.mandatory_review_triggers
            .filter((t) => t.fired)
            .map((t) => t.trigger)
            .join(', ')}
        </Alert>
      ) : null}

      {/* human review form */}
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
        <PanelTitle title="Human Review — final authorization" origin="deterministic" />
        <Typography variant="caption" component="div" sx={{ color: '#8a949e', mb: 1 }}>
          No agent output authorizes a launch. Authorization = deterministic policy outcome above +
          this recorded human decision (against policy {evaluation.policy_version}).
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <TextField size="small" label="Reviewer" value={reviewer} onChange={(e) => setReviewer(e.target.value)} sx={{ width: 180 }} />
          <TextField size="small" select label="Decision" value={decision} onChange={(e) => setDecision(e.target.value)} sx={{ width: 210 }}>
            {DECISIONS.map((d) => (
              <MenuItem key={d} value={d}>
                {d.replace(/_/g, ' ')}
              </MenuItem>
            ))}
          </TextField>
          <TextField size="small" label="Rationale" value={rationale} onChange={(e) => setRationale(e.target.value)} sx={{ flex: 1, minWidth: 260 }} multiline />
          <TextField size="small" label="Override reason (only if overriding policy)" value={overrideReason} onChange={(e) => setOverrideReason(e.target.value)} sx={{ minWidth: 260 }} />
          <Button variant="contained" size="small" disabled={!reviewer || !rationale || submitting} onClick={submit}>
            Record decision
          </Button>
        </Box>
        {error ? (
          <Alert severity="error" variant="outlined" sx={{ mt: 1, py: 0.25 }}>
            {error}
          </Alert>
        ) : null}
        {decisions.length > 0 ? (
          <Box sx={{ mt: 1 }}>
            {decisions.map((d) => (
              <Typography key={d.review_id} variant="caption" component="div" sx={{ color: '#c3ccd4' }}>
                <b>{d.reviewer}</b> — {d.decision.replace(/_/g, ' ')}
                {d.approved_option ? ` (${d.approved_option})` : ''} · {d.timestamp} · policy{' '}
                <span style={{ fontFamily: 'monospace' }}>{d.policy_version || '—'}</span>
                {d.override_reason ? ` · OVERRIDE: ${d.override_reason}` : ''} — “{d.rationale}”
              </Typography>
            ))}
          </Box>
        ) : null}
      </Paper>

      {/* audit trail */}
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
        <PanelTitle
          title="Audit Trail (append-only, hash-chained)"
          origin="deterministic"
          extra={
            <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center' }}>
              {auditValid !== null ? (
                <Chip
                  size="small"
                  label={auditValid ? 'CHAIN VALID' : 'CHAIN BROKEN'}
                  sx={{ height: 18, fontSize: 10, fontWeight: 800, bgcolor: auditValid ? '#1b5e20' : '#b71c1c', color: '#fff' }}
                />
              ) : null}
              <Button size="small" onClick={() => setShowAudit((s) => !s)}>
                {showAudit ? 'Hide' : `Show ${audit.length} records`}
              </Button>
            </Box>
          }
        />
        {showAudit ? (
          <Box sx={{ maxHeight: 280, overflow: 'auto' }}>
            {audit.map((r) => (
              <Typography key={r.seq} variant="caption" component="div" sx={{ fontFamily: 'monospace', fontSize: 10.5, color: '#c3ccd4' }}>
                #{r.seq} [{r.timestamp}] {r.event_type} · {r.actor} · {r.detail}{' '}
                <span style={{ color: '#5c6770' }}>({r.prev_hash} → {r.hash})</span>
              </Typography>
            ))}
          </Box>
        ) : null}
      </Paper>
    </Box>
  );
}
