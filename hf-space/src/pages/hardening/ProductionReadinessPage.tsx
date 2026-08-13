/**
 * Production Readiness: audit browser (filterable findings with file:line
 * refs), readiness scorecard grid, data-funnel visualization, and a
 * remediation kanban (fix-now vs follow-up, effort badges).
 *
 * All data is computed server-side from docs/hardening/audit.json — this
 * page never invents a status.
 */

import { useEffect, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { getAudit, getFunnel, getReadiness } from '../../services/hardening';
import type {
  AuditDocument,
  AuditFinding,
  FunnelResponse,
  ReadinessScorecard,
} from '../../types/hardening';

const SEVERITY_COLORS: Record<AuditFinding['severity'], string> = {
  Critical: '#d32f2f',
  High: '#e65100',
  Medium: '#f9a825',
  Low: '#607d8b',
};

const STATUS_STYLES: Record<string, { label: string; color: string; bg: string }> = {
  blocked_critical: { label: 'BLOCKED (CRITICAL)', color: '#ff8a80', bg: 'rgba(211,47,47,0.15)' },
  gaps_open: { label: 'GAPS OPEN', color: '#ffb74d', bg: 'rgba(230,81,0,0.15)' },
  partially_hardened: { label: 'PARTIALLY HARDENED', color: '#fff176', bg: 'rgba(249,168,37,0.12)' },
  closed: { label: 'CLOSED', color: '#81c784', bg: 'rgba(46,125,50,0.15)' },
  no_findings: { label: 'NO FINDINGS', color: '#90a4ae', bg: 'rgba(96,125,139,0.15)' },
};

const DISPOSITION_LABELS: Record<AuditFinding['disposition'], string> = {
  fix_now: 'fixed now',
  fix_now_partial: 'partially fixed',
  fix_now_layered: 'fixed (layered)',
  follow_up: 'follow-up',
};

const EFFORT_COLORS: Record<AuditFinding['effort'], string> = {
  S: '#2e7d32',
  M: '#e65100',
  L: '#b71c1c',
};

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <Typography variant="subtitle1" sx={{ fontWeight: 800, mt: 3, mb: 1 }}>
      {children}
    </Typography>
  );
}

function FindingCard({ f }: { f: AuditFinding }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.25, mb: 1, bgcolor: '#161c22' }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5, flexWrap: 'wrap' }}>
        <Typography sx={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 12 }}>{f.id}</Typography>
        <Chip
          size="small"
          label={f.severity}
          sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: SEVERITY_COLORS[f.severity], color: '#fff' }}
        />
        <Chip
          size="small"
          label={`effort ${f.effort}`}
          sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: EFFORT_COLORS[f.effort], color: '#fff' }}
        />
        <Typography sx={{ fontSize: 11, color: '#8a949e' }}>{f.area}</Typography>
      </Stack>
      <Typography sx={{ fontSize: 12.5, mb: 0.5 }}>{f.problem}</Typography>
      <Typography sx={{ fontSize: 11.5, color: '#8a949e' }}>{f.disposition_reason}</Typography>
    </Paper>
  );
}

export default function ProductionReadinessPage() {
  const [audit, setAudit] = useState<AuditDocument | null>(null);
  const [readiness, setReadiness] = useState<ReadinessScorecard | null>(null);
  const [funnel, setFunnel] = useState<FunnelResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [areaFilter, setAreaFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [dispositionFilter, setDispositionFilter] = useState('all');

  useEffect(() => {
    Promise.all([getAudit(), getReadiness(), getFunnel()])
      .then(([a, r, f]) => {
        setAudit(a);
        setReadiness(r);
        setFunnel(f);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const areas = useMemo(
    () => Array.from(new Set((audit?.findings ?? []).map((f) => f.area))).sort(),
    [audit]
  );

  const filtered = useMemo(
    () =>
      (audit?.findings ?? []).filter(
        (f) =>
          (areaFilter === 'all' || f.area === areaFilter) &&
          (severityFilter === 'all' || f.severity === severityFilter) &&
          (dispositionFilter === 'all' ||
            (dispositionFilter === 'fixed'
              ? f.disposition !== 'follow_up'
              : f.disposition === 'follow_up'))
      ),
    [audit, areaFilter, severityFilter, dispositionFilter]
  );

  if (error) {
    return <Typography color="error">Failed to load audit data: {error}</Typography>;
  }
  if (!audit || !readiness) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 6 }}>
        <CircularProgress size={28} />
      </Box>
    );
  }

  const s = audit.summary;
  const fixNow = audit.findings.filter((f) => f.disposition !== 'follow_up');
  const followUp = audit.findings.filter((f) => f.disposition === 'follow_up');
  const maxFunnel = Math.max(...(funnel?.stages ?? []).map((st) => st.count), 1);

  return (
    <Box>
      {/* ---- overall banner + severity summary ---- */}
      <Paper
        variant="outlined"
        sx={{
          p: 1.5,
          mb: 2,
          bgcolor: 'rgba(211,47,47,0.08)',
          border: '1px solid rgba(211,47,47,0.4)',
        }}
      >
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ flexWrap: 'wrap' }}>
          <Chip
            label={readiness.overall_status.replace(/_/g, ' ')}
            sx={{ fontWeight: 800, bgcolor: '#d32f2f', color: '#fff' }}
          />
          <Typography sx={{ fontSize: 12.5, color: '#c0c8d0' }}>{readiness.rule}</Typography>
          <Box sx={{ flex: 1 }} />
          {(['critical', 'high', 'medium', 'low'] as const).map((sev) => (
            <Chip
              key={sev}
              size="small"
              label={`${sev}: ${s[sev]}`}
              sx={{
                fontWeight: 700,
                textTransform: 'capitalize',
                bgcolor: SEVERITY_COLORS[(sev[0].toUpperCase() + sev.slice(1)) as AuditFinding['severity']],
                color: '#fff',
              }}
            />
          ))}
          <Chip
            size="small"
            label={`fixed now: ${s.fixed_now + s.fixed_now_partial + s.fixed_now_layered}`}
            sx={{ fontWeight: 700, bgcolor: '#2e7d32', color: '#fff' }}
          />
          <Chip size="small" label={`deferred: ${s.deferred}`} sx={{ fontWeight: 700, bgcolor: '#455a64', color: '#fff' }} />
        </Stack>
      </Paper>

      {/* ---- readiness scorecard grid ---- */}
      <SectionTitle>Readiness scorecard (computed from audit.json)</SectionTitle>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr', lg: '1fr 1fr 1fr 1fr' },
          gap: 1.25,
        }}
      >
        {readiness.categories.map((c) => {
          const st = STATUS_STYLES[c.status] ?? STATUS_STYLES.no_findings;
          return (
            <Paper key={c.category} variant="outlined" sx={{ p: 1.25, bgcolor: '#161c22' }}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.75 }}>
                <Typography sx={{ fontWeight: 800, fontSize: 13, flex: 1 }}>{c.category}</Typography>
                <Chip size="small" label={st.label} sx={{ height: 18, fontSize: 9.5, fontWeight: 800, color: st.color, bgcolor: st.bg }} />
              </Stack>
              <Typography sx={{ fontSize: 11.5, color: '#8a949e', mb: 0.5 }}>
                <b>Prototype:</b> {c.prototype}
              </Typography>
              <Typography sx={{ fontSize: 11.5, color: '#8a949e', mb: 0.5 }}>
                <b>Production:</b> {c.production_requirement}
              </Typography>
              <Typography sx={{ fontSize: 11.5, color: c.gap_count ? '#ffb74d' : '#81c784' }}>
                <b>Gap:</b>{' '}
                {c.gap_count
                  ? `${c.gap_count} open (${c.open_finding_ids.join(', ')})`
                  : 'no open findings'}
                {c.partially_fixed_ids.length ? ` · partial: ${c.partially_fixed_ids.join(', ')}` : ''}
              </Typography>
            </Paper>
          );
        })}
      </Box>

      {/* ---- data funnel ---- */}
      <SectionTitle>Data funnel (live from {funnel?.available ? funnel.store : 'store unavailable'})</SectionTitle>
      {funnel?.available && funnel.stages ? (
        <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161c22' }}>
          {funnel.stages.map((st) => (
            <Stack key={st.stage} direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
              <Typography sx={{ width: 160, fontSize: 12, color: '#c0c8d0' }}>{st.stage}</Typography>
              <Box
                sx={{
                  height: 14,
                  width: `${Math.max((st.count / maxFunnel) * 100, 0.5)}%`,
                  bgcolor: '#4fc3f7',
                  borderRadius: 0.5,
                  opacity: 0.85,
                }}
              />
              <Typography sx={{ fontSize: 12, fontFamily: 'monospace' }}>{st.count.toLocaleString()}</Typography>
            </Stack>
          ))}
          {funnel.triage_breakdown ? (
            <Typography sx={{ fontSize: 11.5, color: '#8a949e', mt: 1 }}>
              triage: {Object.entries(funnel.triage_breakdown).map(([k, v]) => `${k}=${v}`).join(' · ')}
            </Typography>
          ) : null}
        </Paper>
      ) : (
        <Typography sx={{ fontSize: 12.5, color: '#8a949e' }}>{funnel?.note ?? 'Funnel store not available.'}</Typography>
      )}

      {/* ---- findings table ---- */}
      <SectionTitle>Audit findings ({filtered.length} of {audit.findings.length})</SectionTitle>
      <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
        <TextField select size="small" label="Area" value={areaFilter} onChange={(e) => setAreaFilter(e.target.value)} sx={{ minWidth: 220 }}>
          <MenuItem value="all">All areas</MenuItem>
          {areas.map((a) => (
            <MenuItem key={a} value={a}>{a}</MenuItem>
          ))}
        </TextField>
        <TextField select size="small" label="Severity" value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)} sx={{ minWidth: 140 }}>
          <MenuItem value="all">All severities</MenuItem>
          {(['Critical', 'High', 'Medium', 'Low'] as const).map((sev) => (
            <MenuItem key={sev} value={sev}>{sev}</MenuItem>
          ))}
        </TextField>
        <TextField select size="small" label="Disposition" value={dispositionFilter} onChange={(e) => setDispositionFilter(e.target.value)} sx={{ minWidth: 140 }}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="fixed">Fixed now</MenuItem>
          <MenuItem value="follow_up">Follow-up</MenuItem>
        </TextField>
      </Stack>
      <Paper variant="outlined" sx={{ bgcolor: '#161c22', overflowX: 'auto' }}>
        <Table size="small" sx={{ minWidth: 900 }}>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 800 }}>ID</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Area</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Severity</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Refs</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Problem</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Correct approach</TableCell>
              <TableCell sx={{ fontWeight: 800 }}>Disposition</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filtered.map((f) => (
              <TableRow key={f.id} hover>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11.5, whiteSpace: 'nowrap' }}>{f.id}</TableCell>
                <TableCell sx={{ fontSize: 11.5 }}>{f.area}</TableCell>
                <TableCell>
                  <Chip size="small" label={f.severity} sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: SEVERITY_COLORS[f.severity], color: '#fff' }} />
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 10.5, maxWidth: 190 }}>
                  {f.refs.map((r) => (
                    <div key={r}>{r}</div>
                  ))}
                </TableCell>
                <TableCell sx={{ fontSize: 11.5, maxWidth: 320 }}>{f.problem}</TableCell>
                <TableCell sx={{ fontSize: 11.5, maxWidth: 280 }}>{f.correct_approach}</TableCell>
                <TableCell sx={{ whiteSpace: 'nowrap' }}>
                  <Chip
                    size="small"
                    label={DISPOSITION_LABELS[f.disposition]}
                    sx={{
                      height: 18,
                      fontSize: 10,
                      fontWeight: 700,
                      bgcolor: f.disposition === 'follow_up' ? '#455a64' : '#2e7d32',
                      color: '#fff',
                    }}
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {/* ---- remediation kanban ---- */}
      <SectionTitle>Remediation plan</SectionTitle>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1.5 }}>
        <Box>
          <Typography sx={{ fontWeight: 800, fontSize: 12.5, color: '#81c784', mb: 0.75 }}>
            FIX-NOW (implemented in this pass) · {fixNow.length}
          </Typography>
          {fixNow.map((f) => (
            <FindingCard key={f.id} f={f} />
          ))}
        </Box>
        <Box>
          <Typography sx={{ fontWeight: 800, fontSize: 12.5, color: '#ffb74d', mb: 0.75 }}>
            FOLLOW-UP (documented, with reasons) · {followUp.length}
          </Typography>
          {followUp.map((f) => (
            <FindingCard key={f.id} f={f} />
          ))}
        </Box>
      </Box>

      {/* ---- strengths ---- */}
      <SectionTitle>Verified strengths (reused, not reinvented)</SectionTitle>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1 }}>
        {audit.strengths.map((st) => (
          <Paper key={st.package} variant="outlined" sx={{ p: 1.25, bgcolor: '#161c22' }}>
            <Typography sx={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 12, color: '#81c784' }}>
              {st.package}
            </Typography>
            <Typography sx={{ fontSize: 11.5, color: '#c0c8d0' }}>{st.what}</Typography>
          </Paper>
        ))}
      </Box>
    </Box>
  );
}
