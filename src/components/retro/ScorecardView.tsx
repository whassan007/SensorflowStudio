/** Scorecard rendering: severity + launch banner, kinematics, statistics. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import { AlertTriangle, UserCheck } from 'lucide-react';
import type { RetrospectiveScorecard } from '../../types/retro';
import { LAUNCH_COLORS, SEVERITY_COLORS } from './tierTheme';

function Stat({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <Box sx={{ minWidth: 130 }}>
      <Typography sx={{ fontSize: 10.5, color: '#8a949e', letterSpacing: 0.6 }}>{label}</Typography>
      <Typography sx={{ fontSize: 13.5, fontFamily: mono ? 'monospace' : undefined, fontWeight: 600 }}>
        {value}
      </Typography>
    </Box>
  );
}

const fmt = (v: number | null | undefined, unit = '', digits = 2) =>
  v === null || v === undefined ? 'UNKNOWN' : `${v.toFixed(digits)}${unit}`;

export default function ScorecardView({ scorecard: sc }: { scorecard: RetrospectiveScorecard }) {
  const sevColor = SEVERITY_COLORS[sc.severity];
  const launchColor = LAUNCH_COLORS[sc.launch_recommendation];

  return (
    <Box>
      {/* severity + launch banner */}
      <Paper sx={{ p: 1.5, mb: 1.5, bgcolor: `${launchColor}14`, border: `1px solid ${launchColor}55`,
                   display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <Chip label={`SEVERITY: ${sc.severity}`}
              sx={{ bgcolor: sevColor, color: '#0b0e11', fontWeight: 800 }} />
        <Chip label={`LAUNCH: ${sc.launch_recommendation}`}
              sx={{ bgcolor: launchColor, color: '#0b0e11', fontWeight: 800 }} />
        {sc.severity_divergence && sc.ai_proposed_severity ? (
          <Chip icon={<AlertTriangle size={14} />}
                label={`AI proposed ${sc.ai_proposed_severity} — policy override recorded`}
                sx={{ bgcolor: '#ffb74d22', color: '#ffb74d', fontWeight: 700 }} />
        ) : null}
        {sc.human_review_required ? (
          <Chip icon={<UserCheck size={14} />}
                label={`HUMAN REVIEW: ${sc.human_review_reasons.join(' · ')}`}
                sx={{ bgcolor: '#e6510022', color: '#ffab73', fontWeight: 700 }} />
        ) : null}
        <Typography sx={{ fontSize: 11.5, color: '#8a949e', width: '100%' }}>
          {sc.launch_rationale.join(' ')}
        </Typography>
      </Paper>

      <Paper sx={{ p: 1.5, mb: 1.5, bgcolor: '#161c23', border: '1px solid #232a31' }}>
        <Typography sx={{ fontSize: 11, color: '#8a949e', fontWeight: 700, mb: 1 }}>
          CONTEXT & KINEMATICS (UNKNOWN means absent telemetry — never guessed)
        </Typography>
        <Grid container spacing={1.5}>
          <Grid item><Stat label="FAILURE TYPE" value={sc.failure_type} /></Grid>
          <Grid item><Stat label="OBJECT (GT)" value={sc.object_class ?? 'UNKNOWN'} /></Grid>
          <Grid item><Stat label="CONFIDENCE" value={fmt(sc.confidence)} /></Grid>
          <Grid item><Stat label="EGO SPEED" value={fmt(sc.ego_speed_mps, ' m/s', 1)} /></Grid>
          <Grid item><Stat label="DISTANCE" value={fmt(sc.distance_to_object_m, ' m', 1)} /></Grid>
          <Grid item><Stat label="REL. VELOCITY" value={fmt(sc.relative_velocity_mps, ' m/s', 1)} /></Grid>
          <Grid item><Stat label="STOPPING DIST" value={fmt(sc.stopping_distance_m, ' m', 1)} /></Grid>
          <Grid item><Stat label="TTC" value={fmt(sc.ttc_s, ' s')} /></Grid>
          <Grid item><Stat label="DISENGAGE P" value={fmt(sc.disengagement_probability)} /></Grid>
          <Grid item><Stat label="SCR IMPACT" value={sc.safety_critical_recall_impact === null ? 'UNKNOWN' : sc.safety_critical_recall_impact.toFixed(5)} /></Grid>
          <Grid item><Stat label="BASELINE" value={sc.baseline_model ?? 'UNKNOWN'} /></Grid>
          <Grid item><Stat label="CANDIDATE" value={sc.candidate_model ?? 'UNKNOWN'} /></Grid>
        </Grid>
        {sc.ttc_validity.length ? (
          <Typography sx={{ fontSize: 11, color: '#8a949e', mt: 1 }}>
            TTC validity: {sc.ttc_validity.join('; ')}
          </Typography>
        ) : null}
      </Paper>

      <Paper sx={{ p: 1.5, mb: 1.5, bgcolor: '#161c23', border: '1px solid #232a31' }}>
        <Typography sx={{ fontSize: 11, color: '#8a949e', fontWeight: 700, mb: 0.5 }}>
          BEHAVIORAL CONSEQUENCE
        </Typography>
        <Typography sx={{ fontSize: 13 }}>{sc.behavioral_consequence}</Typography>
        {sc.planner_response ? (
          <Typography sx={{ fontSize: 11.5, color: '#8a949e', mt: 0.5, fontFamily: 'monospace' }}>
            observed planner: {JSON.stringify(sc.planner_response)}
          </Typography>
        ) : null}
      </Paper>

      <Paper sx={{ p: 1.5, mb: 1.5, bgcolor: '#161c23', border: '1px solid #232a31' }}>
        <Typography sx={{ fontSize: 11, color: '#8a949e', fontWeight: 700, mb: 0.5 }}>
          STATISTICS
        </Typography>
        <Typography sx={{ fontSize: 12.5, fontFamily: 'monospace' }}>
          metric_delta: {sc.metric_delta ? JSON.stringify(sc.metric_delta) : 'UNKNOWN'}
        </Typography>
        <Typography sx={{ fontSize: 12.5 }}>
          significance: {sc.statistical_significance
            ? `${sc.statistical_significance.method} → ${
                sc.statistical_significance.significant === null
                  ? 'not evaluable'
                  : sc.statistical_significance.significant
                    ? 'SIGNIFICANT'
                    : 'not significant'
              } (${sc.statistical_significance.detail})`
            : 'UNKNOWN'}
        </Typography>
        {sc.distribution_shift ? (
          <Typography sx={{ fontSize: 12.5, fontFamily: 'monospace' }}>
            distribution_shift: {JSON.stringify(sc.distribution_shift)}
          </Typography>
        ) : null}
      </Paper>

      {sc.uncertainty.missing_fields.length || sc.uncertainty.unknown_metrics.length ? (
        <Paper sx={{ p: 1.5, bgcolor: '#90a4ae14', border: '1px solid #90a4ae55' }}>
          <Typography sx={{ fontSize: 11, color: '#90a4ae', fontWeight: 700, mb: 0.5 }}>
            UNCERTAINTY / MISSING EVIDENCE
          </Typography>
          {sc.uncertainty.missing_fields.map((f) => (
            <Typography key={f} sx={{ fontSize: 12, fontFamily: 'monospace' }}>missing field: {f} → UNKNOWN</Typography>
          ))}
          {sc.uncertainty.unknown_metrics.map((m) => (
            <Typography key={m} sx={{ fontSize: 12, fontFamily: 'monospace' }}>metric not computable: {m} → UNKNOWN</Typography>
          ))}
          {sc.uncertainty.notes.map((n, i) => (
            <Typography key={i} sx={{ fontSize: 12, color: '#aab4be' }}>{n}</Typography>
          ))}
        </Paper>
      ) : null}

      <Typography sx={{ fontSize: 10.5, color: '#5c6770', mt: 1.5 }}>
        {sc.policy_version} · {sc.agent_version} · backend {sc.backend_used} · {sc.created_at}
      </Typography>
    </Box>
  );
}
