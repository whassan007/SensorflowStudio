import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import { Rocket, ArrowRight } from 'lucide-react';
import {
  getOverview,
  getFunnel,
  getAlerts,
  generateDataset,
  runPipeline,
  usePoll,
} from '../services/labeleval';
import { useLabelEval, resolvePageId } from '../context/LabelEvalContext';
import VerificationFunnel from '../components/labeleval/VerificationFunnel';
import QueueStatus from '../components/labeleval/QueueStatus';
import {
  MetricCard,
  SectionCard,
  StatusChip,
  LoadingBox,
  ErrorNote,
  EmptyState,
  fmtInt,
  fmtPct,
  fmtNum,
} from '../components/labeleval/shared';

export default function OverviewPage() {
  const { stream, navigate, setActiveDatasetId } = useLabelEval();
  const overview = usePoll(getOverview, 3000);
  const funnel = usePoll(getFunnel, 5000);
  const alerts = usePoll(getAlerts, 5000);
  const [bootstrapping, setBootstrapping] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);

  const data = overview.data;
  const pipeline = stream?.pipeline ?? null;
  const counters = data?.counters;
  const allZero =
    counters !== undefined && Object.values(counters).every((v) => v === 0);

  const bootstrap = async () => {
    setBootstrapping(true);
    setBootError(null);
    try {
      const ds = await generateDataset({ num_sequences: 8, frames_per_sequence: 25, seed: 42 });
      setActiveDatasetId(ds.dataset_id);
      await runPipeline(ds.dataset_id);
      overview.refresh();
    } catch (err) {
      setBootError(err instanceof Error ? err.message : String(err));
    } finally {
      setBootstrapping(false);
    }
  };

  if (overview.loading && !data) return <LoadingBox label="Loading overview…" />;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {overview.error && !data ? <ErrorNote error={overview.error} /> : null}

      {pipeline?.running ? (
        <Alert severity="info" icon={<CircularProgress size={18} />} sx={{ bgcolor: '#12314a' }}>
          Pipeline running — stage: <strong>{pipeline.stage}</strong>. Counters update live.
          <LinearProgress sx={{ mt: 1, height: 6, borderRadius: 1 }} />
        </Alert>
      ) : null}

      {data && allZero && !pipeline?.running ? (
        <SectionCard title="Welcome to Sensorflow Studio">
          <EmptyState
            title="No data yet"
            message="The evaluation platform has not processed any frames. Generate a synthetic multi-sensor dataset and run the full label-evaluation pipeline to see the platform in action."
            action={
              <Box>
                {bootError ? (
                  <Alert severity="error" variant="outlined" sx={{ mb: 1 }}>
                    {bootError}
                  </Alert>
                ) : null}
                <Button
                  variant="contained"
                  size="large"
                  startIcon={bootstrapping ? <CircularProgress size={18} color="inherit" /> : <Rocket size={18} />}
                  disabled={bootstrapping}
                  onClick={() => void bootstrap()}
                  title="Create a small synthetic multi-sensor dataset and run the full label-evaluation pipeline"
                  aria-label="Generate synthetic dataset and run pipeline"
                >
                  {bootstrapping ? 'Generating & starting…' : 'Generate synthetic dataset & run pipeline'}
                </Button>
              </Box>
            }
          />
        </SectionCard>
      ) : null}

      {counters ? (
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          <MetricCard label="Frames processed" value={fmtInt(counters.frames_processed)} info="Sensor frames the pipeline has ingested and pushed through auto-labeling." />
          <MetricCard label="Auto-labeled" value={fmtInt(counters.auto_labeled)} info="Candidate labels produced by the auto-labeler, before any quality gating." />
          <MetricCard label="Auto-graded" value={fmtInt(counters.auto_graded)} accent="#66bb6a" term="status_auto_graded" />
          <MetricCard label="Flagged" value={fmtInt(counters.flagged)} accent="#ffa726" term="status_flagged" />
          <MetricCard label="Verified" value={fmtInt(counters.verified)} accent="#66bb6a" term="status_verified" />
          <MetricCard label="HITL" value={fmtInt(counters.in_hitl)} accent="#42a5f5" info="Labels currently sitting in the human review queue." />
          <MetricCard label="Rejected" value={fmtInt(counters.rejected)} accent="#ef5350" term="status_rejected" />
          <MetricCard label="Rare events" value={fmtInt(counters.rare_events)} accent="#ef5350" info="Samples flagged as rare/anomalous by the ensemble detectors — candidates for targeted training data." />
        </Box>
      ) : null}

      {data ? (
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          <MetricCard label="Precision" value={fmtPct(data.metrics.precision)} term="precision" />
          <MetricCard label="Recall" value={fmtPct(data.metrics.recall)} term="recall" />
          <MetricCard label="Safety recall" value={fmtPct(data.metrics.safety_critical_recall)} accent="#ffa726" term="safety_recall" />
          <MetricCard label="mAP (3D)" value={fmtPct(data.metrics.map_3d)} term="map_3d" />
          <MetricCard label="3D IoU" value={fmtNum(data.metrics.mean_iou_3d)} term="iou_3d" />
          <MetricCard label="IDF1" value={fmtPct(data.metrics.idf1)} term="idf1" />
          <MetricCard label="Anomaly rate" value={fmtPct(data.metrics.anomaly_rate)} term="anomaly_rate" />
          <MetricCard label="Grader consensus" value={fmtPct(data.metrics.grader_consensus)} term="grader_consensus" />
          <MetricCard label="Verification rate" value={fmtPct(data.verification_rate)} accent="#66bb6a" term="verification_rate" />
          <MetricCard label="Automation rate" value={fmtPct(data.automation_rate)} term="automation_rate" />
          <MetricCard label="Process units" value={fmtInt(data.process_units_total)} sub="total consumed" term="process_units" />
        </Box>
      ) : null}

      <VerificationFunnel funnel={funnel.data} />

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <SectionCard
          title={`Alerts (${alerts.data?.alerts.length ?? 0})`}
          help="Active alerts from anomaly detection, regression tracking and distribution-shift checks. Hover a severity chip for what it means; Evidence deep-links to the page that raised the alert with the offending entity selected."
          sx={{ flex: '1 1 420px' }}
        >
          {!alerts.data || alerts.data.alerts.length === 0 ? (
            <Typography variant="body2" sx={{ color: '#8a949e' }}>
              No active alerts — anomaly, regression and shift monitors are all quiet. Alerts will appear here as the
              pipeline processes data.
            </Typography>
          ) : (
            <Table size="small">
              <TableBody>
                {alerts.data.alerts.map((a) => (
                  <TableRow key={a.alert_id} hover>
                    <TableCell sx={{ width: 90 }}>
                      <StatusChip status={a.severity} />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{a.message}</Typography>
                      <Typography variant="caption" sx={{ color: '#8a949e' }}>
                        {a.kind} · {new Date(a.created_at).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell align="right" sx={{ width: 130 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        endIcon={<ArrowRight size={14} />}
                        onClick={() => navigate(resolvePageId(a.evidence_link.page), a.evidence_link.id)}
                        title={`Open ${a.evidence_link.page} with this alert’s evidence selected`}
                        aria-label="Open alert evidence"
                      >
                        Evidence
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </SectionCard>
        <Box sx={{ flex: '1 1 380px' }}>
          <QueueStatus queue={stream?.pipeline.queue ?? undefined} />
        </Box>
      </Box>
    </Box>
  );
}
