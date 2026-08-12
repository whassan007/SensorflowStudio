import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import { Sparkles, X } from 'lucide-react';
import type { CopilotExplainRequest, EvaluationRecord, RareEvent } from '../../types/labeleval';
import { StatusChip, GateLineList, fmtNum, fmtPct, HBar } from './shared';

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <TableRow>
      <TableCell sx={{ color: '#8a949e', border: 0, py: 0.4 }}>{label}</TableCell>
      <TableCell align="right" sx={{ fontFamily: 'monospace', border: 0, py: 0.4 }}>
        {value}
      </TableCell>
    </TableRow>
  );
}

export function evidenceCopilotRequest(record: EvaluationRecord): CopilotExplainRequest {
  const primary = record.decision?.primary_failure_reason ?? null;
  let contextType: CopilotExplainRequest['context_type'] = 'general';
  if (record.anomaly.is_anomaly) contextType = 'anomaly';
  if (primary === 'GRADER_DISAGREEMENT') contextType = 'disagreement';
  if (primary === 'MODEL_REGRESSION') contextType = 'regression';
  return {
    context_type: contextType,
    annotation_id: record.annotation_id,
    model_version: record.model_version,
  };
}

export default function EvidencePanel({
  open,
  onClose,
  record,
  event,
  onAskCopilot,
}: {
  open: boolean;
  onClose: () => void;
  record?: EvaluationRecord | null;
  event?: RareEvent | null;
  onAskCopilot: (request: CopilotExplainRequest) => void;
}) {
  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: 440, bgcolor: '#12171d' } }}>
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1.5, overflowY: 'auto' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6">Evidence</Typography>
          <IconButton size="small" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </Box>

        {record ? (
          <>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
              <Chip size="small" label={record.annotation_id} sx={{ bgcolor: '#232a31', fontFamily: 'monospace' }} />
              <Chip size="small" label={record.object_class} sx={{ bgcolor: '#232a31' }} />
              <Chip size="small" label={record.model_version} sx={{ bgcolor: '#232a31' }} />
              {record.decision ? <StatusChip status={record.decision.status} /> : null}
            </Box>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              Frame {record.frame_id} · Dataset {record.dataset_id} · Detection confidence{' '}
              {fmtNum(record.detection.confidence)}
              {record.ground_truth_type ? ` · GT: ${record.ground_truth_type}` : ' · No ground truth reference'}
            </Typography>

            <Divider />
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Geometry
            </Typography>
            <Table size="small">
              <TableBody>
                <MetricRow label="3D IoU" value={fmtNum(record.geometry.iou_3d)} />
                <MetricRow label="Position error (m)" value={fmtNum(record.geometry.position_error)} />
                <MetricRow label="Orientation error (deg)" value={fmtNum(record.geometry.orientation_error_deg)} />
                <MetricRow label="Dimension error" value={fmtNum(record.geometry.dimension_error)} />
                <MetricRow label="Point density" value={fmtNum(record.geometry.point_density)} />
                <MetricRow label="Point-in-box ratio" value={fmtNum(record.geometry.point_in_box_ratio)} />
                <MetricRow label="Ground contact error" value={fmtNum(record.geometry.ground_contact_error)} />
              </TableBody>
            </Table>

            <Divider />
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Anomaly ensemble — {record.anomaly.ensemble_strategy.replace(/_/g, ' ')} · threshold{' '}
              {fmtNum(record.anomaly.decision_threshold)}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2">
                Score <strong>{fmtNum(record.anomaly.score)}</strong>
              </Typography>
              <Chip
                size="small"
                label={record.anomaly.is_anomaly ? 'ANOMALY' : 'normal'}
                sx={{
                  bgcolor: record.anomaly.is_anomaly ? '#e65100' : '#232a31',
                  color: record.anomaly.is_anomaly ? '#ffe0b2' : '#aab4be',
                  fontWeight: 700,
                }}
              />
            </Box>
            {Object.entries(record.anomaly.detector_scores).map(([name, score]) => (
              <HBar
                key={name}
                label={name}
                value={record.anomaly.normalized_scores[name] ?? score}
                max={1}
                color="#ffa726"
                valueLabel={`raw ${fmtNum(score)} · norm ${fmtNum(record.anomaly.normalized_scores[name] ?? null)}`}
              />
            ))}

            <Divider />
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Grading ({record.grading.grader_count} graders)
            </Typography>
            <Table size="small">
              <TableBody>
                <MetricRow label="Consensus" value={fmtPct(record.grading.consensus)} />
                <MetricRow label="Class agreement" value={fmtPct(record.grading.class_agreement)} />
                <MetricRow label="Spatial agreement" value={fmtPct(record.grading.spatial_agreement)} />
                <MetricRow label="Temporal agreement" value={fmtPct(record.grading.temporal_agreement)} />
              </TableBody>
            </Table>
            {record.grading.disagreement_types.length > 0 ? (
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {record.grading.disagreement_types.map((d) => (
                  <Chip key={d} size="small" label={d} sx={{ bgcolor: '#4a2c00', color: '#ffcc80' }} />
                ))}
              </Box>
            ) : null}

            <Divider />
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Tracking
            </Typography>
            <Typography variant="body2" sx={{ color: '#aab4be' }}>
              ID switch: {record.tracking.id_switch ? 'yes' : 'no'} · Fragmentation:{' '}
              {record.tracking.fragmentation ? 'yes' : 'no'} · Track quality: {fmtNum(record.tracking.track_quality)}
            </Typography>

            <Divider />
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Quality gates {record.validation.passed ? '— PASSED' : '— FAILED'}
            </Typography>
            <GateLineList checks={record.validation.checks} />

            {record.decision ? (
              <>
                <Divider />
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                  Triage decision — policy {record.decision.policy_id}
                </Typography>
                {record.decision.primary_failure_reason ? (
                  <Typography variant="body2" sx={{ color: '#ef9a9a' }}>
                    Primary failure: {record.decision.primary_failure_reason.replace(/_/g, ' ')}
                  </Typography>
                ) : null}
                <GateLineList checks={record.decision.gate_lines} />
              </>
            ) : null}

            {record.injected_errors.length > 0 ? (
              <>
                <Divider />
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  Synthetic injected defects (demo transparency): {record.injected_errors.join(', ')}
                </Typography>
              </>
            ) : null}

            <Button
              variant="outlined"
              startIcon={<Sparkles size={16} />}
              onClick={() => onAskCopilot(evidenceCopilotRequest(record))}
            >
              Ask Copilot
            </Button>
          </>
        ) : null}

        {event ? (
          <>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
              <Chip size="small" label={event.event_id} sx={{ bgcolor: '#232a31', fontFamily: 'monospace' }} />
              <Chip size="small" label={event.scenario_type.replace(/_/g, ' ')} sx={{ bgcolor: '#232a31' }} />
              <StatusChip status={event.severity} />
              {event.verified ? <StatusChip status="VERIFIED" /> : null}
            </Box>
            <Typography variant="body2">{event.description}</Typography>
            <Table size="small">
              <TableBody>
                <MetricRow label="Rarity score" value={fmtNum(event.rarity_score)} />
                <MetricRow label="Anomaly score" value={fmtNum(event.anomaly_score)} />
                <MetricRow label="Confidence" value={fmtPct(event.confidence)} />
                <MetricRow label="Dataset" value={event.dataset_id} />
              </TableBody>
            </Table>
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Evidence frames ({event.evidence_frames.length})
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {event.evidence_frames.map((f) => (
                <Chip key={f} size="small" label={f} sx={{ bgcolor: '#232a31', fontFamily: 'monospace' }} />
              ))}
            </Box>
            {Object.keys(event.sensor_evidence).length > 0 ? (
              <>
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                  Sensor evidence
                </Typography>
                <Table size="small">
                  <TableBody>
                    {Object.entries(event.sensor_evidence).map(([sensor, note]) => (
                      <MetricRow key={sensor} label={sensor} value={note} />
                    ))}
                  </TableBody>
                </Table>
              </>
            ) : null}
            <Button
              variant="outlined"
              startIcon={<Sparkles size={16} />}
              onClick={() => onAskCopilot({ context_type: 'anomaly', event_id: event.event_id })}
            >
              Ask Copilot
            </Button>
          </>
        ) : null}

        {!record && !event ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No evidence selected.
          </Typography>
        ) : null}
      </Box>
    </Drawer>
  );
}
