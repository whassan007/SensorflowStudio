/**
 * Review-queue card for one track-level candidate: schematic scene, the three
 * separate confidence gauges, modality-tagged evidence, alternative hypotheses
 * with retained/rejected reasons, observed vs predicted failure (kept strictly
 * apart), approve/reject with note, and the lineage/governance panel.
 */
import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { CheckCircle2, ChevronDown, ChevronUp, ShieldAlert, XCircle } from 'lucide-react';
import type { LineageRecord, SceneView, TrackCandidateSummary } from '../../types/raremine';
import {
  getCandidateScene,
  getCandidate,
  governanceOverride,
  promoteToTraining,
  reviewCandidate,
} from '../../services/raremine';
import ConfidenceGauges from './ConfidenceGauges';
import SceneCanvas from './SceneCanvas';
import { DifficultyChip, PriorityChip, SectionLabel } from './shared';

const DESTINATIONS = [
  'RARE_EVENT_DATASET',
  'HARD_EXAMPLE_DATASET',
  'REGRESSION_EVALUATION_SET',
  'SAFETY_CRITICAL_EVALUATION_SET',
];

function EvidenceList({ items }: { items: { modality: string; description: string }[] }) {
  if (!items.length) {
    return (
      <Typography variant="caption" sx={{ color: '#5c6a76' }}>
        none recorded
      </Typography>
    );
  }
  return (
    <Box component="ul" sx={{ m: 0, pl: 2 }}>
      {items.map((e, i) => (
        <Typography key={i} component="li" variant="body2" sx={{ color: '#c3ccd5', fontSize: 12.5 }}>
          <Box component="span" sx={{ color: '#4fc3f7', fontFamily: 'monospace', fontSize: 11 }}>
            [{e.modality}]
          </Box>{' '}
          {e.description}
        </Typography>
      ))}
    </Box>
  );
}

function LineagePanel({
  lineage,
  onOverride,
  onPromote,
  busy,
}: {
  lineage: LineageRecord;
  onOverride: () => void;
  onPromote: () => void;
  busy: boolean;
}) {
  return (
    <Box sx={{ p: 1.5, bgcolor: '#10151b', border: '1px solid #232a31', borderRadius: 1 }}>
      <SectionLabel>Lineage &amp; governance</SectionLabel>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 1 }}>
        <Chip size="small" label={`validation: ${lineage.validation_status}`} sx={{ bgcolor: '#232a31', fontSize: 11 }} />
        <Chip
          size="small"
          label={`training ${lineage.training_eligible ? 'ELIGIBLE' : 'NOT eligible'}`}
          sx={{
            bgcolor: lineage.training_eligible ? '#1b5e20' : '#37474f',
            color: '#fff',
            fontSize: 11,
          }}
        />
        <Chip
          size="small"
          label={`evaluation ${lineage.evaluation_eligible ? 'eligible' : 'not eligible'}`}
          sx={{ bgcolor: '#232a31', fontSize: 11 }}
        />
        {lineage.protected_evaluation ? (
          <Chip
            size="small"
            icon={<ShieldAlert size={13} />}
            label="PROTECTED EVAL SET"
            sx={{ bgcolor: '#4a148c', color: '#fff', fontSize: 11 }}
          />
        ) : null}
      </Box>
      <Typography variant="caption" sx={{ color: '#5c6a76', display: 'block' }}>
        source frame {lineage.source_frame_id} · sequence {lineage.source_sequence_id} · {lineage.dataset_version} ·
        curator {lineage.curator} · {new Date(lineage.curation_timestamp).toLocaleString()}
      </Typography>
      {lineage.governance_overrides.length ? (
        <Typography variant="caption" sx={{ color: '#ffb74d', display: 'block', mt: 0.5 }}>
          override on record: {lineage.governance_overrides[0].actor} — “{lineage.governance_overrides[0].reason}”
        </Typography>
      ) : null}
      {lineage.validation_status === 'APPROVED' ? (
        <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
          <Button size="small" variant="outlined" color="warning" disabled={busy} onClick={onPromote}>
            Promote to training
          </Button>
          {lineage.protected_evaluation && !lineage.governance_overrides.length ? (
            <Button size="small" color="secondary" disabled={busy} onClick={onOverride}>
              Request governance override…
            </Button>
          ) : null}
        </Box>
      ) : null}
    </Box>
  );
}

export default function CandidateCard({
  summary,
  onChanged,
}: {
  summary: TrackCandidateSummary;
  onChanged: () => void;
}) {
  const c = summary.candidate;
  const [scene, setScene] = useState<SceneView | null>(null);
  const [lineage, setLineage] = useState<LineageRecord | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [note, setNote] = useState('');
  const [destination, setDestination] = useState(
    DESTINATIONS.includes(c.recommended_dataset_destination)
      ? c.recommended_dataset_destination
      : 'RARE_EVENT_DATASET'
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [overrideActor, setOverrideActor] = useState('');
  const [overrideReason, setOverrideReason] = useState('');

  useEffect(() => {
    getCandidateScene(summary.track_candidate_id).then(setScene).catch(() => setScene(null));
    getCandidate(summary.track_candidate_id)
      .then((d) => setLineage(d.lineage))
      .catch(() => undefined);
  }, [summary.track_candidate_id, summary.stage, summary.destination]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      const d = await getCandidate(summary.track_candidate_id);
      setLineage(d.lineage);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const decided = summary.human_validation !== null;
  const behavior = c.observed_model_behavior;

  return (
    <Card sx={{ bgcolor: '#141a20', border: '1px solid #232a31' }}>
      <CardContent sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {/* left: schematic scene */}
        <Box sx={{ flexShrink: 0 }}>
          {scene ? (
            <SceneCanvas scene={scene} />
          ) : (
            <Box sx={{ width: 300, height: 260, bgcolor: '#0d1117', borderRadius: 1 }} />
          )}
        </Box>

        {/* right: judgment */}
        <Box sx={{ flex: 1, minWidth: 320 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
            <PriorityChip value={c.curation_priority} />
            <DifficultyChip value={c.perception_difficulty} />
            <Chip size="small" label={`silhouette ${c.silhouette_deviation}`} sx={{ bgcolor: '#232a31', fontSize: 10.5, height: 20 }} />
            <Chip
              size="small"
              label={`occlusion ${c.occlusion_level} (${c.occlusion_source.toLowerCase().replace(/_/g, ' ')})`}
              sx={{ bgcolor: '#232a31', fontSize: 10.5, height: 20 }}
            />
            <Chip size="small" label={`evidence ${c.evidence_quality}`} sx={{ bgcolor: '#232a31', fontSize: 10.5, height: 20 }} />
            {summary.frame_count > 1 ? (
              <Chip size="small" label={`track · ${summary.frame_count} frames`} sx={{ bgcolor: '#0d47a1', color: '#90caf9', fontSize: 10.5, height: 20 }} />
            ) : (
              <Chip size="small" label="single frame" sx={{ bgcolor: '#232a31', fontSize: 10.5, height: 20 }} />
            )}
            {summary.stage === 'CURATED' ? (
              <Chip size="small" icon={<CheckCircle2 size={13} />} label={summary.destination} sx={{ bgcolor: '#1b5e20', color: '#fff', fontSize: 10.5, height: 20 }} />
            ) : null}
            {summary.stage === 'ARCHIVED' ? (
              <Chip size="small" icon={<XCircle size={13} />} label="rejected" sx={{ bgcolor: '#5d1616', color: '#fff', fontSize: 10.5, height: 20 }} />
            ) : null}
          </Box>

          <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.25 }}>
            {c.event_type === 'costumed_pedestrian'
              ? `Costumed pedestrian — ${c.costume_type.join(' / ') || 'family undetermined'}`
              : 'No confirmed edge case'}
            <Typography component="span" variant="caption" sx={{ color: '#5c6a76', ml: 1, fontFamily: 'monospace' }}>
              {summary.track_candidate_id}
            </Typography>
          </Typography>
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
            {c.location.context} · {c.location.distance_m?.toFixed(0)} m · {c.location.lighting} · {c.location.weather} ·
            recommended → <b>{c.recommended_dataset_destination}</b>
          </Typography>

          <Box sx={{ maxWidth: 380, mb: 1 }}>
            <ConfidenceGauges candidate={c} />
          </Box>

          {/* observed vs predicted — strictly separated */}
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 1 }}>
            <Box sx={{ flex: '1 1 240px', p: 1, bgcolor: '#10151b', borderRadius: 1, border: '1px solid #232a31' }}>
              <SectionLabel>Observed model behavior (measured)</SectionLabel>
              {behavior === null ? (
                <Typography variant="caption" sx={{ color: '#5c6a76' }}>
                  NOT AVAILABLE — no baseline predictions were supplied for this scene. Nothing is inferred.
                </Typography>
              ) : behavior.failure_observed ? (
                <>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 0.5 }}>
                    {behavior.failure_modes.map((m) => (
                      <Chip key={m} size="small" label={m} sx={{ bgcolor: '#5d1616', color: '#ff8a80', fontSize: 10, height: 18 }} />
                    ))}
                  </Box>
                  {behavior.details.map((d, i) => (
                    <Typography key={i} variant="caption" sx={{ color: '#c3ccd5', display: 'block' }}>
                      {d}
                    </Typography>
                  ))}
                </>
              ) : (
                <Typography variant="caption" sx={{ color: '#66bb6a' }}>
                  {behavior.details[0] ?? 'no failure observed'}
                </Typography>
              )}
            </Box>
            <Box sx={{ flex: '1 1 240px', p: 1, bgcolor: '#10151b', borderRadius: 1, border: '1px solid #232a31' }}>
              <SectionLabel>Predicted failure (miner forecast)</SectionLabel>
              <Typography variant="caption" sx={{ color: c.predicted_failure_mode ? '#ffb74d' : '#5c6a76' }}>
                {c.predicted_failure_mode ?? 'no failure predicted'}
              </Typography>
              <Typography variant="caption" sx={{ color: '#5c6a76', display: 'block', mt: 0.5 }}>
                A forecast from evidence rules — kept separate from measured behavior on the left.
              </Typography>
            </Box>
          </Box>

          <Button
            size="small"
            onClick={() => setExpanded((v) => !v)}
            endIcon={expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            sx={{ textTransform: 'none', color: '#8a949e', fontSize: 12, mb: 0.5 }}
          >
            Evidence, alternative hypotheses &amp; lineage
          </Button>
          <Collapse in={expanded} unmountOnExit>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 1 }}>
              <Box sx={{ flex: '1 1 260px' }}>
                <SectionLabel>Visual evidence</SectionLabel>
                <EvidenceList items={c.visual_evidence} />
                <Box sx={{ mt: 1 }}>
                  <SectionLabel>Human-identity evidence</SectionLabel>
                  <EvidenceList items={c.human_identity_evidence} />
                </Box>
                <Box sx={{ mt: 1 }}>
                  <SectionLabel>Temporal validation</SectionLabel>
                  <Typography variant="caption" sx={{ color: c.temporal_validation.available ? '#c3ccd5' : '#5c6a76' }}>
                    {c.temporal_validation.status}
                    {c.temporal_validation.evidence.length ? ` — ${c.temporal_validation.evidence.join('; ')}` : ''}
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ flex: '1 1 260px' }}>
                <SectionLabel>Alternative hypotheses</SectionLabel>
                {c.alternative_hypotheses.map((a, i) => (
                  <Box key={i} sx={{ mb: 0.75 }}>
                    <Typography variant="body2" sx={{ fontSize: 12.5, color: a.status === 'RETAINED' ? '#ffb74d' : '#5c6a76' }}>
                      <b>{a.hypothesis}</b> · {a.status} · {(a.confidence * 100).toFixed(0)}%
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#8a949e' }}>
                      {a.reason}
                    </Typography>
                  </Box>
                ))}
                <SectionLabel>Priority reasoning</SectionLabel>
                <Typography variant="caption" sx={{ color: '#c3ccd5' }}>
                  {c.priority_reason}
                </Typography>
              </Box>
            </Box>
            {lineage ? (
              <LineagePanel
                lineage={lineage}
                busy={busy}
                onPromote={() => act(() => promoteToTraining(summary.track_candidate_id))}
                onOverride={() => setOverrideOpen(true)}
              />
            ) : null}
          </Collapse>

          {/* review actions */}
          {!decided ? (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mt: 1 }}>
              <TextField
                size="small"
                placeholder="review note…"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                sx={{ width: 220 }}
              />
              <TextField
                size="small"
                select
                label="destination"
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                sx={{ width: 250 }}
              >
                {DESTINATIONS.map((d) => (
                  <MenuItem key={d} value={d}>
                    {d}
                  </MenuItem>
                ))}
              </TextField>
              <Button
                size="small"
                variant="contained"
                color="success"
                disabled={busy}
                startIcon={<CheckCircle2 size={14} />}
                onClick={() => act(() => reviewCandidate(summary.track_candidate_id, 'approve', note, destination))}
              >
                Approve
              </Button>
              <Button
                size="small"
                variant="outlined"
                color="error"
                disabled={busy}
                startIcon={<XCircle size={14} />}
                onClick={() => act(() => reviewCandidate(summary.track_candidate_id, 'reject', note))}
              >
                Reject
              </Button>
            </Box>
          ) : null}
          {error ? (
            <Typography variant="caption" sx={{ color: '#ff8a80', display: 'block', mt: 0.5 }}>
              {error}
            </Typography>
          ) : null}
        </Box>
      </CardContent>

      {/* governance override confirmation */}
      <Dialog open={overrideOpen} onClose={() => setOverrideOpen(false)}>
        <DialogTitle>Governance override</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, minWidth: 380, pt: '10px !important' }}>
          <Typography variant="body2" sx={{ color: '#aab4be' }}>
            This example belongs to a <b>protected evaluation set</b>. Releasing it for training risks leaking
            evaluation data into the model. The override is recorded permanently with your name and reason.
          </Typography>
          <TextField size="small" label="your name (actor)" value={overrideActor} onChange={(e) => setOverrideActor(e.target.value)} />
          <TextField
            size="small"
            label="reason"
            multiline
            minRows={2}
            value={overrideReason}
            onChange={(e) => setOverrideReason(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOverrideOpen(false)}>Cancel</Button>
          <Button
            color="warning"
            variant="contained"
            disabled={!overrideActor.trim() || !overrideReason.trim() || busy}
            onClick={() =>
              act(async () => {
                await governanceOverride(summary.track_candidate_id, overrideActor, overrideReason);
                setOverrideOpen(false);
              })
            }
          >
            Record override
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
}
