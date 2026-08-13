import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import type { TriageDecision } from '../types/labeleval';
import {
  getQualityGroups,
  getQualityGroupDetail,
  getEvaluation,
  usePoll,
} from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import TriageDecisionPanel from '../components/labeleval/TriageDecisionPanel';
import { LoadingBox, ErrorNote } from '../components/labeleval/shared';

const MAX_DECISIONS = 40;

/**
 * The API has no dedicated decision-list endpoint, so recent decisions are
 * assembled from the quality groups: sample annotation ids across groups and
 * load their evaluation records (which embed the triage decision).
 */
async function fetchRecentDecisions(datasetId: string | null): Promise<TriageDecision[]> {
  const groups = await getQualityGroups(datasetId);
  const details = await Promise.all(
    groups.groups.filter((g) => g.count > 0).map((g) => getQualityGroupDetail(g.group_id))
  );
  const perGroup = Math.max(4, Math.ceil(MAX_DECISIONS / Math.max(1, details.length)));
  const ids: string[] = [];
  for (const d of details) {
    ids.push(...d.annotation_ids.slice(0, perGroup));
  }
  const records = await Promise.all(
    ids.slice(0, MAX_DECISIONS).map((id) => getEvaluation(id).catch(() => null))
  );
  const decisions = records
    .map((r) => r?.decision ?? null)
    .filter((d): d is TriageDecision => d !== null);
  decisions.sort((a, b) => b.decided_at.localeCompare(a.decided_at));
  return decisions;
}

export default function TriagePage() {
  const { activeDatasetId } = useLabelEval();
  const decisions = usePoll(() => fetchRecentDecisions(activeDatasetId), 10000, [activeDatasetId]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="body2" sx={{ color: '#8a949e' }}>
        The triage engine applies the quality policy to every annotation&apos;s evidence. Each decision is fully
        explainable: every gate shows its actual value vs. threshold, and the primary failure reason is spec-critical.
      </Typography>
      {decisions.loading && !decisions.data ? <LoadingBox label="Assembling recent triage decisions…" /> : null}
      {decisions.error && !decisions.data ? <ErrorNote error={decisions.error} /> : null}
      <TriageDecisionPanel decisions={decisions.data ?? []} />
    </Box>
  );
}
