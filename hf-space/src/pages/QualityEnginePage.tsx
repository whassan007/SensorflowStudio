import Box from '@mui/material/Box';
import { getQualityMetrics, getPipeline, usePoll } from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import LabelEvaluationEngine from '../components/labeleval/LabelEvaluationEngine';
import StrictQualityValidation from '../components/labeleval/StrictQualityValidation';
import GroundTruthComparison from '../components/labeleval/GroundTruthComparison';
import GraderDisagreement from '../components/labeleval/GraderDisagreement';
import GateStatusPanel from '../components/platform/GateStatusPanel';
import { LoadingBox, ErrorNote } from '../components/labeleval/shared';

export default function QualityEnginePage() {
  const { activeDatasetId, stream } = useLabelEval();
  const metrics = usePoll(() => getQualityMetrics(activeDatasetId), 5000, [activeDatasetId]);
  // Use the stream's pipeline state when available; otherwise poll.
  const pipelinePoll = usePoll(getPipeline, stream ? null : 3000, [stream === null]);
  const pipeline = stream?.pipeline ?? pipelinePoll.data;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {metrics.loading && !metrics.data ? <LoadingBox label="Loading quality metrics…" /> : null}
      {metrics.error && !metrics.data ? <ErrorNote error={metrics.error} /> : null}

      <LabelEvaluationEngine pipeline={pipeline ?? null} />
      <GateStatusPanel candidateRunId={null} baselineRunId={null} />
      <StrictQualityValidation metrics={metrics.data} />
      <GroundTruthComparison metrics={metrics.data} />
      <GraderDisagreement metrics={metrics.data} />
    </Box>
  );
}
