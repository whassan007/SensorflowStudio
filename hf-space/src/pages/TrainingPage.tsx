import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import { Rocket } from 'lucide-react';
import type { TrainingJobStatus } from '../types/labeleval';
import {
  getOverview,
  getDatasets,
  getModels,
  getTrainJobs,
  getTrainJob,
  postTrain,
  usePoll,
} from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import TrainingFlywheel from '../components/labeleval/TrainingFlywheel';
import TrainingLogViewer from '../components/labeleval/TrainingLogViewer';
import {
  SectionCard,
  MetricCard,
  StatusChip,
  ErrorNote,
  fmtInt,
  fmtNum,
  fmtPct,
} from '../components/labeleval/shared';

export default function TrainingPage() {
  const { activeDatasetId, stream } = useLabelEval();
  const overview = usePoll(getOverview, 5000);
  const datasets = usePoll(getDatasets, 10000);
  const models = usePoll(getModels, 10000);
  const jobs = usePoll(getTrainJobs, 3000);

  const [form, setForm] = useState({ dataset: '', epochs: 10, batch_size: 32 });
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [watchedJobId, setWatchedJobId] = useState<string | null>(null);
  const [polledJob, setPolledJob] = useState<TrainingJobStatus | null>(null);

  const jobList = jobs.data?.jobs ?? [];
  const runningJob = jobList.find((j) => j.status === 'running' || j.status === 'queued') ?? null;
  const activeJobId = watchedJobId ?? runningJob?.job_id ?? null;

  // Prefer live SSE training payload; otherwise poll the job every second while active.
  const streamJob = stream?.training && stream.training.job_id === activeJobId ? stream.training : null;
  useEffect(() => {
    if (!activeJobId || streamJob) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const j = await getTrainJob(activeJobId);
        if (!cancelled) {
          setPolledJob(j);
          if (j.status !== 'running' && j.status !== 'queued') {
            clearInterval(timer);
          }
        }
      } catch {
        /* keep last snapshot */
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeJobId, streamJob !== null]);

  const activeJob: TrainingJobStatus | null =
    streamJob ?? (polledJob && polledJob.job_id === activeJobId ? polledJob : null) ?? runningJob;

  const activeDataset = datasets.data?.datasets.find((d) => d.dataset_id === activeDatasetId) ?? null;
  const lineage = activeDataset?.lineage ?? null;

  const submit = async () => {
    if (!form.dataset) {
      setSubmitError('Select a dataset to train on.');
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await postTrain({
        dataset_version: form.dataset,
        training_parameters: { epochs: form.epochs, batch_size: form.batch_size },
      });
      setWatchedJobId(res.job_id);
      jobs.refresh();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const counters = overview.data?.counters;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <Box sx={{ flex: '1 1 480px' }}>
          <TrainingFlywheel
            counts={{
              verified: counters?.verified ?? 0,
              datasets: datasets.data?.datasets.length ?? 0,
              activeJobs: jobList.filter((j) => j.status === 'running' || j.status === 'queued').length,
              models: models.data?.models.length ?? 0,
              evaluated: counters?.evaluated ?? 0,
            }}
          />
        </Box>

        <Box sx={{ flex: '1 1 380px', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <SectionCard title="Dataset Lineage" helpTerm="lineage">
            {lineage ? (
              <>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>{activeDataset?.name}</strong> ({activeDataset?.version})
                  {lineage.generated_from_model ? ` ← ${lineage.generated_from_model}` : ''}
                  {lineage.corrected_by_review_batch ? `, corrected by ${lineage.corrected_by_review_batch}` : ''}
                  {lineage.validated_by_policy ? `, validated by ${lineage.validated_by_policy}` : ''}
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {lineage.parent_dataset ? (
                    <Chip size="small" label={`parent: ${lineage.parent_dataset}`} sx={{ bgcolor: '#232a31' }} />
                  ) : null}
                  {lineage.generated_from_model ? (
                    <Chip size="small" label={`model: ${lineage.generated_from_model}`} sx={{ bgcolor: '#232a31' }} />
                  ) : null}
                  {lineage.validated_by_policy ? (
                    <Chip size="small" label={`policy: ${lineage.validated_by_policy}`} sx={{ bgcolor: '#232a31' }} />
                  ) : null}
                </Box>
              </>
            ) : (
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                Select a dataset (Datasets page) to see its lineage: which model generated it, which review batch
                corrected it and which policy validated it.
              </Typography>
            )}
          </SectionCard>

          <SectionCard
            title="Train a New Model"
            help="Launches a training job on the selected dataset. Only verified labels are used — unverified data would teach the model its own mistakes. The resulting model version registers in Models and gets evaluated back through the platform."
          >
            {submitError ? (
              <Alert severity="error" variant="outlined" onClose={() => setSubmitError(null)} sx={{ mb: 1 }}>
                {submitError}
              </Alert>
            ) : null}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Select
                size="small"
                displayEmpty
                value={form.dataset}
                onChange={(e) => setForm({ ...form, dataset: e.target.value })}
              >
                <MenuItem value="">
                  <em>Select training dataset (verified data only)</em>
                </MenuItem>
                {(datasets.data?.datasets ?? []).map((d) => (
                  <MenuItem key={d.dataset_id} value={d.dataset_id}>
                    {d.name} ({d.version}) — {fmtInt(d.num_annotations)} annotations
                  </MenuItem>
                ))}
              </Select>
              <Box sx={{ display: 'flex', gap: 1.5 }}>
                <TextField
                  size="small"
                  type="number"
                  label="Epochs"
                  value={form.epochs}
                  onChange={(e) => setForm({ ...form, epochs: Number(e.target.value) })}
                  sx={{ flex: 1 }}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Batch size"
                  value={form.batch_size}
                  onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })}
                  sx={{ flex: 1 }}
                />
              </Box>
              <Button
                variant="contained"
                startIcon={submitting ? <CircularProgress size={16} color="inherit" /> : <Rocket size={16} />}
                disabled={submitting}
                onClick={() => void submit()}
                title="Start a training job on the selected verified dataset (unverified labels are excluded)"
                aria-label="Start training job"
              >
                {submitting ? 'Starting…' : 'Start Training'}
              </Button>
            </Box>
          </SectionCard>
        </Box>
      </Box>

      <SectionCard
        title={
          activeJob
            ? `Training Job ${activeJob.job_id} — ${activeJob.model_version}`
            : 'Training Job'
        }
        action={activeJob ? <StatusChip status={activeJob.status} /> : undefined}
      >
        {jobs.error && !jobs.data ? <ErrorNote error={jobs.error} /> : null}
        {activeJob ? (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
              <Typography variant="body2" sx={{ color: '#8a949e' }}>
                Epoch {activeJob.epoch} / {activeJob.total_epochs} · dataset {activeJob.dataset_version}
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={activeJob.total_epochs > 0 ? Math.min(100, (activeJob.epoch / activeJob.total_epochs) * 100) : 0}
              sx={{ height: 8, borderRadius: 1, mb: 2 }}
            />
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
              <MetricCard
                label="Loss"
                value={fmtNum(activeJob.loss, 4)}
                info="Training objective value — should trend down as the model fits the data. Plateaus or spikes suggest learning-rate or data problems."
              />
              <MetricCard label="Rare recall" value={fmtPct(activeJob.rare_recall)} accent="#ef5350" term="rare_recall" />
              <MetricCard label="Safety recall" value={fmtPct(activeJob.safety_recall)} accent="#ffa726" term="safety_recall" />
              <MetricCard label="Process units" value={fmtInt(activeJob.process_units)} term="process_units" />
            </Box>
            <TrainingLogViewer logs={activeJob.logs} />
          </>
        ) : (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No active training job. Past jobs: {jobList.length === 0 ? 'none' : ''}
          </Typography>
        )}
        {!activeJob && jobList.length > 0 ? (
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 1 }}>
            {jobList.map((j) => (
              <Chip
                key={j.job_id}
                size="small"
                label={`${j.job_id} · ${j.model_version} · ${j.status}`}
                onClick={() => setWatchedJobId(j.job_id)}
                sx={{ bgcolor: '#232a31', cursor: 'pointer' }}
              />
            ))}
          </Box>
        ) : null}
      </SectionCard>
    </Box>
  );
}
