/**
 * Evaluation Command Center — the aggregate-first landing page.
 *
 * Mental model surfaced in the UI:
 *   Dataset (population) → Evaluation Run → Population → Cohort → Container → Annotation
 * Aggregates come first; individual annotations are the deepest drill-down.
 *
 * Runs are async: launched runs are queued, progress streams over SSE
 * (stream.megaeval.active_runs) with a 1.5s polling fallback on GET /runs/{id},
 * and the page auto-refreshes when the run publishes.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import type { EvaluationRunInfo, ReviewState, RunStatus } from '../types/megaeval';
import {
  getCacheStats,
  getPopulations,
  getRun,
  getRunReview,
  getRuns,
} from '../services/megaeval';
import { usePoll } from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import { EmptyState, ErrorNote, LoadingBox, SectionCard } from '../components/labeleval/shared';
import HeaderBar, {
  GeneratePopulationButton,
  NewRunButton,
} from '../components/megaeval/HeaderBar';
import RunProgressCard from '../components/megaeval/RunProgressCard';
import HeroRow from '../components/megaeval/HeroRow';
import QualityTab from '../components/megaeval/QualityTab';
import CohortTab from '../components/megaeval/CohortTab';
import ContainersTab from '../components/megaeval/ContainersTab';
import InvestigationTab from '../components/megaeval/InvestigationTab';
import CompareTab from '../components/megaeval/CompareTab';
import ReviewTab from '../components/megaeval/ReviewTab';
import ShiftTab from '../components/megaeval/ShiftTab';
import LineageCard from '../components/megaeval/LineageCard';

const ACTIVE_STATUSES: RunStatus[] = ['created', 'queued', 'running', 'reducing', 'materializing'];

type TabId =
  | 'quality'
  | 'cohorts'
  | 'containers'
  | 'investigation'
  | 'compare'
  | 'review'
  | 'shift'
  | 'lineage';

export default function CommandCenterPage() {
  const { stream, entityId, navigate } = useLabelEval();

  const populationsPoll = usePoll(getPopulations, 15000);
  const runsPoll = usePoll(getRuns, 8000);
  const cachePoll = usePoll(getCacheStats, 10000);

  const populations = useMemo(() => {
    const list = populationsPoll.data?.populations ?? [];
    return [...list].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
  }, [populationsPoll.data]);
  const runs = useMemo(() => runsPoll.data?.runs ?? [], [runsPoll.data]);

  const [populationId, setPopulationId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(entityId);
  const [baselineRunId, setBaselineRunId] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>('quality');
  const [drillContainerId, setDrillContainerId] = useState<number | null>(null);
  const [publishTick, setPublishTick] = useState(0);
  const [review, setReview] = useState<ReviewState | null>(null);
  const [selectedRun, setSelectedRun] = useState<EvaluationRunInfo | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // Explicit selection persists the run in the URL hash (#/command/<run_id>).
  const selectRun = useCallback(
    (id: string) => {
      setRunId(id);
      setDrillContainerId(null);
      navigate('command', id);
    },
    [navigate]
  );

  // Follow hash changes (back/forward navigation, deep links).
  useEffect(() => {
    if (entityId && entityId !== runId) {
      setRunId(entityId);
      setDrillContainerId(null);
    }
  }, [entityId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Default selections once data arrives.
  useEffect(() => {
    if (!populationId && populations.length) setPopulationId(populations[0].population_id);
  }, [populations, populationId]);

  useEffect(() => {
    if (!runId && runs.length) {
      const byCreated = [...runs].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
      const pick = byCreated.find((r) => r.status === 'published') ?? byCreated[0];
      if (pick) setRunId(pick.run_id);
    }
  }, [runs, runId]);

  // Selected run detail: 1.5s polling while the run is in flight (SSE-independent fallback).
  useEffect(() => {
    setSelectedRun(null);
    setRunError(null);
    if (!runId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const load = async () => {
      try {
        const info = await getRun(runId);
        if (cancelled) return;
        setSelectedRun(info);
        setRunError(null);
        if (info.status !== 'published' && info.status !== 'failed') timer = setTimeout(() => void load(), 1500);
      } catch (err) {
        if (cancelled) return;
        setRunError(err instanceof Error ? err.message : String(err));
        timer = setTimeout(() => void load(), 3000);
      }
    };
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId]);

  // Auto-refresh all panels when the selected run publishes.
  const prevStatusRef = useRef<RunStatus | null>(null);
  const runsRefresh = runsPoll.refresh;
  useEffect(() => {
    const status = selectedRun?.status ?? null;
    if (status === 'published' && prevStatusRef.current && prevStatusRef.current !== 'published') {
      setPublishTick((t) => t + 1);
      runsRefresh();
    }
    prevStatusRef.current = status;
  }, [selectedRun?.status, runsRefresh]);

  const published = selectedRun?.status === 'published';

  // Review state (shared by the hero CI annotations and the Review tab).
  useEffect(() => {
    setReview(null);
    if (!runId || !published) return;
    let cancelled = false;
    getRunReview(runId)
      .then((r) => {
        if (!cancelled) setReview(r);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [runId, published, publishTick]);

  const megaStatus = stream?.megaeval ?? null;
  const cache = megaStatus?.cache ?? cachePoll.data ?? null;

  // Prefer SSE progress for the selected run when it is fresher than the poll.
  const streamProgress = megaStatus?.active_runs.find((r) => r.run_id === runId) ?? null;
  const selectedInFlight =
    selectedRun && ACTIVE_STATUSES.includes(selectedRun.status) ? streamProgress ?? selectedRun : null;
  const otherActiveRuns = (megaStatus?.active_runs ?? []).filter((r) => r.run_id !== runId);

  const openContainer = useCallback((cid: number) => {
    setDrillContainerId(cid);
    setTab('containers');
  }, []);

  const initialLoading =
    populationsPoll.loading && !populationsPoll.data && runsPoll.loading && !runsPoll.data;
  const noPopulation = populationsPoll.data !== null && populations.length === 0;
  const noRuns = !noPopulation && runsPoll.data !== null && runs.length === 0;

  const onPopulationGenerated = useCallback(
    (pop: { population_id: string }) => {
      populationsPoll.refresh();
      setPopulationId(pop.population_id);
    },
    [populationsPoll]
  );

  const onRunCreated = useCallback(
    (run: EvaluationRunInfo) => {
      runsRefresh();
      selectRun(run.run_id);
    },
    [runsRefresh, selectRun]
  );

  if (initialLoading) return <LoadingBox label="Loading command center…" />;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {populationsPoll.error && !populationsPoll.data ? <ErrorNote error={populationsPoll.error} /> : null}

      <HeaderBar
        populations={populations}
        populationId={populationId}
        onPopulationChange={setPopulationId}
        runs={runs}
        runId={runId}
        onRunChange={selectRun}
        baselineRunId={baselineRunId}
        onBaselineChange={setBaselineRunId}
        selectedRun={selectedRun}
        cache={cache}
        onGenerated={onPopulationGenerated}
        onRunCreated={onRunCreated}
      />

      <Typography variant="caption" sx={{ color: '#8a949e', mt: -1 }}>
        Dataset (population) → Evaluation run → Population → Cohort → Container → Annotation — aggregates first,
        individual annotations only at the deepest drill-down.
      </Typography>

      {/* Empty states: nothing generated / no runs yet */}
      {noPopulation ? (
        <SectionCard title="Welcome to the Evaluation Command Center">
          <EmptyState
            title="No evaluation population yet"
            message="Generate a synthetic mega-scale population (hundreds of thousands of annotated objects across weather, lighting, scenario and sensor cohorts), then launch an evaluation run against it."
            action={
              <Box sx={{ display: 'flex', gap: 1.5, justifyContent: 'center' }}>
                <GeneratePopulationButton variant="contained" size="large" onGenerated={onPopulationGenerated} />
              </Box>
            }
          />
        </SectionCard>
      ) : null}

      {noRuns ? (
        <SectionCard title="Population ready — no evaluation runs yet">
          <EmptyState
            title="Launch your first evaluation run"
            message="An evaluation run scores every object in the population against a model version, then materializes the aggregate cube, error index and sketches that power this page."
            action={
              <Box sx={{ display: 'flex', gap: 1.5, justifyContent: 'center' }}>
                <NewRunButton variant="contained" size="large" populationId={populationId} onCreated={onRunCreated} />
                <GeneratePopulationButton variant="outlined" size="large" onGenerated={onPopulationGenerated} />
              </Box>
            }
          />
        </SectionCard>
      ) : null}

      {runError && !selectedRun ? <ErrorNote error={runError} /> : null}

      {/* Live progress for the selected run + any other active runs */}
      {selectedInFlight ? <RunProgressCard progress={selectedInFlight} /> : null}
      {otherActiveRuns.map((r) => (
        <RunProgressCard key={r.run_id} progress={r} />
      ))}

      {selectedRun?.status === 'failed' ? (
        <Alert severity="error" variant="outlined">
          Run {selectedRun.run_id.slice(0, 8)} failed{selectedRun.error ? `: ${selectedRun.error}` : '.'} Launch a new
          evaluation run from the header.
        </Alert>
      ) : null}

      {/* Hero aggregates + tab sections (published runs only) */}
      {published && selectedRun ? (
        <>
          <HeroRow run={selectedRun} review={review} refreshKey={publishTick} />

          <Tabs
            value={tab}
            onChange={(_, v: TabId) => setTab(v)}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              minHeight: 36,
              borderBottom: '1px solid #232a31',
              '& .MuiTab-root': { minHeight: 36, py: 0.5 },
            }}
          >
            <Tab label="Quality" value="quality" title="Per-class quality, error distribution, quality funnel, run trend and sketch distributions" />
            <Tab label="Cohorts" value="cohorts" title="Drill the population by dimension (class, weather, lighting…) and run Why? decompositions" />
            <Tab label="Containers" value="containers" title="Per-scene quality table with risk sort presets; drill into the forensic object table" />
            <Tab label="Investigation" value="investigation" title="Multi-criteria search over the error index: worst containers and top examples" />
            <Tab label="Compare" value="compare" title="Candidate vs baseline: headline and per-class deltas, cohort regressions, promotion verdict" />
            <Tab label="Review" value="review" title="Statistical review sampling: plan, execute, and read precision/recall with confidence intervals" />
            <Tab label="Shift" value="shift" title="Train-vs-eval cohort mix shifts annotated with recall impact" />
            <Tab label="Lineage" value="lineage" title="Full reproducibility record: versions, config, seed" />
          </Tabs>

          {tab === 'quality' ? (
            <QualityTab runId={selectedRun.run_id} runs={runs} refreshKey={publishTick} />
          ) : null}
          {tab === 'cohorts' ? <CohortTab runId={selectedRun.run_id} refreshKey={publishTick} /> : null}
          {tab === 'containers' ? (
            <ContainersTab
              runId={selectedRun.run_id}
              refreshKey={publishTick}
              drillContainerId={drillContainerId}
              onDrill={setDrillContainerId}
            />
          ) : null}
          {tab === 'investigation' ? (
            <InvestigationTab runId={selectedRun.run_id} onOpenContainer={openContainer} />
          ) : null}
          {tab === 'compare' ? (
            <CompareTab runId={selectedRun.run_id} baselineRunId={baselineRunId} refreshKey={publishTick} />
          ) : null}
          {tab === 'review' ? (
            <ReviewTab runId={selectedRun.run_id} review={review} onReview={setReview} />
          ) : null}
          {tab === 'shift' ? <ShiftTab runId={selectedRun.run_id} refreshKey={publishTick} /> : null}
          {tab === 'lineage' ? <LineageCard lineage={selectedRun.lineage} /> : null}
        </>
      ) : null}
    </Box>
  );
}
