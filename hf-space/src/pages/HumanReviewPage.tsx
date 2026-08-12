import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import type { FrameSummary, ReviewTask } from '../types/labeleval';
import { getReviewTasks, getReviewTask, getFrame, usePoll } from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import HITLReview from '../components/labeleval/HITLReview';
import RelabelingPanel from '../components/labeleval/RelabelingPanel';
import { LoadingBox, ErrorNote } from '../components/labeleval/shared';

export default function HumanReviewPage() {
  const { entityId } = useLabelEval();
  const tasks = usePoll(getReviewTasks, 5000);
  const [selectedTask, setSelectedTask] = useState<ReviewTask | null>(null);
  const [frame, setFrame] = useState<FrameSummary | null>(null);
  const [prevFrame, setPrevFrame] = useState<FrameSummary | null>(null);
  const [nextFrame, setNextFrame] = useState<FrameSummary | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const selectTask = async (taskId: string) => {
    setLoadError(null);
    setFrameLoading(true);
    setFrame(null);
    setPrevFrame(null);
    setNextFrame(null);
    try {
      const task = await getReviewTask(taskId);
      setSelectedTask(task);
      const res = await getFrame(task.frame_id);
      setFrame(res.frame);
      const [prev, next] = await Promise.all([
        res.prev ? getFrame(res.prev).then((r) => r.frame).catch(() => null) : Promise.resolve(null),
        res.next ? getFrame(res.next).then((r) => r.frame).catch(() => null) : Promise.resolve(null),
      ]);
      setPrevFrame(prev);
      setNextFrame(next);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      setFrameLoading(false);
    }
  };

  // Deep-link support: navigate('review', taskId).
  useEffect(() => {
    if (entityId) void selectTask(entityId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entityId]);

  const annotation =
    (frame && selectedTask
      ? frame.annotations.find((a) => a.annotation_id === selectedTask.annotation_id)
      : null) ?? null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {tasks.loading && !tasks.data ? <LoadingBox label="Loading review tasks…" /> : null}
      {tasks.error && !tasks.data ? <ErrorNote error={tasks.error} /> : null}
      {loadError ? <ErrorNote error={loadError} /> : null}

      <HITLReview
        tasks={tasks.data?.tasks ?? []}
        selectedTask={selectedTask}
        onSelectTask={(id) => void selectTask(id)}
        frame={frame}
        prevFrame={prevFrame}
        nextFrame={nextFrame}
        frameLoading={frameLoading}
      />

      <RelabelingPanel
        task={selectedTask}
        annotation={annotation}
        onResolved={() => {
          tasks.refresh();
          if (selectedTask) {
            // Reload the task so its resolution/status is up to date.
            getReviewTask(selectedTask.task_id)
              .then(setSelectedTask)
              .catch(() => undefined);
          }
        }}
      />
    </Box>
  );
}
