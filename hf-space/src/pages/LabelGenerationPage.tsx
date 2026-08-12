import { useState } from 'react';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import type { FrameResponse } from '../services/labeleval';
import { getDatasets, getFrameIds, getFrame, usePoll } from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import InputStage from '../components/labeleval/InputStage';
import { SectionCard, StatusChip, LoadingBox, ErrorNote, fmtNum } from '../components/labeleval/shared';

export default function LabelGenerationPage() {
  const { activeDatasetId, setActiveDatasetId } = useLabelEval();
  const datasets = usePoll(getDatasets, 10000);
  const frameIds = usePoll(
    () => (activeDatasetId ? getFrameIds(activeDatasetId) : Promise.resolve({ frame_ids: [] as string[] })),
    null,
    [activeDatasetId]
  );
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);
  const [frameData, setFrameData] = useState<FrameResponse | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState<string | null>(null);

  const activeDataset = datasets.data?.datasets.find((d) => d.dataset_id === activeDatasetId) ?? null;

  const loadFrame = async (frameId: string) => {
    setSelectedFrame(frameId);
    setFrameLoading(true);
    setFrameError(null);
    try {
      setFrameData(await getFrame(frameId));
    } catch (err) {
      setFrameData(null);
      setFrameError(err instanceof Error ? err.message : String(err));
    } finally {
      setFrameLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <InputStage dataset={activeDataset} />

      <SectionCard title="Frame Browser — automated label hypotheses">
        <Alert severity="info" variant="outlined" sx={{ mb: 2 }}>
          Auto-generated labels are <strong>hypotheses, not truth</strong> — every annotation below must pass the
          evaluation pipeline (anomaly detection, grading, strict validation and triage) before it can be verified.
        </Alert>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            Dataset:
          </Typography>
          <Select
            size="small"
            value={activeDatasetId ?? ''}
            displayEmpty
            onChange={(e) => {
              setActiveDatasetId(e.target.value || null);
              setSelectedFrame(null);
              setFrameData(null);
            }}
            sx={{ minWidth: 240 }}
          >
            <MenuItem value="">
              <em>Select a dataset</em>
            </MenuItem>
            {(datasets.data?.datasets ?? []).map((d) => (
              <MenuItem key={d.dataset_id} value={d.dataset_id}>
                {d.name} ({d.version})
              </MenuItem>
            ))}
          </Select>
        </Box>

        {!activeDatasetId ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            Select a dataset to browse its frames.
          </Typography>
        ) : (
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <Box sx={{ flex: '0 1 240px', minWidth: 200 }}>
              <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
                FRAMES ({frameIds.data?.frame_ids.length ?? 0})
              </Typography>
              {frameIds.loading ? <LoadingBox label="Loading frames…" /> : null}
              {frameIds.error && !frameIds.data ? <ErrorNote error={frameIds.error} /> : null}
              <List dense sx={{ maxHeight: 420, overflowY: 'auto', bgcolor: '#12171d', borderRadius: 1, p: 0, mt: 0.5 }}>
                {(frameIds.data?.frame_ids ?? []).map((fid) => (
                  <ListItemButton
                    key={fid}
                    dense
                    selected={fid === selectedFrame}
                    onClick={() => void loadFrame(fid)}
                  >
                    <ListItemText primaryTypographyProps={{ fontFamily: 'monospace', fontSize: 12 }} primary={fid} />
                  </ListItemButton>
                ))}
              </List>
            </Box>

            <Box sx={{ flex: '1 1 480px', minWidth: 380 }}>
              {frameLoading ? <LoadingBox label="Loading frame…" /> : null}
              {frameError ? <ErrorNote error={frameError} /> : null}
              {frameData && !frameLoading ? (
                <>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                    {frameData.frame.frame_id} — scene {frameData.frame.scene_id}, seq {frameData.frame.sequence_id}
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
                    t = {frameData.frame.timestamp_us.toLocaleString()} µs ·{' '}
                    {frameData.frame.num_lidar_points.toLocaleString()} lidar points · ego speed{' '}
                    {fmtNum(frameData.frame.ego_pose.speed_mps, 1)} m/s · prev: {frameData.prev ?? '—'} · next:{' '}
                    {frameData.next ?? '—'}
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Annotation</TableCell>
                        <TableCell>Class</TableCell>
                        <TableCell align="right">Confidence</TableCell>
                        <TableCell>Model</TableCell>
                        <TableCell>Version</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {frameData.frame.annotations.map((a) => (
                        <TableRow key={a.annotation_id} hover>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{a.annotation_id}</TableCell>
                          <TableCell>{a.class_name}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                            {fmtNum(a.confidence)}
                          </TableCell>
                          <TableCell sx={{ fontSize: 12 }}>{a.model}</TableCell>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{a.model_version}</TableCell>
                          <TableCell>
                            <StatusChip status={a.status} />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {frameData.frame.annotations.length === 0 ? (
                    <Typography variant="body2" sx={{ color: '#8a949e', mt: 1 }}>
                      No annotations in this frame.
                    </Typography>
                  ) : null}
                </>
              ) : !frameLoading && !frameError ? (
                <Typography variant="body2" sx={{ color: '#8a949e' }}>
                  Select a frame to inspect its auto-label hypotheses.
                </Typography>
              ) : null}
            </Box>
          </Box>
        )}
      </SectionCard>
    </Box>
  );
}
