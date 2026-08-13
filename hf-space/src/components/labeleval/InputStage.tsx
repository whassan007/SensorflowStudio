import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import { Radar, Camera, Map as MapIcon, Compass, Activity, Clock, Route } from 'lucide-react';
import type { DatasetSummary } from '../../types/labeleval';
import { SectionCard, fmtInt } from './shared';

export default function InputStage({ dataset }: { dataset: DatasetSummary | null }) {
  const frames = dataset?.num_frames ?? 0;
  const annotations = dataset?.num_annotations ?? 0;
  const sequences = dataset?.num_sequences ?? 0;
  const scenes = dataset?.num_scenes ?? 0;

  const modalities = [
    { name: 'LiDAR', icon: <Radar size={18} />, detail: `${fmtInt(frames)} sweeps (1 per frame)` },
    { name: 'Camera', icon: <Camera size={18} />, detail: `${fmtInt(frames)} images (1 per frame)` },
    { name: 'Maps', icon: <MapIcon size={18} />, detail: `${fmtInt(scenes)} HD map tiles (1 per scene)` },
    { name: 'Ego Pose', icon: <Compass size={18} />, detail: `${fmtInt(frames)} poses @ frame rate` },
    { name: 'Telemetry', icon: <Activity size={18} />, detail: `${fmtInt(frames)} CAN/IMU samples` },
    { name: 'Timestamps', icon: <Clock size={18} />, detail: `${fmtInt(frames)} synchronized (µs)` },
    { name: 'Tracks', icon: <Route size={18} />, detail: `${fmtInt(annotations)} annotations with track ids` },
  ];

  return (
    <SectionCard
      title="Input Stage — Sensor Modalities"
      help="What the pipeline ingests per frame: LiDAR sweep, camera image, HD map tile, ego pose, telemetry and synchronized timestamps. Auto-labeling and every downstream check (point support, camera–LiDAR consistency, kinematics) consume these modalities."
    >
      <Typography variant="body2" sx={{ color: '#8a949e', mb: 1.5 }}>
        Hierarchy: <strong>Dataset</strong> ({dataset ? dataset.name : 'none selected'}) → <strong>Scene</strong> (
        {fmtInt(scenes)}) → <strong>Sequence</strong> ({fmtInt(sequences)}) → <strong>Frame</strong> ({fmtInt(frames)}
        ). Every frame bundles all sensor modalities below, time-synchronized, before auto-labeling.
      </Typography>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {modalities.map((m) => (
          <Card key={m.name} variant="outlined" sx={{ flex: '1 1 150px', minWidth: 150, bgcolor: '#12171d' }}>
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#4fc3f7', mb: 0.5 }}>
                {m.icon}
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                  {m.name}
                </Typography>
              </Box>
              <Typography variant="caption" sx={{ color: '#8a949e' }}>
                {m.detail}
              </Typography>
            </CardContent>
          </Card>
        ))}
      </Box>
    </SectionCard>
  );
}
