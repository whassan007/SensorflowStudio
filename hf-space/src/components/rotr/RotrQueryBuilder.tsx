/**
 * Structured-query builder for taxonomy mining: one dropdown chip per axis
 * plus a free-text field that the backend parses DETERMINISTICALLY (keyword
 * map, no LLM) into the same filterable query object.
 */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';

const AXIS_OPTIONS: Record<string, string[]> = {
  actor: ['pedestrian', 'cyclist', 'vehicle', 'none'],
  vulnerability: ['VRU', 'NON_VRU'],
  legality: ['YIELD', 'RESTRICTED_PATH', 'LANE_ASSOCIATION', 'SIGNAL', 'MERGE', 'STOP'],
  interaction: ['CROSSING', 'MERGING', 'FOLLOWING', 'OTHER', 'NONE'],
  behavior: [
    'proceed_without_yield',
    'enter_restricted_path',
    'unpermitted_maneuver',
    'run_red_signal',
    'insufficient_gap_merge',
    'rolling_stop',
  ],
  road_geometry: ['uncontrolled', 'controlled', 'none'],
  traffic_control: ['none', 'signal', 'stop_sign'],
  visibility: ['clear', 'low'],
  lighting: ['day', 'dusk', 'night'],
  weather: ['clear', 'rain'],
  consequence_class: [
    'NO_MATERIAL_CONSEQUENCE',
    'DEGRADED_COMFORT',
    'PLANNER_INTERVENTION',
    'SAFETY_CRITICAL',
  ],
  primary_layer: [
    'perception',
    'prediction',
    'planning',
    'localization',
    'map',
    'control',
    'policy_rule',
    'data_label',
  ],
};

interface Props {
  onRun: (text: string | undefined, filters: Record<string, string>) => void;
  busy?: boolean;
}

export default function RotrQueryBuilder({ onRun, busy }: Props) {
  const [text, setText] = useState(
    'failed to yield to pedestrian at uncontrolled intersection during low visibility'
  );
  const [filters, setFilters] = useState<Record<string, string>>({});

  const setAxis = (axis: string, value: string) => {
    setFilters((f) => {
      const next = { ...f };
      if (value) next[axis] = value;
      else delete next[axis];
      return next;
    });
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
        <TextField
          size="small"
          fullWidth
          label="natural-language query (parsed by a deterministic keyword map — no LLM)"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <Button variant="contained" disabled={busy} onClick={() => onRun(text || undefined, filters)}>
          Run query
        </Button>
      </Box>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {Object.entries(AXIS_OPTIONS).map(([axis, options]) => (
          <FormControl key={axis} size="small" sx={{ minWidth: 148 }}>
            <InputLabel sx={{ fontSize: 12 }}>{axis}</InputLabel>
            <Select
              label={axis}
              value={filters[axis] ?? ''}
              onChange={(e) => setAxis(axis, e.target.value)}
              sx={{ fontSize: 12 }}
            >
              <MenuItem value="">
                <em>any</em>
              </MenuItem>
              {options.map((o) => (
                <MenuItem key={o} value={o} sx={{ fontSize: 12 }}>
                  {o}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        ))}
      </Box>
      {Object.keys(filters).length ? (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 1, alignItems: 'center' }}>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            active chips:
          </Typography>
          {Object.entries(filters).map(([axis, value]) => (
            <Chip
              key={axis}
              size="small"
              label={`${axis}=${value}`}
              onDelete={() => setAxis(axis, '')}
              sx={{ height: 20, fontSize: 10, fontFamily: 'monospace' }}
            />
          ))}
        </Box>
      ) : null}
    </Box>
  );
}
