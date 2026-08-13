/**
 * Attribution-matrix heatmap: one row per violation, one column per causal
 * layer. Cell color encodes tri-state evidence (SUPPORTED / RULED_OUT /
 * UNKNOWN); the primary layer carries a ring. Clicking a row selects the
 * violation for the consequence replay view.
 */

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import type { RotrAttributionMatrix } from '../../types/rotr';

const LAYERS = [
  'perception',
  'prediction',
  'planning',
  'localization',
  'map',
  'control',
  'policy_rule',
  'data_label',
];

const STATUS_COLOR: Record<string, string> = {
  SUPPORTED: '#c62828',
  RULED_OUT: '#1b5e20',
  UNKNOWN: '#616161',
};

interface Props {
  matrix: RotrAttributionMatrix;
  selectedViolationId: string | null;
  onSelect: (violationId: string) => void;
}

export default function RotrAttributionHeatmap({ matrix, selectedViolationId, onSelect }: Props) {
  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Box sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        {Object.entries(STATUS_COLOR).map(([status, color]) => (
          <Chip
            key={status}
            size="small"
            label={status}
            sx={{ bgcolor: color, color: '#fff', fontSize: 10, height: 18 }}
          />
        ))}
        <Typography variant="caption" sx={{ color: '#8a949e' }}>
          ring = primary layer (highest-confidence positive evidence)
        </Typography>
      </Box>
      <table style={{ borderCollapse: 'separate', borderSpacing: 3 }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left', fontSize: 11, color: '#8a949e', paddingRight: 8 }}>
              violation
            </th>
            {LAYERS.map((l) => (
              <th
                key={l}
                style={{
                  fontSize: 10,
                  color: '#8a949e',
                  writingMode: 'vertical-rl',
                  transform: 'rotate(180deg)',
                  padding: '2px 0',
                }}
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => {
            const selected = row.violation_id === selectedViolationId;
            return (
              <tr
                key={row.violation_id}
                onClick={() => onSelect(row.violation_id)}
                style={{ cursor: 'pointer' }}
              >
                <td
                  style={{
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: selected ? '#4fc3f7' : '#c9d1d9',
                    paddingRight: 8,
                    whiteSpace: 'nowrap',
                    borderLeft: selected ? '3px solid #4fc3f7' : '3px solid transparent',
                    paddingLeft: 5,
                  }}
                >
                  {row.violation_id.replace(/^bank-[0-9a-f]+-/, '')}
                </td>
                {LAYERS.map((layer) => {
                  const cell = row.layers[layer];
                  const status = cell?.status ?? 'UNKNOWN';
                  const isPrimary = row.primary_layer === layer;
                  return (
                    <Tooltip
                      key={layer}
                      title={
                        <Box sx={{ maxWidth: 340 }}>
                          <b>
                            {layer}: {status}
                          </b>{' '}
                          (conf {cell ? cell.confidence.toFixed(2) : '—'})
                          <br />
                          {cell?.evidence ?? 'no evidence recorded'}
                        </Box>
                      }
                      arrow
                    >
                      <td
                        style={{
                          width: 26,
                          height: 20,
                          background: STATUS_COLOR[status],
                          opacity: status === 'SUPPORTED' ? (cell ? 0.45 + 0.55 * cell.confidence : 1) : 0.85,
                          borderRadius: 3,
                          outline: isPrimary ? '2px solid #ffb300' : 'none',
                          outlineOffset: -2,
                        }}
                      />
                    </Tooltip>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
        {matrix.invariant}
      </Typography>
    </Box>
  );
}
