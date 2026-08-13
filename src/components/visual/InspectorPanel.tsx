/**
 * Property inspector panel (direct-manipulation primitive).
 *
 * Click an object on any canvas → this panel shows its editable properties;
 * every change calls back immediately so the canvas re-renders live (WYSIWYG).
 * Field types: text, number, slider, select, toggle, readonly.
 */
import type { ReactNode } from 'react';
import Box from '@mui/material/Box';
import MenuItem from '@mui/material/MenuItem';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { InfoDot } from '../help/InfoTip';
import { tokens } from '../../theme';

export type InspectorField =
  | { type: 'text'; key: string; label: string; value: string; help?: string }
  | { type: 'number'; key: string; label: string; value: number; min?: number; max?: number; step?: number; help?: string }
  | { type: 'slider'; key: string; label: string; value: number; min: number; max: number; step?: number; unit?: string; help?: string }
  | { type: 'select'; key: string; label: string; value: string; options: Array<{ value: string; label: string }>; help?: string }
  | { type: 'toggle'; key: string; label: string; value: boolean; help?: string }
  | { type: 'readonly'; key: string; label: string; value: ReactNode; help?: string };

interface InspectorPanelProps {
  title: string;
  subtitle?: string;
  accent?: string;
  fields: InspectorField[];
  onChange: (key: string, value: string | number | boolean) => void;
  footer?: ReactNode;
  emptyHint?: string;
}

function FieldLabel({ label, help }: { label: string; help?: string }) {
  return (
    <Typography variant="caption" sx={{ color: tokens.color.neutral, display: 'block', mb: 0.25 }}>
      {label}
      {help ? <InfoDot title={label} detail={help} size={11} /> : null}
    </Typography>
  );
}

export default function InspectorPanel({ title, subtitle, accent, fields, onChange, footer, emptyHint }: InspectorPanelProps) {
  return (
    <Box
      sx={{
        border: `1px solid ${tokens.color.border}`,
        borderLeft: `3px solid ${accent ?? tokens.color.info}`,
        borderRadius: 1,
        bgcolor: tokens.color.surface,
        p: 1.5,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
        transition: `border-color ${tokens.motion.normal}`,
      }}
    >
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
          {title}
        </Typography>
        {subtitle ? (
          <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
            {subtitle}
          </Typography>
        ) : null}
      </Box>

      {fields.length === 0 && emptyHint ? (
        <Typography variant="body2" sx={{ color: tokens.color.neutral, fontSize: 12.5 }}>
          {emptyHint}
        </Typography>
      ) : null}

      {fields.map((f) => {
        switch (f.type) {
          case 'text':
            return (
              <Box key={f.key}>
                <FieldLabel label={f.label} help={f.help} />
                <TextField size="small" fullWidth value={f.value} onChange={(e) => onChange(f.key, e.target.value)} />
              </Box>
            );
          case 'number':
            return (
              <Box key={f.key}>
                <FieldLabel label={f.label} help={f.help} />
                <TextField
                  size="small"
                  fullWidth
                  type="number"
                  value={Number.isFinite(f.value) ? f.value : ''}
                  inputProps={{ min: f.min, max: f.max, step: f.step ?? 'any' }}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (!Number.isNaN(v)) onChange(f.key, v);
                  }}
                />
              </Box>
            );
          case 'slider':
            return (
              <Box key={f.key}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                  <FieldLabel label={f.label} help={f.help} />
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', color: tokens.color.text }}>
                    {f.value}
                    {f.unit ?? ''}
                  </Typography>
                </Box>
                <Slider
                  size="small"
                  value={f.value}
                  min={f.min}
                  max={f.max}
                  step={f.step ?? 0.1}
                  onChange={(_, v) => onChange(f.key, v as number)}
                  sx={{ py: 0.5 }}
                />
              </Box>
            );
          case 'select':
            return (
              <Box key={f.key}>
                <FieldLabel label={f.label} help={f.help} />
                <TextField size="small" select fullWidth value={f.value} onChange={(e) => onChange(f.key, e.target.value)}>
                  {f.options.map((o) => (
                    <MenuItem key={o.value} value={o.value}>
                      {o.label}
                    </MenuItem>
                  ))}
                </TextField>
              </Box>
            );
          case 'toggle':
            return (
              <Box key={f.key} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <FieldLabel label={f.label} help={f.help} />
                <Switch size="small" checked={f.value} onChange={(e) => onChange(f.key, e.target.checked)} />
              </Box>
            );
          case 'readonly':
            return (
              <Box key={f.key}>
                <FieldLabel label={f.label} help={f.help} />
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, color: tokens.color.textDim }}>
                  {f.value}
                </Typography>
              </Box>
            );
          default:
            return null;
        }
      })}

      {footer}
    </Box>
  );
}
