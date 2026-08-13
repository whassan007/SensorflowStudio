import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { X } from 'lucide-react';
import { annotateStreet } from '../../services/api';
import { useFilters } from '../../context/FilterContext';
import { SEVERITY_COLORS } from '../../types';
import { Term } from '../help/InfoTip';

export default function SeverityDrawer() {
  const { selected, setSelected } = useFilters();
  const [annotation, setAnnotation] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveState, setSaveState] = useState<'idle' | 'saved' | 'error'>('idle');

  useEffect(() => {
    setAnnotation(selected?.manual_annotation ?? '');
    setSaveState('idle');
  }, [selected?.id]);

  const handleSave = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await annotateStreet(selected.street_name, annotation);
      setSaveState('saved');
    } catch {
      setSaveState('error');
    } finally {
      setSaving(false);
    }
  };

  const severityColor = selected ? SEVERITY_COLORS[selected.severity_label] : '#888';

  return (
    <Drawer
      anchor="right"
      variant="persistent"
      open={selected !== null}
      PaperProps={{ sx: { width: 340, p: 2.5, boxSizing: 'border-box', top: 'auto', position: 'absolute' } }}
      sx={{ position: 'relative' }}
    >
      {selected && (
        <>
          <div className="drawer-header">
            <div>
              <Typography variant="h6" lineHeight={1.2}>
                {selected.street_name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {selected.county} County · {selected.id}
              </Typography>
            </div>
            <IconButton size="small" onClick={() => setSelected(null)} aria-label="Close details">
              <X size={18} />
            </IconButton>
          </div>

          <div className="severity-block">
            <Typography variant="overline" color="text.secondary">
              <Term k="severity_index">Severity index</Term>
            </Typography>
            <div className="severity-value" style={{ color: severityColor }}>
              {selected.severity_index.toFixed(2)}
              <span className="severity-label" style={{ backgroundColor: severityColor }}>
                {selected.severity_label}
              </span>
            </div>
            <LinearProgress
              variant="determinate"
              value={selected.severity_index * 100}
              sx={{
                height: 8,
                borderRadius: 4,
                mt: 1,
                '& .MuiLinearProgress-bar': { backgroundColor: severityColor },
              }}
            />
          </div>

          <Divider sx={{ my: 2 }} />

          <dl className="metric-list">
            <div>
              <dt>Conflict type</dt>
              <dd>{selected.conflict_type}</dd>
            </div>
            <div>
              <dt>
                <Term k="ttc">Min time-to-collision</Term>
              </dt>
              <dd>{selected.min_ttc.toFixed(1)} s</dd>
            </div>
            <div>
              <dt>
                <Term k="pet">Min post-encroachment time</Term>
              </dt>
              <dd>{selected.min_pet.toFixed(1)} s</dd>
            </div>
            <div>
              <dt>Max approach speed</dt>
              <dd>{selected.max_speed.toFixed(1)} m/s</dd>
            </div>
            <div>
              <dt>Location</dt>
              <dd>
                {selected.lat.toFixed(4)}, {selected.lng.toFixed(4)}
              </dd>
            </div>
          </dl>

          <Divider sx={{ my: 2 }} />

          <Typography variant="overline" color="text.secondary">
            Analyst annotation
          </Typography>
          <TextField
            multiline
            minRows={3}
            fullWidth
            size="small"
            placeholder="Observations, mitigation notes…"
            value={annotation}
            onChange={(e) => {
              setAnnotation(e.target.value);
              setSaveState('idle');
            }}
            sx={{ mt: 1 }}
          />
          <Button variant="contained" size="small" onClick={handleSave} disabled={saving} sx={{ mt: 1.5, alignSelf: 'flex-start' }}>
            {saving ? 'Saving…' : 'Save annotation'}
          </Button>
          {saveState === 'saved' && (
            <Alert severity="success" sx={{ mt: 1.5 }}>
              Annotation saved.
            </Alert>
          )}
          {saveState === 'error' && (
            <Alert severity="error" sx={{ mt: 1.5 }}>
              Failed to save annotation.
            </Alert>
          )}
        </>
      )}
    </Drawer>
  );
}
