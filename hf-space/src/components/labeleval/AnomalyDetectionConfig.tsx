import Box from '@mui/material/Box';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import type { AnomalyConfig } from '../../types/labeleval';

type Detectors = AnomalyConfig['detectors'];

function DetectorRow({
  label,
  enabled,
  onToggle,
  children,
}: {
  label: string;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  children?: React.ReactNode;
}) {
  return (
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {label}
        </Typography>
        <Switch size="small" checked={enabled} onChange={(e) => onToggle(e.target.checked)} />
      </Box>
      {enabled ? <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>{children}</Box> : null}
    </Box>
  );
}

function NumField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <TextField
      size="small"
      type="number"
      label={label}
      value={value}
      inputProps={{ step: step ?? 1 }}
      onChange={(e) => onChange(Number(e.target.value))}
      sx={{ flex: 1, '& input': { fontSize: 13 } }}
    />
  );
}

/** Column 2 of the rare-event configuration grid: classical anomaly detectors. */
export default function AnomalyDetectionConfig({
  detectors,
  onChange,
}: {
  detectors: Detectors;
  onChange: (detectors: Detectors) => void;
}) {
  const set = <K extends keyof Detectors>(key: K, patch: Partial<Detectors[K]>) =>
    onChange({ ...detectors, [key]: { ...detectors[key], ...patch } });

  return (
    <Box>
      <DetectorRow label="KNN" enabled={detectors.knn.enabled} onToggle={(v) => set('knn', { enabled: v })}>
        <NumField label="k" value={detectors.knn.k} onChange={(v) => set('knn', { k: v })} />
      </DetectorRow>
      <DetectorRow label="LOF" enabled={detectors.lof.enabled} onToggle={(v) => set('lof', { enabled: v })}>
        <NumField
          label="n_neighbors"
          value={detectors.lof.n_neighbors}
          onChange={(v) => set('lof', { n_neighbors: v })}
        />
      </DetectorRow>
      <DetectorRow
        label="Isolation Forest"
        enabled={detectors.isolation_forest.enabled}
        onToggle={(v) => set('isolation_forest', { enabled: v })}
      >
        <NumField
          label="n_estimators"
          value={detectors.isolation_forest.n_estimators}
          onChange={(v) => set('isolation_forest', { n_estimators: v })}
        />
      </DetectorRow>
      <DetectorRow label="One-Class SVM" enabled={detectors.ocsvm.enabled} onToggle={(v) => set('ocsvm', { enabled: v })}>
        <NumField label="nu" value={detectors.ocsvm.nu} step={0.01} onChange={(v) => set('ocsvm', { nu: v })} />
      </DetectorRow>
      <DetectorRow
        label="DBSCAN"
        enabled={detectors.dbscan.enabled}
        onToggle={(v) => set('dbscan', { enabled: v })}
      >
        <NumField label="eps" value={detectors.dbscan.eps} step={0.05} onChange={(v) => set('dbscan', { eps: v })} />
        <NumField
          label="min_samples"
          value={detectors.dbscan.min_samples}
          onChange={(v) => set('dbscan', { min_samples: v })}
        />
      </DetectorRow>
    </Box>
  );
}
