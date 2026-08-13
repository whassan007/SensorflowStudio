import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Switch from '@mui/material/Switch';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import { Save, Play } from 'lucide-react';
import type { AnomalyConfig, EnsembleStrategy } from '../../types/labeleval';
import { getAnomalyConfig, postAnomalyConfig, runPipeline } from '../../services/labeleval';
import { SectionCard } from './shared';
import AnomalyDetectionConfig from './AnomalyDetectionConfig';

const DEFAULT_CONFIG: AnomalyConfig = {
  imbalance: { method: 'smote', minority_boost: 2 },
  detectors: {
    knn: { enabled: true, k: 5 },
    lof: { enabled: true, n_neighbors: 20 },
    isolation_forest: { enabled: true, n_estimators: 100 },
    ocsvm: { enabled: false, nu: 0.05 },
    dbscan: { enabled: false, eps: 0.5, min_samples: 5 },
  },
  deep: {
    autoencoder: { enabled: true, latent_dim: 16, epochs: 20 },
    vae: { enabled: false, latent_dim: 8 },
    gan: { enabled: false },
    reconstruction_threshold: 0.5,
  },
  advanced: {
    few_shot: { enabled: false, support_per_class: 5 },
    ensemble_strategy: 'weighted_average',
    decision_threshold: 0.6,
  },
};

function Column({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Box
      sx={{
        flex: '1 1 220px',
        minWidth: 220,
        bgcolor: '#12171d',
        border: '1px solid #232a31',
        borderRadius: 1,
        p: 1.5,
      }}
    >
      <Typography
        variant="caption"
        sx={{ color: '#4fc3f7', fontWeight: 700, letterSpacing: 0.8, display: 'block', mb: 1 }}
      >
        {title}
      </Typography>
      {children}
    </Box>
  );
}

export default function RareEventConfig({
  activeDatasetId,
  onRunStarted,
}: {
  activeDatasetId: string | null;
  onRunStarted?: (runId: string) => void;
}) {
  const [config, setConfig] = useState<AnomalyConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<'save' | 'run' | null>(null);
  const [notice, setNotice] = useState<{ severity: 'success' | 'error' | 'warning'; text: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getAnomalyConfig()
      .then((c) => {
        if (!cancelled) setConfig(c);
      })
      .catch(() => {
        /* backend offline — keep defaults, save will still POST */
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const save = async () => {
    setBusy('save');
    setNotice(null);
    try {
      const saved = await postAnomalyConfig(config);
      setConfig(saved);
      setNotice({ severity: 'success', text: 'Configuration saved.' });
    } catch (err) {
      setNotice({ severity: 'error', text: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(null);
    }
  };

  const rerun = async () => {
    if (!activeDatasetId) {
      setNotice({ severity: 'warning', text: 'Select or generate a dataset first (Datasets page).' });
      return;
    }
    setBusy('run');
    setNotice(null);
    try {
      const res = await runPipeline(activeDatasetId);
      setNotice({ severity: 'success', text: `Evaluation run ${res.run_id} started (${res.status}).` });
      onRunStarted?.(res.run_id);
    } catch (err) {
      setNotice({ severity: 'error', text: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <SectionCard
      title="Rare Event Detection Configuration"
      help="The anomaly ensemble, configured in four stages: (1) class-imbalance handling before detection, (2) classical detectors (KNN, LOF, Isolation Forest, OC-SVM, DBSCAN), (3) deep reconstruction detectors (autoencoder/VAE/GAN), (4) fusion — how individual scores combine into one anomaly score and the decision threshold. Save persists the config; Re-run applies it to the active dataset."
      action={
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            size="small"
            variant="outlined"
            startIcon={busy === 'save' ? <CircularProgress size={14} /> : <Save size={14} />}
            disabled={busy !== null || loading}
            onClick={() => void save()}
          >
            Save
          </Button>
          <Button
            size="small"
            variant="contained"
            startIcon={busy === 'run' ? <CircularProgress size={14} /> : <Play size={14} />}
            disabled={busy !== null}
            onClick={() => void rerun()}
          >
            Re-run evaluation
          </Button>
        </Box>
      }
    >
      {notice ? (
        <Alert severity={notice.severity} variant="outlined" onClose={() => setNotice(null)} sx={{ mb: 1 }}>
          {notice.text}
        </Alert>
      ) : null}
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {/* Column 1: imbalanced data */}
        <Column title="IMBALANCED DATA">
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Method
          </Typography>
          <Select
            size="small"
            fullWidth
            value={config.imbalance.method}
            onChange={(e) =>
              setConfig({
                ...config,
                imbalance: { ...config.imbalance, method: e.target.value as AnomalyConfig['imbalance']['method'] },
              })
            }
            sx={{ mb: 2, fontSize: 13 }}
          >
            <MenuItem value="none">None</MenuItem>
            <MenuItem value="smote">SMOTE</MenuItem>
            <MenuItem value="oversample">Oversample</MenuItem>
            <MenuItem value="undersample">Undersample</MenuItem>
            <MenuItem value="class_weights">Class weights</MenuItem>
          </Select>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Minority boost ×{config.imbalance.minority_boost.toFixed(1)}
          </Typography>
          <Slider
            size="small"
            min={1}
            max={10}
            step={0.5}
            value={config.imbalance.minority_boost}
            onChange={(_, v) =>
              setConfig({ ...config, imbalance: { ...config.imbalance, minority_boost: v as number } })
            }
          />
        </Column>

        {/* Column 2: classical anomaly detectors */}
        <Column title="ANOMALY DETECTION">
          <AnomalyDetectionConfig
            detectors={config.detectors}
            onChange={(detectors) => setConfig({ ...config, detectors })}
          />
        </Column>

        {/* Column 3: deep learning */}
        <Column title="DEEP LEARNING">
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Autoencoder
            </Typography>
            <Switch
              size="small"
              checked={config.deep.autoencoder.enabled}
              onChange={(e) =>
                setConfig({
                  ...config,
                  deep: { ...config.deep, autoencoder: { ...config.deep.autoencoder, enabled: e.target.checked } },
                })
              }
            />
          </Box>
          {config.deep.autoencoder.enabled ? (
            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
              <TextField
                size="small"
                type="number"
                label="latent"
                value={config.deep.autoencoder.latent_dim}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    deep: {
                      ...config.deep,
                      autoencoder: { ...config.deep.autoencoder, latent_dim: Number(e.target.value) },
                    },
                  })
                }
                sx={{ flex: 1 }}
              />
              <TextField
                size="small"
                type="number"
                label="epochs"
                value={config.deep.autoencoder.epochs}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    deep: {
                      ...config.deep,
                      autoencoder: { ...config.deep.autoencoder, epochs: Number(e.target.value) },
                    },
                  })
                }
                sx={{ flex: 1 }}
              />
            </Box>
          ) : null}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              VAE
            </Typography>
            <Switch
              size="small"
              checked={config.deep.vae.enabled}
              onChange={(e) =>
                setConfig({
                  ...config,
                  deep: { ...config.deep, vae: { ...config.deep.vae, enabled: e.target.checked } },
                })
              }
            />
          </Box>
          {config.deep.vae.enabled ? (
            <TextField
              size="small"
              type="number"
              label="latent"
              value={config.deep.vae.latent_dim}
              onChange={(e) =>
                setConfig({
                  ...config,
                  deep: { ...config.deep, vae: { ...config.deep.vae, latent_dim: Number(e.target.value) } },
                })
              }
              sx={{ mb: 1, width: '100%' }}
            />
          ) : null}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              GAN
            </Typography>
            <Switch
              size="small"
              checked={config.deep.gan.enabled}
              onChange={(e) =>
                setConfig({ ...config, deep: { ...config.deep, gan: { enabled: e.target.checked } } })
              }
            />
          </Box>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Reconstruction threshold {config.deep.reconstruction_threshold.toFixed(2)}
          </Typography>
          <Slider
            size="small"
            min={0}
            max={1}
            step={0.01}
            value={config.deep.reconstruction_threshold}
            onChange={(_, v) =>
              setConfig({ ...config, deep: { ...config.deep, reconstruction_threshold: v as number } })
            }
          />
        </Column>

        {/* Column 4: advanced / ensemble */}
        <Column title="ADVANCED / ENSEMBLE">
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Few-shot
            </Typography>
            <Switch
              size="small"
              checked={config.advanced.few_shot.enabled}
              onChange={(e) =>
                setConfig({
                  ...config,
                  advanced: {
                    ...config.advanced,
                    few_shot: { ...config.advanced.few_shot, enabled: e.target.checked },
                  },
                })
              }
            />
          </Box>
          {config.advanced.few_shot.enabled ? (
            <TextField
              size="small"
              type="number"
              label="support / class"
              value={config.advanced.few_shot.support_per_class}
              onChange={(e) =>
                setConfig({
                  ...config,
                  advanced: {
                    ...config.advanced,
                    few_shot: { ...config.advanced.few_shot, support_per_class: Number(e.target.value) },
                  },
                })
              }
              sx={{ mb: 1, width: '100%' }}
            />
          ) : null}
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Ensemble strategy
          </Typography>
          <Select
            size="small"
            fullWidth
            value={config.advanced.ensemble_strategy}
            onChange={(e) =>
              setConfig({
                ...config,
                advanced: { ...config.advanced, ensemble_strategy: e.target.value as EnsembleStrategy },
              })
            }
            sx={{ mb: 2, fontSize: 13 }}
          >
            <MenuItem value="majority_vote">Majority vote</MenuItem>
            <MenuItem value="weighted_average">Weighted average</MenuItem>
            <MenuItem value="meta_classifier">Meta classifier</MenuItem>
          </Select>
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            Decision threshold {config.advanced.decision_threshold.toFixed(2)}
          </Typography>
          <Slider
            size="small"
            min={0}
            max={1}
            step={0.01}
            value={config.advanced.decision_threshold}
            onChange={(_, v) =>
              setConfig({ ...config, advanced: { ...config.advanced, decision_threshold: v as number } })
            }
          />
        </Column>
      </Box>
    </SectionCard>
  );
}
