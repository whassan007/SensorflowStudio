import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { ArrowRight, ShieldAlert } from 'lucide-react';
import { MetricCard } from '../../components/labeleval/shared';
import { VitisSection } from '../../components/vitis/Section';
import {
  generateAugmentBatch,
  getAugmentRecipes,
  runIsp,
} from '../../services/vitis';
import type { AugmentBatch, AugmentationSpec, IspRun } from '../../types/vitis';

function psnrColor(psnr: number): string {
  if (psnr >= 45) return '#66bb6a';
  if (psnr >= 35) return '#ffb74d';
  return '#ef5350';
}

export default function IspTab({ device }: { device: string }) {
  const [widthBits, setWidthBits] = useState(12);
  const [nFrames, setNFrames] = useState(4);
  const [seed, setSeed] = useState(11);
  const [busy, setBusy] = useState<'isp' | 'aug' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<IspRun | null>(null);

  const [recipes, setRecipes] = useState<AugmentationSpec[]>([]);
  const [selected, setSelected] = useState<string[]>(['low_light', 'glare', 'motion_blur']);
  const [nVariants, setNVariants] = useState(8);
  const [batch, setBatch] = useState<AugmentBatch | null>(null);

  useEffect(() => {
    getAugmentRecipes()
      .then((r) => setRecipes(r.augmentations))
      .catch(() => undefined);
  }, []);

  const doIsp = async () => {
    setBusy('isp');
    setError(null);
    try {
      setRun(
        await runIsp({
          n_frames: nFrames,
          seed,
          width_bits: widthBits,
          int_bits: 4,
          device,
          include_previews: true,
        })
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const doAugment = async () => {
    setBusy('aug');
    setError(null);
    try {
      setBatch(
        await generateAugmentBatch({
          recipes: selected.map((aug) => ({ aug })),
          n_variants: nVariants,
          seed,
          backend: 'vitis_emulated',
          width_bits: widthBits,
          device,
          include_thumbnails: true,
        })
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <VitisSection
        title="ISP pipeline"
        subtitle="Defective RAW Bayer → bad-pixel correction → demosaic → HDR tone-map → gain → denoise → resize, on both backends. Per-stage PSNR/SSIM is emulated-vs-reference."
      >
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', mb: 1.5 }}>
          <TextField select size="small" label="Bit width W" value={widthBits}
            onChange={(e) => setWidthBits(Number(e.target.value))} sx={{ width: 130 }}>
            {[8, 10, 12, 14, 16].map((w) => (
              <MenuItem key={w} value={w}>{w} bits</MenuItem>
            ))}
          </TextField>
          <TextField size="small" type="number" label="Frames" value={nFrames}
            onChange={(e) => setNFrames(Math.max(1, Math.min(24, Number(e.target.value))))} sx={{ width: 90 }} />
          <TextField size="small" type="number" label="Seed" value={seed}
            onChange={(e) => setSeed(Number(e.target.value))} sx={{ width: 90 }} />
          <Button variant="contained" size="small" onClick={doIsp} disabled={busy !== null}>
            {busy === 'isp' ? <CircularProgress size={16} /> : 'Run ISP on both backends'}
          </Button>
        </Box>

        {run ? (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1.5 }}>
              {run.stage_report.map((s, i) => (
                <Box key={s.stage} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Tooltip title={`SSIM ${s.ssim.toFixed(4)} · CPU ${s.measured_cpu_ms.toFixed(2)} ms · modeled FPGA ${s.modeled_fpga_ms?.toFixed(3)} ms (${s.modeled_placement}) — modeled, not measured`}>
                    <Chip
                      size="small"
                      label={`${s.stage} · ${s.psnr_db.toFixed(1)} dB`}
                      sx={{ bgcolor: '#161b21', border: `1px solid ${psnrColor(s.psnr_db)}`, color: psnrColor(s.psnr_db), fontFamily: 'monospace', fontSize: 11 }}
                    />
                  </Tooltip>
                  {i < run.stage_report.length - 1 ? <ArrowRight size={14} color="#8a949e" /> : null}
                </Box>
              ))}
            </Box>
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 1.5 }}>
              <MetricCard label="CPU (measured)" value={`${run.throughput.measured_cpu_fps.toFixed(0)} fps`}
                sub={`${run.throughput.measured_cpu_ms_per_frame.toFixed(2)} ms/frame — wall clock`}
                info="Actually measured wall-clock time of the reference float32 pipeline on this machine's CPU." />
              <MetricCard label="FPGA (modeled)" value={`${run.throughput.modeled_fpga_fps_pipelined.toFixed(0)} fps`}
                sub={`${run.throughput.modeled_fpga_ms_per_frame_serial.toFixed(3)} ms serial · dataflow-pipelined`}
                accent="#ffb74d"
                info={run.throughput.note} />
              <MetricCard label="Modeled speedup" value={`${run.throughput.modeled_speedup_x_serial.toFixed(1)}×`}
                sub="MODELED, NOT MEASURED" accent="#ffb74d"
                info="Analytical pixels/cycle × clock model — no FPGA hardware was involved. Validate on silicon before quoting." />
            </Box>
            {run.previews.length > 0 ? (
              <Box>
                <Typography variant="caption" sx={{ color: '#8a949e' }}>
                  Per-stage outputs (frame {run.previews[0].frame_id}, {run.previews[0].cohort}) — reference / emulated / |diff|×8:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mt: 0.5 }}>
                  {run.previews[0].stages.map((s) => (
                    <Box key={s.stage} sx={{ textAlign: 'center' }}>
                      <Typography variant="caption" sx={{ display: 'block', color: '#8a949e', fontSize: 10 }}>{s.stage}</Typography>
                      <Box sx={{ display: 'flex', gap: 0.25 }}>
                        <img src={s.reference_png} alt={`${s.stage} reference`} width={86} style={{ imageRendering: 'pixelated' }} />
                        <img src={s.vitis_png} alt={`${s.stage} emulated`} width={86} style={{ imageRendering: 'pixelated' }} />
                        <img src={s.diff_png} alt={`${s.stage} diff`} width={86} style={{ imageRendering: 'pixelated' }} />
                      </Box>
                    </Box>
                  ))}
                </Box>
              </Box>
            ) : null}
          </>
        ) : (
          <Alert severity="info">Run the pipeline to see per-stage quality, throughput comparison and stage outputs.</Alert>
        )}
      </VitisSection>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <VitisSection
        title="Synthetic edge-case generation"
        subtitle="Hardware-accelerated augmentations mint stress variants of evaluation frames. Every variant carries full lineage and is evaluation-only by default (leakage guard)."
      >
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', mb: 1.5 }}>
          <TextField
            select size="small" label="Recipes" value={selected}
            onChange={(e) => setSelected(typeof e.target.value === 'string' ? e.target.value.split(',') : (e.target.value as string[]))}
            SelectProps={{ multiple: true }}
            sx={{ minWidth: 280 }}
          >
            {recipes.map((r) => (
              <MenuItem key={r.name} value={r.name}>
                <Tooltip title={r.description} placement="right">
                  <span>{r.name}</span>
                </Tooltip>
              </MenuItem>
            ))}
          </TextField>
          <TextField size="small" type="number" label="Variants" value={nVariants}
            onChange={(e) => setNVariants(Math.max(1, Math.min(64, Number(e.target.value))))} sx={{ width: 100 }} />
          <Button variant="contained" size="small" onClick={doAugment} disabled={busy !== null || selected.length === 0}>
            {busy === 'aug' ? <CircularProgress size={16} /> : 'Generate variants'}
          </Button>
        </Box>

        {batch ? (
          <>
            <Alert severity={batch.raremine_hook.available ? 'success' : 'info'} sx={{ mb: 1.5 }}>
              {batch.raremine_hook.available
                ? `${batch.raremine_hook.routed_candidates} pedestrian-bearing variants routed into the raremine candidate flow.`
                : batch.raremine_hook.note}
            </Alert>
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
              {batch.variants.map((v) => (
                <Box key={v.variant_id} sx={{ width: 150, bgcolor: '#161b21', border: '1px solid #232a31', borderRadius: 1, p: 1 }}>
                  {v.thumbnail_png ? (
                    <img src={v.thumbnail_png} alt={v.variant_id} width="100%" style={{ imageRendering: 'pixelated', borderRadius: 3 }} />
                  ) : null}
                  <Typography variant="caption" sx={{ display: 'block', fontFamily: 'monospace', fontSize: 10 }}>
                    {v.variant_id}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                    {v.recipe.map((r, i) => (
                      <Chip key={i} size="small" label={r.aug} sx={{ height: 16, fontSize: 9, bgcolor: '#232a31' }} />
                    ))}
                  </Box>
                  <Tooltip
                    title={`Lineage — source: ${v.lineage.source_frame_id} · seed: ${v.lineage.seed} · backend: ${v.lineage.backend} · destination: ${v.recommended_dataset_destination} · never training-eligible`}
                  >
                    <Chip
                      size="small"
                      icon={<ShieldAlert size={11} />}
                      label="EVAL-ONLY"
                      sx={{ mt: 0.5, height: 18, fontSize: 9, fontWeight: 700, bgcolor: '#4a1f1f', color: '#ef9a9a' }}
                    />
                  </Tooltip>
                </Box>
              ))}
            </Box>
          </>
        ) : (
          <Alert severity="info">Pick recipes and generate a batch — the gallery shows lineage and the evaluation-only leakage guard per variant.</Alert>
        )}
      </VitisSection>
    </Box>
  );
}
