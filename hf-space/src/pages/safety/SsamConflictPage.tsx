/**
 * SSAM Conflict Panel — extended surrogate-safety analysis: scenario
 * selector, conflict list with TTC/PET/DRAC/ΔS/CSI chips, a mini
 * conflict-point map, and CSI comparison bars.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import MenuItem from '@mui/material/MenuItem';
import Slider from '@mui/material/Slider';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { PlayCircle } from 'lucide-react';
import { analyzeSsam, type SsamAnalysis, type SsamConflict } from '../../services/safety';
import { ChartSkeleton, IllustratedEmpty } from '../../components/visual/Feedback';
import { ErrorNote, HBar, MetricCard, SectionCard } from '../../components/labeleval/shared';
import { InfoDot } from '../../components/help/InfoTip';
import { tokens } from '../../theme';

const SCENARIOS = [
  { value: 'mixed', label: 'Mixed traffic (all conflict types)' },
  { value: 'crossing', label: 'Crossing conflict (intersection)' },
  { value: 'rear_end', label: 'Rear-end conflict (car following)' },
  { value: 'lane_change', label: 'Lane change conflict' },
];

const TYPE_COLORS: Record<string, string> = {
  crossing: '#ef5350',
  rear_end: '#ffa726',
  lane_change: '#4fc3f7',
};

function severityColor(c: SsamConflict): string {
  if (c.csi >= 2 || (c.min_ttc_s !== null && c.min_ttc_s < 0.5)) return tokens.color.danger;
  if (c.csi >= 0.75 || (c.min_ttc_s !== null && c.min_ttc_s < 1.0)) return '#ffa726';
  return tokens.color.warn;
}

function MeasureChip({ label, value, unit, term, danger }: { label: string; value: number | null; unit: string; term: string; danger?: boolean }) {
  return (
    <Chip
      size="small"
      label={
        <span>
          {label} <strong>{value === null ? '—' : value.toFixed(2)}</strong>
          {value === null ? '' : unit}
        </span>
      }
      title={term}
      sx={{
        height: 20,
        fontSize: 10.5,
        fontFamily: 'monospace',
        bgcolor: danger ? tokens.color.dangerBg : tokens.color.surfaceRaised,
        color: danger ? tokens.color.danger : tokens.color.textDim,
        border: `1px solid ${danger ? tokens.color.danger : tokens.color.border}`,
      }}
    />
  );
}

// ------------------------------------------------------------ conflict map

function ConflictMap({ conflicts, selected, onSelect }: { conflicts: SsamConflict[]; selected: number | null; onSelect: (i: number) => void }) {
  const pts = conflicts.filter((c) => c.conflict_point);
  const xs = pts.map((c) => c.conflict_point![0]);
  const ys = pts.map((c) => c.conflict_point![1]);
  const pad = 8;
  const xLo = Math.min(...xs, -10) - pad;
  const xHi = Math.max(...xs, 10) + pad;
  const yLo = Math.min(...ys, -10) - pad;
  const yHi = Math.max(...ys, 10) + pad;
  const W = 320;
  const H = 240;
  const sx = (x: number) => ((x - xLo) / (xHi - xLo)) * (W - 20) + 10;
  const sy = (y: number) => H - (((y - yLo) / (yHi - yLo)) * (H - 20) + 10);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: 380, display: 'block' }}>
      <rect x={0} y={0} width={W} height={H} rx={6} fill={tokens.color.surfaceSunken} stroke={tokens.color.border} />
      {/* road cross reference */}
      <line x1={sx(xLo + pad)} y1={sy(0)} x2={sx(xHi - pad)} y2={sy(0)} stroke={tokens.color.borderStrong} strokeDasharray="4 4" />
      <line x1={sx(0)} y1={sy(yLo + pad)} x2={sx(0)} y2={sy(yHi - pad)} stroke={tokens.color.borderStrong} strokeDasharray="4 4" />
      <text x={sx(xHi - pad) - 2} y={sy(0) - 4} textAnchor="end" fontSize={8.5} fill={tokens.color.textFaint}>x (m)</text>
      <text x={sx(0) + 4} y={sy(yHi - pad) + 10} fontSize={8.5} fill={tokens.color.textFaint}>y (m)</text>
      {conflicts.map((c, i) => {
        if (!c.conflict_point) return null;
        const r = 4 + Math.min(10, c.csi * 3);
        const isSel = selected === i;
        return (
          <g key={i} onClick={() => onSelect(i)} style={{ cursor: 'pointer' }}>
            <circle
              cx={sx(c.conflict_point[0])}
              cy={sy(c.conflict_point[1])}
              r={r}
              fill={TYPE_COLORS[c.conflict_type] ?? tokens.color.neutral}
              opacity={isSel ? 0.95 : 0.55}
              stroke={isSel ? '#fff' : 'none'}
              strokeWidth={1.5}
            >
              <title>{`${c.conflict_type}: ${c.vehicle_a} vs ${c.vehicle_b} · CSI ${c.csi.toFixed(2)}`}</title>
            </circle>
            {isSel ? (
              <text x={sx(c.conflict_point[0])} y={sy(c.conflict_point[1]) - r - 4} textAnchor="middle" fontSize={9} fontWeight={700} fill="#fff">
                CSI {c.csi.toFixed(2)}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------------------------------------------ page

export default function SsamConflictPage() {
  const [scenario, setScenario] = useState('mixed');
  const [seed, setSeed] = useState(0);
  const [reaction, setReaction] = useState(0.6);
  const [analysis, setAnalysis] = useState<SsamAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);

  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    analyzeSsam({ scenario, seed, reaction_delay_s: reaction })
      .then((a) => {
        setAnalysis(a);
        setSelected(a.conflicts.length ? 0 : null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [scenario, seed, reaction]);

  useEffect(run, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sortedByCsi = useMemo(
    () => (analysis ? [...analysis.conflicts].map((c, i) => ({ c, i })).sort((a, b) => b.c.csi - a.c.csi) : []),
    [analysis]
  );
  const agg = analysis?.aggregate;
  const glossary = analysis?.measures_glossary ?? {};

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}

      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <TextField select size="small" label="Scenario" value={scenario} onChange={(e) => setScenario(e.target.value)} sx={{ minWidth: 260 }}>
          {SCENARIOS.map((s) => (
            <MenuItem key={s.value} value={s.value}>{s.label}</MenuItem>
          ))}
        </TextField>
        <TextField size="small" label="Seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} sx={{ width: 90 }} />
        <Box sx={{ width: 200 }}>
          <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
            Reaction delay: {reaction.toFixed(1)}s
            <InfoDot title="Reaction delay" detail="Simulated driver/AV reaction latency injected into the generated trajectories — longer delays produce more severe conflicts." size={11} />
          </Typography>
          <Slider size="small" value={reaction} min={0} max={2} step={0.1} onChange={(_, v) => setReaction(v as number)} sx={{ py: 0.5 }} />
        </Box>
        <Button variant="contained" startIcon={<PlayCircle size={16} />} disabled={loading} onClick={run}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </Button>
        {analysis?.generated ? (
          <Chip size="small" label="simulated trajectories" sx={{ bgcolor: tokens.color.warnBg, color: tokens.color.warn, fontSize: 10.5 }} />
        ) : null}
      </Box>

      {loading && !analysis ? <ChartSkeleton height={200} /> : null}

      {agg ? (
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
          <MetricCard label="Conflicts" value={agg.num_conflicts} sub={Object.entries(agg.by_type).map(([t, n]) => `${t}: ${n}`).join(' · ')} info="Vehicle-pair interactions crossing the surrogate-safety thresholds (TTC/PET) during the observed window." />
          <MetricCard label="Aggregate CSI" value={agg.aggregate_csi.toFixed(2)} sub={`mean ${agg.mean_csi_per_conflict.toFixed(2)} per conflict`} accent={agg.aggregate_csi > 3 ? tokens.color.danger : tokens.color.warn} info={glossary.CSI ?? 'Conflict Severity Index'} />
          <MetricCard label="Min TTC" value={agg.min_ttc_s !== null ? `${agg.min_ttc_s.toFixed(2)}s` : '—'} accent={agg.min_ttc_s !== null && agg.min_ttc_s < 1 ? tokens.color.danger : undefined} info={glossary.TTC ?? 'Time-to-collision'} />
          <MetricCard label="Min PET" value={agg.min_pet_s !== null ? `${agg.min_pet_s.toFixed(2)}s` : '—'} info={glossary.PET ?? 'Post-encroachment time'} />
          <MetricCard label="Max DRAC" value={agg.max_drac_mps2 !== null ? `${agg.max_drac_mps2.toFixed(1)} m/s²` : '—'} info={glossary.DRAC ?? 'Deceleration rate to avoid crash'} />
        </Box>
      ) : null}

      {analysis && !analysis.conflicts.length ? (
        <SectionCard title="Conflicts">
          <IllustratedEmpty art="map" title="No conflicts detected" message="The generated trajectories never crossed the TTC/PET thresholds. Try a different scenario, seed, or a longer reaction delay." />
        </SectionCard>
      ) : null}

      {analysis && analysis.conflicts.length ? (
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <SectionCard
            title="Conflict list"
            sx={{ flex: '2 1 480px' }}
            help="Each row is one vehicle-pair conflict with its surrogate measures. Red chips mark values past critical thresholds (TTC < 1.5s is the conflict definition; < 1s is severe). Click a row to highlight it on the map."
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {analysis.conflicts.map((c, i) => (
                <Box
                  key={i}
                  onClick={() => setSelected(i)}
                  sx={{
                    p: 1.25,
                    border: `1px solid ${selected === i ? tokens.color.info : tokens.color.border}`,
                    borderLeft: `3px solid ${severityColor(c)}`,
                    borderRadius: 1,
                    cursor: 'pointer',
                    bgcolor: selected === i ? tokens.color.surfaceRaised : 'transparent',
                    transition: `border-color ${tokens.motion.fast}, background-color ${tokens.motion.fast}`,
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5, flexWrap: 'wrap' }}>
                    <Chip size="small" label={c.conflict_type} sx={{ height: 18, fontSize: 10, fontWeight: 700, bgcolor: TYPE_COLORS[c.conflict_type] ?? tokens.color.neutral, color: '#0e1114' }} />
                    <Typography variant="caption" sx={{ fontFamily: 'monospace', color: tokens.color.textDim }}>
                      {c.vehicle_a} × {c.vehicle_b}
                    </Typography>
                    <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                      t = {c.t_start_s.toFixed(1)}–{c.t_end_s.toFixed(1)}s · P(collision) {(c.collision_probability * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    <MeasureChip label="TTC" value={c.min_ttc_s} unit="s" term={glossary.TTC ?? ''} danger={c.min_ttc_s !== null && c.min_ttc_s < 1.0} />
                    <MeasureChip label="PET" value={c.pet_s} unit="s" term={glossary.PET ?? ''} danger={c.pet_s !== null && c.pet_s < 1.0} />
                    <MeasureChip label="DRAC" value={c.max_drac_mps2} unit="m/s²" term={glossary.DRAC ?? ''} danger={c.max_drac_mps2 !== null && c.max_drac_mps2 > 6} />
                    <MeasureChip label="ΔS" value={c.delta_s_mps} unit="m/s" term={glossary.DeltaS ?? ''} />
                    <MeasureChip label="CSI" value={c.csi} unit="" term={glossary.CSI ?? ''} danger={c.csi >= 2} />
                  </Box>
                </Box>
              ))}
            </Box>
          </SectionCard>

          <Box sx={{ flex: '1 1 340px', display: 'flex', flexDirection: 'column', gap: 2 }}>
            <SectionCard
              title="Conflict-point map"
              help="Where each conflict happens in the scene (meters). Marker color = conflict type, size = CSI severity. Click a marker to select the conflict."
            >
              <ConflictMap conflicts={analysis.conflicts} selected={selected} onSelect={setSelected} />
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                {Object.entries(TYPE_COLORS).map(([t, color]) => (
                  <Box key={t} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 9, height: 9, borderRadius: '50%', bgcolor: color }} />
                    <Typography variant="caption" sx={{ color: tokens.color.neutral, fontSize: 10.5 }}>{t}</Typography>
                  </Box>
                ))}
              </Box>
            </SectionCard>

            <SectionCard
              title="CSI comparison"
              help="Conflict Severity Index per conflict, sorted — CSI integrates relative speed, collision probability and exposure time, so it ranks conflicts more faithfully than TTC alone."
            >
              {sortedByCsi.map(({ c, i }) => (
                <HBar
                  key={i}
                  label={`${c.conflict_type} · ${c.vehicle_b.split(':').pop()}`}
                  value={c.csi}
                  max={Math.max(...analysis.conflicts.map((x) => x.csi), 0.001)}
                  color={selected === i ? tokens.color.info : severityColor(c)}
                  valueLabel={c.csi.toFixed(3)}
                />
              ))}
            </SectionCard>
          </Box>
        </Box>
      ) : null}
    </Box>
  );
}
