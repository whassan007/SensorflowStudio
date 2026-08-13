/** Table-style stage views: comparison validity, offline audit, population,
 * serving parity, shadow traffic, label integrity. Each receives the
 * diagnostics payload for its stage. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import {
  DeltaBar,
  Explainer,
  KV,
  PsiBadge,
  SectionCard,
  ShareBar,
  StatusPill,
  tableSx,
} from './common';

const num = (v: unknown, digits = 2): string =>
  typeof v === 'number' ? v.toFixed(digits) : String(v ?? '—');

// ------------------------------------------------ stage 0: comparison validity

export function ComparisonValidityView({ data }: { data: any }) {
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard
        title="Comparison-validity matrix"
        subtitle="Offline harness vs shadow serving, dimension by dimension. Mismatch rows are candidate explanations that must be ruled out before believing either headline number."
      >
        <Box component="table" sx={tableSx}>
          <thead>
            <tr>
              <th>Dimension</th>
              <th>Offline</th>
              <th>Shadow</th>
              <th>Verdict</th>
              <th>If mismatched</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r: any) => (
              <tr key={r.field}>
                <td style={{ fontWeight: 600 }}>{r.label}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11.5 }}>{String(r.offline ?? 'unrecorded')}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11.5 }}>{String(r.shadow ?? 'unrecorded')}</td>
                <td><StatusPill status={r.status} /></td>
                <td>
                  <Typography variant="caption" sx={{ color: r.criticality === 'CRITICAL' ? '#ef5350' : '#8a949e' }}>
                    {r.criticality}
                  </Typography>
                </td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}

// ---------------------------------------------------- stage 1: offline audit

export function OfflineAuditView({ data }: { data: any }) {
  const r = data.reproducibility;
  const l = data.leakage;
  return (
    <>
      <Explainer text={data.explainer} />
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
        <SectionCard title="Reproducibility fingerprint">
          <KV k="Claimed offline delta" v={`${num(r.original_delta_pp)}pp`} />
          <KV k="Re-run delta" v={`${num(r.rerun_delta_pp)}pp`} />
          <KV k="|difference|" v={`${num(r.abs_diff_pp)}pp`} />
          <KV k="Reproduces (<0.5pp)" v={<StatusPill status={r.reproduces ? 'PASS' : 'MISMATCH'} />} />
          <KV k="Version pins present" v={<StatusPill status={r.pins_present ? 'PASS' : 'UNKNOWN'} />} />
          <KV k="Environment lock" v={r.environment_lock} />
        </SectionCard>
        <SectionCard title="Leakage scan (train ∩ eval)">
          <KV k="Near-duplicate eval units" v={`${l.n_duplicates} (${(l.dup_fraction * 100).toFixed(1)}% of ${l.offline_n})`} />
          <KV k="Entities also in B's train set" v={l.overlap_entity_count} />
          <KV k="Delta on leaked units" v={<DeltaBar value={l.dup_delta_pp} max={25} />} />
          <KV k="Delta on clean units" v={<DeltaBar value={l.clean_delta_pp} max={25} />} />
          {l.overlap_examples?.length ? (
            <KV k="Examples" v={l.overlap_examples.join(', ')} />
          ) : null}
        </SectionCard>
        <SectionCard title="Split-boundary analysis">
          <KV k="Split strategy" v={<StatusPill status={data.split.strategy === 'entity_level' ? 'PASS' : 'MISMATCH'} />} />
          <KV k="Strategy" v={data.split.strategy} />
          <KV k="Eval entities" v={data.split.n_entities} />
          <KV k="Mean units / entity" v={num(data.split.mean_units_per_entity, 1)} />
        </SectionCard>
        <SectionCard title="Temporal leakage">
          <KV k="Eval window after B train end" v={<StatusPill status={data.temporal.ok ? 'PASS' : 'MISMATCH'} />} />
          <KV k="B train window" v={data.temporal.b_train_window} />
        </SectionCard>
      </Box>
    </>
  );
}

// ---------------------------------------------- stage 2: population validation

export function PopulationView({ data }: { data: any }) {
  const v = data.volumes;
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard title="Volumes">
        <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <KV k="Offline eval units" v={v.offline_n.toLocaleString()} />
          <KV k="Shadow eligible" v={v.shadow_eligible_n.toLocaleString()} />
          <KV k="Shadow scored" v={v.shadow_scored_n.toLocaleString()} />
          <KV k="Sampling rate" v={`${(v.sampling_rate * 100).toFixed(1)}%`} />
          <KV k="Entities (off / sh)" v={`${v.offline_entities} / ${v.shadow_entities}`} />
          <KV k="Entity overlap" v={v.entity_overlap} />
        </Box>
      </SectionCard>
      {Object.entries(data.dimensions as Record<string, any[]>).map(([dim, rows]) => (
        <SectionCard key={dim} title={`Composition: ${dim}`}>
          <Box component="table" sx={tableSx}>
            <thead>
              <tr><th>Value</th><th>Offline share</th><th>Shadow share</th></tr>
            </thead>
            <tbody>
              {rows.map((r: any) => (
                <tr key={r.value}>
                  <td style={{ fontWeight: 600 }}>{r.value}</td>
                  <td><ShareBar value={r.offline_share} color="#78909c" /></td>
                  <td><ShareBar value={r.shadow_share} /></td>
                </tr>
              ))}
            </tbody>
          </Box>
        </SectionCard>
      ))}
    </>
  );
}

// ----------------------------------------------------- stage 8: serving parity

export function ServingParityView({ data }: { data: any }) {
  return (
    <>
      <Explainer text={data.explainer} />
      <SectionCard
        title="Serving artifact / config diff"
        subtitle="The exact stack difference between what was evaluated offline and what shadow actually runs."
      >
        <Box component="table" sx={tableSx}>
          <thead>
            <tr><th>Artifact / config</th><th>Offline harness</th><th>Shadow serving</th><th>Verdict</th><th>Criticality</th></tr>
          </thead>
          <tbody>
            {data.rows.map((r: any) => (
              <tr key={r.field}>
                <td style={{ fontWeight: 600 }}>{r.label}</td>
                <td style={{ fontFamily: 'monospace' }}>{String(r.offline ?? 'unrecorded')}</td>
                <td style={{ fontFamily: 'monospace' }}>{String(r.shadow ?? 'unrecorded')}</td>
                <td><StatusPill status={r.status} /></td>
                <td>
                  <Typography variant="caption" sx={{ color: r.criticality === 'CRITICAL' ? '#ef5350' : '#8a949e' }}>
                    {r.criticality}
                  </Typography>
                </td>
              </tr>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}

// ----------------------------------------------------- stage 9: shadow traffic

export function TrafficAuditView({ data }: { data: any }) {
  const t = data.traffic;
  const s = data.selection;
  return (
    <>
      <Explainer text={data.explainer} />
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
        <SectionCard title="Traffic accounting">
          <KV k="Eligible units" v={t.eligible_count.toLocaleString()} />
          <KV k="Sampled / scored" v={`${t.sampled_count.toLocaleString()} (${(t.sampling_rate * 100).toFixed(1)}%)`} />
          <KV k="Dropped" v={t.dropped_count.toLocaleString()} />
          <KV k="Timeouts" v={t.timeout_count.toLocaleString()} />
          <KV k="Fallback engaged" v={t.fallback_count.toLocaleString()} />
          <KV k="Sampler" v={t.sampler} />
          <Box sx={{ mt: 0.5 }}>
            {t.eligibility_filters.map((f: string) => (
              <Chip key={f} size="small" label={f} sx={{ mr: 0.5, mb: 0.5, height: 20, fontSize: 10.5, bgcolor: '#232a31' }} />
            ))}
          </Box>
        </SectionCard>
        <SectionCard
          title="Selection-bias indicators"
          subtitle="Does the scored sample statistically match the eligible stream it claims to represent?"
        >
          <KV k="Difficulty shift (sampled vs eligible)" v={<PsiBadge psi={s.difficulty_psi} />} />
          <KV k="Confidence shift" v={<PsiBadge psi={s.conf_psi} />} />
          <KV k="Segment mix shift" v={<PsiBadge psi={s.segment_psi} />} />
          <KV k="Δ on scored sample" v={<DeltaBar value={s.sampled_delta_pp} max={8} />} />
          <KV k="Δ on unsampled remainder" v={<DeltaBar value={s.unsampled_delta_pp} max={8} />} />
          <KV k="Δ on full eligible stream" v={<DeltaBar value={s.eligible_delta_pp} max={8} />} />
          <KV k="Dropped+fallback+timeout rate" v={`${(s.drop_rate * 100).toFixed(1)}%`} />
        </SectionCard>
      </Box>
    </>
  );
}

// --------------------------------------------------- stage 10: label integrity

export function LabelIntegrityView({ data }: { data: any }) {
  const maxCount = Math.max(...data.age_histogram.map((h: any) => h.count), 1);
  return (
    <>
      <Explainer text={data.explainer} />
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
        <SectionCard title="Label age distribution" subtitle={`Maturity threshold: ${data.maturity_hours}h (audited labels).`}>
          {data.age_histogram.map((h: any) => (
            <Box key={h.bucket} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 0.2 }}>
              <Typography variant="caption" sx={{ minWidth: 70, color: '#8a949e', fontFamily: 'monospace' }}>
                {h.bucket}
              </Typography>
              <Box sx={{ flex: 1, height: 10, bgcolor: '#0d1116', borderRadius: 1 }}>
                <Box sx={{ width: `${(h.count / maxCount) * 100}%`, height: '100%', bgcolor: '#4fc3f7', borderRadius: 1 }} />
              </Box>
              <Typography variant="caption" sx={{ fontFamily: 'monospace', minWidth: 44, textAlign: 'right' }}>
                {h.count}
              </Typography>
            </Box>
          ))}
          <Box sx={{ mt: 1 }}>
            <KV k="Provisional fraction" v={`${(data.provisional_fraction * 100).toFixed(1)}%`} />
            <KV k="Labeling policy (off / sh)" v={`${data.policy.offline} / ${data.policy.shadow}`} />
            <KV k="Policy differs" v={<StatusPill status={data.policy.differs ? 'MISMATCH' : 'PASS'} />} />
          </Box>
        </SectionCard>
        <SectionCard
          title="Verdict by label maturity"
          subtitle="If the conclusion flips between mature and provisional labels, the -2% is a property of the labels, not the model."
        >
          <KV k={`Mature labels (n=${data.mature_n})`} v={<DeltaBar value={data.mature_delta_pp} max={10} />} />
          <KV k={`Provisional labels (n=${data.provisional_n})`} v={<DeltaBar value={data.provisional_delta_pp} max={10} />} />
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1.5, mb: 0.5, fontWeight: 700 }}>
            BY DIFFICULTY QUARTILE
          </Typography>
          <Box component="table" sx={tableSx}>
            <thead>
              <tr><th>Quartile</th><th>Provisional %</th><th>Δ (all)</th><th>Δ (provisional)</th></tr>
            </thead>
            <tbody>
              {data.by_difficulty_quartile.map((q: any) => (
                <tr key={q.quartile}>
                  <td style={{ fontWeight: 600 }}>{q.quartile}</td>
                  <td>{(q.provisional_frac * 100).toFixed(0)}%</td>
                  <td><DeltaBar value={q.delta_pp} max={15} width={90} /></td>
                  <td>
                    {q.provisional_delta_pp == null ? (
                      <Typography variant="caption" sx={{ color: '#5c666f' }}>n too small</Typography>
                    ) : (
                      <DeltaBar value={q.provisional_delta_pp} max={15} width={90} />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </Box>
        </SectionCard>
      </Box>
    </>
  );
}
