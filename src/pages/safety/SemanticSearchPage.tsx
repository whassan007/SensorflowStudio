/**
 * Semantic Search — neuro-symbolic concept mining over containers or the
 * scenario DB: concept input, per-stage pipeline explanation, provider
 * badge, and result cards with per-stage explanations.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Typography from '@mui/material/Typography';
import { BrainCircuit, Search } from 'lucide-react';
import { semanticSearch, type SemanticSearchResponse } from '../../services/safety';
import { RunSelect, usePublishedRuns } from '../../components/safety/shared';
import { IllustratedEmpty, PanelSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { InfoDot } from '../../components/help/InfoTip';
import { useLabelEval } from '../../context/LabelEvalContext';
import { tokens } from '../../theme';

const EXAMPLES = [
  'pedestrians at night in the rain',
  'occluded cyclists on the highway',
  'construction zones with anomalies',
  'high-risk intersections at dusk',
];

interface ContainerResult {
  container_id: number;
  weather?: string;
  lighting?: string;
  road_type?: string;
  scenario?: string;
  n_objects?: number;
  fn?: number;
  fp?: number;
  anomalies?: number;
  safety_n?: number;
  risk_score?: number;
  score: number;
  rule_score?: number;
  embedding_similarity?: number | null;
  explanations?: { stage1_symbolic?: string; stage2_reasoning?: Array<Record<string, unknown>> };
  [k: string]: unknown;
}

export default function SemanticSearchPage() {
  const { navigate } = useLabelEval();
  const { runs, error: runsError } = usePublishedRuns();
  const [runId, setRunId] = useState<string | null>(null);
  const [target, setTarget] = useState<'containers' | 'scenarios'>('containers');
  const [concept, setConcept] = useState(EXAMPLES[0]);
  const [llmMode, setLlmMode] = useState<'auto' | 'on' | 'off'>('auto');
  const [response, setResponse] = useState<SemanticSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId && runs?.length) setRunId(runs[0].run_id);
  }, [runs, runId]);

  const run = useCallback(() => {
    if (!concept.trim()) return;
    setLoading(true);
    setError(null);
    semanticSearch({
      concept: concept.trim(),
      target,
      run: target === 'containers' ? runId ?? undefined : undefined,
      k: 12,
      use_llm: llmMode === 'auto' ? null : llmMode === 'on',
    })
      .then(setResponse)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [concept, target, runId, llmMode]);

  const stage1 = response?.stage1 as { applied_filters?: Record<string, unknown>; candidates_before?: number; candidates_after?: number } | undefined;
  const stage2 = response?.stage2 as { provider?: string; note?: string } | undefined;
  const hybrid = response?.hybrid as { rule_weight?: number; embedding_weight?: number; embedding_source?: string } | undefined;
  const llm = response?.llm as { provider?: string; rationale?: string } | null | undefined;
  const results = (response?.results ?? []) as ContainerResult[];
  const maxScore = useMemo(() => Math.max(...results.map((r) => r.score), 0.001), [results]);
  const providerLabel = llm?.provider ?? stage2?.provider ?? 'offline_deterministic';
  const isLlm = providerLabel !== 'offline_deterministic';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {runsError ? <ErrorNote error={runsError} /> : null}
      {error ? <ErrorNote error={error} /> : null}

      <SectionCard
        title="Concept"
        help="Describe what you want to find in plain language. Stage 1 applies any structured filters symbolically; stage 2 scores candidates with a deterministic reasoning lexicon over their structured evidence, blended with embedding similarity. An LLM (local Ollama), when reachable, adds rationale text but never reorders results — rankings stay reproducible."
      >
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField
            size="small"
            value={concept}
            onChange={(e) => setConcept(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') run(); }}
            placeholder="e.g. pedestrians at night in the rain"
            sx={{ minWidth: 360, flex: '1 1 360px', maxWidth: 560 }}
          />
          <ToggleButtonGroup size="small" exclusive value={target} onChange={(_, v) => v && setTarget(v)}>
            <ToggleButton value="containers" sx={{ textTransform: 'none', py: 0.4 }}>Containers</ToggleButton>
            <ToggleButton value="scenarios" sx={{ textTransform: 'none', py: 0.4 }}>Scenario DB</ToggleButton>
          </ToggleButtonGroup>
          {target === 'containers' ? <RunSelect label="Run" value={runId} onChange={setRunId} runs={runs} /> : null}
          <TextField select size="small" label="LLM rationale" value={llmMode} onChange={(e) => setLlmMode(e.target.value as 'auto' | 'on' | 'off')} sx={{ width: 130 }}>
            <MenuItem value="auto">auto</MenuItem>
            <MenuItem value="on">force on</MenuItem>
            <MenuItem value="off">off</MenuItem>
          </TextField>
          <Button variant="contained" startIcon={<Search size={15} />} disabled={loading || !concept.trim()} onClick={run}>
            {loading ? 'Searching…' : 'Search'}
          </Button>
        </Box>
        <Box sx={{ display: 'flex', gap: 0.75, mt: 1, flexWrap: 'wrap' }}>
          {EXAMPLES.map((ex) => (
            <Chip key={ex} size="small" label={ex} onClick={() => setConcept(ex)} sx={{ height: 22, fontSize: 11, cursor: 'pointer', bgcolor: tokens.color.surfaceRaised }} />
          ))}
        </Box>
      </SectionCard>

      {loading ? <PanelSkeleton rows={5} /> : null}

      {response && !loading ? (
        <>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Chip size="small" label={`stage 1 · symbolic filter: ${stage1?.candidates_before ?? '—'} → ${stage1?.candidates_after ?? '—'} candidates`} sx={{ bgcolor: tokens.color.surfaceRaised, fontFamily: 'monospace', fontSize: 11 }} />
            <Chip
              size="small"
              icon={<BrainCircuit size={13} />}
              label={`stage 2 · ${isLlm ? `LLM rationale: ${providerLabel}` : 'deterministic reasoning scorer'}`}
              sx={{ bgcolor: isLlm ? tokens.color.infoBg : tokens.color.surfaceRaised, color: isLlm ? tokens.color.info : tokens.color.textDim, fontFamily: 'monospace', fontSize: 11 }}
            />
            {hybrid ? (
              <Chip size="small" label={`hybrid: ${hybrid.rule_weight} rules + ${hybrid.embedding_weight} embeddings`} sx={{ bgcolor: tokens.color.surfaceRaised, fontFamily: 'monospace', fontSize: 11 }} />
            ) : null}
            <InfoDot title="Provider badge" detail={stage2?.note ?? 'The reasoning stage is deterministic; an LLM only decorates results with rationale text and never changes the ranking.'} />
          </Box>

          {llm?.rationale ? (
            <SectionCard title="LLM rationale" help="Free-text rationale from the local LLM about why these results match the concept. Advisory only — the ranking above it is deterministic.">
              <Typography variant="body2" sx={{ color: tokens.color.textDim, whiteSpace: 'pre-wrap' }}>{llm.rationale}</Typography>
            </SectionCard>
          ) : null}

          {!results.length ? (
            <IllustratedEmpty art="search" title="No matches" message="Nothing scored above zero for this concept. Try different words (the lexicon knows weather, lighting, road types, classes, risk and anomaly vocabulary) or drop filters." />
          ) : (
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 1.5 }}>
              {results.map((r, i) => {
                const dims = [r.weather, r.lighting, r.road_type, r.scenario].filter(Boolean) as string[];
                const matched = (r.explanations?.stage2_reasoning ?? []) as Array<{ term?: string; matched?: string; reason?: string; weight?: number; [k: string]: unknown }>;
                return (
                  <Box key={`${r.container_id ?? r.scenario_id ?? i}`} sx={{ border: `1px solid ${tokens.color.border}`, borderRadius: 1, p: 1.5, bgcolor: tokens.color.surface, display: 'flex', flexDirection: 'column', gap: 0.75, transition: `border-color ${tokens.motion.fast}`, '&:hover': { borderColor: tokens.color.borderStrong } }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 800, fontFamily: 'monospace' }}>
                        {r.container_id !== undefined ? `container #${r.container_id}` : String(r.scenario_id ?? `result ${i + 1}`)}
                      </Typography>
                      <Box sx={{ flex: 1 }} />
                      <Typography variant="caption" sx={{ fontFamily: 'monospace', color: tokens.color.info, fontWeight: 700 }}>
                        score {r.score.toFixed(3)}
                      </Typography>
                    </Box>
                    <Box sx={{ height: 5, bgcolor: tokens.color.border, borderRadius: 1, overflow: 'hidden' }}>
                      <Box sx={{ height: '100%', width: `${(r.score / maxScore) * 100}%`, bgcolor: tokens.color.info }} />
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {dims.map((d) => (
                        <Chip key={d} size="small" label={d} sx={{ height: 18, fontSize: 10, bgcolor: tokens.color.surfaceRaised }} />
                      ))}
                      {typeof r.n_objects === 'number' ? <Chip size="small" label={`${r.n_objects} objects`} sx={{ height: 18, fontSize: 10, bgcolor: tokens.color.surfaceRaised }} /> : null}
                      {typeof r.fn === 'number' && r.fn > 0 ? <Chip size="small" label={`${r.fn} FN`} sx={{ height: 18, fontSize: 10, bgcolor: tokens.color.dangerBg, color: tokens.color.danger }} /> : null}
                      {typeof r.anomalies === 'number' && r.anomalies > 0 ? <Chip size="small" label={`${r.anomalies} anomalies`} sx={{ height: 18, fontSize: 10, bgcolor: tokens.color.warnBg, color: tokens.color.warn }} /> : null}
                    </Box>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                      <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                        <strong>stage 1:</strong> {r.explanations?.stage1_symbolic ?? '—'}
                      </Typography>
                      <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
                        <strong>stage 2:</strong>{' '}
                        {matched.length
                          ? matched.map((m) => String(m.reason ?? m.matched ?? m.term ?? '')).filter(Boolean).join('; ')
                          : 'embedding similarity only'}
                      </Typography>
                      <Typography variant="caption" sx={{ color: tokens.color.textFaint, fontFamily: 'monospace' }}>
                        rules {r.rule_score?.toFixed(3) ?? '—'} · embedding {r.embedding_similarity != null ? r.embedding_similarity.toFixed(3) : '—'}
                      </Typography>
                    </Box>
                    {r.container_id !== undefined ? (
                      <Button size="small" sx={{ alignSelf: 'flex-start', textTransform: 'none', fontSize: 11.5, p: 0.25 }} onClick={() => navigate('command', runId)}>
                        Investigate in Command Center →
                      </Button>
                    ) : null}
                  </Box>
                );
              })}
            </Box>
          )}
        </>
      ) : null}
    </Box>
  );
}
