/**
 * The traceable-chain view: a vertical evidence flow from raw failure to
 * human decision, each stage expandable into tier-color-coded items.
 */
import { useState } from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import { ChevronDown } from 'lucide-react';
import type { EvidenceItem, RetrospectiveScorecard } from '../../types/retro';
import { DETERMINATION_COLOR, LAUNCH_COLORS, SEVERITY_COLORS, TIER_META } from './tierTheme';

interface StageDef {
  id: string;
  title: string;
  color: string;
  badge: string;
  items: { primary: string; secondary: string; color: string }[];
}

function evidenceRow(e: EvidenceItem) {
  const meta = TIER_META[e.tier];
  return {
    primary: `${e.key}${e.value !== null ? ` = ${e.value}` : ''}`,
    secondary: `${e.statement} — ${e.provenance}`,
    color: meta.color,
  };
}

function buildStages(sc: RetrospectiveScorecard): StageDef[] {
  const byTier = (tier: EvidenceItem['tier']) =>
    sc.evidence.filter((e) => e.tier === tier && !e.key.startsWith('policy_') && e.key !== 'severity_divergence');
  const observed = byTier('TIER1_OBSERVED');
  const derived = byTier('TIER2_DERIVED').filter((e) => e.key !== 'behavioral_impact');
  const behavioral = sc.evidence.filter((e) => e.key === 'behavioral_impact');
  const retrieved = byTier('TIER3_RETRIEVED');
  const policyItems = sc.evidence.filter(
    (e) => e.key === 'policy_severity' || e.key === 'severity_divergence'
  );

  return [
    {
      id: 'raw',
      title: '1 · Raw failure artifact',
      color: '#8a949e',
      badge: sc.failure_type,
      items: [
        {
          primary: sc.evaluation_id,
          secondary: `${String(sc.scenario?.description ?? 'evaluation log')} (weather: ${String(
            sc.scenario?.weather ?? '?'
          )}, ODD: ${String(sc.scenario?.odd_status ?? '?')})`,
          color: '#8a949e',
        },
      ],
    },
    {
      id: 'observed',
      title: `2 · Observed evidence — FACT (${observed.length})`,
      color: TIER_META.TIER1_OBSERVED.color,
      badge: 'TIER 1',
      items: observed.map(evidenceRow),
    },
    {
      id: 'derived',
      title: `3 · Derived metrics — DERIVED FACT (${derived.length})`,
      color: TIER_META.TIER2_DERIVED.color,
      badge: 'TIER 2',
      items: derived.map(evidenceRow),
    },
    {
      id: 'retrieved',
      title: `4 · Retrieved engineering evidence (${retrieved.length})`,
      color: TIER_META.TIER3_RETRIEVED.color,
      badge: 'TIER 3',
      items: retrieved.map(evidenceRow),
    },
    {
      id: 'hypothesis',
      title: `5 · Agent hypotheses — INFERENCE (${sc.root_cause_hypotheses.length})`,
      color: TIER_META.TIER4_AI_HYPOTHESIS.color,
      badge: 'TIER 4',
      items: sc.root_cause_hypotheses.map((h) => ({
        primary: `${Math.round(h.confidence * 100)}% confidence`,
        secondary: `${h.hypothesis} [supported by: ${h.supporting_evidence_keys.join(', ') || 'n/a'}; missing: ${
          h.missing_evidence.join(', ') || 'none'
        }]`,
        color: TIER_META.TIER4_AI_HYPOTHESIS.color,
      })),
    },
    {
      id: 'behavioral',
      title: '6 · Behavioral analysis',
      color: TIER_META.TIER2_DERIVED.color,
      badge: 'DERIVED',
      items: behavioral.length
        ? behavioral.map(evidenceRow)
        : [{ primary: 'behavioral consequence', secondary: sc.behavioral_consequence, color: TIER_META.TIER2_DERIVED.color }],
    },
    {
      id: 'policy',
      title: '7 · Safety policy (deterministic)',
      color: DETERMINATION_COLOR,
      badge: sc.policy_version,
      items: [
        ...policyItems.map(evidenceRow),
        ...sc.launch_rationale.map((r) => ({
          primary: 'launch gate',
          secondary: r,
          color: DETERMINATION_COLOR,
        })),
      ],
    },
    {
      id: 'scorecard',
      title: '8 · Retrospective scorecard → human decision',
      color: LAUNCH_COLORS[sc.launch_recommendation],
      badge: sc.launch_recommendation,
      items: [
        {
          primary: `severity ${sc.severity} · launch ${sc.launch_recommendation}`,
          secondary: sc.human_review_required
            ? `HUMAN REVIEW REQUIRED: ${sc.human_review_reasons.join('; ')}`
            : 'no human review trigger fired — decision still rests with a human',
          color: SEVERITY_COLORS[sc.severity],
        },
      ],
    },
  ];
}

export default function EvidenceChain({ scorecard }: { scorecard: RetrospectiveScorecard }) {
  const [expanded, setExpanded] = useState<string[]>(['observed', 'derived', 'policy', 'scorecard']);
  const stages = buildStages(scorecard);

  const toggle = (id: string) =>
    setExpanded((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 1, mb: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
        <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
          EVIDENCE TIER LEGEND:
        </Typography>
        {Object.values(TIER_META).map((m) => (
          <Chip key={m.label} size="small" label={m.label}
                sx={{ bgcolor: `${m.color}22`, color: m.color, fontWeight: 700, fontSize: 10 }} />
        ))}
        <Chip size="small" label="DETERMINATION"
              sx={{ bgcolor: `${DETERMINATION_COLOR}22`, color: DETERMINATION_COLOR, fontWeight: 700, fontSize: 10 }} />
      </Box>

      {stages.map((stage, i) => (
        <Box key={stage.id} sx={{ display: 'flex', gap: 1.5 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 18, pt: 1 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: stage.color, flexShrink: 0 }} />
            {i < stages.length - 1 ? (
              <Box sx={{ width: 2, flex: 1, bgcolor: '#232a31', minHeight: 18 }} />
            ) : null}
          </Box>
          <Accordion
            disableGutters
            expanded={expanded.includes(stage.id)}
            onChange={() => toggle(stage.id)}
            sx={{ flex: 1, mb: 1, bgcolor: '#161c23', border: '1px solid #232a31', '&:before': { display: 'none' } }}
          >
            <AccordionSummary expandIcon={<ChevronDown size={16} />} sx={{ minHeight: 40 }}>
              <Typography sx={{ fontSize: 13.5, fontWeight: 700, flex: 1 }}>{stage.title}</Typography>
              <Chip size="small" label={stage.badge}
                    sx={{ bgcolor: `${stage.color}22`, color: stage.color, fontWeight: 700, fontSize: 10, mr: 1 }} />
            </AccordionSummary>
            <AccordionDetails sx={{ pt: 0 }}>
              {stage.items.length === 0 ? (
                <Typography sx={{ fontSize: 12, color: '#8a949e' }}>no items</Typography>
              ) : (
                stage.items.map((item, j) => (
                  <Box key={j} sx={{ borderLeft: `3px solid ${item.color}`, pl: 1.25, py: 0.5, mb: 0.5 }}>
                    <Typography sx={{ fontSize: 12.5, fontFamily: 'monospace', color: item.color }}>
                      {item.primary}
                    </Typography>
                    <Typography sx={{ fontSize: 12, color: '#aab4be' }}>{item.secondary}</Typography>
                  </Box>
                ))
              )}
            </AccordionDetails>
          </Accordion>
        </Box>
      ))}
    </Box>
  );
}
