/** Agent outputs, each clearly labeled as advisory AI analysis with its
 * epistemic status, confidence basis and escalation triggers. */
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import { ChevronDown } from 'lucide-react';
import type { AgentResult } from '../../types/agentic';
import { EvidenceStatusChip, KV, OriginBadge, PanelTitle } from './common';

const AGENT_LABELS: Record<string, string> = {
  failure_detection: 'Failure Detection Agent',
  vlm_scene_analysis: 'VLM Scene Analysis Agent',
  sensor_fusion_verification: 'Sensor Fusion Verification Agent',
  scenario_mining: 'Scenario Mining Agent',
  statistical_regression: 'Statistical Regression Agent',
  safety_impact: 'Safety Impact Agent',
  launch_decision: 'Launch Decision Agent (narrative only)',
  eval_flywheel: 'Eval Flywheel Agent',
};

function summarize(agent: AgentResult): string {
  const o = agent.output;
  switch (agent.agent) {
    case 'vlm_scene_analysis': {
      const hyps = (o.hypotheses as { hypothesis: string }[] | undefined) ?? [];
      return hyps.map((h) => h.hypothesis).slice(0, 2).join(' · ') || 'no hypotheses';
    }
    case 'sensor_fusion_verification':
      return `verdict: ${String(o.overall_verdict)}`;
    case 'scenario_mining':
      return `novelty: ${String(o.novelty)} · ${String((o.clusters as unknown[] | undefined)?.length ?? 0)} cluster(s)`;
    case 'safety_impact':
      return String(o.assessment ?? '');
    case 'launch_decision':
      return `restates policy outcome ${String(o.restated_outcome)} (cannot change it)`;
    case 'eval_flywheel':
      return o.proposal ? `proposes suite ${String((o.proposal as Record<string, unknown>).suite_name)}` : String(o.reason ?? '');
    default:
      return '';
  }
}

export default function AgentOutputsPanel({ agents }: { agents: AgentResult[] }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: '#161b22' }}>
      <PanelTitle title="Agent Analyses" origin="ai" />
      {agents.map((a) => (
        <Accordion key={a.agent} disableGutters sx={{ bgcolor: '#12171d', '&:before': { display: 'none' } }}>
          <AccordionSummary expandIcon={<ChevronDown size={16} color="#8a949e" />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1, minWidth: 0 }}>
              <Typography variant="caption" sx={{ fontWeight: 800, minWidth: 210 }}>
                {AGENT_LABELS[a.agent] ?? a.agent}
              </Typography>
              <OriginBadge origin="ai" />
              <EvidenceStatusChip status={a.epistemic_status} />
              {a.escalation.required ? (
                <Chip
                  size="small"
                  label="ESCALATED"
                  sx={{ height: 18, fontSize: 10, fontWeight: 800, bgcolor: '#e65100', color: '#fff' }}
                />
              ) : null}
              <Typography
                variant="caption"
                sx={{ color: '#8a949e', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              >
                {summarize(a)}
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ pt: 0 }}>
            <KV k="Authority" v={`${a.authority} — cannot authorize a launch`} />
            <KV k="Confidence" v={`${a.confidence} (${a.confidence_basis})`} />
            {a.escalation.required ? (
              <KV
                k="Human-review triggers"
                v={a.escalation.human_review_triggers.join(', ') + ' — ' + a.escalation.reasons.join('; ')}
              />
            ) : null}
            {a.llm_rationale ? <KV k="LLM rationale" v={a.llm_rationale} /> : null}
            <Box
              component="pre"
              sx={{
                m: 0,
                mt: 0.5,
                p: 1,
                bgcolor: '#0d1117',
                borderRadius: 1,
                fontSize: 10.5,
                maxHeight: 240,
                overflow: 'auto',
              }}
            >
              {JSON.stringify(a.output, null, 2)}
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
    </Paper>
  );
}
