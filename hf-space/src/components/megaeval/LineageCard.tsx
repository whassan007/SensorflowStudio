/**
 * Collapsible reproducibility card: every version pin that produced this
 * evaluation run (dataset, model, labels, evaluator code, metric definitions,
 * thresholds, sampling config, seed, hardware).
 */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { RunLineage } from '../../types/megaeval';
import { SectionCard } from '../labeleval/shared';
import { InfoDot } from '../help/InfoTip';

function Field({ label, value, info }: { label: string; value: string | number; info?: string }) {
  return (
    <Box sx={{ minWidth: 220, flex: '1 1 220px' }}>
      <Typography variant="caption" sx={{ color: '#8a949e', textTransform: 'uppercase', fontSize: 9.5 }}>
        {label}
        {info ? <InfoDot title={label} detail={info} size={10} /> : null}
      </Typography>
      <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
        {value}
      </Typography>
    </Box>
  );
}

export default function LineageCard({ lineage }: { lineage: RunLineage }) {
  const [open, setOpen] = useState(true);
  return (
    <SectionCard
      title="Run lineage & reproducibility"
      helpTerm="lineage"
      action={
        <IconButton size="small" onClick={() => setOpen((o) => !o)}>
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </IconButton>
      }
    >
      <Collapse in={open}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
          <Field label="Evaluation ID" value={lineage.evaluation_id} info="Unique identifier of this evaluation run; referenced by all query APIs." />
          <Field label="Dataset version" value={lineage.dataset_version} info="The immutable population this run scored — same ID means byte-identical inputs." />
          <Field label="Model version" value={lineage.model_version} info="Human-readable version of the model under evaluation." />
          <Field label="Model checkpoint" value={lineage.model_checkpoint} info="Content hash of the exact weights file — two runs on the same checkpoint are directly comparable." />
          <Field label="Label version" value={lineage.label_version} info="Version of the label/reference set used for matching." />
          <Field label="Evaluator code version" value={lineage.evaluator_code_version} info="Version of the evaluation code; metric bugs are pinned to this." />
          <Field label="Metric version" value={lineage.metric_version} info="Version of the metric definitions (matching rules, thresholds semantics)." />
          <Field label="Seed" value={lineage.seed} info="RNG seed for every stochastic step (sampling, tie-breaking). Same seed + inputs = identical outputs." />
          <Field label="Hardware" value={lineage.hardware} info="Execution platform; recorded because numerics can differ across architectures." />
          <Field label="Timestamp" value={new Date(lineage.timestamp).toLocaleString()} />
        </Box>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Box sx={{ flex: '1 1 320px' }}>
            <Typography variant="caption" sx={{ color: '#8a949e', textTransform: 'uppercase', fontSize: 9.5 }}>
              Threshold config
            </Typography>
            <Box
              component="pre"
              sx={{
                m: 0,
                p: 1,
                bgcolor: '#12171d',
                border: '1px solid #232a31',
                borderRadius: 1,
                fontSize: 11.5,
                fontFamily: 'monospace',
                overflowX: 'auto',
              }}
            >
              {JSON.stringify(lineage.threshold_config, null, 2)}
            </Box>
          </Box>
          <Box sx={{ flex: '1 1 320px' }}>
            <Typography variant="caption" sx={{ color: '#8a949e', textTransform: 'uppercase', fontSize: 9.5 }}>
              Sampling config
            </Typography>
            <Box
              component="pre"
              sx={{
                m: 0,
                p: 1,
                bgcolor: '#12171d',
                border: '1px solid #232a31',
                borderRadius: 1,
                fontSize: 11.5,
                fontFamily: 'monospace',
                overflowX: 'auto',
              }}
            >
              {JSON.stringify(lineage.sampling_config, null, 2)}
            </Box>
          </Box>
        </Box>
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1.5 }}>
          Identical inputs + seed reproduce this evaluation bit-for-bit.
        </Typography>
      </Collapse>
    </SectionCard>
  );
}
