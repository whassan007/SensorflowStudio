import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { BrainCircuit, TrendingDown, Users } from 'lucide-react';
import type { PipelineStateResponse, ServiceStatus } from '../../types/labeleval';
import { SectionCard, StatusChip, fmtInt } from './shared';

function findService(pipeline: PipelineStateResponse | null, ...needles: string[]): ServiceStatus | null {
  if (!pipeline) return null;
  const lower = needles.map((n) => n.toLowerCase());
  return (
    pipeline.services.find((s) => {
      const name = s.service.toLowerCase();
      return lower.some((n) => name.includes(n));
    }) ?? null
  );
}

function EngineCard({
  icon,
  title,
  service,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  service: ServiceStatus | null;
  description: string;
}) {
  return (
    <Card variant="outlined" sx={{ flex: '1 1 260px', minWidth: 260, bgcolor: '#12171d' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Box sx={{ color: '#4fc3f7' }}>{icon}</Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, flex: 1 }}>
            {title}
          </Typography>
          <StatusChip status={service?.state ?? 'IDLE'} />
        </Box>
        <Typography variant="body2" sx={{ color: '#aab4be', mb: 1 }}>
          {description}
        </Typography>
        {service ? (
          <Typography variant="caption" sx={{ color: '#8a949e', display: 'block' }}>
            {fmtInt(service.processed)} / {fmtInt(service.total)} processed · {fmtInt(service.process_units)} process
            units{service.detail ? ` · ${service.detail}` : ''}
          </Typography>
        ) : (
          <Typography variant="caption" sx={{ color: '#8a949e' }}>
            No live status yet.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

export default function LabelEvaluationEngine({ pipeline }: { pipeline: PipelineStateResponse | null }) {
  return (
    <SectionCard title="Label Evaluation Engine — three independent evidence sources">
      <Typography variant="body2" sx={{ color: '#8a949e', mb: 1.5 }}>
        Each engine produces <strong>independent evidence</strong> that feeds the Quality Gate. There is never a single
        blended score: a label is only verified when every applicable gate passes on its own evidence.
      </Typography>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        <EngineCard
          icon={<BrainCircuit size={20} />}
          title="ML Anomaly Detection"
          service={findService(pipeline, 'anomaly')}
          description="Ensemble of classical + deep detectors scoring every annotation for statistical abnormality relative to the fleet distribution."
        />
        <EngineCard
          icon={<TrendingDown size={20} />}
          title="Regression Tracking"
          service={findService(pipeline, 'regression')}
          description="Compares the current model's per-class and per-scenario metrics against the baseline; any tolerated-delta breach is evidence."
        />
        <EngineCard
          icon={<Users size={20} />}
          title="Grader Disagreement"
          service={findService(pipeline, 'grader', 'consensus', 'disagreement')}
          description="Multiple automated graders vote on class, spatial and temporal correctness; low consensus is evidence of an unreliable label."
        />
      </Box>
    </SectionCard>
  );
}
