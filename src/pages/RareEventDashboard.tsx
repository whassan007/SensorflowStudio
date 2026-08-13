import { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import type {
  CopilotExplainRequest,
  EvaluationRecord,
  HaystackPoint,
  RareEvent,
} from '../types/labeleval';
import {
  getHaystack,
  getRareEvents,
  getBenchmark,
  getEvaluation,
  usePoll,
} from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import RareEventConfig from '../components/labeleval/RareEventConfig';
import HaystackVisualizer from '../components/labeleval/HaystackVisualizer';
import BenchmarkResults from '../components/labeleval/BenchmarkResults';
import EvidencePanel from '../components/labeleval/EvidencePanel';
import CopilotDrawer from '../components/labeleval/CopilotDrawer';
import { SectionCard, StatusChip, ErrorNote, fmtNum, fmtPct } from '../components/labeleval/shared';
import { HeadCell } from '../components/help/InfoTip';

export default function RareEventDashboard() {
  const { activeDatasetId } = useLabelEval();
  const haystack = usePoll(() => getHaystack(activeDatasetId), 5000, [activeDatasetId]);
  const rareEvents = usePoll(getRareEvents, 5000);
  const benchmark = usePoll(getBenchmark, null);

  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidenceRecord, setEvidenceRecord] = useState<EvaluationRecord | null>(null);
  const [evidenceEvent, setEvidenceEvent] = useState<RareEvent | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotRequest, setCopilotRequest] = useState<CopilotExplainRequest | null>(null);

  const openEventEvidence = (event: RareEvent) => {
    setEvidenceRecord(null);
    setEvidenceEvent(event);
    setEvidenceError(null);
    setEvidenceOpen(true);
  };

  const openPointEvidence = async (point: HaystackPoint) => {
    setEvidenceError(null);
    if (point.kind === 'rare_event') {
      const event = rareEvents.data?.events.find((e) => e.event_id === point.id) ?? null;
      if (event) {
        openEventEvidence(event);
        return;
      }
    }
    setEvidenceEvent(null);
    setEvidenceRecord(null);
    setEvidenceOpen(true);
    try {
      setEvidenceRecord(await getEvaluation(point.id));
    } catch (err) {
      setEvidenceError(err instanceof Error ? err.message : String(err));
    }
  };

  const askCopilot = (request: CopilotExplainRequest) => {
    setCopilotRequest(request);
    setCopilotOpen(true);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <RareEventConfig activeDatasetId={activeDatasetId} />

      {haystack.error && !haystack.data ? <ErrorNote error={haystack.error} /> : null}
      <HaystackVisualizer points={haystack.data?.points ?? []} onPointClick={(p) => void openPointEvidence(p)} />

      <SectionCard
        title={`Rare Events (${rareEvents.data?.events.length ?? 0})`}
        help="Confirmed rare-event candidates mined by the anomaly ensemble, ranked by rarity and severity. Click a row to open its full evidence (frames, scores, detector votes). These samples carry the highest training value per label."
      >
        {rareEvents.error && !rareEvents.data ? <ErrorNote error={rareEvents.error} /> : null}
        {!rareEvents.data || rareEvents.data.events.length === 0 ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No rare events detected yet — run the evaluation pipeline (Datasets → Run Evaluation) and the anomaly
            ensemble will surface candidates here.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Event</TableCell>
                <TableCell>Scenario</TableCell>
                <TableCell>
                  <HeadCell
                    label="Severity"
                    title="Severity"
                    detail="Impact class of the event (critical / high / medium / low) combining safety relevance and rarity. Hover the chip for the level's meaning."
                  />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Rarity" term="rarity_score" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Anomaly score" term="anomaly_score" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Confidence" term="confidence" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell
                    label="Evidence frames"
                    title="Evidence frames"
                    detail="Number of frames captured as supporting evidence for this event."
                  />
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rareEvents.data.events.map((e) => (
                <TableRow key={e.event_id} hover sx={{ cursor: 'pointer' }} onClick={() => openEventEvidence(e)}>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{e.event_id}</TableCell>
                  <TableCell>{e.scenario_type.replace(/_/g, ' ')}</TableCell>
                  <TableCell>
                    <StatusChip status={e.severity} />
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtNum(e.rarity_score)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtNum(e.anomaly_score)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {fmtPct(e.confidence)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {e.evidence_frames.length}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </SectionCard>

      <BenchmarkResults
        benchmark={benchmark.data}
        activeDatasetId={activeDatasetId}
        onRefresh={benchmark.refresh}
      />

      {evidenceError && evidenceOpen ? <ErrorNote error={evidenceError} /> : null}
      <EvidencePanel
        open={evidenceOpen}
        onClose={() => setEvidenceOpen(false)}
        record={evidenceRecord}
        event={evidenceEvent}
        onAskCopilot={askCopilot}
      />
      <CopilotDrawer open={copilotOpen} onClose={() => setCopilotOpen(false)} request={copilotRequest} />
    </Box>
  );
}
