import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TablePagination from '@mui/material/TablePagination';
import type { CopilotExplainRequest, EvaluationRecord } from '../types/labeleval';
import {
  getDatasets,
  getQualityGroups,
  getQualityGroupDetail,
  getEvaluation,
  usePoll,
} from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import EvidencePanel from '../components/labeleval/EvidencePanel';
import CopilotDrawer from '../components/labeleval/CopilotDrawer';
import { SectionCard, StatusChip, LoadingBox, ErrorNote, fmtNum, fmtPct } from '../components/labeleval/shared';
import { ExplainTip, HeadCell } from '../components/help/InfoTip';
import { glossaryKeyForStatus } from '../content/glossary';

const PAGE_SIZE = 15;

export default function EvaluationPage() {
  const { activeDatasetId, setActiveDatasetId } = useLabelEval();
  const datasets = usePoll(getDatasets, 10000);

  const [annotationIds, setAnnotationIds] = useState<string[]>([]);
  const [idsLoading, setIdsLoading] = useState(false);
  const [idsError, setIdsError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [records, setRecords] = useState<EvaluationRecord[]>([]);
  const [recordsLoading, setRecordsLoading] = useState(false);

  const [selected, setSelected] = useState<EvaluationRecord | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotRequest, setCopilotRequest] = useState<CopilotExplainRequest | null>(null);

  // Collect annotation ids for the chosen dataset via its quality groups.
  useEffect(() => {
    let cancelled = false;
    setAnnotationIds([]);
    setPage(0);
    setIdsError(null);
    if (!activeDatasetId) return;
    setIdsLoading(true);
    (async () => {
      try {
        const groups = await getQualityGroups(activeDatasetId);
        const details = await Promise.all(
          groups.groups.filter((g) => g.count > 0).map((g) => getQualityGroupDetail(g.group_id))
        );
        const ids = details.flatMap((d) => d.annotation_ids);
        if (!cancelled) setAnnotationIds(ids);
      } catch (err) {
        if (!cancelled) setIdsError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setIdsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeDatasetId]);

  // Load the current page of evaluation records.
  useEffect(() => {
    let cancelled = false;
    const pageIds = annotationIds.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    if (pageIds.length === 0) {
      setRecords([]);
      return;
    }
    setRecordsLoading(true);
    Promise.all(pageIds.map((id) => getEvaluation(id).catch(() => null)))
      .then((rows) => {
        if (!cancelled) setRecords(rows.filter((r): r is EvaluationRecord => r !== null));
      })
      .finally(() => {
        if (!cancelled) setRecordsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [annotationIds, page]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title="Evaluation Record Explorer"
        help="One row per annotation: the evidence every engine recorded (geometry, anomaly, consensus) and the triage verdict. Click a row to open the full evidence panel with gate lines and per-engine details; Copilot can narrate the WHY. For population-scale analysis use the Command Center instead — this explorer is for individual records."
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            Dataset:
          </Typography>
          <Select
            size="small"
            value={activeDatasetId ?? ''}
            displayEmpty
            onChange={(e) => setActiveDatasetId(e.target.value || null)}
            sx={{ minWidth: 260 }}
          >
            <MenuItem value="">
              <em>Select a dataset</em>
            </MenuItem>
            {(datasets.data?.datasets ?? []).map((d) => (
              <MenuItem key={d.dataset_id} value={d.dataset_id}>
                {d.name} ({d.version})
              </MenuItem>
            ))}
          </Select>
        </Box>

        {idsLoading ? <LoadingBox label="Collecting annotation ids…" /> : null}
        {idsError ? <ErrorNote error={idsError} /> : null}

        {!activeDatasetId ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            Pick a dataset to explore its annotation-level evaluation records.
          </Typography>
        ) : annotationIds.length === 0 && !idsLoading ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No evaluation records for this dataset yet — run the evaluation pipeline first.
          </Typography>
        ) : (
          <>
            {recordsLoading ? <LoadingBox label="Loading records…" /> : null}
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Annotation</TableCell>
                  <TableCell>Class</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell align="right">
                    <HeadCell label="3D IoU" term="iou_3d" />
                  </TableCell>
                  <TableCell align="right">
                    <HeadCell label="Anomaly" term="anomaly_score" />
                  </TableCell>
                  <TableCell align="right">
                    <HeadCell label="Consensus" term="grader_consensus" />
                  </TableCell>
                  <TableCell>
                    <HeadCell
                      label="Status"
                      title="Triage status"
                      detail="The routing decision the quality gate produced for this label. Hover the chip for what each status means; click the row for the gate-by-gate breakdown."
                    />
                  </TableCell>
                  <TableCell>Primary failure</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {records.map((r) => (
                  <TableRow
                    key={r.annotation_id}
                    hover
                    sx={{ cursor: 'pointer' }}
                    onClick={() => {
                      setSelected(r);
                      setPanelOpen(true);
                    }}
                  >
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{r.annotation_id}</TableCell>
                    <TableCell>{r.object_class}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{r.model_version}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                      {fmtNum(r.geometry.iou_3d)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                      {fmtNum(r.anomaly.score)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                      {fmtPct(r.grading.consensus)}
                    </TableCell>
                    <TableCell>{r.decision ? <StatusChip status={r.decision.status} /> : '—'}</TableCell>
                    <TableCell sx={{ fontSize: 12, color: '#ef9a9a' }}>
                      {r.decision?.primary_failure_reason ? (
                        <ExplainTip term={glossaryKeyForStatus(r.decision.primary_failure_reason) ?? undefined}>
                          <Box component="span" sx={{ borderBottom: '1px dotted #5c6873', cursor: 'help' }}>
                            {r.decision.primary_failure_reason.replace(/_/g, ' ')}
                          </Box>
                        </ExplainTip>
                      ) : (
                        '—'
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <TablePagination
              component="div"
              count={annotationIds.length}
              page={page}
              onPageChange={(_, p) => setPage(p)}
              rowsPerPage={PAGE_SIZE}
              rowsPerPageOptions={[PAGE_SIZE]}
            />
          </>
        )}
      </SectionCard>

      <EvidencePanel
        open={panelOpen}
        onClose={() => setPanelOpen(false)}
        record={selected}
        onAskCopilot={(request) => {
          setCopilotRequest(request);
          setCopilotOpen(true);
        }}
      />
      <CopilotDrawer open={copilotOpen} onClose={() => setCopilotOpen(false)} request={copilotRequest} />
    </Box>
  );
}
