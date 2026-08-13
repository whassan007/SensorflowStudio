/**
 * Container explorer ("Quality by Container"): sortable container table with
 * risk presets. Drilling into a container opens the forensic object table —
 * intentionally the deepest level of the app — plus nearest-neighbor
 * "similar containers" retrieval for casting a wider net.
 */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { ArrowLeft, Search, ShieldAlert } from 'lucide-react';
import type { ContainerSortPreset, SimilarityResponse } from '../../types/megaeval';
import {
  findSimilarContainers,
  fmtCompact,
  getContainerObjects,
  getRunContainers,
} from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, LoadingBox, SectionCard, fmtNum, fmtPct } from '../labeleval/shared';
import { HeadCell } from '../help/InfoTip';
import { ContainerStatusIcon, DimChips, OutcomeChip } from './shared';

const SORT_PRESETS: Array<{ value: ContainerSortPreset; label: string }> = [
  { value: 'highest_risk', label: 'Highest risk' },
  { value: 'worst_recall', label: 'Worst recall' },
  { value: 'worst_precision', label: 'Worst precision' },
  { value: 'worst_iou', label: 'Worst IoU' },
  { value: 'most_anomalies', label: 'Most anomalies' },
  { value: 'least_verified', label: 'Least verified' },
];

// ---------------------------------------------------------------- forensic view

function ForensicView({
  runId,
  containerId,
  onBack,
  onDrill,
}: {
  runId: string;
  containerId: number;
  onBack: () => void;
  onDrill: (cid: number) => void;
}) {
  const objects = usePoll(() => getContainerObjects(runId, containerId), null, [runId, containerId]);
  const [similar, setSimilar] = useState<SimilarityResponse | null>(null);
  const [simBusy, setSimBusy] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  const findSimilar = async () => {
    setSimBusy(true);
    setSimError(null);
    try {
      setSimilar(await findSimilarContainers({ run_id: runId, container_id: containerId, k: 10 }));
    } catch (err) {
      setSimError(err instanceof Error ? err.message : String(err));
    } finally {
      setSimBusy(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <SectionCard
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Button size="small" startIcon={<ArrowLeft size={14} />} onClick={onBack}>
              Containers
            </Button>
            <span>Forensic object table · container #{containerId}</span>
          </Box>
        }
        action={
          <Button
            size="small"
            variant="outlined"
            startIcon={simBusy ? <CircularProgress size={14} color="inherit" /> : <Search size={14} />}
            disabled={simBusy}
            onClick={() => void findSimilar()}
          >
            Find similar containers
          </Button>
        }
        help="Every annotation in this one container with its evaluation outcome. This is the deepest level of the app — per-object data is loaded on demand only here, never scanned in aggregate views. The shield icon marks safety-critical objects."
        sx={{ flex: '2 1 620px' }}
      >
        <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mb: 1 }}>
          Deepest drill-down level: individual annotations, loaded on demand for this container only.
        </Typography>
        {objects.error ? <ErrorNote error={objects.error} /> : null}
        {objects.loading && !objects.data ? <LoadingBox /> : null}
        {objects.data ? (
          <Box sx={{ maxHeight: 460, overflowY: 'auto' }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>
                    <HeadCell
                      label="Outcome"
                      title="Outcome"
                      detail="Evaluation verdict for this annotation: TP (correct), FN (missed object), FP (phantom), LOCALIZATION (matched but geometry off), LOW_CONF. Hover each chip for its definition."
                    />
                  </TableCell>
                  <TableCell>Class</TableCell>
                  <TableCell align="right">
                    <HeadCell label="IoU" term="iou_3d" />
                  </TableCell>
                  <TableCell align="right">
                    <HeadCell label="Conf" term="confidence" />
                  </TableCell>
                  <TableCell align="center">
                    <HeadCell label="Anomaly" term="anomaly_score" />
                  </TableCell>
                  <TableCell>
                    <HeadCell
                      label="Dims"
                      title="Dimensions"
                      detail="The cohort dimensions this object belongs to (weather, lighting, scenario, sensor, distance band, occlusion) plus evidence flags like sensor-disagree."
                    />
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {objects.data.objects.map((o) => (
                  <TableRow key={o.annotation_id} hover>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <OutcomeChip outcome={o.outcome} />
                        {o.safety_critical ? <ShieldAlert size={14} color="#ef5350" /> : null}
                      </Box>
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>{o.class}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {fmtNum(o.iou, 3)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {fmtNum(o.confidence, 2)}
                    </TableCell>
                    <TableCell align="center">
                      {o.anomaly ? (
                        <Typography component="span" sx={{ color: '#ec407a', fontSize: 13 }}>
                          ●
                        </Typography>
                      ) : (
                        <Typography component="span" sx={{ color: '#37474f', fontSize: 13 }}>
                          ○
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <DimChips
                        values={[
                          o.weather,
                          o.lighting,
                          o.scenario,
                          o.sensor,
                          o.distance_band,
                          o.occlusion,
                          o.sensor_disagree ? 'sensor-disagree' : undefined,
                        ]}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        ) : null}
      </SectionCard>

      {simError ? (
        <Box sx={{ flex: '1 1 340px' }}>
          <ErrorNote error={simError} />
        </Box>
      ) : null}
      {similar ? (
        <SectionCard
          title={`Similar containers${similar.retrieval ? ` · ${similar.retrieval}` : ''}`}
          helpTerm="embedding_similarity"
          sx={{ flex: '1 1 380px' }}
        >
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Container</TableCell>
                <TableCell align="right">Similarity</TableCell>
                <TableCell>Context</TableCell>
                <TableCell align="right">FN/FP</TableCell>
                <TableCell align="right">Risk</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {similar.results.map((s) => (
                <TableRow key={s.container_id} hover onClick={() => onDrill(s.container_id)} sx={{ cursor: 'pointer' }}>
                  <TableCell sx={{ fontFamily: 'monospace', color: '#4fc3f7' }}>#{s.container_id}</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(s.similarity)}
                  </TableCell>
                  <TableCell>
                    <DimChips values={[s.scenario, s.lighting, s.weather]} />
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {s.fn}/{s.fp}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtNum(s.risk_score, 2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </SectionCard>
      ) : null}
    </Box>
  );
}

// ---------------------------------------------------------------- containers table

export default function ContainersTab({
  runId,
  refreshKey,
  drillContainerId,
  onDrill,
}: {
  runId: string;
  refreshKey: number;
  drillContainerId: number | null;
  onDrill: (cid: number | null) => void;
}) {
  const [sort, setSort] = useState<ContainerSortPreset>('highest_risk');
  const containers = usePoll(() => getRunContainers(runId, sort, 50), null, [runId, sort, refreshKey]);

  if (drillContainerId !== null) {
    return (
      <ForensicView
        runId={runId}
        containerId={drillContainerId}
        onBack={() => onDrill(null)}
        onDrill={(cid) => onDrill(cid)}
      />
    );
  }

  return (
    <SectionCard
      title="Quality by container"
      helpTerm="container"
      action={
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {containers.data ? (
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              {fmtCompact(containers.data.total)} containers (exact) · showing top {containers.data.rows.length}
            </Typography>
          ) : null}
          <TextField
            select
            size="small"
            label="Sort"
            value={sort}
            onChange={(e) => setSort(e.target.value as ContainerSortPreset)}
            sx={{ minWidth: 180 }}
          >
            {SORT_PRESETS.map((p) => (
              <MenuItem key={p.value} value={p.value}>
                {p.label}
              </MenuItem>
            ))}
          </TextField>
        </Box>
      }
    >
      {containers.error ? <ErrorNote error={containers.error} /> : null}
      {containers.loading && !containers.data ? <LoadingBox /> : null}
      {containers.data ? (
        <Box sx={{ maxHeight: 520, overflowY: 'auto' }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell />
                <TableCell>
                  <HeadCell label="Container" term="container" />
                </TableCell>
                <TableCell>Context</TableCell>
                <TableCell align="right">Objects</TableCell>
                <TableCell align="right">
                  <HeadCell label="Verified" term="status_verified" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Precision" term="precision" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Recall" term="recall" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Mean IoU" term="iou_3d" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Anomalies" term="anomaly_score" />
                </TableCell>
                <TableCell align="right">
                  <HeadCell label="Risk" term="risk_score" />
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {containers.data.rows.map((c) => (
                <TableRow key={c.container_id} hover onClick={() => onDrill(c.container_id)} sx={{ cursor: 'pointer' }}>
                  <TableCell sx={{ width: 30 }}>
                    <ContainerStatusIcon status={c.status} />
                  </TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', color: '#4fc3f7' }}>#{c.container_id}</TableCell>
                  <TableCell>
                    <DimChips values={[c.scenario, c.lighting, c.weather, c.road_type]} />
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtCompact(c.n_objects)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {c.verified}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(c.precision)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtPct(c.recall)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtNum(c.mean_iou, 3)}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', color: c.anomalies > 0 ? '#ec407a' : undefined }}>
                    {c.anomalies}
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {fmtNum(c.risk_score, 2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      ) : null}
      <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
        Click a container to open its forensic object table (deepest drill-down).
      </Typography>
    </SectionCard>
  );
}
