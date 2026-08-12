import { Fragment, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import { AlertTriangle, ChevronDown, ChevronRight, Sparkles } from 'lucide-react';
import type { CopilotExplainRequest, RegressionEntry, RegressionResponse } from '../../types/labeleval';
import { SectionCard, fmtNum } from './shared';

function DeltaTable({ entry }: { entry: RegressionEntry }) {
  return (
    <Table size="small" sx={{ my: 1, bgcolor: '#12171d' }}>
      <TableHead>
        <TableRow>
          <TableCell>Metric</TableCell>
          <TableCell align="right">Baseline</TableCell>
          <TableCell align="right">Current</TableCell>
          <TableCell align="right">Delta</TableCell>
          <TableCell align="right">Tolerance</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {entry.deltas.map((d) => (
          <TableRow key={d.metric} sx={{ bgcolor: d.regressed ? 'rgba(183,28,28,0.18)' : undefined }}>
            <TableCell sx={{ fontWeight: 600 }}>{d.metric}</TableCell>
            <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
              {fmtNum(d.baseline)}
            </TableCell>
            <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
              {fmtNum(d.current)}
            </TableCell>
            <TableCell
              align="right"
              sx={{ fontFamily: 'monospace', fontSize: 12, color: d.regressed ? '#ef5350' : '#66bb6a' }}
            >
              {d.delta > 0 ? '+' : ''}
              {fmtNum(d.delta)}
            </TableCell>
            <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
              ±{fmtNum(d.tolerance)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function RegressionTracking({
  regression,
  onAskCopilot,
}: {
  regression: RegressionResponse | null;
  onAskCopilot: (request: CopilotExplainRequest) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <>
      {regression?.current_alert ? (
        <Box
          sx={{
            bgcolor: '#b71c1c',
            color: '#fff',
            borderRadius: 1,
            p: 2,
            mb: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
          }}
        >
          <AlertTriangle size={28} />
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: 1 }}>
              REGRESSION DETECTED
            </Typography>
            <Typography variant="body2">
              The current model performs worse than its baseline beyond configured tolerances. Affected labels are
              blocked from auto-verification until resolved.
            </Typography>
          </Box>
        </Box>
      ) : null}

      <SectionCard title="Regression Tracking — model vs. baseline">
        {!regression || regression.entries.length === 0 ? (
          <Typography variant="body2" sx={{ color: '#8a949e' }}>
            No regression comparisons yet — they appear after evaluating a model that has a baseline.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell />
                <TableCell>Model</TableCell>
                <TableCell>Baseline</TableCell>
                <TableCell>Dataset</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Kinds</TableCell>
                <TableCell>Affected classes</TableCell>
                <TableCell>Affected scenarios</TableCell>
                <TableCell>Status</TableCell>
                <TableCell />
              </TableRow>
            </TableHead>
            <TableBody>
              {regression.entries.map((e) => {
                const key = e.run_id;
                const isOpen = expanded === key;
                return (
                  <Fragment key={key}>
                    <TableRow hover onClick={() => setExpanded(isOpen ? null : key)} sx={{ cursor: 'pointer' }}>
                      <TableCell sx={{ width: 32 }}>
                        {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{e.model_version}</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                        {e.baseline_version ?? '—'}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{e.dataset_version}</TableCell>
                      <TableCell sx={{ fontSize: 12 }}>{new Date(e.date).toLocaleString()}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                          {e.kinds.map((k) => (
                            <Chip key={k} size="small" label={k} sx={{ bgcolor: '#232a31', fontSize: 10 }} />
                          ))}
                        </Box>
                      </TableCell>
                      <TableCell sx={{ fontSize: 12 }}>{e.affected_classes.join(', ') || '—'}</TableCell>
                      <TableCell sx={{ fontSize: 12 }}>{e.affected_scenarios.join(', ') || '—'}</TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          label={e.regression_detected ? 'REGRESSED' : 'OK'}
                          sx={{
                            bgcolor: e.regression_detected ? '#b71c1c' : '#1b5e20',
                            color: e.regression_detected ? '#ffcdd2' : '#a5d6a7',
                            fontWeight: 700,
                          }}
                        />
                      </TableCell>
                      <TableCell onClick={(ev) => ev.stopPropagation()}>
                        {e.regression_detected ? (
                          <Button
                            size="small"
                            variant="outlined"
                            startIcon={<Sparkles size={14} />}
                            onClick={() =>
                              onAskCopilot({
                                context_type: 'regression',
                                model_version: e.model_version,
                                extra: { run_id: e.run_id, dataset_version: e.dataset_version },
                              })
                            }
                          >
                            Explain via Copilot
                          </Button>
                        ) : null}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell colSpan={10} sx={{ p: 0, border: 0 }}>
                        <Collapse in={isOpen} unmountOnExit>
                          <Box sx={{ px: 4, py: 1 }}>
                            <DeltaTable entry={e} />
                          </Box>
                        </Collapse>
                      </TableCell>
                    </TableRow>
                  </Fragment>
                );
              })}
            </TableBody>
          </Table>
        )}
      </SectionCard>
    </>
  );
}
