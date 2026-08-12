/**
 * Investigation panel: multi-criteria error search against the inverted error
 * index. Results stay aggregate-first (counts, by-type shares, worst
 * containers) — clicking a container jumps to the forensic drill-down.
 */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Checkbox from '@mui/material/Checkbox';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { SearchCode, ShieldAlert } from 'lucide-react';
import type { DimName, ErrorSearchResponse, ErrorType } from '../../types/megaeval';
import { fmtCompact, getDimensions, searchErrors } from '../../services/megaeval';
import { usePoll } from '../../services/labeleval';
import { ErrorNote, MetricCard, SectionCard, fmtNum } from '../labeleval/shared';
import { DimChips, ERROR_TYPE_COLORS, OutcomeChip } from './shared';

const ALL_ERROR_TYPES: ErrorType[] = ['FN', 'FP', 'LOCALIZATION', 'ANOMALY', 'LOW_CONF'];

function MultiSelect({
  label,
  options,
  value,
  onChange,
  minWidth = 170,
}: {
  label: string;
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
  minWidth?: number;
}) {
  return (
    <TextField
      select
      size="small"
      label={label}
      value={value}
      onChange={(e) => {
        const v = e.target.value as unknown;
        onChange(typeof v === 'string' ? v.split(',').filter(Boolean) : (v as string[]));
      }}
      SelectProps={{ multiple: true }}
      sx={{ minWidth }}
    >
      {options.map((o) => (
        <MenuItem key={o} value={o}>
          {o}
        </MenuItem>
      ))}
    </TextField>
  );
}

export default function InvestigationTab({
  runId,
  onOpenContainer,
}: {
  runId: string;
  onOpenContainer: (cid: number) => void;
}) {
  const dims = usePoll(getDimensions, null, []);
  const dimOptions = (d: DimName): string[] => dims.data?.dimensions[d] ?? [];

  const [errorTypes, setErrorTypes] = useState<string[]>([]);
  const [classes, setClasses] = useState<string[]>([]);
  const [lightings, setLightings] = useState<string[]>([]);
  const [weathers, setWeathers] = useState<string[]>([]);
  const [confMax, setConfMax] = useState('');
  const [riskMin, setRiskMin] = useState('');
  const [safetyOnly, setSafetyOnly] = useState(false);

  const [result, setResult] = useState<ErrorSearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSearch = async () => {
    setBusy(true);
    setError(null);
    try {
      const filters: Partial<Record<DimName, string[]>> = {};
      if (classes.length) filters.class = classes;
      if (lightings.length) filters.lighting = lightings;
      if (weathers.length) filters.weather = weathers;
      const confParsed = Number(confMax);
      const riskParsed = Number(riskMin);
      setResult(
        await searchErrors({
          run_id: runId,
          ...(errorTypes.length ? { error_types: errorTypes as ErrorType[] } : {}),
          ...(Object.keys(filters).length ? { filters } : {}),
          ...(confMax.trim() !== '' && Number.isFinite(confParsed) ? { confidence_max: confParsed } : {}),
          ...(riskMin.trim() !== '' && Number.isFinite(riskParsed) ? { risk_min: riskParsed } : {}),
          ...(safetyOnly ? { safety_only: true } : {}),
          limit_containers: 20,
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard title="Error search">
        <Box sx={{ display: 'flex', gap: 1.25, flexWrap: 'wrap', alignItems: 'center' }}>
          <MultiSelect label="Error types" options={ALL_ERROR_TYPES} value={errorTypes} onChange={setErrorTypes} />
          <MultiSelect label="Class" options={dimOptions('class')} value={classes} onChange={setClasses} />
          <MultiSelect label="Lighting" options={dimOptions('lighting')} value={lightings} onChange={setLightings} />
          <MultiSelect label="Weather" options={dimOptions('weather')} value={weathers} onChange={setWeathers} />
          <TextField
            size="small"
            label="Max confidence"
            type="number"
            value={confMax}
            onChange={(e) => setConfMax(e.target.value)}
            sx={{ width: 130 }}
            inputProps={{ step: 0.05, min: 0, max: 1 }}
          />
          <TextField
            size="small"
            label="Min risk"
            type="number"
            value={riskMin}
            onChange={(e) => setRiskMin(e.target.value)}
            sx={{ width: 110 }}
            inputProps={{ step: 0.1, min: 0 }}
          />
          <FormControlLabel
            control={<Checkbox size="small" checked={safetyOnly} onChange={(e) => setSafetyOnly(e.target.checked)} />}
            label={<Typography variant="body2">Safety-critical only</Typography>}
          />
          <Button
            variant="contained"
            size="small"
            startIcon={busy ? <CircularProgress size={14} color="inherit" /> : <SearchCode size={15} />}
            disabled={busy}
            onClick={() => void runSearch()}
          >
            Search errors
          </Button>
        </Box>
        {error ? <ErrorNote error={error} /> : null}
      </SectionCard>

      {result ? (
        <>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
            <MetricCard label="Matched errors" value={fmtCompact(result.matched_errors)} accent="#ef5350" />
            <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center' }}>
              {ALL_ERROR_TYPES.filter((t) => (result.by_type[t] ?? 0) > 0).map((t) => (
                <Chip
                  key={t}
                  size="small"
                  label={`${t}: ${fmtCompact(result.by_type[t] ?? 0)}`}
                  sx={{
                    fontFamily: 'monospace',
                    fontWeight: 700,
                    bgcolor: '#232a31',
                    color: ERROR_TYPE_COLORS[t],
                  }}
                />
              ))}
            </Box>
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <SectionCard title={`Worst containers (${result.worst_containers.length})`} sx={{ flex: '1 1 480px' }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Container</TableCell>
                    <TableCell>Context</TableCell>
                    <TableCell align="right">Errors</TableCell>
                    <TableCell align="right">Mean risk</TableCell>
                    <TableCell align="right">Max severity</TableCell>
                    <TableCell align="right">Safety hits</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.worst_containers.map((c) => (
                    <TableRow
                      key={c.container_id}
                      hover
                      onClick={() => onOpenContainer(c.container_id)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell sx={{ fontFamily: 'monospace', color: '#4fc3f7' }}>#{c.container_id}</TableCell>
                      <TableCell>
                        <DimChips values={[c.scenario, c.lighting, c.weather]} />
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {c.error_count}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {fmtNum(c.mean_risk, 2)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {fmtNum(c.max_severity, 2)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace', color: c.safety_hits > 0 ? '#ef5350' : undefined }}>
                        {c.safety_hits}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <Typography variant="caption" sx={{ color: '#8a949e', display: 'block', mt: 1 }}>
                Click a container to open its forensic object table.
              </Typography>
            </SectionCard>

            <SectionCard title={`Top examples (${result.examples.length})`} sx={{ flex: '1 1 420px' }}>
              <Box sx={{ maxHeight: 420, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 0.75 }}>
                {result.examples.map((ex) => (
                  <Box
                    key={ex.error_id}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      p: 0.75,
                      border: '1px solid #232a31',
                      borderRadius: 1,
                      cursor: 'pointer',
                      '&:hover': { bgcolor: '#1b222a' },
                    }}
                    onClick={() => onOpenContainer(ex.container_id)}
                  >
                    <OutcomeChip outcome={ex.error_type} />
                    {ex.safety_critical ? <ShieldAlert size={14} color="#ef5350" /> : null}
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {ex.class}
                    </Typography>
                    <DimChips values={[ex.scenario, ex.lighting, ex.weather]} />
                    <Box sx={{ flex: 1 }} />
                    <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#8a949e' }}>
                      sev {fmtNum(ex.severity, 2)} · conf {fmtNum(ex.confidence, 2)} · risk {fmtNum(ex.risk_score, 2)} · #
                      {ex.container_id}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </SectionCard>
          </Box>
        </>
      ) : (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          Define criteria and search the error index — results are served from the inverted index, not a scan.
        </Typography>
      )}
    </Box>
  );
}
