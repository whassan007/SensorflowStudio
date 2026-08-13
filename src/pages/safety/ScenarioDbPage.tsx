/**
 * Scenario Database browser — Safety Pool-inspired local scenario library:
 * toggleable filter chips + text search + export bundle, populated from rare
 * events, discrepancy mining and ODD gap fills.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { Download, FolderInput, Search } from 'lucide-react';
import {
  exportScenarios,
  populateScenarios,
  searchScenarios,
  type ScenarioCounts,
  type ScenarioRecord,
} from '../../services/safety';
import { downloadJson } from '../../components/safety/shared';
import { IllustratedEmpty, PanelSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, MetricCard, SectionCard, StatusChip } from '../../components/labeleval/shared';
import { tokens } from '../../theme';

function FilterChipRow({
  label,
  options,
  active,
  onToggle,
}: {
  label: string;
  options: Array<{ value: string; count: number }>;
  active: string | null;
  onToggle: (v: string | null) => void;
}) {
  if (!options.length) return null;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
      <Typography variant="caption" sx={{ color: tokens.color.neutral, width: 74, flexShrink: 0 }}>
        {label}
      </Typography>
      {options.map((o) => (
        <Chip
          key={o.value}
          size="small"
          label={`${o.value} (${o.count})`}
          onClick={() => onToggle(active === o.value ? null : o.value)}
          sx={{
            height: 22,
            fontSize: 11,
            cursor: 'pointer',
            bgcolor: active === o.value ? tokens.color.infoBg : tokens.color.surfaceRaised,
            color: active === o.value ? tokens.color.info : tokens.color.textDim,
            border: `1px solid ${active === o.value ? tokens.color.info : tokens.color.border}`,
            transition: `all ${tokens.motion.fast}`,
          }}
        />
      ))}
    </Box>
  );
}

export default function ScenarioDbPage() {
  const [counts, setCounts] = useState<ScenarioCounts | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioRecord[] | null>(null);
  const [severity, setSeverity] = useState<string | null>(null);
  const [type, setType] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [text, setText] = useState('');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    searchScenarios({
      severity: severity ?? undefined,
      scenario_type: type ?? undefined,
      source: source ?? undefined,
      text: query || undefined,
      limit: 200,
    })
      .then((r) => {
        setCounts(r.counts);
        setScenarios(r.scenarios);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [severity, type, source, query]);

  useEffect(refresh, [refresh]);

  const populate = () => {
    setBusy('populate');
    populateScenarios()
      .then(() => refresh())
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(null));
  };

  const doExport = () => {
    setBusy('export');
    exportScenarios({ severity: severity ?? undefined, scenario_type: type ?? undefined, source: source ?? undefined, text: query || undefined })
      .then((bundle) => downloadJson('scenario-db-export.json', bundle))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(null));
  };

  const opts = useMemo(
    () => ({
      severity: Object.entries(counts?.by_severity ?? {}).map(([value, count]) => ({ value, count })),
      type: Object.entries(counts?.by_type ?? {}).map(([value, count]) => ({ value, count })),
      source: Object.entries(counts?.by_source ?? {}).map(([value, count]) => ({ value, count })),
    }),
    [counts]
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <ErrorNote error={error} /> : null}

      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
        <MetricCard label="Scenarios" value={counts?.total ?? '—'} info="All scenario records in the local database, across sources (rare-event mining, discrepancy mining, ODD gap fills)." />
        <Box sx={{ flex: 1 }} />
        <Button variant="outlined" size="small" startIcon={<FolderInput size={15} />} disabled={busy !== null} onClick={populate}>
          {busy === 'populate' ? 'Importing…' : 'Import rare events'}
        </Button>
        <Button variant="outlined" size="small" startIcon={<Download size={15} />} disabled={busy !== null || !scenarios?.length} onClick={doExport}>
          {busy === 'export' ? 'Exporting…' : 'Export filtered bundle'}
        </Button>
      </Box>

      <SectionCard
        title="Filters"
        help="Chips toggle exact filters over the scenario index; the text box searches descriptions and tags. Filters combine (AND). Export downloads exactly the filtered set as a JSON bundle with provenance."
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <FilterChipRow label="severity" options={opts.severity} active={severity} onToggle={setSeverity} />
          <FilterChipRow label="type" options={opts.type} active={type} onToggle={setType} />
          <FilterChipRow label="source" options={opts.source} active={source} onToggle={setSource} />
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5 }}>
            <TextField
              size="small"
              placeholder="Search descriptions and tags…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') setQuery(text);
              }}
              sx={{ width: 340 }}
            />
            <Button size="small" variant="contained" startIcon={<Search size={14} />} onClick={() => setQuery(text)}>
              Search
            </Button>
            {query ? (
              <Chip size="small" label={`text: "${query}"`} onDelete={() => { setQuery(''); setText(''); }} sx={{ height: 22, fontSize: 11 }} />
            ) : null}
          </Box>
        </Box>
      </SectionCard>

      <SectionCard title={`Scenarios ${scenarios ? `(${scenarios.length}${scenarios.length === 200 ? '+' : ''})` : ''}`}>
        {loading ? <PanelSkeleton rows={6} header={false} /> : null}
        {!loading && scenarios && !scenarios.length ? (
          <IllustratedEmpty
            art="data"
            title={counts && counts.total > 0 ? 'No scenarios match the filters' : 'Scenario database is empty'}
            message={
              counts && counts.total > 0
                ? 'Relax a filter chip or clear the text search.'
                : 'Populate it by importing mined rare events, running discrepancy mining (Safety → Discrepancy), or filling ODD coverage gaps — each source registers scenarios here automatically.'
            }
            action={
              counts && counts.total === 0 ? (
                <Button variant="contained" size="small" startIcon={<FolderInput size={15} />} onClick={populate}>
                  Import rare events
                </Button>
              ) : undefined
            }
          />
        ) : null}
        {!loading && scenarios?.length ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Severity</TableCell>
                <TableCell>Source</TableCell>
                <TableCell>ODD tags</TableCell>
                <TableCell>Description</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {scenarios.map((s) => (
                <TableRow key={s.scenario_id} hover>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{s.scenario_id.slice(0, 14)}</TableCell>
                  <TableCell sx={{ fontSize: 12 }}>{s.scenario_type}</TableCell>
                  <TableCell><StatusChip status={s.severity} /></TableCell>
                  <TableCell sx={{ fontSize: 12, color: tokens.color.textDim }}>{s.source}</TableCell>
                  <TableCell>
                    {Object.entries(s.odd_tags ?? {}).map(([k, v]) => (
                      <Chip key={k} size="small" label={`${k}=${v}`} sx={{ height: 17, fontSize: 9.5, mr: 0.5, bgcolor: tokens.color.surfaceRaised }} />
                    ))}
                  </TableCell>
                  <TableCell sx={{ fontSize: 12, color: tokens.color.textDim, maxWidth: 420 }}>{s.description}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : null}
      </SectionCard>
    </Box>
  );
}
