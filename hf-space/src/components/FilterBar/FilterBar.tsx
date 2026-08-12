import Autocomplete from '@mui/material/Autocomplete';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Slider from '@mui/material/Slider';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { RotateCcw, Search, ShieldAlert } from 'lucide-react';
import { useFilters } from '../../context/FilterContext';
import { SEVERITY_COLORS, SeverityLabel } from '../../types';

const SEVERITY_ORDER: SeverityLabel[] = ['Critical', 'High', 'Medium', 'Low'];

export default function FilterBar() {
  const { filters, setFilters, resetFilters, data, loading } = useFilters();
  const options = data?.filter_options;

  return (
    <header className="filter-bar">
      <div className="filter-bar-title">
        <ShieldAlert size={22} color="#ff9100" />
        <div>
          <Typography variant="subtitle1" fontWeight={700} lineHeight={1.2}>
            SSAM Safety Dashboard
          </Typography>
          <Typography variant="caption" color="text.secondary">
            California statewide surrogate safety conflicts
          </Typography>
        </div>
      </div>

      <TextField
        size="small"
        placeholder="Search street or county…"
        value={filters.search}
        onChange={(e) => setFilters({ search: e.target.value })}
        InputProps={{ startAdornment: <Search size={16} style={{ marginRight: 6, opacity: 0.6 }} /> }}
        sx={{ minWidth: 220 }}
      />

      <Autocomplete
        multiple
        size="small"
        limitTags={1}
        options={options?.counties ?? []}
        value={filters.counties}
        onChange={(_, v) => setFilters({ counties: v })}
        renderInput={(params) => <TextField {...params} label="County" />}
        sx={{ minWidth: 200 }}
      />

      <Autocomplete
        multiple
        size="small"
        limitTags={1}
        options={options?.conflict_types ?? []}
        value={filters.conflictTypes}
        onChange={(_, v) => setFilters({ conflictTypes: v })}
        renderInput={(params) => <TextField {...params} label="Conflict type" />}
        sx={{ minWidth: 190 }}
      />

      <Autocomplete
        multiple
        size="small"
        limitTags={1}
        options={options?.severity_labels ?? SEVERITY_ORDER}
        value={filters.severityLabels}
        onChange={(_, v) => setFilters({ severityLabels: v })}
        renderInput={(params) => <TextField {...params} label="Severity" />}
        sx={{ minWidth: 180 }}
      />

      <div className="slider-group">
        <Typography variant="caption" color="text.secondary">
          Max TTC: {filters.ttcMax ?? '—'}s
        </Typography>
        <Slider
          size="small"
          min={0.3}
          max={1.7}
          step={0.1}
          value={filters.ttcMax ?? 1.7}
          onChange={(_, v) => setFilters({ ttcMax: v === 1.7 ? null : (v as number) })}
          sx={{ width: 110 }}
        />
      </div>

      <div className="slider-group">
        <Typography variant="caption" color="text.secondary">
          Min speed: {filters.speedMin ?? '—'} m/s
        </Typography>
        <Slider
          size="small"
          min={0}
          max={18}
          step={0.5}
          value={filters.speedMin ?? 0}
          onChange={(_, v) => setFilters({ speedMin: v === 0 ? null : (v as number) })}
          sx={{ width: 110 }}
        />
      </div>

      <div className="summary-chips">
        {SEVERITY_ORDER.map((label) => (
          <Chip
            key={label}
            size="small"
            label={`${label} ${data?.summary?.[label] ?? 0}`}
            onClick={() =>
              setFilters({
                severityLabels: filters.severityLabels.includes(label)
                  ? filters.severityLabels.filter((l) => l !== label)
                  : [...filters.severityLabels, label],
              })
            }
            variant={filters.severityLabels.includes(label) ? 'filled' : 'outlined'}
            sx={{
              borderColor: SEVERITY_COLORS[label],
              color: filters.severityLabels.includes(label) ? '#111' : SEVERITY_COLORS[label],
              bgcolor: filters.severityLabels.includes(label) ? SEVERITY_COLORS[label] : 'transparent',
              fontWeight: 600,
            }}
          />
        ))}
      </div>

      <Tooltip title="Reset all filters">
        <span>
          <IconButton size="small" onClick={resetFilters} disabled={loading}>
            <RotateCcw size={16} />
          </IconButton>
        </span>
      </Tooltip>
    </header>
  );
}
