import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TablePagination from '@mui/material/TablePagination';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import Typography from '@mui/material/Typography';
import { useFilters } from '../../context/FilterContext';
import { SEVERITY_COLORS } from '../../types';

const COLUMNS: { id: string; label: string; numeric?: boolean; sortable?: boolean }[] = [
  { id: 'street_name', label: 'Intersection', sortable: true },
  { id: 'county', label: 'County', sortable: true },
  { id: 'conflict_type', label: 'Conflict' },
  { id: 'min_ttc', label: 'TTC (s)', numeric: true, sortable: true },
  { id: 'min_pet', label: 'PET (s)', numeric: true, sortable: true },
  { id: 'max_speed', label: 'Speed (m/s)', numeric: true, sortable: true },
  { id: 'severity_index', label: 'Severity', numeric: true, sortable: true },
];

export default function DataGrid() {
  const { data, error, page, setPage, pageSize, setPageSize, sortBy, sortDir, setSort, selected, setSelected } =
    useFilters();

  if (error) {
    return (
      <section className="data-grid data-grid-message">
        <Typography color="error">Failed to load SSAM data: {error}</Typography>
      </section>
    );
  }

  return (
    <section className="data-grid">
      <TableContainer sx={{ flex: 1, overflow: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              {COLUMNS.map((col) => (
                <TableCell key={col.id} align={col.numeric ? 'right' : 'left'} sortDirection={sortBy === col.id ? sortDir : false}>
                  {col.sortable ? (
                    <TableSortLabel
                      active={sortBy === col.id}
                      direction={sortBy === col.id ? sortDir : 'desc'}
                      onClick={() => setSort(col.id)}
                    >
                      {col.label}
                    </TableSortLabel>
                  ) : (
                    col.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {(data?.rows ?? []).map((row) => (
              <TableRow
                key={row.id}
                hover
                selected={selected?.id === row.id}
                onClick={() => setSelected(row)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>{row.street_name}</TableCell>
                <TableCell>{row.county}</TableCell>
                <TableCell>{row.conflict_type}</TableCell>
                <TableCell align="right">{row.min_ttc.toFixed(1)}</TableCell>
                <TableCell align="right">{row.min_pet.toFixed(1)}</TableCell>
                <TableCell align="right">{row.max_speed.toFixed(1)}</TableCell>
                <TableCell align="right">
                  <span className="severity-pill" style={{ backgroundColor: SEVERITY_COLORS[row.severity_label] }}>
                    {row.severity_index.toFixed(2)} {row.severity_label}
                  </span>
                </TableCell>
              </TableRow>
            ))}
            {data && data.rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={COLUMNS.length} align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                    No conflicts match the current filters.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component="div"
        count={data?.total ?? 0}
        page={page - 1}
        onPageChange={(_, p) => setPage(p + 1)}
        rowsPerPage={pageSize}
        onRowsPerPageChange={(e) => {
          setPageSize(parseInt(e.target.value, 10));
          setPage(1);
        }}
        rowsPerPageOptions={[10, 25, 50, 100]}
      />
    </section>
  );
}
