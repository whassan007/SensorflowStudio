/** Tool-call audit trail: args, result hash, timing, permission outcomes. */
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { AuditRecord } from '../../types/retro';

const STATUS_COLORS: Record<string, string> = {
  ok: '#66bb6a',
  error: '#ef5350',
  timeout: '#ffb74d',
  denied: '#e65100',
};

export default function AuditTrail({ records }: { records: AuditRecord[] }) {
  if (!records.length) {
    return <Typography sx={{ fontSize: 13, color: '#8a949e' }}>No audit records.</Typography>;
  }
  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Typography sx={{ fontSize: 11.5, color: '#8a949e', mb: 1 }}>
        Every tool call — including denials and failures — is appended with its
        arguments, a SHA-256 result hash, and a UTC timestamp.
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            {['time (UTC)', 'tool', 'status', 'ms', 'args', 'result hash', 'write?'].map((h) => (
              <TableCell key={h} sx={{ color: '#8a949e', fontSize: 11, whiteSpace: 'nowrap' }}>{h}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {records.map((r) => (
            <TableRow key={r.call_id} hover>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, whiteSpace: 'nowrap' }}>
                {r.timestamp.replace('T', ' ').slice(0, 19)}
              </TableCell>
              <TableCell sx={{ fontSize: 12, fontWeight: 600 }}>{r.tool}</TableCell>
              <TableCell>
                <Chip size="small" label={r.status}
                      sx={{ bgcolor: `${STATUS_COLORS[r.status] ?? '#8a949e'}22`,
                            color: STATUS_COLORS[r.status] ?? '#8a949e',
                            fontWeight: 700, fontSize: 10 }} />
              </TableCell>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{r.elapsed_ms.toFixed(1)}</TableCell>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 10.5, maxWidth: 320,
                               overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {JSON.stringify(r.args)}
              </TableCell>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 10.5 }}>
                {r.result_hash ? `${r.result_hash.slice(0, 12)}…` : r.error ? `⚠ ${r.error.slice(0, 60)}` : '—'}
              </TableCell>
              <TableCell sx={{ fontSize: 11 }}>{r.authorized_write ? 'AUTHORIZED' : ''}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}
