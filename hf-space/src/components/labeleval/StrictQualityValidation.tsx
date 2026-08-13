import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import type { QualityMetrics } from '../../types/labeleval';
import { SectionCard, fmtNum } from './shared';
import { Term } from '../help/InfoTip';

interface CheckRow {
  check: string;
  tolerance: string;
  observed: (m: QualityMetrics) => string;
  /** Glossary key explaining the check on hover. */
  term?: string;
}

/**
 * The geometric validation checks with their configured tolerances.
 * The API only exposes aggregate metrics, so the "observed" column surfaces
 * the closest aggregate where one exists.
 */
const CHECKS: CheckRow[] = [
  {
    check: 'Point-in-box ratio',
    tolerance: '≥ 0.60 of box volume occupied by LiDAR returns',
    observed: () => '— (per-annotation, see evidence panels)',
    term: 'point_in_box_ratio',
  },
  {
    check: 'Point density',
    tolerance: '≥ 5 pts/m³ at range-adjusted threshold',
    observed: () => '— (per-annotation, see evidence panels)',
    term: 'point_density',
  },
  {
    check: 'Occupancy',
    tolerance: 'No overlap > 0.30 IoU with another box of same class',
    observed: (m) => `mean 3D IoU ${fmtNum(m.global.mean_iou_3d)}`,
    term: 'iou_3d',
  },
  {
    check: 'Dimensions',
    tolerance: 'Within class prior ± 2.5σ (l, w, h)',
    observed: () => '— (per-annotation, see evidence panels)',
    term: 'dimension_error',
  },
  {
    check: 'Orientation',
    tolerance: '≤ 15° error vs. reference / motion heading',
    observed: (m) => `mean ${fmtNum(m.global.mean_orientation_error_deg, 1)}°`,
    term: 'orientation_error',
  },
  {
    check: 'Centroid',
    tolerance: '≤ 0.5 m position error vs. reference',
    observed: (m) => `mean ${fmtNum(m.global.mean_position_error)} m`,
    term: 'position_error',
  },
  {
    check: 'Ground contact',
    tolerance: 'Box bottom within ± 0.3 m of ground plane',
    observed: () => '— (per-annotation, see evidence panels)',
    term: 'ground_contact',
  },
  {
    check: 'Sensor consistency',
    tolerance: 'LiDAR vs. camera class/extent agreement required',
    observed: (m) => `anomaly rate ${(m.global.anomaly_rate * 100).toFixed(1)}%`,
    term: 'sensor_consistency',
  },
];

export default function StrictQualityValidation({ metrics }: { metrics: QualityMetrics | null }) {
  return (
    <SectionCard
      title="Strict Quality Validation — geometric gates"
      help="GT-free validation: every label is checked against physics and sensor evidence, no reference truth needed. Hover a check name for what it measures. Tolerances come from the active quality policy; per-annotation verdicts appear in each label's evidence panel and drive the geometric_validation gate in Triage."
    >
      <Typography variant="body2" sx={{ color: '#8a949e', mb: 1 }}>
        Every annotation must pass all applicable geometric checks. Failing any single gate flags the label — gates are
        never averaged away.
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Check</TableCell>
            <TableCell>Configured tolerance</TableCell>
            <TableCell align="right">Observed aggregate</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {CHECKS.map((c) => (
            <TableRow key={c.check} hover>
              <TableCell sx={{ fontWeight: 600 }}>
                {c.term ? <Term k={c.term}>{c.check}</Term> : c.check}
              </TableCell>
              <TableCell sx={{ color: '#aab4be', fontSize: 12 }}>{c.tolerance}</TableCell>
              <TableCell align="right" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                {metrics ? c.observed(metrics) : '—'}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </SectionCard>
  );
}
