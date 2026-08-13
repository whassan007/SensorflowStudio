import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardActionArea from '@mui/material/CardActionArea';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import type { QualityGroupDetail } from '../types/labeleval';
import {
  getDatasets,
  getQualityGroups,
  getQualityGroupDetail,
  usePoll,
} from '../services/labeleval';
import { useLabelEval } from '../context/LabelEvalContext';
import { glossaryKeyForStatus } from '../content/glossary';
import DatasetSelector from '../components/labeleval/DatasetSelector';
import {
  SectionCard,
  LoadingBox,
  ErrorNote,
  HBar,
  fmtInt,
  fmtPct,
  fmtNum,
  MetricCard,
} from '../components/labeleval/shared';

const GROUP_COLORS: Record<string, string> = {
  verified: '#66bb6a',
  non_verified: '#ffa726',
  hitl: '#42a5f5',
  rejected: '#ef5350',
};

export default function DatasetsPage() {
  const { activeDatasetId, setActiveDatasetId } = useLabelEval();
  const datasets = usePoll(getDatasets, 5000);
  const groups = usePoll(
    () => getQualityGroups(activeDatasetId),
    activeDatasetId ? 5000 : null,
    [activeDatasetId]
  );
  const [groupDetail, setGroupDetail] = useState<QualityGroupDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [verificationRates, setVerificationRates] = useState<Record<string, number>>({});

  // DatasetSummary carries no verification rate, so pull it from each
  // dataset's quality groups whenever the dataset list changes.
  const datasetIdsKey = (datasets.data?.datasets ?? []).map((d) => d.dataset_id).join(',');
  useEffect(() => {
    let cancelled = false;
    const ids = datasetIdsKey ? datasetIdsKey.split(',') : [];
    if (ids.length === 0) return;
    Promise.all(
      ids.map((id) =>
        getQualityGroups(id)
          .then((g) => [id, g.verification_rate] as const)
          .catch(() => null)
      )
    ).then((entries) => {
      if (cancelled) return;
      const next: Record<string, number> = {};
      for (const e of entries) {
        if (e) next[e[0]] = e[1];
      }
      setVerificationRates(next);
    });
    return () => {
      cancelled = true;
    };
  }, [datasetIdsKey]);

  useEffect(() => {
    setGroupDetail(null);
    setDetailError(null);
  }, [activeDatasetId]);

  const openGroup = async (groupId: string) => {
    setDetailLoading(true);
    setDetailError(null);
    try {
      setGroupDetail(await getQualityGroupDetail(groupId));
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetailLoading(false);
    }
  };

  const failureRows = groupDetail ? Object.entries(groupDetail.failure_reason_counts).sort((a, b) => b[1] - a[1]) : [];
  const maxFailure = Math.max(1, ...failureRows.map(([, c]) => c));

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {datasets.loading && !datasets.data ? <LoadingBox label="Loading datasets…" /> : null}
      {datasets.error && !datasets.data ? <ErrorNote error={datasets.error} /> : null}

      {datasets.data ? (
        <DatasetSelector
          datasets={datasets.data.datasets}
          selectedId={activeDatasetId}
          onSelect={setActiveDatasetId}
          onChanged={datasets.refresh}
          verificationRates={verificationRates}
        />
      ) : null}

      {activeDatasetId ? (
        <SectionCard
          title={`Quality Groups — ${activeDatasetId}`}
          help="The dataset's annotations partitioned by lifecycle outcome: verified (human-confirmed), non-verified (auto-graded but untouched by humans), HITL (in review) and rejected. Click a group to see its quality profile and which gates its labels failed."
        >
          {groups.error && !groups.data ? <ErrorNote error={groups.error} /> : null}
          {groups.data ? (
            <>
              <Typography variant="body2" sx={{ color: '#8a949e', mb: 1.5 }}>
                {fmtInt(groups.data.total)} annotations · verification rate {fmtPct(groups.data.verification_rate)}.
                Click a group for its quality profile.
              </Typography>
              <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
                {groups.data.groups.map((g) => (
                  <Card
                    key={g.group_id}
                    variant="outlined"
                    sx={{
                      flex: '1 1 160px',
                      minWidth: 160,
                      bgcolor: '#12171d',
                      borderColor: groupDetail?.group_id === g.group_id ? '#4fc3f7' : undefined,
                    }}
                  >
                    <CardActionArea onClick={() => void openGroup(g.group_id)}>
                      <CardContent>
                        <Typography
                          variant="subtitle2"
                          sx={{ fontWeight: 700, color: GROUP_COLORS[g.name] ?? '#e6e9ec', textTransform: 'uppercase' }}
                        >
                          {g.name.replace(/_/g, ' ')}
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 800 }}>
                          {fmtInt(g.count)}
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#8a949e' }}>
                          {fmtPct(g.pct)} of annotations
                        </Typography>
                      </CardContent>
                    </CardActionArea>
                  </Card>
                ))}
              </Box>
            </>
          ) : !groups.error ? (
            <LoadingBox label="Loading groups…" />
          ) : null}

          {detailLoading ? <LoadingBox label="Loading group detail…" /> : null}
          {detailError ? <ErrorNote error={detailError} /> : null}
          {groupDetail && !detailLoading ? (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                {groupDetail.name.replace(/_/g, ' ').toUpperCase()} — quality profile
              </Typography>
              <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
                <MetricCard label="Precision" value={fmtPct(groupDetail.precision)} term="precision" />
                <MetricCard label="Recall" value={fmtPct(groupDetail.recall)} term="recall" />
                <MetricCard label="F1" value={fmtPct(groupDetail.f1)} term="f1" />
                <MetricCard label="3D IoU" value={fmtNum(groupDetail.mean_iou_3d)} term="iou_3d" />
                <MetricCard label="Consensus" value={fmtPct(groupDetail.mean_consensus)} term="grader_consensus" />
                <MetricCard label="Anomaly" value={fmtNum(groupDetail.mean_anomaly_score)} term="anomaly_score" />
                <MetricCard label="Tracking quality" value={fmtNum(groupDetail.tracking_quality)} term="track_quality" />
              </Box>
              {failureRows.length > 0 ? (
                <>
                  <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>
                    FAILURE REASON BREAKDOWN
                  </Typography>
                  <Box sx={{ mt: 0.5, maxWidth: 520 }}>
                    {failureRows.map(([reason, count]) => (
                      <HBar
                        key={reason}
                        label={reason.replace(/_/g, ' ')}
                        term={glossaryKeyForStatus(reason) ?? undefined}
                        value={count}
                        max={maxFailure}
                        color="#ef5350"
                      />
                    ))}
                  </Box>
                </>
              ) : (
                <Typography variant="body2" sx={{ color: '#8a949e' }}>
                  No failure reasons recorded for this group.
                </Typography>
              )}
            </Box>
          ) : null}
        </SectionCard>
      ) : (
        <Typography variant="body2" sx={{ color: '#8a949e' }}>
          Select a dataset above to drill into its quality groups.
        </Typography>
      )}
    </Box>
  );
}
