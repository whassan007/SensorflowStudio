/**
 * Safety Evidence Package viewer — rendered markdown from
 * /api/safety/evidence/{run}?format=markdown, JSON download, and a prominent
 * disclaimer about the simulated provenance of the underlying data.
 */
import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import { Download, ShieldCheck } from 'lucide-react';
import { getEvidenceJson, getEvidenceMarkdown } from '../../services/safety';
import { MarkdownLite, RunSelect, downloadJson, usePublishedRuns } from '../../components/safety/shared';
import { IllustratedEmpty, PanelSkeleton } from '../../components/visual/Feedback';
import { ErrorNote, SectionCard } from '../../components/labeleval/shared';
import { useLabelEval } from '../../context/LabelEvalContext';
import { tokens } from '../../theme';

export default function EvidencePage() {
  const { entityId, navigate } = useLabelEval();
  const { runs, error: runsError } = usePublishedRuns();
  const [runId, setRunId] = useState<string | null>(entityId);
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId && runs?.length) setRunId(entityId ?? runs[0].run_id);
  }, [runs, runId, entityId]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    setLoading(true);
    setMissing(false);
    setError(null);
    setMarkdown(null);
    getEvidenceMarkdown(runId)
      .then((r) => {
        if (!cancelled) setMarkdown(r.markdown);
      })
      .catch((e) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        if (msg.startsWith('API 404')) setMissing(true);
        else setError(msg);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const download = () => {
    if (!runId) return;
    getEvidenceJson(runId)
      .then((pkg) => downloadJson(`safety-evidence-${runId}.json`, pkg))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {runsError ? <ErrorNote error={runsError} /> : null}
      {error ? <ErrorNote error={error} /> : null}

      <Alert severity="warning" variant="outlined" icon={<ShieldCheck size={20} />} sx={{ borderWidth: 2 }}>
        <AlertTitle sx={{ fontWeight: 800 }}>Demonstration artifact — not a certification document</AlertTitle>
        This Safety Evidence Package is generated from <strong>synthetic evaluation data</strong> inside Sensorflow
        Studio. Its structure follows UL 4600 / ISO 26262-style evidence practices (claims backed by measurements,
        versioned policy, reproducible lineage), but nothing here has been reviewed by an assessor and it must not be
        used to support a real deployment decision.
      </Alert>

      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <RunSelect label="Candidate run" value={runId} onChange={(id) => setRunId(id)} runs={runs} />
        <Button variant="outlined" size="small" startIcon={<Download size={15} />} disabled={!markdown} onClick={download}>
          Download JSON
        </Button>
        <Typography variant="caption" sx={{ color: tokens.color.neutral }}>
          Packages are produced by the Release Gates evaluation; the markdown below is rendered from the same data as
          the JSON.
        </Typography>
      </Box>

      <SectionCard title="Safety Evidence Package">
        {loading ? <PanelSkeleton rows={8} /> : null}
        {missing && !loading ? (
          <IllustratedEmpty
            art="search"
            title="No evidence package for this run"
            message="Evidence packages are created when the release gates are evaluated for a candidate run. Run the gate evaluation first, then the package appears here."
            action={
              <Button variant="contained" size="small" onClick={() => navigate('safety-gates')}>
                Go to Release Gates
              </Button>
            }
          />
        ) : null}
        {markdown ? (
          <Box sx={{ maxWidth: 980, px: 1 }}>
            <MarkdownLite text={markdown} />
          </Box>
        ) : null}
      </SectionCard>
    </Box>
  );
}
