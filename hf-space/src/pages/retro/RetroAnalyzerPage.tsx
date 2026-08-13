/**
 * Retrospective Analyzer — agentic failure retrospectives with a traceable
 * evidence chain, deterministic policy gate, and full tool-call audit trail.
 */
import { useCallback, useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Select from '@mui/material/Select';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import { History, Play, ShieldCheck } from 'lucide-react';
import AuditTrail from '../../components/retro/AuditTrail';
import EvidenceChain from '../../components/retro/EvidenceChain';
import HardwareCard from '../../components/retro/HardwareCard';
import ScorecardView from '../../components/retro/ScorecardView';
import StandardsPanel from '../../components/retro/StandardsPanel';
import {
  analyzeFixture,
  getRetroAnalysis,
  getRetroAudit,
  getRetroBackends,
  getRetroFixtures,
  listRetroAnalyses,
} from '../../services/retro';
import type {
  AnalysisSummary,
  AuditRecord,
  BackendStatus,
  FixtureInfo,
  RetrospectiveScorecard,
} from '../../types/retro';

export default function RetroAnalyzerPage() {
  const [fixtures, setFixtures] = useState<FixtureInfo[]>([]);
  const [backends, setBackends] = useState<BackendStatus[]>([]);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [fixtureId, setFixtureId] = useState('');
  const [backend, setBackend] = useState('mock');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scorecard, setScorecard] = useState<RetrospectiveScorecard | null>(null);
  const [audit, setAudit] = useState<AuditRecord[]>([]);
  const [tab, setTab] = useState(0);

  const refreshHistory = useCallback(() => {
    listRetroAnalyses().then((r) => setHistory(r.analyses)).catch(() => undefined);
  }, []);

  useEffect(() => {
    getRetroFixtures()
      .then((r) => {
        setFixtures(r.fixtures);
        if (r.fixtures.length) setFixtureId(r.fixtures[0].fixture_id);
      })
      .catch((e: Error) => setError(e.message));
    getRetroBackends().then((r) => setBackends(r.backends)).catch(() => undefined);
    refreshHistory();
  }, [refreshHistory]);

  const runAnalysis = () => {
    if (!fixtureId) return;
    setRunning(true);
    setError(null);
    analyzeFixture(fixtureId, backend)
      .then((res) => {
        setScorecard(res.scorecard);
        setTab(0);
        refreshHistory();
        return getRetroAudit(res.scorecard.evaluation_id).then((a) => setAudit(a.records));
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setRunning(false));
  };

  const loadHistorical = (evaluationId: string) => {
    setError(null);
    getRetroAnalysis(evaluationId)
      .then((res) => {
        setScorecard(res.scorecard);
        setTab(0);
        return getRetroAudit(evaluationId).then((a) => setAudit(a.records));
      })
      .catch((e: Error) => setError(e.message));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ShieldCheck size={20} color="#4fc3f7" />
        <Typography variant="h6" sx={{ fontWeight: 700 }}>Retrospective Analyzer</Typography>
        <Typography sx={{ fontSize: 12, color: '#8a949e' }}>
          LLM interprets & hypothesizes · deterministic code computes metrics & owns the safety boundary
        </Typography>
      </Box>

      <HardwareCard />

      {/* picker + analyze action */}
      <Paper sx={{ p: 1.5, bgcolor: '#161c23', border: '1px solid #232a31',
                   display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <Box>
          <Typography sx={{ fontSize: 10.5, color: '#8a949e' }}>FAILURE FIXTURE</Typography>
          <Select size="small" value={fixtureId} onChange={(e) => setFixtureId(e.target.value)}
                  sx={{ minWidth: 340, fontSize: 13 }}>
            {fixtures.map((f) => (
              <MenuItem key={f.fixture_id} value={f.fixture_id} sx={{ fontSize: 13 }}>
                {f.fixture_id} — {f.description.slice(0, 60)}
              </MenuItem>
            ))}
          </Select>
        </Box>
        <Box>
          <Typography sx={{ fontSize: 10.5, color: '#8a949e' }}>INFERENCE BACKEND</Typography>
          <Select size="small" value={backend} onChange={(e) => setBackend(e.target.value)}
                  sx={{ minWidth: 260, fontSize: 13 }}>
            {backends.map((b) => (
              <MenuItem key={b.backend} value={b.backend} disabled={!b.available} sx={{ fontSize: 13 }}>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  {b.backend}
                  <Chip size="small" label={b.available ? 'AVAILABLE' : 'UNAVAILABLE'}
                        sx={{ height: 16, fontSize: 9, fontWeight: 700,
                              bgcolor: b.available ? '#66bb6a22' : '#ef535022',
                              color: b.available ? '#66bb6a' : '#ef5350' }} />
                  <Typography sx={{ fontSize: 10.5, color: '#8a949e' }}>{b.detail.slice(0, 48)}</Typography>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </Box>
        <Button variant="contained" onClick={runAnalysis} disabled={running || !fixtureId}
                startIcon={running ? <CircularProgress size={14} /> : <Play size={14} />}
                sx={{ mt: 1.5, fontWeight: 700 }}>
          {running ? 'Analyzing…' : 'Analyze failure'}
        </Button>
        {history.length ? (
          <Box sx={{ ml: 'auto' }}>
            <Typography sx={{ fontSize: 10.5, color: '#8a949e', display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <History size={12} /> PRIOR ANALYSES
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', maxWidth: 420 }}>
              {history.slice(0, 6).map((h) => (
                <Chip key={h.evaluation_id} size="small" onClick={() => loadHistorical(h.evaluation_id)}
                      label={`${h.evaluation_id.slice(0, 24)} · ${h.severity ?? '?'}`}
                      sx={{ fontSize: 10, cursor: 'pointer', bgcolor: '#232a31' }} />
              ))}
            </Box>
          </Box>
        ) : null}
      </Paper>

      {error ? (
        <Paper sx={{ p: 1.5, bgcolor: '#ef535014', border: '1px solid #ef535055' }}>
          <Typography sx={{ fontSize: 12.5, color: '#ef5350' }}>{error}</Typography>
        </Paper>
      ) : null}

      {scorecard ? (
        <Paper sx={{ p: 2, bgcolor: '#11161c', border: '1px solid #232a31' }}>
          <Tabs value={tab} onChange={(_, v: number) => setTab(v)} sx={{ mb: 2, minHeight: 36 }}>
            <Tab label="Evidence chain" sx={{ fontSize: 12, minHeight: 36 }} />
            <Tab label="Scorecard" sx={{ fontSize: 12, minHeight: 36 }} />
            <Tab label={`Retrieved standards (${scorecard.retrieved_standards.length})`}
                 sx={{ fontSize: 12, minHeight: 36 }} />
            <Tab label={`Audit trail (${audit.length})`} sx={{ fontSize: 12, minHeight: 36 }} />
          </Tabs>
          {tab === 0 ? <EvidenceChain scorecard={scorecard} /> : null}
          {tab === 1 ? <ScorecardView scorecard={scorecard} /> : null}
          {tab === 2 ? <StandardsPanel standards={scorecard.retrieved_standards} /> : null}
          {tab === 3 ? <AuditTrail records={audit} /> : null}
        </Paper>
      ) : (
        <Paper sx={{ p: 3, bgcolor: '#11161c', border: '1px dashed #232a31', textAlign: 'center' }}>
          <Typography sx={{ fontSize: 13, color: '#8a949e' }}>
            Pick a failure fixture and run an analysis. The result renders as a traceable chain:
            RAW FAILURE → OBSERVED → DERIVED → RETRIEVED → HYPOTHESIS → BEHAVIORAL → POLICY → SCORECARD → HUMAN DECISION.
          </Typography>
        </Paper>
      )}
    </Box>
  );
}
