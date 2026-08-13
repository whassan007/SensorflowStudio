/**
 * Rare-Event Miner: multimodal mining + perception QA for costumed
 * pedestrians. Four views: mining run panel, candidate review queue,
 * track view, and coverage & curator-quality statistics.
 */
import { useCallback, useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import MenuItem from '@mui/material/MenuItem';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { Pickaxe, RefreshCcw } from 'lucide-react';
import type { StatusResponse, TrackCandidateSummary, TrackView } from '../../types/raremine';
import {
  generateScenes,
  getImprovementReport,
  getStatus,
  getTracks,
  listCandidates,
  runMining,
} from '../../services/raremine';
import CandidateCard from '../../components/raremine/CandidateCard';
import CoverageQualityTab from '../../components/raremine/CoverageQualityTab';
import TrackStrip from '../../components/raremine/TrackStrip';
import { Explainer, StatChip, PriorityChip } from '../../components/raremine/shared';

const PRIORITIES = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
const COSTUMES = ['mascot', 'inflatable', 'animal', 'character', 'robot_armor', 'oversized', 'large_prop'];
const DIFFICULTIES = ['EXTREME', 'HARD', 'MODERATE', 'EASY'];

export default function RareMinePage() {
  const [tab, setTab] = useState(0);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [candidates, setCandidates] = useState<TrackCandidateSummary[]>([]);
  const [tracks, setTracks] = useState<TrackView[]>([]);
  const [nScenes, setNScenes] = useState(60);
  const [seed, setSeed] = useState(7);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [fPriority, setFPriority] = useState('');
  const [fCostume, setFCostume] = useState('');
  const [fDifficulty, setFDifficulty] = useState('');

  const refresh = useCallback(() => {
    getStatus().then(setStatus).catch(() => setStatus(null));
    listCandidates({
      priority: fPriority || undefined,
      costume: fCostume || undefined,
      difficulty: fDifficulty || undefined,
    })
      .then((r) => setCandidates(r.candidates))
      .catch(() => setCandidates([]));
    getTracks()
      .then((r) => setTracks(r.tracks))
      .catch(() => setTracks([]));
    setRefreshKey((k) => k + 1);
  }, [fPriority, fCostume, fDifficulty]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const mineAll = () =>
    run('mine', async () => {
      await generateScenes(nScenes, seed);
      await runMining();
    });

  const remineImproved = () =>
    run('remine', async () => {
      const imp = await getImprovementReport();
      await runMining(12, imp.next_run_config as Record<string, unknown>);
    });

  const hist = status?.priority_histogram ?? {};

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {/* ---- mining run panel */}
      <Box sx={{ p: 1.5, bgcolor: '#141a20', border: '1px solid #232a31', borderRadius: 1 }}>
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField
            size="small"
            label="scenes (n)"
            type="number"
            value={nScenes}
            onChange={(e) => setNScenes(Number(e.target.value) || 60)}
            sx={{ width: 110 }}
          />
          <TextField
            size="small"
            label="seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 7)}
            sx={{ width: 90 }}
          />
          <Button
            variant="contained"
            size="small"
            disabled={busy !== null}
            startIcon={busy === 'mine' ? <CircularProgress size={14} /> : <Pickaxe size={15} />}
            onClick={mineAll}
          >
            Generate bank + run miner
          </Button>
          <Button
            variant="outlined"
            size="small"
            disabled={busy !== null || !status?.last_run}
            startIcon={busy === 'remine' ? <CircularProgress size={14} /> : <RefreshCcw size={14} />}
            onClick={remineImproved}
          >
            Re-mine with improvements
          </Button>
          {error ? (
            <Typography variant="caption" sx={{ color: '#ff8a80' }}>
              {error}
            </Typography>
          ) : null}
        </Box>
        {status?.bank ? (
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mt: 1.25, alignItems: 'center' }}>
            <StatChip value={status.bank.num_scenes} label="scenes" />
            <StatChip value={status.bank.num_sequences} label="sequences" />
            <StatChip value={status.last_run?.num_frame_candidates ?? 0} label="frame proposals" />
            <StatChip value={status.num_detected} label="track candidates" />
            <StatChip value={status.dedup_report?.dedup_savings ?? 0} label="duplicates removed" color="#66bb6a" />
            <StatChip value={status.last_run?.num_diversity_selected ?? 0} label="diversity-selected" />
            <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', ml: 1 }}>
              {PRIORITIES.map((p) =>
                hist[p] ? (
                  <Box key={p} sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                    <PriorityChip value={p} />
                    <Typography variant="caption" sx={{ color: '#c3ccd5', fontWeight: 700 }}>
                      {hist[p]}
                    </Typography>
                  </Box>
                ) : null
              )}
            </Box>
          </Box>
        ) : (
          <Typography variant="caption" sx={{ color: '#5c6a76', display: 'block', mt: 1 }}>
            No scene bank yet — generate one to start mining.
          </Typography>
        )}
      </Box>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ borderBottom: '1px solid #232a31', minHeight: 38 }}>
        <Tab label={`Review queue (${candidates.length})`} sx={{ minHeight: 38, textTransform: 'none' }} />
        <Tab label={`Tracks (${tracks.length})`} sx={{ minHeight: 38, textTransform: 'none' }} />
        <Tab label="Coverage & curator quality" sx={{ minHeight: 38, textTransform: 'none' }} />
      </Tabs>

      {tab === 0 ? (
        <Box>
          <Explainer>
            Every card is a PROPOSAL from the rule-based miner — not ground truth. The miner only claims what its
            available sensors support: three separate confidences (a human is present / a costume is present / this is
            a rare event), modality-tagged evidence, and alternative hypotheses it could not rule out. Automated
            validation has already measured each proposal against ground truth where available; your approve/reject is
            the human-validation stage. Approving routes the example to a dataset destination with full lineage —
            training eligibility stays governed separately.
          </Explainer>
          <Box sx={{ display: 'flex', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
            <TextField size="small" select label="priority" value={fPriority} onChange={(e) => setFPriority(e.target.value)} sx={{ width: 140 }}>
              <MenuItem value="">all</MenuItem>
              {PRIORITIES.map((p) => (
                <MenuItem key={p} value={p}>
                  {p}
                </MenuItem>
              ))}
            </TextField>
            <TextField size="small" select label="costume" value={fCostume} onChange={(e) => setFCostume(e.target.value)} sx={{ width: 150 }}>
              <MenuItem value="">all</MenuItem>
              {COSTUMES.map((p) => (
                <MenuItem key={p} value={p}>
                  {p}
                </MenuItem>
              ))}
            </TextField>
            <TextField size="small" select label="difficulty" value={fDifficulty} onChange={(e) => setFDifficulty(e.target.value)} sx={{ width: 140 }}>
              <MenuItem value="">all</MenuItem>
              {DIFFICULTIES.map((p) => (
                <MenuItem key={p} value={p}>
                  {p}
                </MenuItem>
              ))}
            </TextField>
          </Box>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            {candidates.map((c) => (
              <CandidateCard key={c.track_candidate_id} summary={c} onChanged={refresh} />
            ))}
            {!candidates.length ? (
              <Typography variant="body2" sx={{ color: '#5c6a76' }}>
                No candidates match — run mining or clear filters.
              </Typography>
            ) : null}
          </Box>
        </Box>
      ) : null}

      {tab === 1 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Explainer>
            Multi-frame sightings of one physical event consolidate into ONE track-level candidate — a 20-frame mascot
            walk is one rare event, not twenty. Each strip shows per-frame difficulty and marks the representative
            frames (best evidence, worst case, observed model failure): the minimal set worth curating.
          </Explainer>
          {tracks.map((t) => (
            <TrackStrip key={t.track_candidate_id} track={t} />
          ))}
          {!tracks.length ? (
            <Typography variant="body2" sx={{ color: '#5c6a76' }}>
              No multi-frame tracks yet — run mining first.
            </Typography>
          ) : null}
        </Box>
      ) : null}

      {tab === 2 ? <CoverageQualityTab refreshKey={refreshKey} /> : null}
    </Box>
  );
}
