/** Design Lab: challenge brief + pragmatic add-node/connect builder + rationale fields -> graded results. */

import { useMemo, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { Plus, Trash2 } from 'lucide-react';
import { ErrorNote, HBar, LoadingBox, SectionCard } from '../../components/labeleval/shared';
import { PassChip, ScoreChip, useAsync } from '../../components/hillclimb/shared';
import { getDesignChallenges, submitDesign } from '../../services/hillclimb';
import type { DesignComponent, DesignEdge, DesignGrade } from '../../types/hillclimb';

const TYPE_COLORS: Record<string, string> = {
  source: '#8d6e63', ingestion: '#4fc3f7', stream: '#29b6f6', batch: '#0288d1',
  storage: '#9575cd', feature: '#7e57c2', training: '#ffb74d', inference: '#ffa726',
  eval: '#66bb6a', monitoring: '#26a69a', feedback: '#ec407a',
};

export default function DesignLabView() {
  const catalog = useAsync(getDesignChallenges);
  const [challengeId, setChallengeId] = useState('petabyte_ingestion');
  const [components, setComponents] = useState<DesignComponent[]>([]);
  const [edges, setEdges] = useState<DesignEdge[]>([]);
  const [rationales, setRationales] = useState<Record<string, string>>({});
  const [newType, setNewType] = useState('source');
  const [newName, setNewName] = useState('');
  const [newNote, setNewNote] = useState('');
  const [edgeFrom, setEdgeFrom] = useState('');
  const [edgeTo, setEdgeTo] = useState('');
  const [grade, setGrade] = useState<DesignGrade | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const challenge = useMemo(
    () => catalog.data?.challenges.find((c) => c.challenge_id === challengeId) ?? null,
    [catalog.data, challengeId]
  );

  const reset = (cid: string) => {
    setChallengeId(cid);
    setComponents([]);
    setEdges([]);
    setRationales({});
    setGrade(null);
  };

  const addComponent = () => {
    const id = `c${components.length + 1}`;
    setComponents([...components, { id, type: newType, name: newName || `${newType}-${id}`, note: newNote }]);
    setNewName('');
    setNewNote('');
  };

  const removeComponent = (id: string) => {
    setComponents(components.filter((c) => c.id !== id));
    setEdges(edges.filter((e) => e.source !== id && e.target !== id));
  };

  const addEdge = () => {
    if (!edgeFrom || !edgeTo || edgeFrom === edgeTo) return;
    if (edges.some((e) => e.source === edgeFrom && e.target === edgeTo)) return;
    setEdges([...edges, { source: edgeFrom, target: edgeTo }]);
  };

  const submit = () => {
    setBusy(true);
    setError(null);
    submitDesign(challengeId, components, edges, rationales)
      .then(setGrade)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  if (catalog.loading && !catalog.data) return <LoadingBox label="Loading challenges…" />;
  if (catalog.error && !catalog.data) return <ErrorNote error={catalog.error} />;

  const nameOf = (id: string) => components.find((c) => c.id === id)?.name ?? id;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <SectionCard
        title="Challenge"
        action={
          <TextField select size="small" value={challengeId} onChange={(e) => reset(e.target.value)} sx={{ minWidth: 280 }}>
            {(catalog.data?.challenges ?? []).map((c) => (
              <MenuItem key={c.challenge_id} value={c.challenge_id}>
                {c.title}
              </MenuItem>
            ))}
          </TextField>
        }
        help="Submit a typed component graph plus a written rationale per key decision. The rule-based grader checks required stages, orphans, feedback-loop closure, single points of failure, capacity math, and rationale coverage of scalability / reliability / latency / cost / observability / failure handling / tradeoffs."
      >
        {challenge ? (
          <>
            <Typography variant="body2" sx={{ color: '#e6e9ec', lineHeight: 1.65, mb: 1 }}>
              {challenge.brief}
            </Typography>
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              Required stages:{' '}
              {challenge.required_stages.map((g) => g.join(' or ')).join(' · ')}
              {challenge.requires_feedback_loop ? ' · must close a feedback loop' : ''}
            </Typography>
          </>
        ) : null}
      </SectionCard>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <SectionCard title={`Components (${components.length})`} sx={{ flex: '1 1 340px' }}>
          <Box sx={{ display: 'flex', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
            <TextField select size="small" label="Type" value={newType} onChange={(e) => setNewType(e.target.value)} sx={{ width: 130 }}>
              {(catalog.data?.component_types ?? []).map((t) => (
                <MenuItem key={t} value={t}>{t}</MenuItem>
              ))}
            </TextField>
            <TextField size="small" label="Name" value={newName} onChange={(e) => setNewName(e.target.value)} sx={{ width: 150 }} />
            <TextField size="small" label="Capacity note (e.g. 2 PB/week)" value={newNote} onChange={(e) => setNewNote(e.target.value)} sx={{ flex: 1, minWidth: 170 }} />
            <Button size="small" variant="outlined" startIcon={<Plus size={14} />} onClick={addComponent}>
              Add
            </Button>
          </Box>
          {components.map((c) => (
            <Box key={c.id} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Chip size="small" label={c.type} sx={{ bgcolor: TYPE_COLORS[c.type] ?? '#37474f', color: '#101418', fontWeight: 700, fontSize: 10.5, width: 88 }} />
              <Typography variant="body2" sx={{ flex: 1 }}>
                {c.name}
                {c.note ? (
                  <Typography component="span" variant="caption" sx={{ color: '#8a949e', ml: 1 }}>
                    {c.note}
                  </Typography>
                ) : null}
              </Typography>
              <IconButton size="small" onClick={() => removeComponent(c.id)}>
                <Trash2 size={14} />
              </IconButton>
            </Box>
          ))}
        </SectionCard>

        <SectionCard title={`Connections (${edges.length})`} sx={{ flex: '1 1 300px' }}>
          <Box sx={{ display: 'flex', gap: 1, mb: 1.5 }}>
            <TextField select size="small" label="From" value={edgeFrom} onChange={(e) => setEdgeFrom(e.target.value)} sx={{ flex: 1 }}>
              {components.map((c) => (
                <MenuItem key={c.id} value={c.id}>{c.name}</MenuItem>
              ))}
            </TextField>
            <TextField select size="small" label="To" value={edgeTo} onChange={(e) => setEdgeTo(e.target.value)} sx={{ flex: 1 }}>
              {components.map((c) => (
                <MenuItem key={c.id} value={c.id}>{c.name}</MenuItem>
              ))}
            </TextField>
            <Button size="small" variant="outlined" onClick={addEdge}>
              Connect
            </Button>
          </Box>
          {edges.map((e, i) => (
            <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.4 }}>
              <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#aab4be', flex: 1 }}>
                {nameOf(e.source)} → {nameOf(e.target)}
              </Typography>
              <IconButton size="small" onClick={() => setEdges(edges.filter((_x, j) => j !== i))}>
                <Trash2 size={13} />
              </IconButton>
            </Box>
          ))}
        </SectionCard>
      </Box>

      <SectionCard title="Rationale per key decision" help="Each decision needs written reasoning. The grader scans for capacity math and coverage of the seven design dimensions — hand-waving scores zero.">
        {(challenge?.key_decisions ?? []).map((d) => (
          <TextField
            key={d}
            label={d}
            multiline
            minRows={2}
            fullWidth
            sx={{ mb: 1.5 }}
            value={rationales[d] ?? ''}
            onChange={(e) => setRationales({ ...rationales, [d]: e.target.value })}
          />
        ))}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button variant="contained" onClick={submit} disabled={busy || components.length === 0}>
            {busy ? 'Grading…' : 'Submit for grading'}
          </Button>
        </Box>
        {error ? <ErrorNote error={error} /> : null}
      </SectionCard>

      {grade ? (
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <SectionCard
            title={
              <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
                Structural checks <ScoreChip score={grade.overall_score} label="Overall" />
              </Box>
            }
            sx={{ flex: '1 1 320px' }}
          >
            <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mb: 1.5 }}>
              <PassChip passed={grade.structural.missing_stages.length === 0} label="required stages" />
              <PassChip passed={grade.structural.orphan_components.length === 0} label="no orphans" />
              <PassChip
                passed={!grade.structural.feedback_loop_required || grade.structural.feedback_loop_closed}
                label="feedback loop"
              />
              <PassChip passed={grade.structural.single_points_of_failure.length === 0} label="no SPOF" />
              <PassChip passed={grade.structural.capacity_math_found} label="capacity math" />
            </Box>
            {grade.gaps.map((g, i) => (
              <Typography key={i} variant="caption" sx={{ color: '#ef9a9a', display: 'block', mb: 0.5 }}>
                • {g}
              </Typography>
            ))}
            {grade.gaps.length === 0 ? (
              <Typography variant="body2" sx={{ color: '#9ccc65' }}>
                No structural gaps found.
              </Typography>
            ) : null}
          </SectionCard>

          <SectionCard title="Per-dimension scores" sx={{ flex: '1 1 340px' }}>
            {grade.dimension_grades.map((d) => (
              <HBar
                key={d.dimension}
                label={d.dimension.replace(/_/g, ' ')}
                value={d.score}
                max={5}
                color={d.score >= 3 ? '#66bb6a' : d.score >= 2 ? '#f9a825' : '#ef5350'}
                valueLabel={`${d.score}/5`}
              />
            ))}
            {grade.evidence_id ? (
              <Alert severity="success" variant="outlined" sx={{ mt: 1, py: 0.25 }}>
                Saved to your evidence library.
              </Alert>
            ) : null}
          </SectionCard>
        </Box>
      ) : null}
    </Box>
  );
}
