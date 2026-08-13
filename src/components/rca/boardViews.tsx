/** Stage 11 views: the 8-hypothesis Root Cause Board (auto + human
 * assessments, evidence links that jump to stage findings) and the decision
 * tree with the live path highlighted. */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { ChevronDown, ChevronRight, Link2 } from 'lucide-react';
import type { Confidence, DecisionTree, EvidenceLink, RootCause, Scoreboard } from '../../types/rca';
import { BORDER, CONFIDENCE_COLORS, Explainer, SectionCard, SEVERITY_COLORS, tableSx } from './common';

function ConfidenceChip({ c }: { c: Confidence | null }) {
  if (!c) return <Typography variant="caption" sx={{ color: '#5c666f' }}>—</Typography>;
  return (
    <Chip size="small" label={c} sx={{
      height: 18, fontSize: 10, fontWeight: 800,
      bgcolor: `${CONFIDENCE_COLORS[c]}22`, color: CONFIDENCE_COLORS[c],
      border: `1px solid ${CONFIDENCE_COLORS[c]}55`,
    }} />
  );
}

function EvidenceList({ items, sign, onJump }: {
  items: EvidenceLink[];
  sign: '+' | '−';
  onJump: (stage: string) => void;
}) {
  if (!items.length) return <Typography variant="caption" sx={{ color: '#5c666f' }}>none</Typography>;
  return (
    <Box>
      {items.map((e) => (
        <Box key={`${e.code}-${e.finding_id}`} sx={{ display: 'flex', alignItems: 'center', gap: 0.5, py: 0.15 }}>
          <Typography variant="caption" sx={{ fontFamily: 'monospace', color: sign === '+' ? '#a5d6a7' : '#ef9a9a', minWidth: 34 }}>
            {sign}{e.weight.toFixed(1)}
          </Typography>
          <Tooltip title={`[${e.code}] from stage "${e.stage}" — click to open that stage`} arrow>
            <Chip
              size="small"
              icon={<Link2 size={11} />}
              label={e.title}
              onClick={() => onJump(e.stage)}
              sx={{
                height: 19, fontSize: 10.5, maxWidth: 340,
                bgcolor: '#232a31',
                color: e.severity === 'CRITICAL' ? SEVERITY_COLORS.CRITICAL : '#cfd8e0',
                '.MuiChip-icon': { color: '#8a949e', ml: 0.5 },
              }}
            />
          </Tooltip>
        </Box>
      ))}
    </Box>
  );
}

export function ScoreBoardView({ board, onJumpToStage, onAssess }: {
  board: Scoreboard;
  onJumpToStage: (stageKey: string) => void;
  onAssess: (hypothesis: RootCause, confidence: Confidence, note: string) => void;
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({ [board.rows[0]?.hypothesis]: true });
  const [notes, setNotes] = useState<Record<string, string>>({});
  return (
    <>
      <Explainer text={board.explainer} />
      <SectionCard title="Root Cause Board — 8 hypotheses, scored from the evidence">
        <Box component="table" sx={tableSx}>
          <thead>
            <tr>
              <th /><th>#</th><th>Hypothesis</th><th>Score</th>
              <th>Auto confidence</th><th>Your assessment</th><th>Next discriminating test</th>
            </tr>
          </thead>
          <tbody>
            {board.rows.map((r) => (
              <>
                <tr key={r.hypothesis} style={r.rank === 1 ? { background: '#4fc3f70d' } : undefined}>
                  <td>
                    <IconButton size="small" onClick={() => setOpen((o) => ({ ...o, [r.hypothesis]: !o[r.hypothesis] }))}>
                      {open[r.hypothesis] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </IconButton>
                  </td>
                  <td style={{ fontWeight: 800 }}>{r.rank}</td>
                  <td>
                    <Typography variant="body2" sx={{ fontWeight: r.rank === 1 ? 800 : 600, fontSize: 12.5 }}>
                      {r.hypothesis}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#8a949e' }}>{r.label}</Typography>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontWeight: 700, color: r.score > 0 ? '#a5d6a7' : '#8a949e' }}>
                    {r.score.toFixed(1)}
                  </td>
                  <td><ConfidenceChip c={r.auto_confidence} /></td>
                  <td>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Select
                        size="small"
                        value={r.human_confidence ?? ''}
                        displayEmpty
                        onChange={(e) => onAssess(r.hypothesis, e.target.value as Confidence, notes[r.hypothesis] ?? r.human_note)}
                        sx={{ fontSize: 11, height: 26, minWidth: 104, '.MuiSelect-select': { py: 0.25 } }}
                      >
                        <MenuItem value="" disabled><em>set…</em></MenuItem>
                        {(['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'] as Confidence[]).map((c) => (
                          <MenuItem key={c} value={c} sx={{ fontSize: 12 }}>{c}</MenuItem>
                        ))}
                      </Select>
                      {r.human_confidence ? <ConfidenceChip c={r.human_confidence} /> : null}
                    </Box>
                  </td>
                  <td>
                    <Typography variant="caption" sx={{ color: '#aab4be', display: 'block', maxWidth: 320 }}>
                      {r.next_discriminating_test}
                    </Typography>
                  </td>
                </tr>
                {open[r.hypothesis] ? (
                  <tr key={`${r.hypothesis}-detail`}>
                    <td />
                    <td colSpan={6}>
                      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2, py: 0.5 }}>
                        <Box>
                          <Typography variant="caption" sx={{ fontWeight: 800, color: '#a5d6a7' }}>EVIDENCE FOR</Typography>
                          <EvidenceList items={r.evidence_for} sign="+" onJump={onJumpToStage} />
                        </Box>
                        <Box>
                          <Typography variant="caption" sx={{ fontWeight: 800, color: '#ef9a9a' }}>EVIDENCE AGAINST</Typography>
                          <EvidenceList items={r.evidence_against} sign="−" onJump={onJumpToStage} />
                        </Box>
                        <Box>
                          <Typography variant="caption" sx={{ fontWeight: 800, color: '#8a949e' }}>YOUR NOTE</Typography>
                          <TextField
                            size="small"
                            fullWidth
                            multiline
                            minRows={2}
                            placeholder="Why do you agree/disagree with the auto assessment?"
                            defaultValue={r.human_note}
                            onChange={(e) => setNotes((n) => ({ ...n, [r.hypothesis]: e.target.value }))}
                            onBlur={(e) => {
                              if (r.human_confidence && e.target.value !== r.human_note) {
                                onAssess(r.hypothesis, r.human_confidence, e.target.value);
                              }
                            }}
                            sx={{ '.MuiInputBase-input': { fontSize: 11.5 } }}
                          />
                        </Box>
                      </Box>
                    </td>
                  </tr>
                ) : null}
              </>
            ))}
          </tbody>
        </Box>
      </SectionCard>
    </>
  );
}

// --------------------------------------------------------------- decision tree

const ANSWER_COLORS = { yes: '#66bb6a', no: '#ef5350', unknown: '#ffb74d' };

export function DecisionTreeView({ tree }: { tree: DecisionTree }) {
  const onPath = new Set(tree.path);
  const terminal = tree.path[tree.path.length - 1];
  return (
    <>
      <Explainer text={tree.explainer} />
      <SectionCard title="Decision chain: measurement validity → distribution → parity → significance">
        <Box sx={{ display: 'flex', flexDirection: 'column' }}>
          {tree.nodes.map((n, i) => {
            const active = onPath.has(n.id);
            const isTerminal = n.id === terminal;
            const stoppedHere = isTerminal && (n.answer === 'no' || n.id === 'significant');
            return (
              <Box key={n.id} sx={{ display: 'flex', gap: 1.5, opacity: active ? 1 : 0.42 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 22 }}>
                  <Box sx={{
                    width: 14, height: 14, borderRadius: '50%', mt: 0.75, flexShrink: 0,
                    bgcolor: ANSWER_COLORS[n.answer],
                    boxShadow: active ? `0 0 8px ${ANSWER_COLORS[n.answer]}88` : 'none',
                  }} />
                  {i < tree.nodes.length - 1 ? (
                    <Box sx={{ width: 2, flex: 1, minHeight: 26, bgcolor: active && !stoppedHere ? '#4fc3f7' : BORDER }} />
                  ) : null}
                </Box>
                <Box sx={{ pb: 2, minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: active ? 700 : 500, fontSize: 12.5 }}>
                    {n.question}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', mt: 0.25 }}>
                    <Chip size="small" label={n.answer.toUpperCase()} sx={{
                      height: 18, fontSize: 10, fontWeight: 800,
                      bgcolor: `${ANSWER_COLORS[n.answer]}22`, color: ANSWER_COLORS[n.answer],
                    }} />
                    {n.basis.map((b) => (
                      <Typography key={b} variant="caption" sx={{ fontFamily: 'monospace', color: '#8a949e', fontSize: 10 }}>
                        {b}
                      </Typography>
                    ))}
                    {n.answer === 'no' && n.conclusion_if_no && active ? (
                      <Chip size="small" label={`→ ${n.conclusion_if_no}`} sx={{ height: 18, fontSize: 10, fontWeight: 800, bgcolor: '#ef535022', color: '#ef9a9a' }} />
                    ) : null}
                  </Box>
                </Box>
              </Box>
            );
          })}
        </Box>
        <Box sx={{ mt: 1, p: 1.25, borderRadius: 1, border: `1px solid ${BORDER}`, bgcolor: '#0d1116' }}>
          <Typography variant="caption" sx={{ color: '#8a949e', fontWeight: 700 }}>CHAIN CONCLUSION</Typography>
          <Typography variant="body2" sx={{ fontWeight: 800, color: tree.conclusion_kind === 'insufficient_evidence' ? '#ffb74d' : '#4fc3f7' }}>
            {tree.conclusion_kind === 'insufficient_evidence'
              ? `INSUFFICIENT EVIDENCE (leaning ${tree.conclusion})`
              : tree.conclusion ?? '—'}
          </Typography>
        </Box>
      </SectionCard>
    </>
  );
}
