/**
 * Hill Climbing EM — section shell with competency-first internal navigation:
 * Assess → Diagnose → Practice → Apply → Evaluate → Improve (not an LMS).
 */

import { useCallback, useState } from 'react';
import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import type { Exercise } from '../../types/hillclimb';
import DashboardView from './DashboardView';
import DiagnosticView from './DiagnosticView';
import PracticeView from './PracticeView';
import StarView from './StarView';
import DesignLabView from './DesignLabView';
import SimulationView from './SimulationView';
import InterviewView from './InterviewView';
import EvidenceView from './EvidenceView';
import MatrixView from './MatrixView';

export type HillClimbView =
  | 'dashboard'
  | 'diagnostic'
  | 'practice'
  | 'star'
  | 'design'
  | 'simulation'
  | 'interview'
  | 'matrix'
  | 'evidence';

const VIEWS: { id: HillClimbView; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'diagnostic', label: 'Diagnostic' },
  { id: 'practice', label: 'Practice' },
  { id: 'star', label: 'STAR Story Box' },
  { id: 'design', label: 'Design Lab' },
  { id: 'simulation', label: 'Simulation' },
  { id: 'interview', label: 'Interview Room' },
  { id: 'matrix', label: 'Competency Matrix' },
  { id: 'evidence', label: 'Evidence Library' },
];

export interface GoFn {
  (view: HillClimbView, exercise?: Exercise | null, competencyId?: string | null): void;
}

export default function HillClimbSection({ initialView }: { initialView?: string | null }) {
  const initial = (VIEWS.some((v) => v.id === initialView) ? initialView : 'dashboard') as HillClimbView;
  const [view, setView] = useState<HillClimbView>(initial);
  const [seedExercise, setSeedExercise] = useState<Exercise | null>(null);
  const [seedCompetency, setSeedCompetency] = useState<string | null>(null);

  const go: GoFn = useCallback((next, exercise = null, competencyId = null) => {
    setSeedExercise(exercise);
    setSeedCompetency(competencyId);
    setView(next);
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Tabs
        value={view}
        onChange={(_e, v: HillClimbView) => go(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          minHeight: 38,
          borderBottom: '1px solid #232a31',
          '& .MuiTab-root': { minHeight: 38, py: 0.5, fontSize: 13, textTransform: 'none' },
        }}
      >
        {VIEWS.map((v) => (
          <Tab key={v.id} value={v.id} label={v.label} />
        ))}
      </Tabs>

      {view === 'dashboard' ? <DashboardView go={go} /> : null}
      {view === 'diagnostic' ? <DiagnosticView go={go} /> : null}
      {view === 'practice' ? (
        <PracticeView go={go} seedExercise={seedExercise} seedCompetency={seedCompetency} />
      ) : null}
      {view === 'star' ? <StarView /> : null}
      {view === 'design' ? <DesignLabView /> : null}
      {view === 'simulation' ? <SimulationView /> : null}
      {view === 'interview' ? <InterviewView /> : null}
      {view === 'matrix' ? <MatrixView go={go} /> : null}
      {view === 'evidence' ? <EvidenceView competencyFilter={seedCompetency} /> : null}
    </Box>
  );
}
