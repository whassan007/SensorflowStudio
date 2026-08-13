/**
 * Closed-Loop Lab — next-generation AV perception evaluation.
 * Counterfactual generation + validity gating, closed-loop behavioral
 * evaluation with causal replay, safety-informed metrics, launch-eval
 * gauntlet with compute dedup, and the architecture decision docs.
 */
import { useState } from 'react';
import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import ArchitectureTab from '../../components/nextgen/ArchitectureTab';
import ClosedLoopTab from '../../components/nextgen/ClosedLoopTab';
import CounterfactualsTab from '../../components/nextgen/CounterfactualsTab';
import GauntletTab from '../../components/nextgen/GauntletTab';
import SafetyMetricsTab from '../../components/nextgen/SafetyMetricsTab';

const TABS = [
  { id: 'counterfactuals', label: 'Counterfactuals' },
  { id: 'closedloop', label: 'Closed-Loop' },
  { id: 'safety', label: 'Safety Metrics' },
  { id: 'gauntlet', label: 'Gauntlet' },
  { id: 'architecture', label: 'Architecture' },
] as const;

export default function ClosedLoopLabPage() {
  const [tab, setTab] = useState<string>('counterfactuals');
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        variant="scrollable"
        sx={{ minHeight: 36, borderBottom: '1px solid #232a31', '& .MuiTab-root': { minHeight: 36, fontSize: 12.5, textTransform: 'none', fontWeight: 700 } }}
      >
        {TABS.map((t) => (
          <Tab key={t.id} value={t.id} label={t.label} />
        ))}
      </Tabs>
      <Box sx={{ pt: 0.5 }}>
        {tab === 'counterfactuals' && <CounterfactualsTab />}
        {tab === 'closedloop' && <ClosedLoopTab />}
        {tab === 'safety' && <SafetyMetricsTab />}
        {tab === 'gauntlet' && <GauntletTab />}
        {tab === 'architecture' && <ArchitectureTab />}
      </Box>
    </Box>
  );
}
