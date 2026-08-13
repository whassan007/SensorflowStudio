import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import MenuItem from '@mui/material/MenuItem';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { Cpu, FlaskConical, Zap } from 'lucide-react';
import { VitisSection } from '../../components/vitis/Section';
import { getBackendStatus } from '../../services/vitis';
import type { BackendStatus } from '../../types/vitis';
import HilTab from './HilTab';
import IspTab from './IspTab';
import PrdTab from './PrdTab';
import TemporalTab from './TemporalTab';

export default function VitisPage() {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState(0);
  const [device, setDevice] = useState('versal-ai-edge');

  useEffect(() => {
    getBackendStatus()
      .then(setStatus)
      .catch((e) => setError((e as Error).message));
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {error ? <Alert severity="error">{error}</Alert> : null}

      <VitisSection
        title="Acceleration backends"
        subtitle={status?.note ?? 'Loading backend capability report…'}
      >
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'stretch' }}>
          {status?.backends.map((b) => (
            <Box key={b.name} sx={{ flex: '1 1 240px', bgcolor: '#161b21', border: '1px solid #232a31', borderRadius: 1, p: 1.5, opacity: b.available ? 1 : 0.6 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                {b.name === 'reference' ? <Cpu size={16} color="#4fc3f7" /> : b.name === 'vitis_emulated' ? <FlaskConical size={16} color="#ffb74d" /> : <Zap size={16} color="#8a949e" />}
                <Typography sx={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 14 }}>{b.name}</Typography>
                {b.available ? (
                  <Chip size="small" label="AVAILABLE" sx={{ height: 18, fontSize: 9, fontWeight: 700, bgcolor: '#1b5e20' }} />
                ) : (
                  <Chip size="small" label="NOT PRESENT" sx={{ height: 18, fontSize: 9, fontWeight: 700, bgcolor: '#232a31', color: '#8a949e' }} />
                )}
                {b.emulated ? (
                  <Tooltip title="Honesty badge: this backend runs on CPU. It faithfully models ap_fixed<W,I> quantization (truncation + saturation), XFCVDEPTH line-buffer limits, and LUT divide/sqrt — but every FPGA latency/speedup it reports is analytically modeled, never measured on silicon.">
                    <Chip size="small" label="EMULATED — modeled, not measured" sx={{ height: 18, fontSize: 9, fontWeight: 700, bgcolor: '#5d4037', color: '#ffcc80' }} />
                  </Tooltip>
                ) : null}
              </Box>
              <Typography variant="caption" sx={{ color: '#8a949e', lineHeight: 1.4, display: 'block' }}>
                {b.description}
              </Typography>
            </Box>
          ))}
        </Box>
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', mt: 1.5 }}>
          <TextField select size="small" label="Target device (latency model)" value={device}
            onChange={(e) => setDevice(e.target.value)} sx={{ width: 260 }}>
            {Object.entries(status?.devices ?? {}).map(([name, d]) => (
              <MenuItem key={name} value={name}>
                {name} — {d.clock_mhz} MHz{d.has_aie ? ' + AIE' : ''}
              </MenuItem>
            ))}
          </TextField>
          {status ? (
            <Typography variant="caption" sx={{ color: '#8a949e' }}>
              {status.devices[device]?.description}
            </Typography>
          ) : null}
        </Box>
      </VitisSection>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} variant="scrollable">
        <Tab label="HIL Quantization Gap" />
        <Tab label="ISP & Synthetic Data" />
        <Tab label="Temporal Stability" />
        <Tab label="PRDs" />
      </Tabs>

      {tab === 0 ? <HilTab device={device} /> : null}
      {tab === 1 ? <IspTab device={device} /> : null}
      {tab === 2 ? <TemporalTab device={device} /> : null}
      {tab === 3 ? <PrdTab /> : null}
    </Box>
  );
}
