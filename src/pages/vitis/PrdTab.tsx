import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import { LoadingBox } from '../../components/labeleval/shared';
import { VitisSection } from '../../components/vitis/Section';
import Markdown from '../../components/vitis/Markdown';
import { getPrd, listPrds } from '../../services/vitis';
import type { PrdDoc, PrdListEntry } from '../../types/vitis';

const TITLES: Record<string, string> = {
  'vitis-hil-regression': 'HIL Quantization Gap',
  'vitis-isp-preprocessing': 'ISP & Synthetic Data',
  'vitis-temporal-stability': 'Temporal Stability',
};

export default function PrdTab() {
  const [prds, setPrds] = useState<PrdListEntry[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [doc, setDoc] = useState<PrdDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listPrds()
      .then((r) => {
        setPrds(r.prds);
        const first = r.prds.find((p) => p.available);
        if (first) setActive(first.id);
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  useEffect(() => {
    if (!active) return;
    setDoc(null);
    getPrd(active)
      .then(setDoc)
      .catch((e) => setError((e as Error).message));
  }, [active]);

  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Tabs value={active ?? false} onChange={(_, v) => setActive(v)} variant="scrollable">
        {prds.map((p) => (
          <Tab key={p.id} value={p.id} label={TITLES[p.id] ?? p.id} disabled={!p.available} />
        ))}
      </Tabs>
      {doc ? (
        <VitisSection title={doc.file} subtitle="Served live from docs/prd/ via GET /api/vitis/prd/{id}.">
          <Markdown source={doc.markdown} />
        </VitisSection>
      ) : active ? (
        <LoadingBox label="Loading PRD…" />
      ) : null}
    </Box>
  );
}
