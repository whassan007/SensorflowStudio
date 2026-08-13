import { useCallback, useEffect, useMemo, useState } from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import ListSubheader from '@mui/material/ListSubheader';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import {
  Gauge,
  LayoutDashboard,
  Database,
  Tag,
  Radar,
  Pickaxe,
  ShieldCheck,
  TrendingDown,
  Microscope,
  Filter,
  UserCheck,
  Rocket,
  Boxes,
  ListChecks,
  ScrollText,
  Workflow,
  Map as MapIcon,
  AppWindow,
  Mountain,
  Undo2,
  CircuitBoard,
} from 'lucide-react';
import { LabelEvalContext, type PageId, ALL_PAGE_IDS } from './context/LabelEvalContext';
import { getOverview, useStream } from './services/labeleval';
import OverviewPage from './pages/OverviewPage';
import DatasetsPage from './pages/DatasetsPage';
import LabelGenerationPage from './pages/LabelGenerationPage';
import RareEventDashboard from './pages/RareEventDashboard';
import QualityEnginePage from './pages/QualityEnginePage';
import RegressionPage from './pages/RegressionPage';
import TriagePage from './pages/TriagePage';
import HumanReviewPage from './pages/HumanReviewPage';
import TrainingPage from './pages/TrainingPage';
import ModelsPage from './pages/ModelsPage';
import EvaluationPage from './pages/EvaluationPage';
import AuditPage from './pages/AuditPage';
import PipelineArchitecturePage from './pages/PipelineArchitecturePage';
import SSAMSafetyDashboard from './pages/SSAMSafetyDashboard';
import LegacyStudioPage from './pages/LegacyStudioPage';
import CommandCenterPage from './pages/CommandCenterPage';
import RootCauseLabPage from './pages/rca/RootCauseLabPage';
import RareMinePage from './pages/raremine/RareMinePage';
import VitisPage from './pages/vitis/VitisPage';
import HillClimbSection from './pages/hillclimb/HillClimbSection';
import RetroAnalyzerPage from './pages/retro/RetroAnalyzerPage';
import PageIntro from './components/help/PageIntro';
import HelpMenu from './components/help/HelpMenu';

const DRAWER_WIDTH = 230;

interface NavItem {
  id: PageId;
  label: string;
  icon: React.ReactNode;
}

const PLATFORM_NAV: NavItem[] = [
  { id: 'command', label: 'Command Center', icon: <Gauge size={18} /> },
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={18} /> },
  { id: 'datasets', label: 'Datasets', icon: <Database size={18} /> },
  { id: 'label-generation', label: 'Label Generation', icon: <Tag size={18} /> },
  { id: 'rare-events', label: 'Rare Events', icon: <Radar size={18} /> },
  { id: 'raremine', label: 'Rare-Event Miner', icon: <Pickaxe size={18} /> },
  { id: 'quality', label: 'Quality Engine', icon: <ShieldCheck size={18} /> },
  { id: 'regression', label: 'Regression', icon: <TrendingDown size={18} /> },
  { id: 'rca', label: 'Root Cause Lab', icon: <Microscope size={18} /> },
  { id: 'triage', label: 'Triage', icon: <Filter size={18} /> },
  { id: 'review', label: 'Human Review', icon: <UserCheck size={18} /> },
  { id: 'training', label: 'Training', icon: <Rocket size={18} /> },
  { id: 'models', label: 'Models', icon: <Boxes size={18} /> },
  { id: 'evaluation', label: 'Evaluation', icon: <ListChecks size={18} /> },
  { id: 'audit', label: 'Audit', icon: <ScrollText size={18} /> },
  { id: 'pipeline', label: 'Pipeline Architecture', icon: <Workflow size={18} /> },
  { id: 'hillclimb', label: 'Hill Climbing EM', icon: <Mountain size={18} /> },
  { id: 'retro', label: 'Retrospective Analyzer', icon: <Undo2 size={18} /> },
  { id: 'vitis', label: 'Hardware Acceleration', icon: <CircuitBoard size={18} /> },
];

const LEGACY_NAV: NavItem[] = [
  { id: 'ssam', label: 'SSAM Safety', icon: <MapIcon size={18} /> },
  { id: 'legacy', label: 'Legacy Studio', icon: <AppWindow size={18} /> },
];

const PAGE_TITLES: Record<PageId, string> = {
  command: 'Evaluation Command Center',
  overview: 'Overview',
  datasets: 'Datasets',
  'label-generation': 'Label Generation',
  'rare-events': 'Rare Event Detection',
  raremine: 'Rare-Event Miner (Costumed Pedestrians)',
  quality: 'Quality Engine',
  regression: 'Regression Tracking',
  rca: 'Root Cause Lab',
  triage: 'Automated Triage',
  review: 'Human Review (HITL)',
  training: 'Training Flywheel',
  models: 'Models',
  evaluation: 'Evaluation Records',
  audit: 'Audit & Process Units',
  pipeline: 'Pipeline Architecture',
  hillclimb: 'Hill Climbing EM',
  retro: 'Retrospective Safety Analyzer',
  vitis: 'Hardware Acceleration (Vitis Vision)',
  ssam: 'SSAM Safety Dashboard',
  legacy: 'Legacy Studio',
};

function parseHash(): { page: PageId; entityId: string | null } {
  const raw = window.location.hash.replace(/^#\/?/, '');
  const [pageRaw, entityRaw] = raw.split('/');
  const page = (ALL_PAGE_IDS as string[]).includes(pageRaw ?? '') ? (pageRaw as PageId) : 'command';
  return { page, entityId: entityRaw ? decodeURIComponent(entityRaw) : null };
}

export default function App() {
  const initial = useMemo(parseHash, []);
  const [page, setPage] = useState<PageId>(initial.page);
  const [entityId, setEntityId] = useState<string | null>(initial.entityId);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const stream = useStream();

  const navigate = useCallback((nextPage: PageId, nextEntityId?: string | null) => {
    setPage(nextPage);
    setEntityId(nextEntityId ?? null);
    window.location.hash = nextEntityId ? `${nextPage}/${encodeURIComponent(nextEntityId)}` : nextPage;
  }, []);

  useEffect(() => {
    const onHashChange = () => {
      const parsed = parseHash();
      setPage(parsed.page);
      setEntityId(parsed.entityId);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // Adopt the backend's active dataset once on startup so pages have context.
  useEffect(() => {
    getOverview()
      .then((o) => {
        if (o.active_dataset) {
          setActiveDatasetId((current) => current ?? o.active_dataset);
        }
      })
      .catch(() => undefined);
  }, []);

  const reviewCount = stream?.pipeline.review_queue_count ?? 0;
  const alertsCount = stream?.alerts_count ?? 0;

  const badgeFor = (id: PageId): number => {
    if (id === 'review') return reviewCount;
    if (id === 'overview') return alertsCount;
    return 0;
  };

  const contextValue = useMemo(
    () => ({ activeDatasetId, setActiveDatasetId, stream, page, entityId, navigate }),
    [activeDatasetId, stream, page, entityId, navigate]
  );

  const renderNavItem = (item: NavItem) => {
    const badge = badgeFor(item.id);
    return (
      <ListItemButton
        key={item.id}
        selected={page === item.id}
        onClick={() => navigate(item.id)}
        sx={{
          borderRadius: 1,
          mx: 0.75,
          my: 0.1,
          py: 0.6,
          '&.Mui-selected': { bgcolor: 'rgba(79,195,247,0.12)', color: '#4fc3f7' },
        }}
      >
        <ListItemIcon sx={{ minWidth: 32, color: page === item.id ? '#4fc3f7' : '#8a949e' }}>
          {item.icon}
        </ListItemIcon>
        <ListItemText primaryTypographyProps={{ fontSize: 13.5 }} primary={item.label} />
        {badge > 0 ? (
          <Chip
            size="small"
            label={badge > 999 ? '999+' : badge.toLocaleString()}
            sx={{
              height: 18,
              fontSize: 10,
              fontWeight: 700,
              bgcolor: item.id === 'review' ? '#e65100' : '#b71c1c',
              color: '#fff',
            }}
          />
        ) : null}
      </ListItemButton>
    );
  };

  const fullBleed = page === 'ssam' || page === 'legacy';

  return (
    <LabelEvalContext.Provider value={contextValue}>
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <AppBar position="fixed" sx={{ zIndex: (t) => t.zIndex.drawer + 1, bgcolor: '#12171d', backgroundImage: 'none' }}>
          <Toolbar variant="dense" sx={{ minHeight: 52 }}>
            <Typography variant="h6" sx={{ fontSize: 16, fontWeight: 700, flex: 1 }}>
              Sensorflow Studio — L4 Perception Label Evaluation
            </Typography>
            {stream?.pipeline.running ? (
              <Chip
                size="small"
                label={`PIPELINE RUNNING · ${stream.pipeline.stage}`}
                sx={{ bgcolor: '#0d47a1', color: '#90caf9', fontWeight: 700 }}
              />
            ) : null}
            {activeDatasetId ? (
              <Chip
                size="small"
                label={`dataset: ${activeDatasetId}`}
                sx={{ ml: 1, bgcolor: '#232a31', fontFamily: 'monospace', fontSize: 11 }}
              />
            ) : null}
            <HelpMenu />
          </Toolbar>
        </AppBar>

        <Drawer
          variant="permanent"
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
              bgcolor: '#12171d',
              borderRight: '1px solid #232a31',
            },
          }}
        >
          <Toolbar variant="dense" sx={{ minHeight: 52 }} />
          <Box sx={{ overflowY: 'auto', pb: 2 }}>
            <List
              dense
              subheader={
                <ListSubheader sx={{ bgcolor: 'transparent', fontSize: 11, letterSpacing: 1.2, lineHeight: '32px' }}>
                  PLATFORM
                </ListSubheader>
              }
            >
              {PLATFORM_NAV.map(renderNavItem)}
            </List>
            <List
              dense
              subheader={
                <ListSubheader sx={{ bgcolor: 'transparent', fontSize: 11, letterSpacing: 1.2, lineHeight: '32px' }}>
                  LEGACY / EXISTING
                </ListSubheader>
              }
            >
              {LEGACY_NAV.map(renderNavItem)}
            </List>
          </Box>
        </Drawer>

        <Box
          component="main"
          sx={{
            flexGrow: 1,
            minWidth: 0,
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Toolbar variant="dense" sx={{ minHeight: 52, flexShrink: 0 }} />
          {fullBleed ? (
            <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ px: 2, pt: 1, flexShrink: 0, borderBottom: '1px solid #232a31', bgcolor: '#12171d' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
                  {PAGE_TITLES[page]}
                </Typography>
                <PageIntro page={page} dense />
              </Box>
              <Box sx={{ flex: 1, minHeight: 0 }} className={page === 'ssam' ? 'ssam-embed' : undefined}>
                {page === 'ssam' ? <SSAMSafetyDashboard /> : <LegacyStudioPage />}
              </Box>
            </Box>
          ) : (
            <Box sx={{ flex: 1, minHeight: 0, overflowY: 'auto', p: 2.5 }}>
              <Typography variant="h5" sx={{ fontWeight: 800, mb: 0.5 }}>
                {PAGE_TITLES[page]}
              </Typography>
              <PageIntro page={page} />
              {page === 'command' ? <CommandCenterPage /> : null}
              {page === 'overview' ? <OverviewPage /> : null}
              {page === 'datasets' ? <DatasetsPage /> : null}
              {page === 'label-generation' ? <LabelGenerationPage /> : null}
              {page === 'rare-events' ? <RareEventDashboard /> : null}
              {page === 'raremine' ? <RareMinePage /> : null}
              {page === 'quality' ? <QualityEnginePage /> : null}
              {page === 'regression' ? <RegressionPage /> : null}
              {page === 'rca' ? <RootCauseLabPage /> : null}
              {page === 'triage' ? <TriagePage /> : null}
              {page === 'review' ? <HumanReviewPage /> : null}
              {page === 'training' ? <TrainingPage /> : null}
              {page === 'models' ? <ModelsPage /> : null}
              {page === 'evaluation' ? <EvaluationPage /> : null}
              {page === 'audit' ? <AuditPage /> : null}
              {page === 'pipeline' ? <PipelineArchitecturePage /> : null}
              {page === 'hillclimb' ? <HillClimbSection initialView={entityId} /> : null}
              {page === 'retro' ? <RetroAnalyzerPage /> : null}
              {page === 'vitis' ? <VitisPage /> : null}
            </Box>
          )}
        </Box>
      </Box>
    </LabelEvalContext.Provider>
  );
}
