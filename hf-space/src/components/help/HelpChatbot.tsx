/**
 * Floating help chatbot: asks POST /api/help/chat and shows answer + sources.
 * Works offline via the backend FAQ matcher when Ollama is unreachable.
 */
import { useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Drawer from '@mui/material/Drawer';
import Fab from '@mui/material/Fab';
import IconButton from '@mui/material/IconButton';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { MessageCircle, Send, X } from 'lucide-react';
import { askHelp, type HelpChatSource } from '../../services/help';
import { useLabelEval, type PageId } from '../../context/LabelEvalContext';
import { PAGE_GUIDES } from '../../help/pageGuides';

interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
  sources?: HelpChatSource[];
  provider?: string;
}

const SUGGESTIONS = [
  'What is Sensorflow Studio?',
  'How do I get started?',
  'When do humans review labels?',
  'What does the Command Center do?',
  'Does help chat need a GPU?',
];

export default function HelpChatbot({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { page, navigate } = useLabelEval();
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([
    {
      role: 'assistant',
      text:
        'Ask about any page or feature. I use the local help index (works on CPU Spaces); if Ollama is available the answer may be enriched.',
    },
  ]);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns, open]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const send = async (raw: string) => {
    const question = raw.trim();
    if (!question || busy) return;
    setError(null);
    setInput('');
    setTurns((t) => [...t, { role: 'user', text: question }]);
    setBusy(true);
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await askHelp({ question, page_id: page }, ac.signal);
      setTurns((t) => [
        ...t,
        {
          role: 'assistant',
          text: res.answer,
          sources: res.sources,
          provider: res.provider,
        },
      ]);
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      // Client-side soft fallback when the API is down
      const guide = PAGE_GUIDES[page];
      const fallback =
        guide != null
          ? `${guide.title}: ${guide.summary}\n\nKey actions:\n${guide.keyActions.map((a) => `• ${a}`).join('\n')}\n\n(Help API unreachable — showing the local page guide.)`
          : 'Help API unreachable. Open Help (?) → Pages for guides, or retry when the backend is up.';
      setTurns((t) => [...t, { role: 'assistant', text: fallback }]);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const jumpToSource = (src: HelpChatSource) => {
    const pageId = (src as HelpChatSource & { page_id?: string }).page_id;
    // sources from API may include page_id; also parse page:id
    const fromId = src.id.startsWith('page:') ? (src.id.slice(5) as PageId) : null;
    const target = (pageId || fromId) as PageId | null;
    if (target && PAGE_GUIDES[target]) {
      onOpenChange(false);
      navigate(target);
    }
  };

  return (
    <>
      <Tooltip title="Ask the help chatbot about Sensorflow Studio">
        <Fab
          size="medium"
          color="primary"
          onClick={() => onOpenChange(true)}
          aria-label="Open help chatbot"
          sx={{
            position: 'fixed',
            right: 20,
            bottom: 20,
            zIndex: (t) => t.zIndex.drawer + 2,
            bgcolor: '#0288d1',
            '&:hover': { bgcolor: '#0277bd' },
          }}
        >
          <MessageCircle size={22} />
        </Fab>
      </Tooltip>

      <Drawer
        anchor="right"
        open={open}
        onClose={() => onOpenChange(false)}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 400 },
            bgcolor: '#12171d',
            borderLeft: '1px solid #232a31',
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 2, py: 1.5, borderBottom: '1px solid #232a31' }}>
          <MessageCircle size={18} color="#4fc3f7" />
          <Typography variant="subtitle1" sx={{ fontWeight: 700, flex: 1, fontSize: 14 }}>
            Help chatbot
          </Typography>
          <Chip
            size="small"
            label={PAGE_GUIDES[page]?.title ?? page}
            sx={{ bgcolor: '#232a31', fontSize: 11, height: 22 }}
            title="Answers can use the page you are on as context"
          />
          <IconButton size="small" onClick={() => onOpenChange(false)} aria-label="Close help chatbot">
            <X size={16} />
          </IconButton>
        </Box>

        <Box sx={{ flex: 1, overflowY: 'auto', px: 2, py: 1.5 }}>
          {turns.map((turn, i) => (
            <Box
              key={i}
              sx={{
                mb: 1.5,
                p: 1.25,
                borderRadius: 1,
                bgcolor: turn.role === 'user' ? 'rgba(79,195,247,0.12)' : '#141a20',
                border: '1px solid #232a31',
              }}
            >
              <Typography variant="caption" sx={{ color: '#5c6873', fontWeight: 700, displayTransform: 'uppercase' }}>
                {turn.role === 'user' ? 'You' : 'Help'}
              </Typography>
              <Typography variant="body2" sx={{ color: '#e6e9ec', whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.55, mt: 0.25 }}>
                {turn.text}
              </Typography>
              {turn.provider ? (
                <Typography variant="caption" sx={{ color: '#5c6873', display: 'block', mt: 0.75 }}>
                  via {turn.provider === 'faq_offline' ? 'local FAQ index' : turn.provider}
                </Typography>
              ) : null}
              {turn.sources && turn.sources.length > 0 ? (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                  {turn.sources.map((s) => (
                    <Chip
                      key={s.id}
                      size="small"
                      label={s.title}
                      onClick={() => jumpToSource(s)}
                      sx={{ bgcolor: '#1d242c', fontSize: 10, height: 22, cursor: s.id.startsWith('page:') ? 'pointer' : 'default' }}
                      title={s.kind}
                    />
                  ))}
                </Box>
              ) : null}
            </Box>
          ))}
          {busy ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#8a949e', py: 1 }}>
              <CircularProgress size={14} />
              <Typography variant="caption">Searching help index…</Typography>
            </Box>
          ) : null}
          {error ? (
            <Typography variant="caption" sx={{ color: '#ffd54f' }}>
              {error}
            </Typography>
          ) : null}
          <div ref={bottomRef} />
        </Box>

        <Box sx={{ px: 2, pb: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          {SUGGESTIONS.map((s) => (
            <Chip
              key={s}
              size="small"
              label={s}
              onClick={() => void send(s)}
              disabled={busy}
              sx={{ bgcolor: '#1d242c', fontSize: 10, height: 22 }}
            />
          ))}
        </Box>

        <Box sx={{ display: 'flex', gap: 1, p: 2, borderTop: '1px solid #232a31' }}>
          <TextField
            size="small"
            fullWidth
            placeholder="Ask about a page or feature…"
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            inputProps={{ 'aria-label': 'Help question' }}
          />
          <Button
            variant="contained"
            disabled={busy || !input.trim()}
            onClick={() => void send(input)}
            aria-label="Send help question"
            title="Send question to the help chatbot"
            sx={{ minWidth: 44 }}
          >
            <Send size={16} />
          </Button>
        </Box>
      </Drawer>
    </>
  );
}
