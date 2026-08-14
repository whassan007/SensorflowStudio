/** Help chatbot client — POST /api/help/chat with graceful client-side fallback. */

export interface HelpChatRequest {
  question: string;
  page_id?: string | null;
}

export interface HelpChatSource {
  id: string;
  title: string;
  kind: string;
  score?: number;
  page_id?: string | null;
}

export interface HelpChatResponse {
  answer: string;
  sources: HelpChatSource[];
  provider?: string;
  page_hint?: string | null;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function askHelp(req: HelpChatRequest, signal?: AbortSignal): Promise<HelpChatResponse> {
  const res = await fetch('/api/help/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: req.question,
      page_id: req.page_id ?? null,
    }),
    signal,
  });
  return handle<HelpChatResponse>(res);
}
