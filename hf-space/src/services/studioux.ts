/**
 * Studio-UX helper service: dashboard layout persistence and capability
 * probing for optional backends.
 *
 * Layout persistence is dual-path: localStorage always (works offline /
 * without the optional router), plus best-effort sync to the small
 * /api/studio-ux backend when it is mounted, so layouts survive browser
 * storage resets and follow the backend store.
 */

const LS_PREFIX = 'sf-studio-ux:';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------- layouts

export function loadLayoutLocal<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(LS_PREFIX + key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function saveLayoutLocal<T>(key: string, value: T): void {
  try {
    localStorage.setItem(LS_PREFIX + key, JSON.stringify(value));
  } catch {
    /* storage full / private mode — backend sync still applies */
  }
}

export async function loadLayoutRemote<T>(key: string): Promise<T | null> {
  try {
    const res = await fetch(`/api/studio-ux/layouts/${encodeURIComponent(key)}`);
    if (!res.ok) return null;
    const body = (await res.json()) as { key: string; value: T | null };
    return body.value ?? null;
  } catch {
    return null;
  }
}

export function saveLayoutRemote<T>(key: string, value: T): void {
  // fire-and-forget; localStorage is the guaranteed path
  fetch(`/api/studio-ux/layouts/${encodeURIComponent(key)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  }).catch(() => undefined);
}

/** Load a layout preferring the backend copy, falling back to localStorage. */
export async function loadLayout<T>(key: string): Promise<T | null> {
  const remote = await loadLayoutRemote<T>(key);
  if (remote !== null) return remote;
  return loadLayoutLocal<T>(key);
}

export function saveLayout<T>(key: string, value: T): void {
  saveLayoutLocal(key, value);
  saveLayoutRemote(key, value);
}

// ---------------------------------------------------------------- capability probe

export interface Capabilities {
  ssamAnalyze: boolean;
  bevfusion: boolean;
  nextgenCounterfactual: boolean;
  studioUx: boolean;
}

let capCache: Capabilities | null = null;

async function probe(url: string, init?: RequestInit): Promise<boolean> {
  try {
    const res = await fetch(url, init);
    // Any non-404 answer means the router is mounted (405/422 count as present).
    return res.status !== 404;
  } catch {
    return false;
  }
}

/**
 * Feature-detect the optional scene-consuming / persistence APIs once per
 * session. Pages hide actions whose backing endpoint is absent.
 */
export async function getCapabilities(force = false): Promise<Capabilities> {
  if (capCache && !force) return capCache;
  const [ssamAnalyze, bevfusion, nextgenCounterfactual, studioUx] = await Promise.all([
    probe('/api/safety/ssam/analyze', { method: 'OPTIONS' }),
    probe('/api/bevfusion/status'),
    probe('/api/nextgen/counterfactual', { method: 'OPTIONS' }),
    probe('/api/studio-ux/layouts/__probe__'),
  ]);
  capCache = { ssamAnalyze, bevfusion, nextgenCounterfactual, studioUx };
  return capCache;
}

export async function fetchJson<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handle<T>(res);
}
