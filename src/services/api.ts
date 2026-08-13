import type { StatewideQuery, StatewideResponse } from '../types';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchStatewide(query: StatewideQuery, signal?: AbortSignal): Promise<StatewideResponse> {
  const res = await fetch('/api/ssam/statewide', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(query),
    signal,
  });
  return handle<StatewideResponse>(res);
}

export async function annotateStreet(streetName: string, annotation: string): Promise<void> {
  const res = await fetch('/api/ssam/annotate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ street_name: streetName, manual_annotation: annotation }),
  });
  await handle(res);
}
