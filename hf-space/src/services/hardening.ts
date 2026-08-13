/** Fetchers for /api/hardening (Production Readiness page). */

import type {
  AuditDocument,
  FunnelResponse,
  HardeningSummary,
  ReadinessScorecard,
} from '../types/hardening';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function getAudit(): Promise<AuditDocument> {
  return handle<AuditDocument>(await fetch('/api/hardening/audit'));
}

export async function getReadiness(): Promise<ReadinessScorecard> {
  return handle<ReadinessScorecard>(await fetch('/api/hardening/readiness'));
}

export async function getFunnel(): Promise<FunnelResponse> {
  return handle<FunnelResponse>(await fetch('/api/hardening/funnel'));
}

export async function getHardeningSummary(): Promise<HardeningSummary> {
  return handle<HardeningSummary>(await fetch('/api/hardening/summary'));
}
