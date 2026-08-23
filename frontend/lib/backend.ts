/**
 * Server-side helpers for Next.js API routes that proxy the backend.
 *
 * - API_URL: server-side target (set in Docker to http://backend:8000).
 *   NEXT_PUBLIC_API_URL kept as a legacy fallback.
 * - BACKEND_API_KEY: forwarded as X-API-Key when the backend requires one.
 */

export const BACKEND_URL =
  process.env.API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://127.0.0.1:8000'

export function backendHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra }
  if (process.env.BACKEND_API_KEY) {
    headers['X-API-Key'] = process.env.BACKEND_API_KEY
  }
  return headers
}
