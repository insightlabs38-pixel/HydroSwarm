/** Runtime LIVE incident selection (submission.txt SUB-12.1 P1 #5).
 *
 * Previously the frontend could only ever load ONE incident, chosen at
 * *build time* via `VITE_INCIDENT_ID` -- meaning a judge could never run
 * `Run Live Example`, have it create a real incident through the real
 * API, and actually see it: there was no way for a freshly-created
 * incident's id to reach the frontend at all without a rebuild.
 *
 * Resolution priority, evaluated fresh on every call (never frozen at
 * module-import time, unlike the old `VITE_INCIDENT_ID`-only approach):
 *   1. `?incident=<id>` in the URL -- explicit, deep-linkable.
 *   2. The session-selected incident (sessionStorage) -- survives a
 *      refresh within the same tab/session, cleared on tab close; set by
 *      selectIncident() whenever the LIVE example (or any future flow)
 *      creates/picks an incident.
 *   3. `VITE_INCIDENT_ID` -- kept ONLY as an optional development
 *      compatibility fallback (e.g. a developer's local .env pointing at
 *      a fixture incident id while iterating), never the judge path.
 */

const SESSION_STORAGE_KEY = 'hydroswarm-selected-incident-id';

function readSessionStorage(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    // sessionStorage can throw (private browsing, disabled storage) --
    // treat as "no session-selected incident", never a hard failure.
    return null;
  }
}

function readUrlIncidentId(): string | null {
  if (typeof window === 'undefined') return null;
  return new URLSearchParams(window.location.search).get('incident');
}

/** The runtime-resolved LIVE incident id, or null if none is selected by
 * any of the three sources above. */
export function requestedIncidentId(): string | null {
  const fromUrl = readUrlIncidentId();
  if (fromUrl) return fromUrl;

  const fromSession = readSessionStorage();
  if (fromSession) return fromSession;

  const buildTimeFallback = import.meta.env.VITE_INCIDENT_ID as string | undefined;
  return buildTimeFallback && buildTimeFallback.length > 0 ? buildTimeFallback : null;
}

/** Whether ANY LIVE incident is currently selectable (URL, session, or
 * the dev build-time fallback) -- used by the first-launch gateway, which
 * must not show for a deep link or a returning session that already has
 * an incident. */
export function hasSelectableIncident(): boolean {
  return requestedIncidentId() !== null;
}

/** Records a newly created/selected incident as the current session's
 * LIVE incident: persists it to sessionStorage (survives a refresh) and
 * reflects it in the URL (`?incident=<id>`, via history.replaceState so
 * it doesn't add a back-button entry) so the choice is deep-linkable and
 * shareable, not just held in memory. */
export function selectIncident(incidentId: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, incidentId);
  } catch {
    // Storage unavailable -- the URL update below still makes the
    // selection work for this page view; it just won't survive a
    // same-tab refresh without the query param.
  }
  const url = new URL(window.location.href);
  url.searchParams.set('incident', incidentId);
  window.history.replaceState(null, '', url);
}

/** Clears the session-selected incident (not the URL -- callers that want
 * to fully back out of LIVE should also clear/replace ?incident=). Mainly
 * for tests and an explicit "start over" action. */
export function clearSelectedIncident(): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Nothing to clear if storage never worked in the first place.
  }
}
