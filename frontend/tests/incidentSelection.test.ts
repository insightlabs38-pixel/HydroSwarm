import {
  clearSelectedIncident,
  hasSelectableIncident,
  requestedIncidentId,
  selectIncident,
} from '../src/incidentSelection';

beforeEach(() => {
  window.sessionStorage.clear();
  window.history.pushState(null, '', '/');
  vi.unstubAllEnvs();
});

test('no incident selected anywhere resolves to null, and the gateway must show', () => {
  expect(requestedIncidentId()).toBeNull();
  expect(hasSelectableIncident()).toBe(false);
});

test('selectIncident() makes a newly created incident immediately resolvable', () => {
  selectIncident('incident-abc');
  expect(requestedIncidentId()).toBe('incident-abc');
  expect(hasSelectableIncident()).toBe(true);
});

test('selectIncident() updates the URL so the choice is deep-linkable', () => {
  selectIncident('incident-abc');
  expect(new URLSearchParams(window.location.search).get('incident')).toBe('incident-abc');
});

test('selection survives a "refresh" (re-reading state fresh, as a new page load would)', () => {
  selectIncident('incident-abc');
  // Simulate a refresh: the URL persists (real browser behavior), and a
  // fresh call re-reads it -- this module never caches a frozen value at
  // import time the way the old build-time VITE_INCIDENT_ID did.
  expect(requestedIncidentId()).toBe('incident-abc');
});

test('?incident=<id> in the URL is honored as a deep link, independent of any prior selection', () => {
  window.history.pushState(null, '', '/?incident=deep-linked-incident');
  expect(requestedIncidentId()).toBe('deep-linked-incident');
});

test('URL ?incident= takes priority over a stale session-selected incident', () => {
  selectIncident('session-incident');
  window.history.pushState(null, '', '/?incident=url-incident');
  expect(requestedIncidentId()).toBe('url-incident');
});

test('no stale incident leaks across sessions: clearSelectedIncident() removes it', () => {
  selectIncident('incident-abc');
  expect(hasSelectableIncident()).toBe(true);

  clearSelectedIncident();
  window.history.pushState(null, '', '/');
  expect(requestedIncidentId()).toBeNull();
  expect(hasSelectableIncident()).toBe(false);
});

test('VITE_INCIDENT_ID is only a last-resort dev fallback, never preferred over a real selection', () => {
  vi.stubEnv('VITE_INCIDENT_ID', 'dev-fallback-incident');
  expect(requestedIncidentId()).toBe('dev-fallback-incident');

  selectIncident('real-selected-incident');
  expect(requestedIncidentId()).toBe('real-selected-incident');
});

test('a fresh, unconfigured clean install has no selectable incident (gateway must show)', () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  expect(hasSelectableIncident()).toBe(false);
});
