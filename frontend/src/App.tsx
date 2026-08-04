import { lazy, Suspense } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchIncidentWithFallback } from './api';
import { useConsoleStore, type Page } from './store';
import { StatusBadge } from './components/StatusBadge';
import type { RuntimeMode } from './types';

function modeTone(mode: RuntimeMode): 'good' | 'warn' | 'danger' | 'info' {
  switch (mode) {
    case 'LIVE':
      return 'good';
    case 'REPLAY':
      return 'info';
    case 'DEMO_FALLBACK':
      return 'warn';
    case 'ERROR':
      return 'danger';
  }
}

const Overview = lazy(() =>
  import('./pages/Overview').then((module) => ({ default: module.Overview })),
);
const AuditPage = lazy(() =>
  import('./pages/AuditPage').then((module) => ({ default: module.AuditPage })),
);
const ValidationPage = lazy(() =>
  import('./pages/ValidationPage').then((module) => ({ default: module.ValidationPage })),
);
const BenchmarkPage = lazy(() =>
  import('./pages/BenchmarkPage').then((module) => ({ default: module.BenchmarkPage })),
);
const TopologyPage = lazy(() =>
  import('./pages/TopologyPage').then((module) => ({ default: module.TopologyPage })),
);

const pages: { id: Page; label: string }[] = [
  { id: 'overview', label: 'Incident' },
  { id: 'audit', label: 'Audit' },
  { id: 'validation', label: 'Validation' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'topology', label: 'Topology' },
];

export default function App() {
  const { page, setPage, reducedMotion, toggleReducedMotion } = useConsoleStore();
  const query = useQuery({
    queryKey: ['active-incident'],
    queryFn: ({ signal }) => fetchIncidentWithFallback(signal),
    staleTime: 5_000,
  });
  if (!query.data)
    return (
      <main className="loading-state" aria-live="polite">
        Loading local incident state…
      </main>
    );
  const incident = query.data;
  return (
    <div className={reducedMotion ? 'app reduced-motion' : 'app'}>
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            H
          </span>
          <div>
            <strong>HydroSwarm</strong>
            <small>Offline Neuro-Hydraulic Incident Intelligence</small>
          </div>
        </div>
        <div className="header-context">
          <span>
            Incident <strong>{incident.id}</strong>
          </span>
          <span>{incident.networkId}</span>
          <span>{incident.modelVersion}</span>
        </div>
        <div className="header-status">
          <StatusBadge tone="good">OFFLINE · LOCAL</StatusBadge>
          <StatusBadge tone={modeTone(incident.mode)}>
            <span aria-label={`Data mode ${incident.mode}`}>{incident.mode}</span>
          </StatusBadge>
          <span
            className="runtime"
            aria-label={`Inference runtime ${incident.runtimeMs} milliseconds`}
          >
            {incident.runtimeMs} ms
          </span>
        </div>
      </header>
      {incident.mode === 'DEMO_FALLBACK' && (
        <div className="fallback-banner" role="status">
          <strong>DETERMINISTIC DEMO FALLBACK</strong>
          <span>{incident.modeReason}</span>
        </div>
      )}
      {incident.mode === 'REPLAY' && (
        <div className="fallback-banner" role="status">
          <strong>REPLAY</strong>
          <span>{incident.modeReason ?? 'Showing a selected stored trajectory, not live telemetry.'}</span>
        </div>
      )}
      {incident.mode === 'ERROR' && (
        <div className="error-banner" role="alert">
          <strong>INCIDENT UNAVAILABLE</strong>
          <span>{incident.modeReason ?? 'The configured incident could not be loaded.'}</span>
          <button type="button" onClick={() => query.refetch()}>
            Retry
          </button>
        </div>
      )}
      <nav className="app-nav" aria-label="Operator console pages">
        {pages.map((item) => (
          <button
            type="button"
            key={item.id}
            onClick={() => setPage(item.id)}
            aria-current={page === item.id ? 'page' : undefined}
          >
            {item.label}
          </button>
        ))}
        <button
          type="button"
          className="motion-toggle"
          onClick={toggleReducedMotion}
          aria-pressed={reducedMotion}
        >
          Reduced motion {reducedMotion ? 'on' : 'off'}
        </button>
      </nav>
      <main id="main-content">
        <Suspense
          fallback={
            <div className="page-loading" role="status">
              Loading local console view…
            </div>
          }
        >
          {page === 'overview' && <Overview incident={incident} />}
          {page === 'audit' && <AuditPage incident={incident} />}
          {page === 'validation' && <ValidationPage incident={incident} />}
          {page === 'benchmarks' && <BenchmarkPage incident={incident} />}
          {page === 'topology' && <TopologyPage incident={incident} />}
        </Suspense>
      </main>
      <footer>
        <span>Decision support only · No autonomous control</span>
        <span>Exact verifier: WNTR / EPANET</span>
        <span>Operator approval required</span>
      </footer>
    </div>
  );
}
