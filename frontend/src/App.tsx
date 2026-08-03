import { useQuery } from '@tanstack/react-query';
import { fetchIncidentWithFallback } from './api';
import { AuditPage } from './pages/AuditPage';
import { BenchmarkPage } from './pages/BenchmarkPage';
import { Overview } from './pages/Overview';
import { TopologyPage } from './pages/TopologyPage';
import { ValidationPage } from './pages/ValidationPage';
import { useConsoleStore, type Page } from './store';
import { StatusBadge } from './components/StatusBadge';

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
          <span
            className="runtime"
            aria-label={`Inference runtime ${incident.runtimeMs} milliseconds`}
          >
            {incident.runtimeMs} ms
          </span>
        </div>
      </header>
      {incident.source === 'demo-fallback' && (
        <div className="fallback-banner" role="status">
          <strong>DETERMINISTIC DEMO FALLBACK</strong>
          <span>
            Live API unavailable or no incident configured. Values below are a frozen,
            simulator-derived fixture—not live telemetry.
          </span>
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
        {page === 'overview' && <Overview incident={incident} />}
        {page === 'audit' && <AuditPage incident={incident} />}
        {page === 'validation' && <ValidationPage incident={incident} />}
        {page === 'benchmarks' && <BenchmarkPage incident={incident} />}
        {page === 'topology' && <TopologyPage incident={incident} />}
      </main>
      <footer>
        <span>Decision support only · No autonomous control</span>
        <span>Exact verifier: WNTR / EPANET</span>
        <span>Operator approval required</span>
      </footer>
    </div>
  );
}
