import { lazy, Suspense } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchIncidentWithFallback } from './api';
import { useConsoleStore, WORKSPACE_LABELS } from './store';
import { MissionHeader } from './shell/MissionHeader';
import { ModeBanner } from './shell/ModeBanner';
import { WorkflowRail } from './shell/WorkflowRail';
import { WorkspaceToolbar } from './shell/WorkspaceToolbar';
import { DecisionInspector } from './shell/DecisionInspector';
import { TechnicalDock } from './shell/TechnicalDock';
import { EmptyState } from './components/common/EmptyState';

const Overview = lazy(() =>
  import('./pages/Overview').then((module) => ({ default: module.Overview })),
);
const ValidationPage = lazy(() =>
  import('./pages/ValidationPage').then((module) => ({ default: module.ValidationPage })),
);
const BenchmarkPage = lazy(() =>
  import('./pages/BenchmarkPage').then((module) => ({ default: module.BenchmarkPage })),
);

const NOT_YET_MIGRATED_DETAIL: Partial<Record<string, string>> = {
  source: 'Planned for UI-3 (governed source-localization workspace).',
  sampling: 'Planned for UI-4 (deterministic evidence-value sampling workspace).',
  response: 'Planned for UI-5 (verified response-plan decision workspace).',
  approval: 'Planned for UI-6 (guarded human plan-approval workflow).',
  replay: 'Planned for UI-8 (deterministic replay and fail-closed demo states).',
  network: 'Planned for UI-9 (network governance and benchmark utilities).',
  authority: 'Planned for UI-9 (network governance and benchmark utilities).',
};

export default function App() {
  const { workspace, reducedMotion, toggleReducedMotion } = useConsoleStore();
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

  let workspaceBody;
  if (workspace === 'incident') {
    workspaceBody = <Overview incident={incident} />;
  } else if (workspace === 'validation') {
    workspaceBody = <ValidationPage incident={incident} />;
  } else if (workspace === 'benchmarks') {
    workspaceBody = <BenchmarkPage incident={incident} />;
  } else {
    workspaceBody = (
      <div className="workspace-placeholder">
        <EmptyState
          title={`${WORKSPACE_LABELS[workspace]} has not been implemented in the mission-control shell yet.`}
          detail={NOT_YET_MIGRATED_DETAIL[workspace]}
        />
      </div>
    );
  }

  return (
    <div className={reducedMotion ? 'mission-shell reduced-motion' : 'mission-shell'}>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <MissionHeader incident={incident} />
      <ModeBanner
        incident={incident}
        onRetry={incident.mode === 'ERROR' ? () => query.refetch() : undefined}
      />
      <div className="mission-shell-body">
        <WorkflowRail incident={incident} />
        <div className="mission-shell-main">
          <WorkspaceToolbar />
          <main id="main-content" className="workspace-content">
            <Suspense
              fallback={
                <div className="page-loading" role="status">
                  Loading local console view…
                </div>
              }
            >
              {workspaceBody}
            </Suspense>
          </main>
        </div>
        <DecisionInspector incident={incident} />
      </div>
      <TechnicalDock incident={incident} />
      <footer className="mission-footer">
        <span>Decision support only · No autonomous control</span>
        <span>Exact verifier: WNTR / EPANET</span>
        <span>Operator approval required</span>
        <button
          type="button"
          className="motion-toggle"
          onClick={toggleReducedMotion}
          aria-pressed={reducedMotion}
        >
          Reduced motion {reducedMotion ? 'on' : 'off'}
        </button>
      </footer>
    </div>
  );
}
