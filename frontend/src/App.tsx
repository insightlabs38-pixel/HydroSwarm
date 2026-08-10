import { lazy, Suspense, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { Experience } from './api/incident';
import {
  fetchIncidentWithFallback,
  hasConfiguredLiveIncident,
  requestedExperience,
} from './api/incident';
import { useConsoleStore, WORKSPACE_LABELS } from './store';
import { useReferenceIncident } from './reference/useReferenceIncident';
import { useLiveExampleFlow } from './liveExample/useLiveExampleFlow';
import { MissionHeader } from './shell/MissionHeader';
import { ModeBanner } from './shell/ModeBanner';
import { WorkflowRail } from './shell/WorkflowRail';
import { WorkspaceToolbar } from './shell/WorkspaceToolbar';
import { DecisionInspector } from './shell/DecisionInspector';
import { TechnicalDock } from './shell/TechnicalDock';
import { EmptyState } from './components/common/EmptyState';
import { FirstLaunchGateway } from './shell/FirstLaunchGateway';
import { LiveExampleProgress } from './shell/LiveExampleProgress';

const Overview = lazy(() =>
  import('./pages/Overview').then((module) => ({ default: module.Overview })),
);
const SourceWorkspace = lazy(() =>
  import('./workspaces/SourceWorkspace').then((module) => ({ default: module.SourceWorkspace })),
);
const SamplingWorkspace = lazy(() =>
  import('./workspaces/SamplingWorkspace').then((module) => ({
    default: module.SamplingWorkspace,
  })),
);
const ResponseWorkspace = lazy(() =>
  import('./workspaces/ResponseWorkspace').then((module) => ({
    default: module.ResponseWorkspace,
  })),
);
const ApprovalWorkspace = lazy(() =>
  import('./workspaces/ApprovalWorkspace').then((module) => ({
    default: module.ApprovalWorkspace,
  })),
);
const ReplayWorkspace = lazy(() =>
  import('./workspaces/ReplayWorkspace').then((module) => ({ default: module.ReplayWorkspace })),
);
const NetworkWorkspace = lazy(() =>
  import('./workspaces/NetworkWorkspace').then((module) => ({
    default: module.NetworkWorkspace,
  })),
);
const AuthorityWorkspace = lazy(() =>
  import('./workspaces/AuthorityWorkspace').then((module) => ({
    default: module.AuthorityWorkspace,
  })),
);
const ValidationPage = lazy(() =>
  import('./pages/ValidationPage').then((module) => ({ default: module.ValidationPage })),
);
const BenchmarkPage = lazy(() =>
  import('./pages/BenchmarkPage').then((module) => ({ default: module.BenchmarkPage })),
);

const NOT_YET_MIGRATED_DETAIL: Partial<Record<string, string>> = {};

/** Sets `?experience=<value>` in the URL (so a refresh/copied link
 * preserves the judge's choice -- submission.txt SS5.3) and forces a
 * re-render by also returning it for local state. */
function setExperienceParam(value: Experience): void {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  url.searchParams.set('experience', value);
  window.history.replaceState(null, '', url);
}

function hasRoutingOverride(): boolean {
  if (typeof window === 'undefined') return false;
  const params = new URLSearchParams(window.location.search);
  return params.has('failure') || params.has('demo');
}

export default function App() {
  const { workspace, setWorkspace, reducedMotion, toggleReducedMotion } = useConsoleStore();
  const [manualExperience, setManualExperience] = useState<Experience | null>(null);
  const experience = manualExperience ?? requestedExperience();
  const isReference = experience === 'reference';
  // "live" (automated Run Live Example) only runs the flow while no
  // incident is selected yet; once useLiveExampleFlow's approve() calls
  // selectIncident(), hasConfiguredLiveIncident() flips true and this
  // becomes false on the next render, falling through to the ordinary
  // LIVE liveQuery path below -- no separate "done" branch needed.
  // "import" (advanced manual network import, SUB-12.1 P1 #6) never runs
  // the automated flow at all, even though it is also headed for LIVE.
  const isLiveExampleFlow = experience === 'live' && !hasConfiguredLiveIncident();

  // First-launch gateway (submission.txt SS5): only for a genuinely
  // unconfigured, unrouted fresh install -- never overrides an explicit
  // ?experience=/?demo=/?failure= request or an already-configured LIVE
  // deployment.
  const showGateway = experience === null && !hasConfiguredLiveIncident() && !hasRoutingOverride();

  function chooseExperience(value: Experience) {
    setExperienceParam(value);
    setManualExperience(value);
  }

  const referenceController = useReferenceIncident(reducedMotion, isReference && !showGateway);
  const liveExampleController = useLiveExampleFlow(isLiveExampleFlow && !showGateway);
  const liveQuery = useQuery({
    queryKey: ['active-incident', experience],
    queryFn: ({ signal }) => fetchIncidentWithFallback(signal),
    staleTime: 5_000,
    enabled: !isReference && !isLiveExampleFlow && !showGateway,
  });

  if (showGateway) {
    return (
      <FirstLaunchGateway
        onRunReference={() => chooseExperience('reference')}
        onRunLive={() => chooseExperience('live')}
        onImportNetwork={() => {
          chooseExperience('import');
          setWorkspace('network');
        }}
        onExploreFallback={() => chooseExperience('fallback')}
      />
    );
  }

  if (isLiveExampleFlow) {
    return (
      <LiveExampleProgress
        controller={liveExampleController}
        onExploreFallback={() => chooseExperience('fallback')}
      />
    );
  }

  if (isReference && referenceController.isError) {
    return (
      <main className="loading-state" role="alert">
        <div>
          <p>Reference incident unavailable.</p>
          <p className="supporting">
            {referenceController.errorMessage ?? 'The reference-demo artifact could not be loaded.'}
          </p>
        </div>
      </main>
    );
  }

  const incident = isReference ? referenceController.incident : (liveQuery.data ?? null);
  if (!incident)
    return (
      <main className="loading-state" aria-live="polite">
        Loading local incident state…
      </main>
    );

  let workspaceBody;
  if (workspace === 'incident') {
    workspaceBody = <Overview incident={incident} />;
  } else if (workspace === 'source') {
    workspaceBody = <SourceWorkspace incident={incident} />;
  } else if (workspace === 'sampling') {
    workspaceBody = <SamplingWorkspace incident={incident} />;
  } else if (workspace === 'response') {
    workspaceBody = <ResponseWorkspace incident={incident} />;
  } else if (workspace === 'approval') {
    workspaceBody = <ApprovalWorkspace incident={incident} />;
  } else if (workspace === 'replay') {
    workspaceBody = <ReplayWorkspace incident={incident} />;
  } else if (workspace === 'network') {
    workspaceBody = <NetworkWorkspace />;
  } else if (workspace === 'authority') {
    workspaceBody = <AuthorityWorkspace incident={incident} />;
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
        onRetry={incident.mode === 'ERROR' ? () => liveQuery.refetch() : undefined}
        reference={isReference ? referenceController : undefined}
        onExploreReplay={() => setWorkspace('replay')}
      />
      <div className="mission-shell-body">
        <WorkflowRail incident={incident} />
        <div className="mission-shell-main">
          <WorkspaceToolbar />
          <main id="main-content" className="workspace-content" tabIndex={-1}>
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
