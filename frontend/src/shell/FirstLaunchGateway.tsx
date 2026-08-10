/**
 * First-launch judge gateway (submission.txt SS5). Shown only when no
 * LIVE incident is configured and no experience/demo/failure routing
 * param is already set -- a clean installation must not silently drop a
 * judge into a confusing ERROR screen or a fully-completed illustrative
 * fallback. Compact, mission-control styled; not a marketing landing page.
 */
export function FirstLaunchGateway({
  onRunReference,
  onRunLive,
  onImportNetwork,
  onExploreFallback,
}: {
  onRunReference: () => void;
  onRunLive: () => void;
  onImportNetwork: () => void;
  onExploreFallback: () => void;
}) {
  return (
    <main className="first-launch-gateway" aria-labelledby="first-launch-title">
      <div className="first-launch-panel">
        <h1 id="first-launch-title">HydroSwarm is ready</h1>
        <p className="supporting">
          Offline mission-control decision support for drinking-water contamination incidents.
        </p>
        <div className="first-launch-actions">
          <button type="button" className="first-launch-primary" onClick={onRunReference}>
            Run Reference Incident
            <span className="first-launch-recommended">Recommended</span>
          </button>
          <button type="button" onClick={onRunLive}>
            Run Live Example
            <span className="first-launch-secondary-label">Real computation, reference inputs</span>
          </button>
          <button type="button" onClick={onImportNetwork}>
            Import Your Own Network
            <span className="first-launch-secondary-label">Advanced</span>
          </button>
        </div>
        <button type="button" className="first-launch-secondary" onClick={onExploreFallback}>
          Explore illustrative fallback
        </button>
      </div>
    </main>
  );
}
