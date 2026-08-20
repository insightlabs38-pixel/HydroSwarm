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
        <div className="first-launch-brand">
          <span className="brand-mark" aria-hidden="true">
            H
          </span>
          <span>HydroSwarm</span>
        </div>
        <h1 id="first-launch-title">
          Local incident decision support, ready when the evidence is.
        </h1>
        <p className="supporting">
          Offline mission-control decision support for drinking-water contamination incidents.
        </p>
        <ul className="first-launch-trust" aria-label="System safeguards">
          <li>Local / offline</li>
          <li>WNTR / EPANET verification</li>
          <li>Human approval required</li>
        </ul>
        <div className="first-launch-actions">
          <button type="button" className="first-launch-primary" onClick={onRunReference}>
            Run Reference Incident
            <span className="first-launch-recommended">
              Recommended · deterministic checksummed workflow replay
            </span>
          </button>
          <button type="button" onClick={onRunLive}>
            Run Live Example
            <span className="first-launch-secondary-label">
              Current HydroCore-v5 runtime · real computation on reference inputs
            </span>
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
